"""One model call that returns a JSON object for a caller's JSON schema (EL-7679).

Method order:

- native clients: one tool whose input schema is the caller's schema, forced with ``tool_choice``;
  the object is read from the tool-call arguments;
- Anthropic with thinking: native ``json_schema`` (Anthropic rejects forced ``tool_choice`` with
  thinking), unless the schema has an open object, which ``json_schema`` empties (#7623);
- OpenAI-compatible passthroughs (Claude via LiteLLM ``/chat/completions``): no tools, JSON prompt,
  because they 400-reject the forced transform (#250);
- a 400 transform rejection, an unparseable structured reply or a reply without the tool call:
  JSON prompt.

Every answer is checked for shape: a JSON object with the schema's required fields. Full validation
is the caller's (its model coerces and filters values). A reply without such an object is sent back
with a correction (no JSON / invalid JSON / missing fields / cut off), up to ``max_attempts``; after
the last attempt the reply text is returned so the caller reports it as it always has. After one
provider rejection the run stays on the JSON prompt.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .utils import (
    extract_json_content,
    has_open_object,
    make_anthropic_compatible_schema_dict,
    prepare_messages_for_model,
)

logger = logging.getLogger(__name__)

TOOL_NAME = "submit_result"
TOOL_DESCRIPTION = "Submit the final result. Call this exactly once with the complete result."

JSON_PROMPT_TEMPLATE = (
    "\n\n**IMPORTANT: Respond with ONLY one valid JSON object that matches this JSON schema:**\n"
    "{schema}\n\n"
    "Rules:\n"
    "1. Output ONLY the JSON object, starting with {{ and ending with }}: no explanations, no text before or after\n"
    "2. Include every required field\n"
    "3. Use proper JSON syntax with double quotes for strings\n"
    "4. Do not wrap the JSON in code blocks or backticks"
)

MAX_ATTEMPTS = 3
MAX_ECHO_CHARS = 2000
MAX_ERROR_CHARS = 600
LENGTH_STOP_REASONS = {'length', 'max_tokens', 'max_output_tokens', 'stop_reason_max_tokens'}  # lower-cased

NO_JSON_CORRECTION = (
    "Your reply contained no JSON object. Reply again with only the JSON object, starting with { and "
    "ending with }, with no text before or after it."
)
INVALID_JSON_CORRECTION = "Your reply was not valid JSON: {error}. Reply again with only the corrected JSON object."
SCHEMA_CORRECTION = (
    "The JSON is missing required fields: {error}. Reply again with only the complete JSON object."
)
CUT_OFF_CORRECTION = (
    "Your reply was cut off before the JSON object was complete. Reply again with a shorter JSON object: "
    "keep every required field and shorten long texts."
)

_TRANSFORM_MARKERS = ('tool_choice', 'response_format', 'json_schema', 'output_format')


@dataclass
class StructuredResult:
    data: Optional[dict]  # the schema-valid object, or None after the last failed attempt
    text: Optional[str]  # the last reply as text (what a caller reports when ``data`` is None)
    method: str  # "tool" | "native_schema" | "json_prompt" — of the last attempt
    finish_reason: Optional[str] = None
    attempts: int = 1


def candidates(client: Any) -> list:
    """``client`` plus its ``.bound`` (a tool-bound ``RunnableBinding`` wraps the real model)."""
    try:
        bound = getattr(client, 'bound', None)
        return [client, bound] if bound is not None else [client]
    except Exception:  # pragma: no cover — defensive
        return [client]


def is_anthropic_client(client: Any) -> bool:
    return any('langchain_anthropic' in (getattr(type(c), '__module__', '') or '') for c in candidates(client))


def is_openai_compatible_client(client: Any) -> bool:
    """True for a passthrough client stamped ``_elitea_openai_compatible`` by ``EliteAClient.get_llm``."""
    return any(getattr(c, '_elitea_openai_compatible', False) for c in candidates(client))


def is_anthropic_thinking_client(client: Any) -> bool:
    for candidate in candidates(client):
        if 'langchain_anthropic' not in (getattr(type(candidate), '__module__', '') or ''):
            continue
        thinking = getattr(candidate, 'thinking', None)
        if isinstance(thinking, dict) and thinking.get('type') in ('enabled', 'adaptive'):
            return True
    return False


def is_structured_transform_rejection(exc: Exception) -> bool:
    """True when a provider 400-rejected the ``tool_choice`` / ``response_format`` / ``json_schema`` transform
    (e.g. Bedrock ``tool_choice.type``, Azure ``tool_choice.function``)."""
    msg = str(getattr(exc, 'message', '') or exc).lower()
    is_bad_request = (
        'badrequest' in type(exc).__name__.lower()
        or 'badrequesterror' in msg
        or '400' in msg
        or 'invalid_request_error' in msg
    )
    return is_bad_request and any(marker in msg for marker in _TRANSFORM_MARKERS)


def structured_text(content: Any) -> str:
    """Text blocks and bare strings of a reply, joined; thinking / reasoning blocks are skipped.

    With thinking on, Claude can return the answer as a bare string beside the thinking block
    (``['', {'type': 'thinking', ...}, '{...}']``) instead of a ``{'type': 'text'}`` block (#7623).
    """
    if not isinstance(content, list):
        return str(content)
    return ''.join(
        block if isinstance(block, str) else block.get('text', '')
        for block in content
        if isinstance(block, str) or (isinstance(block, dict) and block.get('type') == 'text')
    )


def _finish_reason(message: Any) -> Optional[str]:
    metadata = getattr(message, 'response_metadata', None) or {}
    return metadata.get('finish_reason') or metadata.get('stop_reason')


def _tool(schema: dict) -> dict:
    return {"type": "function", "function": {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": schema}}


def _with_json_instruction(messages: List, schema: dict) -> List:
    instruction = JSON_PROMPT_TEMPLATE.format(schema=json.dumps(schema, ensure_ascii=False))
    prompt = list(messages)
    last = prompt[-1] if prompt else None
    if isinstance(last, HumanMessage) and isinstance(last.content, str):
        prompt[-1] = HumanMessage(content=last.content + instruction)
    else:
        prompt.append(HumanMessage(content=instruction))
    return prompt


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _missing_required(data: dict, schema: dict) -> List[str]:
    """Top-level required fields that are absent or null.

    Shape only: the caller's model coerces and filters values (``"12"`` → 12, unknown items dropped),
    so full JSON-schema validation here would reject answers the caller accepts.
    """
    return [name for name in schema.get("required") or () if data.get(name) is None]


@dataclass
class _Reply:
    data: Optional[dict]
    text: str
    method: str
    finish_reason: Optional[str]
    rejected: bool = False  # the provider 400-rejected the tool / schema transform on this call


def _call_json_prompt(llm, messages, schema, config) -> _Reply:
    reply = llm.invoke(prepare_messages_for_model(_with_json_instruction(messages, schema), model=llm), config=config)
    text = structured_text(getattr(reply, 'content', reply)).strip()
    return _Reply(data=None, text=text, method="json_prompt", finish_reason=_finish_reason(reply))


def _call_once(llm, messages, schema, wire_schema, config, json_only: bool) -> _Reply:
    """One model call by the best method this client allows; ``data`` is set when it returned an object."""
    if json_only:
        return _call_json_prompt(llm, messages, schema, config)
    prepared = prepare_messages_for_model(messages, model=llm)

    if is_anthropic_client(llm) and is_anthropic_thinking_client(llm):
        try:
            data = llm.with_structured_output(wire_schema, method='json_schema').invoke(prepared, config=config)
        except Exception as exc:
            if not (isinstance(exc, OutputParserException) or is_structured_transform_rejection(exc)):
                raise
            logger.warning("Structured output via json_schema failed; using the JSON prompt")
            reply = _call_json_prompt(llm, messages, schema, config)
            reply.rejected = not isinstance(exc, OutputParserException)
            return reply
        if isinstance(data, dict):
            return _Reply(data=data, text=json.dumps(data, ensure_ascii=False), method="native_schema", finish_reason=None)
        return _call_json_prompt(llm, messages, schema, config)

    try:
        reply = llm.bind_tools([_tool(wire_schema)], tool_choice=TOOL_NAME).invoke(prepared, config=config)
    except Exception as exc:
        if not is_structured_transform_rejection(exc):
            raise
        logger.warning("Forced tool call rejected by the provider; using the JSON prompt")
        reply = _call_json_prompt(llm, messages, schema, config)
        reply.rejected = True
        return reply
    finish_reason = _finish_reason(reply)
    for call in getattr(reply, 'tool_calls', None) or []:
        if call.get('name') == TOOL_NAME and isinstance(call.get('args'), dict):
            return _Reply(data=call['args'], text=json.dumps(call['args'], ensure_ascii=False), method="tool",
                          finish_reason=finish_reason)
    if (finish_reason or "").lower() in LENGTH_STOP_REASONS:
        # the arguments were cut off (they land in invalid_tool_calls): ask for a shorter answer next
        partial = next((c.get('args') for c in getattr(reply, 'invalid_tool_calls', None) or []
                        if isinstance(c.get('args'), str)), '')
        return _Reply(data=None, text=partial, method="tool", finish_reason=finish_reason)
    logger.warning("Model answered without the forced tool call; using the JSON prompt")
    return _call_json_prompt(llm, messages, schema, config)


def _is_cut_off(reply: _Reply) -> bool:
    return (reply.finish_reason or "").lower() in LENGTH_STOP_REASONS


def _check(reply: _Reply, schema: dict):
    """(object, None) when the reply holds an object with the required fields, else (None, correction)."""
    data = reply.data
    if data is None:
        if _is_cut_off(reply):
            return None, CUT_OFF_CORRECTION
        if "{" not in reply.text:
            return None, NO_JSON_CORRECTION
        try:
            data = extract_json_content(reply.text)
        except (ValueError, json.JSONDecodeError) as exc:
            return None, INVALID_JSON_CORRECTION.format(error=_truncate(str(exc), MAX_ERROR_CHARS))
        if not isinstance(data, dict):
            return None, INVALID_JSON_CORRECTION.format(error="the reply is not a JSON object")
    missing = _missing_required(data, schema)
    if missing:
        return None, SCHEMA_CORRECTION.format(error=_truncate(", ".join(missing), MAX_ERROR_CHARS))
    return data, None


def invoke_structured(llm: Any, messages: List, schema: dict, config: Any = None,
                      max_attempts: int = MAX_ATTEMPTS) -> StructuredResult:
    """Ask ``llm`` for one object matching ``schema``, correcting and retrying up to ``max_attempts``.

    See the module docstring for the method order. Each failed attempt adds the model's reply (capped)
    and a correction naming the error; the last reply's text is returned when every attempt fails.
    """
    anthropic = is_anthropic_client(llm)
    wire_schema = make_anthropic_compatible_schema_dict(schema) if anthropic else schema
    json_only = is_openai_compatible_client(llm) or (
        anthropic and is_anthropic_thinking_client(llm) and has_open_object(wire_schema)
    )

    conversation = list(messages)
    reply = None
    for attempt in range(1, max_attempts + 1):
        reply = _call_once(llm, conversation, schema, wire_schema, config, json_only)
        json_only = json_only or reply.rejected  # one rejected call per run, not per attempt
        data, correction = _check(reply, schema)
        logger.info("structured output: attempt=%d method=%s ok=%s", attempt, reply.method, data is not None)
        if data is not None:
            return StructuredResult(data=data, text=reply.text, method=reply.method,
                                    finish_reason=reply.finish_reason, attempts=attempt)
        conversation += [AIMessage(content=_truncate(reply.text, MAX_ECHO_CHARS)), HumanMessage(content=correction)]
    return StructuredResult(data=None, text=reply.text, method=reply.method,
                            finish_reason=reply.finish_reason, attempts=max_attempts)


class StructuredAnswerRunnable:
    """What ``predict_agent(output_schema=...)`` returns: one structured answer, no tools, no graph.

    ``invoke`` takes the agent's input (``{"messages": [...]}``) and returns the keys the worker reads
    from an agent run (``output``, ``messages``, ``thread_id``). ``output`` is the validated object as
    JSON text, or the last reply text when every attempt failed — so callers parse it as before.
    """

    def __init__(self, llm: Any, instructions: str, schema: dict, max_attempts: int = MAX_ATTEMPTS):
        self.llm = llm
        self.instructions = instructions
        self.schema = schema
        self.max_attempts = max_attempts

    def invoke(self, input: dict, config: Any = None) -> dict:  # pylint: disable=W0622
        history = list((input or {}).get("messages") or [])
        messages = ([SystemMessage(content=self.instructions)] if self.instructions else []) + history
        result = invoke_structured(self.llm, messages, self.schema, config=config, max_attempts=self.max_attempts)
        output = json.dumps(result.data, ensure_ascii=False) if result.data is not None else (result.text or "")
        thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
        # structured_ok: a provider rejection the run recovered from must not fail it (the callback
        # records every failed call as the run's error)
        return {"output": output, "messages": history + [AIMessage(content=output)], "thread_id": thread_id,
                "structured_ok": result.data is not None}
