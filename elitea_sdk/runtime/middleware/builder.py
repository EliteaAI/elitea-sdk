"""Elitea Builder middleware (EliteaAI/elitea_issues#7383).

The platform's Builder tools (builder_find / read / propose / apply / reference) never pause on their own: they
return a status. This middleware turns two of them into the existing human-in-the-loop cards, so clarification and
confirmation happen the same way on every model:

* ``needs_input``        -> the Ask User ``clarifying_question`` interrupt; the answer becomes the tool result.
* ``needs_confirmation`` -> a ``builder_changeset`` interrupt (Changes card). Apply calls ``builder_apply`` with the
  changeset id; Cancel returns ``cancelled`` and nothing is written. The platform also refuses to apply a destructive
  changeset the user has not confirmed, so a model can't skip the card.

It also places the Builder core prompt in the system prompt.
"""
import json
import logging
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from .base import Middleware
from .tool_exception_handler import ToolExceptionHandlerMiddleware
from .tool_patching import rebind_invoke_after_copy
from ..tools.ask_user import ASK_USER_ANSWER_ACTION, CLARIFYING_QUESTION_GUARDRAIL

logger = logging.getLogger(__name__)

BUILDER_CHANGESET_GUARDRAIL = "builder_changeset"
APPLY_ACTION = "apply"
CANCEL_ACTION = "cancel"
_ACTION_RE = re.compile(r"(?:^|_)builder_(find|read|propose|apply|ref|reference)(?:_|$)")
_NON_INTERACTIVE = ("The user is not available to confirm in this non-interactive run, so nothing was changed. "
                    "Tell the user the change needs their confirmation in an interactive chat.")


def builder_action(tool_name: str) -> Optional[str]:
    """The Builder action of an MCP tool name (``post_elitea_core_builder_propose`` -> ``propose``), else None."""
    match = _ACTION_RE.search(str(tool_name or "").lower())
    return match.group(1) if match else None


def _parse(result: Any) -> Optional[dict]:
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _capture_pending_messages() -> List[dict]:
    """In-flight sibling tool messages, as ask_user and the sensitive-tool guard embed them in the interrupt."""
    try:
        from langchain_core.messages import message_to_dict
        from ..tools.llm import _PENDING_TOOL_MESSAGES
    except Exception:  # pragma: no cover - defensive
        return []
    serialized = []
    for msg in _PENDING_TOOL_MESSAGES.get([]) or []:
        try:
            serialized.append(message_to_dict(msg))
        except Exception:  # pylint: disable=W0703
            pass
    return serialized


def _questions(raw: List[dict]) -> List[dict]:
    questions = []
    for index, item in enumerate(raw or [], start=1):
        options = [{"label": str(o.get("label") or o.get("value")), "description": str(o.get("value", ""))}
                   for o in item.get("options") or []]
        questions.append({
            "id": f"q{index}", "question": item.get("question") or "", "header": item.get("header") or "",
            "options": options, "multiSelect": False, "allow_other": True,
        })
    return questions[:4]


def _answer_text(questions: List[dict], value: Any) -> str:
    if isinstance(value, dict):
        lines = []
        for question in questions:
            answer = value.get(question["id"])
            if isinstance(answer, list):
                answer = ", ".join(str(a) for a in answer)
            lines.append(f"- {question['question']}: {answer if answer not in (None, '') else '(no answer)'}")
        body = "\n".join(lines)
    else:
        body = str(value or "(no answer)")
    return f"User answered:\n{body}\nUse these answers and call post_elitea_core_builder_propose again."


class BuilderMiddleware(Middleware):
    def __init__(self, core_prompt: str = "", interactive: bool = True, conversation_id: Optional[str] = None,
                 **kwargs):
        super().__init__(conversation_id=conversation_id, **kwargs)
        self.core_prompt = core_prompt or ""
        self.interactive = interactive
        self._apply_tool: Optional[BaseTool] = None
        self._seen_playbooks: set = set()

    def get_tools(self) -> List[BaseTool]:
        return []

    def get_system_prompt(self) -> str:
        return self.core_prompt

    def bind_tools(self, tools: List[BaseTool]) -> None:
        """Remember builder_apply so the Changes card can apply what the user approved."""
        for tool in tools or []:
            if builder_action(getattr(tool, "name", "")) == "apply":
                self._apply_tool = tool

    # --- wrapping ------------------------------------------------------------------------------------------------

    def wrap_tool(self, tool: BaseTool) -> BaseTool:
        action = builder_action(getattr(tool, "name", ""))
        if action is None:
            return tool
        if action == "apply":
            self._apply_tool = tool
        if action not in ("propose", "apply", "find", "read"):
            return tool
        sync_target = ToolExceptionHandlerMiddleware._sync_target(tool)
        async_target = ToolExceptionHandlerMiddleware._async_target(tool)
        if sync_target is None and async_target is None:
            return tool
        copied = tool.model_copy()
        rebind_invoke_after_copy(copied)
        if sync_target is not None:
            object.__setattr__(copied, sync_target[0], self._sync(sync_target[1], tool.name))
        if async_target is not None:
            object.__setattr__(copied, async_target[0], self._async(async_target[1], tool.name))
        return copied

    def _sync(self, original: Callable, tool_name: str) -> Callable:
        @wraps(original)
        def run(*args, **kwargs):
            return self._handle(original(*args, **kwargs), tool_name, kwargs)
        return run

    def _async(self, original: Callable, tool_name: str) -> Callable:
        @wraps(original)
        async def run(*args, **kwargs):
            return self._handle(await original(*args, **kwargs), tool_name, kwargs)
        return run

    # --- statuses ------------------------------------------------------------------------------------------------

    def _handle(self, result: Any, tool_name: str, tool_args: dict) -> Any:
        parsed = _parse(result)
        if not parsed:
            return result
        playbook = parsed.get("playbook")
        if playbook:
            if playbook in self._seen_playbooks:
                parsed = {k: v for k, v in parsed.items() if k != "playbook"}
                stripped = json.dumps(parsed)
                result = (stripped, *result[1:]) if isinstance(result, tuple) else stripped
            else:
                self._seen_playbooks.add(playbook)
        status = parsed.get("status")
        if status == "needs_input":
            return self._ask(parsed, result, tool_name, tool_args)
        if status == "needs_confirmation":
            return self._confirm(parsed, tool_name, tool_args)
        return result

    @staticmethod
    def _resume_fields(tool_name: str, tool_args: dict) -> dict:
        """What the graph resume needs to re-issue this same Builder call without a new LLM turn."""
        fields = {"builder_tool_name": tool_name, "tool_args": dict(tool_args or {})}
        pending = _capture_pending_messages()
        if pending:
            fields["_pending_messages"] = pending
        return fields

    def _ask(self, parsed: Dict, original: Any, tool_name: str, tool_args: dict) -> Any:
        questions = _questions(parsed.get("questions"))
        if not questions or not self.interactive:
            return original
        payload = {
            "type": "hitl", "interrupt_id": f"hitl_{uuid4().hex}", "guardrail_type": CLARIFYING_QUESTION_GUARDRAIL,
            "builder": True, "node_name": "builder", "tool_name": tool_name, "message": questions[0]["question"],
            "questions": questions, "available_actions": [ASK_USER_ANSWER_ACTION], "routes": {},
            **self._resume_fields(tool_name, tool_args),
        }
        logger.info("[BUILDER] needs_input -> clarifying question (%d)", len(questions))
        value = interrupt(payload)
        if isinstance(value, dict) and "value" in value and "action" in value:
            value = value["value"]
        return _answer_text(questions, value)

    def _confirm(self, parsed: Dict, tool_name: str, tool_args: dict) -> str:
        changeset_id = parsed.get("changeset_id")
        if not self.interactive:
            return json.dumps({**parsed, "message": _NON_INTERACTIVE})
        payload = {
            "type": "hitl", "interrupt_id": f"hitl_{uuid4().hex}", "guardrail_type": BUILDER_CHANGESET_GUARDRAIL,
            "node_name": "builder", "tool_name": tool_name, "changeset_id": changeset_id,
            "builder_project_id": parsed.get("project_id"),
            "message": parsed.get("summary") or "Confirm this change", "summary": parsed.get("summary") or "",
            "diff": parsed.get("diff") or [], "arguments": parsed.get("arguments"),
            "available_actions": [APPLY_ACTION, CANCEL_ACTION], "routes": {},
            **self._resume_fields(tool_name, tool_args),
        }
        logger.info("[BUILDER] needs_confirmation -> Changes card (changeset=%s)", changeset_id)
        decision = interrupt(payload)
        action = (decision or {}).get("action") if isinstance(decision, dict) else decision
        if action != APPLY_ACTION:
            return json.dumps({"status": "cancelled", "changeset_id": changeset_id,
                               "message": "The user cancelled the change; nothing was changed."})
        if self._apply_tool is None:
            return json.dumps({"status": "error", "message": "builder_apply is not available in this chat."})
        return self._apply_tool.invoke({"changeset_id": changeset_id})
