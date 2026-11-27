"""EL-7679: invoke_structured — one model call that returns a JSON object for a schema.

Method order (plan.md, option B):
- native Anthropic / OpenAI: one forced tool whose input schema is the caller's schema;
- Anthropic with thinking: native ``json_schema`` (forced tool_choice is rejected with thinking);
- OpenAI-compatible passthrough: no tools, JSON prompt;
- a 400 transform rejection or a reply without the tool call: JSON prompt.
"""
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from elitea_sdk.runtime.langchain.structured import TOOL_NAME, invoke_structured

SCHEMA = {
    "title": "Draft",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Agent name"},
        "instructions": {"type": "string"},
    },
    "required": ["name", "instructions"],
}
MESSAGES = [SystemMessage(content="Build an agent."), HumanMessage(content="Business value calculator")]


class _FakeBadRequest(Exception):
    pass


class _Recorder:
    """Chat-model stub recording bind_tools / with_structured_output / invoke."""

    _module = "tests.fake"

    def __init__(self, replies=None, bound_replies=None, structured_reply=None, bound_error=None):
        self.replies = list(replies or [])
        self.bound_replies = list(bound_replies or [])
        self.structured_reply = structured_reply
        self.bound_error = bound_error
        self.bind_calls = []
        self.structured_calls = []
        self.invocations = []

    def bind_tools(self, tools, **kwargs):
        self.bind_calls.append((tools, kwargs))
        recorder = self

        class _Bound:
            bound = recorder

            def invoke(self, messages, config=None):
                recorder.invocations.append(("bound", list(messages)))
                if recorder.bound_error:
                    raise recorder.bound_error
                return recorder.bound_replies.pop(0)

        return _Bound()

    def with_structured_output(self, schema, method="function_calling", **kwargs):
        self.structured_calls.append((schema, method))
        recorder = self

        class _Struct:
            def invoke(self, messages, config=None):
                recorder.invocations.append(("structured", list(messages)))
                return recorder.structured_reply

        return _Struct()

    def invoke(self, messages, config=None):
        self.invocations.append(("plain", list(messages)))
        return self.replies.pop(0)


def _anthropic(thinking=None, **kwargs):
    cls = type("ChatAnthropic", (_Recorder,), {"__module__": "langchain_anthropic.chat_models"})
    llm = cls(**kwargs)
    llm.thinking = thinking
    return llm


def _passthrough(**kwargs):
    llm = _Recorder(**kwargs)
    llm._elitea_openai_compatible = True
    return llm


def _tool_call_reply(args):
    return AIMessage(content="", tool_calls=[{"name": TOOL_NAME, "args": args, "id": "call_1"}])


VALID = {"name": "Business Value Calculator", "instructions": "Compute ROI."}


# AC-3: native client — one forced tool with the caller's schema; data from tool-call args
class TestForcedTool:
    def test_native_client_forces_single_tool_and_reads_args(self):
        llm = _Recorder(bound_replies=[_tool_call_reply(VALID)])

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert len(llm.bind_calls) == 1
        tools, kwargs = llm.bind_calls[0]
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == TOOL_NAME
        assert tools[0]["function"]["parameters"]["properties"] == SCHEMA["properties"]
        assert kwargs["tool_choice"] == TOOL_NAME
        assert result.data == VALID
        assert result.method == "tool"

    def test_anthropic_without_thinking_forces_tool(self):
        llm = _anthropic(thinking=None, bound_replies=[_tool_call_reply(VALID)])

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert llm.bind_calls and llm.bind_calls[0][1]["tool_choice"] == TOOL_NAME
        assert llm.structured_calls == []
        assert result.method == "tool" and result.data == VALID

    def test_no_tool_call_in_reply_falls_back_to_json_prompt(self):
        llm = _Recorder(
            bound_replies=[AIMessage(content="Sure! Here is your agent.")],
            replies=[AIMessage(content='{"name": "X", "instructions": "Y"}')],
        )

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.method == "json_prompt"
        assert result.data == {"name": "X", "instructions": "Y"}


# AC-4: Anthropic thinking — native json_schema, no forced tool_choice
class TestAnthropicThinking:
    @pytest.mark.parametrize("thinking_type", ["enabled", "adaptive"])
    def test_thinking_uses_json_schema(self, thinking_type):
        llm = _anthropic(thinking={"type": thinking_type}, structured_reply=dict(VALID))

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert llm.bind_calls == []
        assert [m for _, m in llm.structured_calls] == ["json_schema"]
        assert result.method == "native_schema"
        assert result.data == VALID


# Option B: OpenAI-compatible passthrough — no tools at all, JSON prompt
class TestPassthrough:
    def test_passthrough_sends_no_tools_and_uses_json_prompt(self):
        llm = _passthrough(replies=[AIMessage(content='```json\n{"name": "X", "instructions": "Y"}\n```')])

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert llm.bind_calls == [] and llm.structured_calls == []
        assert result.method == "json_prompt"
        assert result.data == {"name": "X", "instructions": "Y"}
        sent = llm.invocations[0][1]
        assert "JSON" in sent[-1].content and "name" in sent[-1].content


# AC-5: provider 400-rejects the tool transform — JSON prompt on the same request
class TestTransformRejection:
    def test_tool_choice_rejection_falls_back(self):
        llm = _Recorder(
            bound_error=_FakeBadRequest("Error code: 400 - BedrockException - tool_choice.type: Field required"),
            replies=[AIMessage(content='{"name": "X", "instructions": "Y"}')],
        )

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.method == "json_prompt"
        assert result.data == {"name": "X", "instructions": "Y"}

    def test_other_400_propagates(self):
        llm = _Recorder(bound_error=_FakeBadRequest("Error code: 400 - context length exceeded"))

        with pytest.raises(_FakeBadRequest):
            invoke_structured(llm, MESSAGES, SCHEMA)


# AC-9: list content — text blocks and bare strings joined, thinking skipped
class TestListContent:
    def test_json_prompt_text_skips_thinking_blocks(self):
        llm = _passthrough(replies=[AIMessage(content=[
            {"type": "thinking", "thinking": "let me think {not json}"},
            {"type": "text", "text": '{"name": "X", '},
            '"instructions": "Y"}',
        ])])

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.data == {"name": "X", "instructions": "Y"}


class TestFinishReason:
    def test_finish_reason_is_reported(self):
        reply = AIMessage(content='{"name": "X"', response_metadata={"finish_reason": "length"})
        llm = _passthrough(replies=[reply])

        result = invoke_structured(llm, MESSAGES, SCHEMA, max_attempts=1)

        assert result.finish_reason == "length"

    def test_anthropic_stop_reason_is_reported(self):
        reply = AIMessage(content='{"name": "X"', response_metadata={"stop_reason": "max_tokens"})
        llm = _passthrough(replies=[reply])

        assert invoke_structured(llm, MESSAGES, SCHEMA, max_attempts=1).finish_reason == "max_tokens"


# client.predict_agent(output_schema=...): a runnable with the agent's invoke contract
class TestPredictAgentStructured:
    def _client(self):
        from elitea_sdk.runtime.clients import client as client_module
        return object.__new__(client_module.EliteAClient)

    def test_returns_structured_runnable_and_answers_with_json_text(self, monkeypatch):
        from elitea_sdk.runtime.langchain import structured
        seen = {}

        def fake_invoke(llm, messages, schema, config=None, max_attempts=3):
            seen.update(messages=messages, schema=schema, config=config)
            return structured.StructuredResult(data=VALID, text=json.dumps(VALID), method="tool")

        monkeypatch.setattr(structured, 'invoke_structured', fake_invoke)
        runnable = self._client().predict_agent(llm=object(), instructions='sys', output_schema=SCHEMA)

        history = [HumanMessage(content='earlier'), AIMessage(content='reply')]
        response = runnable.invoke({"messages": history + [HumanMessage(content='u')]},
                                   {"configurable": {"thread_id": "t1"}, "callbacks": []})

        kinds = [(type(m).__name__, m.content) for m in seen['messages']]
        assert kinds == [('SystemMessage', 'sys'), ('HumanMessage', 'earlier'), ('AIMessage', 'reply'),
                         ('HumanMessage', 'u')]
        assert seen['schema'] == SCHEMA
        assert json.loads(response["output"]) == VALID
        assert response["thread_id"] == "t1"
        assert isinstance(response["messages"][-1], AIMessage)

    def test_failed_attempts_answer_with_last_text(self, monkeypatch):
        from elitea_sdk.runtime.langchain import structured
        monkeypatch.setattr(structured, 'invoke_structured', lambda *a, **k: structured.StructuredResult(
            data=None, text=PROSE, method="json_prompt", attempts=3))
        runnable = self._client().predict_agent(llm=object(), instructions='sys', output_schema=SCHEMA)

        response = runnable.invoke({"messages": [HumanMessage(content='u')]}, {})

        assert response["output"] == PROSE


# AC-6 / AC-7: correction retries inside the SDK
PROSE = "Sure! I updated the role to a Business value calculator agent."


class TestRetry:
    def test_prose_then_valid_on_passthrough(self):
        llm = _passthrough(replies=[AIMessage(content=PROSE), AIMessage(content='{"name": "X", "instructions": "Y"}')])

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.data == {"name": "X", "instructions": "Y"}
        assert result.attempts == 2 and result.method == "json_prompt"
        second = llm.invocations[1][1]
        assert isinstance(second[-2], AIMessage) and second[-2].content == PROSE
        assert "no JSON object" in second[-1].content

    def test_missing_field_then_valid(self):
        llm = _passthrough(replies=[AIMessage(content='{"name": "X"}'),
                                    AIMessage(content='{"name": "X", "instructions": "Y"}')])

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.data == {"name": "X", "instructions": "Y"} and result.attempts == 2
        assert "instructions" in llm.invocations[1][1][-1].content

    def test_tool_args_missing_field_are_corrected(self):
        llm = _Recorder(bound_replies=[_tool_call_reply({"name": "X"}), _tool_call_reply(VALID)])

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.data == VALID and result.attempts == 2 and result.method == "tool"

    def test_cut_off_asks_for_shorter_answer(self):
        llm = _passthrough(replies=[
            AIMessage(content='{"name": "X", "instructions": "lo', response_metadata={"finish_reason": "length"}),
            AIMessage(content='{"name": "X", "instructions": "Y"}'),
        ])

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.data is not None
        assert "cut off" in llm.invocations[1][1][-1].content

    def test_bad_reply_echo_is_capped(self):
        from elitea_sdk.runtime.langchain.structured import MAX_ECHO_CHARS
        llm = _passthrough(replies=[AIMessage(content="x" * 10_000), AIMessage(content=json.dumps(VALID))])

        invoke_structured(llm, MESSAGES, SCHEMA)

        assert len(llm.invocations[1][1][-2].content) <= MAX_ECHO_CHARS + 1

    def test_gives_up_after_max_attempts_and_returns_last_text(self):
        llm = _passthrough(replies=[AIMessage(content=PROSE)] * 3)

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.data is None and result.attempts == 3 and result.text == PROSE
        assert len(llm.invocations) == 3

    def test_max_attempts_is_configurable(self):
        llm = _passthrough(replies=[AIMessage(content=PROSE)])

        result = invoke_structured(llm, MESSAGES, SCHEMA, max_attempts=1)

        assert result.attempts == 1 and len(llm.invocations) == 1

    def test_fenced_json_is_accepted(self):
        llm = _passthrough(replies=[AIMessage(content='Here:\n```json\n{"name": "X", "instructions": "Y"}\n```')])

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.data == {"name": "X", "instructions": "Y"} and result.attempts == 1


# Persona: a structured run sends the endpoint's instructions verbatim (no EliteA chat template, #7467)
def test_structured_run_has_no_persona_template(monkeypatch):
    from elitea_sdk.runtime.clients import client as client_module
    from elitea_sdk.runtime.langchain import structured
    seen = {}
    monkeypatch.setattr(structured, 'invoke_structured', lambda llm, messages, schema, **k: (
        seen.setdefault('messages', messages), structured.StructuredResult(data=VALID, text='', method='tool'))[1])

    runnable = object.__new__(client_module.EliteAClient).predict_agent(
        llm=object(), instructions='Reply with one JSON object.', output_schema=SCHEMA, persona='generic')
    runnable.invoke({"messages": [HumanMessage(content='u')]}, {})

    system = seen['messages'][0]
    assert isinstance(system, SystemMessage) and system.content == 'Reply with one JSON object.'
    assert 'EliteA' not in system.content


# Code review fixes
class TestReviewFixes:
    def test_shape_check_accepts_values_the_endpoint_model_coerces(self):
        """#2: extra keys, wrong scalar types and nested shapes are the endpoint's to coerce or reject."""
        schema = {**SCHEMA, "properties": {**SCHEMA["properties"], "count": {"type": "integer"},
                                           "items": {"type": "array", "items": {"type": "object"}}}}
        loose = {"name": "X", "instructions": "Y", "count": "12", "items": [None, "mcp"], "extra": True}
        llm = _passthrough(replies=[AIMessage(content=json.dumps(loose))])

        result = invoke_structured(llm, MESSAGES, schema)

        assert result.data == loose and result.attempts == 1

    def test_required_field_set_to_null_is_corrected(self):
        llm = _passthrough(replies=[AIMessage(content='{"name": null, "instructions": "Y"}'),
                                    AIMessage(content=json.dumps(VALID))])

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.data == VALID and result.attempts == 2
        assert "name" in llm.invocations[1][1][-1].content

    def test_structured_module_does_not_import_jsonschema(self):
        """#8: jsonschema is not a declared SDK dependency."""
        import elitea_sdk.runtime.langchain.structured as module
        source = open(module.__file__).read()
        assert "jsonschema" not in source

    def test_after_a_rejection_later_attempts_skip_the_tool(self):
        """#3: one rejected call per run, not one per attempt."""
        llm = _Recorder(
            bound_error=_FakeBadRequest("Error code: 400 - tool_choice.type: Field required"),
            replies=[AIMessage(content=PROSE), AIMessage(content=json.dumps(VALID))],
        )

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.data == VALID and result.attempts == 2
        assert [kind for kind, _ in llm.invocations] == ["bound", "plain", "plain"]

    def test_tool_call_cut_off_by_max_tokens_asks_for_shorter_answer(self):
        """#6: a truncated tool call is 'cut off', not 'no tool call'."""
        truncated = AIMessage(content="", invalid_tool_calls=[{"name": TOOL_NAME, "args": '{"name": "X", "instr',
                                                               "id": "c1", "error": "json"}],
                              response_metadata={"stop_reason": "max_tokens"})
        llm = _Recorder(bound_replies=[truncated, _tool_call_reply(VALID)])

        result = invoke_structured(llm, MESSAGES, SCHEMA)

        assert result.data == VALID and result.attempts == 2
        assert [kind for kind, _ in llm.invocations] == ["bound", "bound"]
        assert "cut off" in llm.invocations[1][1][-1].content

    @pytest.mark.parametrize("reason", ["MAX_TOKENS", "max_tokens", "length", "Length"])
    def test_length_stop_reasons_any_case(self, reason):
        """#7: Gemini reports MAX_TOKENS."""
        llm = _passthrough(replies=[AIMessage(content='{"name": "X", "ins', response_metadata={"finish_reason": reason}),
                                    AIMessage(content=json.dumps(VALID))])

        invoke_structured(llm, MESSAGES, SCHEMA)

        assert "cut off" in llm.invocations[1][1][-1].content

    def test_runnable_reports_structured_ok(self, monkeypatch):
        """#1: the worker needs to know the run recovered from a rejected call."""
        from elitea_sdk.runtime.langchain import structured
        monkeypatch.setattr(structured, 'invoke_structured', lambda *a, **k: structured.StructuredResult(
            data=VALID, text='', method="json_prompt", attempts=1))
        ok = structured.StructuredAnswerRunnable(llm=object(), instructions='s', schema=SCHEMA).invoke(
            {"messages": [HumanMessage(content='u')]}, {})
        monkeypatch.setattr(structured, 'invoke_structured', lambda *a, **k: structured.StructuredResult(
            data=None, text=PROSE, method="json_prompt", attempts=3))
        failed = structured.StructuredAnswerRunnable(llm=object(), instructions='s', schema=SCHEMA).invoke(
            {"messages": [HumanMessage(content='u')]}, {})

        assert ok["structured_ok"] is True and failed["structured_ok"] is False
