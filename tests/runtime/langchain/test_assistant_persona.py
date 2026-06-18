"""Tests for persona-driven system-prompt composition (issue #5264).

Covers the 'bare' persona (no Elitea identity, functional addons only) and asserts
that existing personas (generic/none) are unchanged. Exercises
``Assistant._compose_system_prompt`` directly via a lightweight instance so we don't
need a full EliteAClient / LLM client just to assert the prompt string.
"""
import pytest
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from elitea_sdk.runtime.langchain.assistant import (
    Assistant,
    APP_TYPE_PREDICT,
    APP_TYPE_AGENT,
)
from elitea_sdk.runtime.langchain.constants import (
    DEFAULT_ASSISTANT,
    PLAN_ADDON,
    TASK_DELEGATION_ADDON,
)


def _make_assistant(persona, app_type=APP_TYPE_PREDICT, middleware_prompt=""):
    """Build an Assistant without running the heavyweight __init__.

    ``_compose_system_prompt`` only reads ``persona``, ``app_type`` and
    ``_middleware_prompt`` off ``self``, so we set just those.
    """
    a = object.__new__(Assistant)
    a.persona = persona
    a.app_type = app_type
    a._middleware_prompt = middleware_prompt
    return a


def _compose(assistant, prompt_instructions="", tool_names=None, agent_tools=None):
    tool_names = tool_names or []
    # simple_tools only needs to be truthy/length-meaningful for the FILE_HANDLING gate;
    # the helper iterates tool_names (not simple_tools) for addon selection.
    simple_tools = [object() for _ in tool_names]
    return assistant._compose_system_prompt(
        prompt_instructions, simple_tools, tool_names, agent_tools or []
    )


# --- bare persona -----------------------------------------------------------

def test_bare_no_instructions_no_tools_is_empty():
    assistant = _make_assistant("bare")
    assert _compose(assistant, prompt_instructions="") == ""


def test_bare_with_custom_instructions_only():
    assistant = _make_assistant("bare")
    out = _compose(assistant, prompt_instructions="You are a JSON formatter.")
    assert out == "You are a JSON formatter."


def test_bare_does_not_include_elitea_identity():
    assistant = _make_assistant("bare")
    out = _compose(assistant, prompt_instructions="some instructions")
    assert "EliteA" not in out
    assert "elitea.ai" not in out


def test_bare_keeps_functional_addons_when_tools_present():
    # planning tool present -> PLAN_ADDON should be injected; agent tool -> delegation.
    assistant = _make_assistant("bare")
    out = _compose(
        assistant,
        prompt_instructions="Custom.",
        tool_names=["update_plan"],
        agent_tools=[object()],
    )
    assert "## Planning" in out          # from PLAN_ADDON
    assert "Sub-agent delegation" in out  # from TASK_DELEGATION_ADDON
    assert "Custom." in out
    assert "EliteA" not in out            # still no Elitea identity


def test_bare_addons_only_when_no_instructions():
    # No custom instructions, but a planning tool is bound -> addon-only prompt.
    assistant = _make_assistant("bare")
    out = _compose(assistant, prompt_instructions="", tool_names=["update_plan"])
    assert out  # not empty
    assert "## Planning" in out
    assert "EliteA" not in out


# --- unchanged personas -----------------------------------------------------

def test_none_persona_still_uses_default_assistant():
    # 'none' is not in persona_templates and instructions are empty -> DEFAULT_ASSISTANT.
    assistant = _make_assistant("none")
    out = _compose(assistant, prompt_instructions="")
    assert "EliteA" in out


def test_generic_persona_still_uses_default_assistant():
    assistant = _make_assistant("generic")
    out = _compose(assistant, prompt_instructions="")
    assert "EliteA" in out


def test_agent_branch_unchanged_for_non_bare():
    # agent/predict + instructions (non-bare) -> own instructions, no Elitea identity.
    assistant = _make_assistant("generic", app_type=APP_TYPE_AGENT)
    out = _compose(assistant, prompt_instructions="Do the thing.")
    assert out.startswith("Do the thing.")
    assert "EliteA" not in out


# --- end-to-end: empty system content omits the SystemMessage (llm.py) -------
#
# Drives a real compiled graph (Assistant.runnable() -> create_graph) with a
# recording LLM, asserting the messages the model actually receives. This covers
# the llm.py change: empty system_content must NOT produce SystemMessage(content="").

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402


class _RecordingLLM:
    """Minimal LLM stub that records the message list it is invoked with and
    returns a terminal answer (no tool calls)."""

    def __init__(self):
        self.invocations = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        self.invocations.append(list(messages))
        return AIMessage(content="ok")


class _DummyRuntime:
    def get_mcp_toolkits(self):
        return []


def _run_bare_chat(instructions):
    """Build a 'bare' predict Assistant with the given instructions (no tools)
    and invoke it once, returning the messages the LLM received."""
    llm = _RecordingLLM()
    assistant = Assistant(
        elitea=_DummyRuntime(),
        data={"instructions": instructions, "tools": [], "meta": {}},
        client=llm,
        tools=[],
        memory=MemorySaver(),
        app_type=APP_TYPE_PREDICT,
        persona="bare",
    )
    runnable = assistant.runnable()
    runnable.invoke(
        {"messages": [HumanMessage(content="Who are you?")]},
        config={"configurable": {"thread_id": "bare-omit-test"}},
    )
    assert llm.invocations, "LLM was never invoked"
    return llm.invocations[0]


def test_bare_empty_omits_system_message_end_to_end():
    msgs = _run_bare_chat(instructions=None)
    assert not any(isinstance(m, SystemMessage) for m in msgs), (
        "bare with no instructions/tools must send NO SystemMessage; "
        f"got: {[type(m).__name__ for m in msgs]}"
    )


def test_bare_with_instructions_sends_one_system_message_end_to_end():
    msgs = _run_bare_chat(instructions="Always respond in Spanish.")
    system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1
    assert "Always respond in Spanish." in str(system_msgs[0].content)
    assert "EliteA" not in str(system_msgs[0].content)
