"""#6982: Web Search module, SDK side.

1. WEB_SEARCH_ADDON is present in the system prompt whenever the `web_search` tool is bound,
   on every composition branch except swarm (documented gap), and absent otherwise.
2. A tool-spend refusal inside a pipeline node does not fail the node: the FunctionTool node
   yields an ERROR outcome carrying the limit message, and the run continues (owner decision,
   replaces the original "node fails explicitly" acceptance wording).
"""
from unittest.mock import patch

import pytest
from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.langchain.assistant import Assistant, APP_TYPE_AGENT, APP_TYPE_PREDICT
from elitea_sdk.runtime.langchain.constants import WEB_SEARCH_ADDON
from elitea_sdk.runtime.tool_outcome import ToolResultStatus
from elitea_sdk.runtime.tools.function import FunctionTool
from elitea_sdk.tools.utils.metering import ToolSpendDenied


def _assistant(persona=None, app_type=APP_TYPE_PREDICT):
    a = object.__new__(Assistant)
    a.persona = persona
    a.app_type = app_type
    a._middleware_prompt = ""
    return a


def _compose(assistant, tool_names, instructions=""):
    return assistant._compose_system_prompt(instructions, [object() for _ in tool_names], tool_names, [])


@pytest.mark.parametrize("persona,app_type,instructions", [
    (None, APP_TYPE_PREDICT, ""),                 # persona-template fallback
    ("bare", APP_TYPE_PREDICT, ""),               # bare, addons only
    ("bare", APP_TYPE_PREDICT, "Be brief."),      # bare with instructions
    (None, APP_TYPE_AGENT, "You are a helper."),  # agent with its own instructions
])
def test_addon_present_when_web_search_bound(persona, app_type, instructions):
    prompt = _compose(_assistant(persona, app_type), ["web_search", "web_extract"], instructions)
    assert WEB_SEARCH_ADDON.strip() in prompt


@pytest.mark.parametrize("tools", [[], ["web_extract"], ["web_crawl", "web_map"], ["read_file"]])
def test_addon_absent_without_web_search(tools):
    for persona, app_type, instr in [(None, APP_TYPE_PREDICT, ""), (None, APP_TYPE_AGENT, "x")]:
        assert "## Web search" not in _compose(_assistant(persona, app_type), tools, instr)


def test_addon_covers_the_safety_rules():
    text = WEB_SEARCH_ADDON.lower()
    assert "web_extract" in text and "markdown links" in text
    assert "untrusted" in text and "confidential" in text


def _denied_search(query: str) -> str:
    raise ToolSpendDenied("tool_spend_limit_reached", "Web search limit reached for this project this month.")


def test_pipeline_node_yields_error_outcome_on_limit():
    tool = StructuredTool.from_function(func=_denied_search, name="web_search", description="search")
    node = FunctionTool(
        name="search_node", tool=tool,
        input_mapping={"query": {"type": "fixed", "value": "news"}},
        input_variables=[], output_variables=["results"],
    )
    with patch("elitea_sdk.runtime.tools.function.dispatch_custom_event"):
        result = node.invoke({})
    outcome = result["last_tool_outcome"]
    status = outcome["status"] if isinstance(outcome, dict) else outcome.status
    message = outcome["message"] if isinstance(outcome, dict) else outcome.message
    assert status in (ToolResultStatus.ERROR, ToolResultStatus.ERROR.value)
    assert "limit reached" in message
    assert "limit reached" in result["messages"][0]["content"]
