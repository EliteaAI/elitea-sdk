"""Unit tests for #6168 toolkit_id/toolkit_type metadata injection.

Provider/dynamic-module toolkits have no `api_wrapper`, so `_inject_toolkit_id`
is a silent no-op for them. `_inject_toolkit_metadata` closes that gap by
writing into `t.metadata` instead, which is what every real consumer reads.
"""
import logging
import types
from unittest.mock import patch

import pytest
from langchain_core.tools import StructuredTool

from elitea_sdk.tools import (
    _inject_display_metadata,
    _inject_toolkit_id,
    _inject_toolkit_metadata,
    get_tools,
)


def _make_tool(name="t1"):
    def _fn(x: str = "") -> str:
        return x
    return StructuredTool.from_function(func=_fn, name=name, description="d")


class ApiWrapper:
    toolkit_id = None


class TestInjectToolkitMetadataNoApiWrapper:
    def test_sets_toolkit_id_and_type_on_metadata_dict(self):
        tool = _make_tool()
        assert not hasattr(tool, "api_wrapper")

        _inject_toolkit_metadata({"id": 42, "type": "imagegen"}, [tool])

        assert tool.metadata == {"toolkit_id": 42, "toolkit_type": "imagegen"}

    def test_id_none_injects_type_only_and_logs_no_error(self, caplog):
        tool = _make_tool()

        with caplog.at_level(logging.ERROR):
            _inject_toolkit_metadata({"id": None, "type": "imagegen"}, [tool])

        assert tool.metadata == {"toolkit_type": "imagegen"}
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)

    def test_type_none_injects_id_only(self):
        tool = _make_tool()
        _inject_toolkit_metadata({"id": 5, "type": None}, [tool])
        assert tool.metadata == {"toolkit_id": 5}

    def test_neither_present_leaves_metadata_untouched(self):
        tool = _make_tool()
        _inject_toolkit_metadata({}, [tool])
        assert tool.metadata == {}

    def test_preexisting_metadata_is_merged_not_replaced(self):
        tool = _make_tool()
        tool.metadata = {"display_name": "My Toolkit"}
        _inject_toolkit_metadata({"id": 1, "type": "jira"}, [tool])
        assert tool.metadata == {"display_name": "My Toolkit", "toolkit_id": 1, "toolkit_type": "jira"}

    def test_skips_tools_without_metadata_attribute(self):
        class NoMetadata:
            pass
        obj = NoMetadata()
        _inject_toolkit_metadata({"id": 1, "type": "jira"}, [obj])
        assert not hasattr(obj, "metadata")


class _ToolWithApiWrapper:
    """StructuredTool (pydantic) rejects extra fields, so real toolkit tools that
    carry an api_wrapper (e.g. jira) are plain objects like this one instead."""
    def __init__(self):
        self.api_wrapper = ApiWrapper()
        self.metadata = {}


class TestExistingBehaviourUnchanged:
    def test_inject_toolkit_id_still_writes_api_wrapper_when_present(self):
        tool = _ToolWithApiWrapper()

        _inject_toolkit_id({"id": 99, "type": "jira"}, [tool])

        assert tool.api_wrapper.toolkit_id == 99

    def test_inject_toolkit_id_no_api_wrapper_logs_error_when_id_not_int(self, caplog):
        tool = _make_tool()
        with caplog.at_level(logging.ERROR):
            _inject_toolkit_id({"id": None, "type": "imagegen", "name": "ImageGen"}, [tool])

        assert any("Toolkit ID is missing or not an integer" in r.message for r in caplog.records)

    def test_inject_display_metadata_still_sets_display_name(self):
        tool = _make_tool()
        _inject_display_metadata({"name": "My Jira Project"}, [tool])
        assert tool.metadata["display_name"] == "My Jira Project"

    def test_display_metadata_and_toolkit_metadata_compose(self):
        tool = _make_tool()
        tool_conf = {"id": 3, "type": "jira", "name": "My Jira Project"}

        _inject_toolkit_id(tool_conf, [tool])
        _inject_display_metadata(tool_conf, [tool])
        _inject_toolkit_metadata(tool_conf, [tool])

        assert tool.metadata == {
            "display_name": "My Jira Project",
            "toolkit_id": 3,
            "toolkit_type": "jira",
        }


class TestDynamicModuleBranchForwardsIdAndType:
    """get_tools()'s custom-toolkit branch must forward the tool's own id/type into
    get_toolkit_params, so provider toolkits can attribute their own shadow logs (#6168).
    Before this fix, only settings (a nested dict without id/type) was copied."""

    def _make_fake_toolkit_module(self, recorded_calls):
        class FakeToolkit:
            @classmethod
            def get_toolkit(cls, **kwargs):
                recorded_calls.append(kwargs)
                instance = cls()
                instance.get_tools = lambda: []
                return instance

        return types.SimpleNamespace(FakeToolkit=FakeToolkit)

    def test_id_and_type_reach_get_toolkit_params(self):
        recorded_calls = []
        fake_module = self._make_fake_toolkit_module(recorded_calls)

        tools_list = [{
            "id": 42,
            "type": "custom_provider",
            "name": "My Provider Toolkit",
            "settings": {
                "module": "fake.module.path",
                "class": "FakeToolkit",
            },
        }]

        with patch("elitea_sdk.tools.import_module", return_value=fake_module):
            get_tools(tools_list, elitea=None, llm=None)

        assert len(recorded_calls) == 1
        assert recorded_calls[0]["id"] == 42
        assert recorded_calls[0]["type"] == "custom_provider"
