"""
Tests for TestRail toolkit get_suites.

get_suites had no unit coverage before; this locks in its core behavior plus the
shared projection treatment (custom_* passthrough + ISO-8601 timestamps) applied
for consistency with the runs/results read tools.

Same fixture pattern as the other testrail tests: bypass the validator chain via
`object.__new__` and inject a mocked TestRail client.
"""
import pytest
from unittest.mock import MagicMock

from langchain_core.tools import ToolException
from testrail_api import StatusCodeError


@pytest.fixture
def wrapper():
    from elitea_sdk.tools.testrail.api_wrapper import TestrailAPIWrapper

    instance = object.__new__(TestrailAPIWrapper)
    instance._client = MagicMock()
    return instance


def _suite(suite_id: int = 6, **overrides) -> dict:
    suite = {
        "id": suite_id,
        "name": f"suite-{suite_id}",
        "description": None,
        "project_id": 10,
        "is_master": True,
        "is_completed": False,
        "completed_on": None,
        "url": "http://x",
    }
    suite.update(overrides)
    return suite


class TestGetSuitesHappyPath:
    def test_json_default_lists_suites(self, wrapper):
        wrapper._client.suites.get_suites.return_value = [_suite(6)]

        result = wrapper.get_suites(project_id="10")

        assert result.startswith("Extracted data:")
        assert "suite-6" in result

    def test_csv_format(self, wrapper):
        wrapper._client.suites.get_suites.return_value = [_suite(6)]

        result = wrapper.get_suites(project_id="10", output_format="csv")

        assert "id,name" in result and "suite-6" in result


class TestGetSuitesProjection:
    def test_non_custom_extra_fields_dropped(self, wrapper):
        suite = _suite(6)
        suite["unexpected"] = "drop-me"
        wrapper._client.suites.get_suites.return_value = [suite]

        result = wrapper.get_suites(project_id="10")

        assert "unexpected" not in result and "suite-6" in result

    def test_custom_fields_pass_through(self, wrapper):
        suite = _suite(6, custom_team="qa")
        wrapper._client.suites.get_suites.return_value = [suite]

        result = wrapper.get_suites(project_id="10")

        assert "custom_team" in result and "qa" in result

    def test_completed_on_rendered_as_iso(self, wrapper):
        wrapper._client.suites.get_suites.return_value = [_suite(6, completed_on=1700000000)]

        result = wrapper.get_suites(project_id="10")

        assert "2023-11-14T22:13:20+00:00" in result
        assert "1700000000" not in result

    def test_null_completed_on_passes_through(self, wrapper):
        wrapper._client.suites.get_suites.return_value = [_suite(6, completed_on=None)]

        result = wrapper.get_suites(project_id="10")

        assert "suite-6" in result  # no crash on null timestamp


class TestGetSuitesErrorPaths:
    def test_empty_returns_tool_exception(self, wrapper):
        """get_suites keeps its existing empty-result contract (ToolException)."""
        wrapper._client.suites.get_suites.return_value = []

        result = wrapper.get_suites(project_id="10")

        assert isinstance(result, ToolException)
        assert "No test suites found" in str(result)

    def test_status_code_error_is_wrapped(self, wrapper):
        wrapper._client.suites.get_suites.side_effect = StatusCodeError(
            400, "GET", "url", b'{"error": "bad"}'
        )

        result = wrapper.get_suites(project_id="999")

        assert isinstance(result, ToolException)
        assert "Unable to extract test suites" in str(result)
