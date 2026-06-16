"""
Tests for TestRail toolkit get_run / get_runs (Runs read tools).

Covers the response-shape handling (bare list vs {'runs': [...]} dict), the
run_filter pass-through (string + dict, forwarded as kwargs to the client),
all three output formats, the empty-result behaviour (empty markup, NOT a
ToolException — matching the get_cases fix for issue #4126), the error paths,
and the field projection allowlist.

Test pattern mirrors `tests/tools/testrail/test_get_sections.py`: bypass the heavy
pydantic/indexer-base validator chain by constructing the wrapper via
`object.__new__` and injecting a mocked TestRail client.
"""
import pytest
from unittest.mock import MagicMock

from langchain_core.tools import ToolException
from testrail_api import StatusCodeError


@pytest.fixture
def wrapper():
    """TestrailAPIWrapper with a mocked TestRail client."""
    from elitea_sdk.tools.testrail.api_wrapper import TestrailAPIWrapper

    instance = object.__new__(TestrailAPIWrapper)
    instance._client = MagicMock()
    return instance


def _run(run_id: int, **overrides) -> dict:
    """Build a minimal TestRail run dict matching the api_wrapper field allowlist."""
    run = {
        "id": run_id,
        "suite_id": 6,
        "name": f"run-{run_id}",
        "description": None,
        "is_completed": False,
        "passed_count": 1,
        "failed_count": 0,
        "untested_count": 2,
        "project_id": 10,
    }
    run.update(overrides)
    return run


def _status_code_error(code: int = 400, body: bytes = b'{"error": "oops"}') -> StatusCodeError:
    """Build a StatusCodeError whose args layout matches what _format_status_error expects."""
    return StatusCodeError(code, "GET", "fake_url", body)


class TestGetRunsResponseShape:
    """testrail_api returns either a bare list or {'runs': [...]}; both must work."""

    def test_handles_dict_response_with_runs_key(self, wrapper):
        wrapper._client.runs.get_runs.return_value = {"runs": [_run(1)], "offset": 0}

        result = wrapper.get_runs(project_id="10")

        assert "run-1" in result

    def test_handles_bare_list_response(self, wrapper):
        wrapper._client.runs.get_runs.return_value = [_run(1)]

        result = wrapper.get_runs(project_id="10")

        assert "run-1" in result


class TestGetRunsFilter:
    """run_filter is parsed and forwarded to the client as keyword args."""

    def test_dict_filter_forwarded_as_kwargs(self, wrapper):
        wrapper._client.runs.get_runs.return_value = [_run(1)]

        wrapper.get_runs(project_id="10", run_filter={"is_completed": 0, "suite_id": 6})

        wrapper._client.runs.get_runs.assert_called_once_with(
            project_id="10", is_completed=0, suite_id=6
        )

    def test_string_filter_forwarded_as_kwargs(self, wrapper):
        wrapper._client.runs.get_runs.return_value = [_run(1)]

        wrapper.get_runs(project_id="10", run_filter='{"is_completed": 1}')

        wrapper._client.runs.get_runs.assert_called_once_with(project_id="10", is_completed=1)

    def test_none_filter_sends_no_extra_kwargs(self, wrapper):
        wrapper._client.runs.get_runs.return_value = [_run(1)]

        wrapper.get_runs(project_id="10")

        wrapper._client.runs.get_runs.assert_called_once_with(project_id="10")

    def test_invalid_json_filter_returns_tool_exception(self, wrapper):
        result = wrapper.get_runs(project_id="10", run_filter="{not-json}")

        assert isinstance(result, ToolException)
        assert "Invalid parameter for run_filter" in str(result)
        wrapper._client.runs.get_runs.assert_not_called()

    def test_non_str_non_dict_filter_returns_tool_exception(self, wrapper):
        result = wrapper.get_runs(project_id="10", run_filter=123)

        assert isinstance(result, ToolException)
        assert "must be a JSON string or dictionary" in str(result)

    @pytest.mark.parametrize("bad", ["[1, 2]", "5", '"hi"', "null", "true"])
    def test_json_string_parsing_to_non_object_returns_tool_exception(self, wrapper, bad):
        """A run_filter that is valid JSON but not an object must not crash with TypeError."""
        result = wrapper.get_runs(project_id="10", run_filter=bad)

        assert isinstance(result, ToolException)
        assert "must be a JSON object" in str(result)
        wrapper._client.runs.get_runs.assert_not_called()


class TestGetRunsOutputFormats:
    def _arrange(self, wrapper):
        wrapper._client.runs.get_runs.return_value = [_run(1)]

    def test_json_default(self, wrapper):
        self._arrange(wrapper)
        assert wrapper.get_runs(project_id="10").startswith("Extracted data:")

    def test_csv(self, wrapper):
        self._arrange(wrapper)
        result = wrapper.get_runs(project_id="10", output_format="csv")
        assert "id,suite_id,name" in result and "run-1" in result

    def test_markdown(self, wrapper):
        self._arrange(wrapper)
        result = wrapper.get_runs(project_id="10", output_format="markdown")
        assert "|" in result and "name" in result and "run-1" in result

    def test_invalid_format_returns_tool_exception(self, wrapper):
        self._arrange(wrapper)
        result = wrapper.get_runs(project_id="10", output_format="xml")
        assert isinstance(result, ToolException)
        assert "Invalid format" in str(result)


class TestGetRunsEmptyAndErrors:
    def test_empty_returns_empty_markup_not_exception(self, wrapper):
        """No runs → empty markup (matches get_cases issue #4126 fix), not a ToolException."""
        wrapper._client.runs.get_runs.return_value = []

        result = wrapper.get_runs(project_id="10")

        assert not isinstance(result, ToolException)
        assert result == "Extracted data:\n[]"

    def test_status_code_error_is_formatted(self, wrapper):
        wrapper._client.runs.get_runs.side_effect = _status_code_error(
            code=400, body=b'{"error": "Field :project_id is not a valid or accessible project."}'
        )

        result = wrapper.get_runs(project_id="999")

        assert isinstance(result, ToolException)
        assert "TestRail API error 400" in str(result)
        assert "Field :project_id" in str(result)


class TestGetRunsFieldProjection:
    def test_non_allowlisted_non_custom_fields_dropped(self, wrapper):
        run = _run(1)
        run["config_ids"] = [1, 2, 3]  # not in allowlist, not custom -> dropped
        wrapper._client.runs.get_runs.return_value = [run]

        result = wrapper.get_runs(project_id="10")

        assert "config_ids" not in result
        assert "run-1" in result

    def test_custom_fields_pass_through(self, wrapper):
        run = _run(1)
        run["custom_status1_count"] = 9
        run["custom_env"] = "staging"
        wrapper._client.runs.get_runs.return_value = [run]

        result = wrapper.get_runs(project_id="10")

        assert "custom_status1_count" in result
        assert "custom_env" in result and "staging" in result


class TestGetRunsTimestampNormalization:
    def test_created_on_rendered_as_iso(self, wrapper):
        wrapper._client.runs.get_runs.return_value = [_run(1, created_on=1700000000)]

        result = wrapper.get_runs(project_id="10")

        assert "2023-11-14T22:13:20+00:00" in result
        assert "1700000000" not in result

    def test_null_completed_on_passes_through(self, wrapper):
        wrapper._client.runs.get_runs.return_value = [_run(1, completed_on=None)]

        result = wrapper.get_runs(project_id="10")

        assert "run-1" in result  # no crash on None timestamp


class TestGetRunsIsCompletedCoercion:
    @pytest.mark.parametrize("value,expected", [
        (True, 1), (False, 0), ("true", 1), ("false", 0),
        ("1", 1), ("0", 0), (1, 1), (0, 0),
    ])
    def test_is_completed_coerced_to_int(self, wrapper, value, expected):
        wrapper._client.runs.get_runs.return_value = []

        wrapper.get_runs(project_id="10", run_filter={"is_completed": value})

        assert wrapper._client.runs.get_runs.call_args.kwargs["is_completed"] == expected


class TestGetRunsProjectIdGuard:
    def test_non_numeric_project_id_returns_tool_exception(self, wrapper):
        result = wrapper.get_runs(project_id="abc")

        assert isinstance(result, ToolException)
        assert "project_id must be numeric" in str(result)
        wrapper._client.runs.get_runs.assert_not_called()


class TestGetRun:
    """Single-run read."""

    def test_happy_path(self, wrapper):
        wrapper._client.runs.get_run.return_value = _run(5)

        result = wrapper.get_run(run_id="5")

        wrapper._client.runs.get_run.assert_called_once_with(run_id=5)
        assert result.startswith("Extracted data:")
        assert "run-5" in result

    def test_non_numeric_run_id_returns_tool_exception(self, wrapper):
        result = wrapper.get_run(run_id="abc")

        assert isinstance(result, ToolException)
        assert "run_id must be numeric" in str(result)
        wrapper._client.runs.get_run.assert_not_called()

    def test_status_code_error_is_formatted(self, wrapper):
        wrapper._client.runs.get_run.side_effect = _status_code_error(
            code=400, body=b'{"error": "Field :run_id is not a valid test run."}'
        )

        result = wrapper.get_run(run_id="999")

        assert isinstance(result, ToolException)
        assert "TestRail API error 400" in str(result)

    def test_output_format_markdown(self, wrapper):
        wrapper._client.runs.get_run.return_value = _run(5)

        result = wrapper.get_run(run_id="5", output_format="markdown")

        assert "|" in result and "run-5" in result

    def test_field_projection(self, wrapper):
        run = _run(5)
        run["config_ids"] = [1]  # not in allowlist
        wrapper._client.runs.get_run.return_value = run

        result = wrapper.get_run(run_id="5")

        assert "config_ids" not in result
