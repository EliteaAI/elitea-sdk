"""
Tests for TestRail toolkit results read tools:
get_results_for_run / get_results_for_case / get_results.

Covers the shared parse/fetch/render path (`_read_results`): result_filter
pass-through (string + dict, forwarded as kwargs), automatic stripping of
limit/offset (the *_bulk endpoints page internally), response-shape handling
(bare list vs {'results': [...]}), all three output formats, empty-result
behaviour (empty markup, NOT a ToolException), error paths, field projection,
and the numeric-id guards on each tool.

Test pattern mirrors `tests/tools/testrail/test_get_runs.py`: bypass the heavy
validator chain via `object.__new__` and inject a mocked TestRail client.
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


def _result(result_id: int, **overrides) -> dict:
    """Minimal TestRail result dict matching the api_wrapper field allowlist."""
    result = {
        "id": result_id,
        "test_id": 7,
        "status_id": 5,
        "comment": f"result-{result_id}",
        "version": "1.0",
        "elapsed": "30s",
        "defects": "TR-1",
        "created_by": 3,
        "created_on": 1700000000,
    }
    result.update(overrides)
    return result


def _status_code_error(code: int = 400, body: bytes = b'{"error": "oops"}') -> StatusCodeError:
    return StatusCodeError(code, "GET", "fake_url", body)


class TestFilterHandling:
    """result_filter parsing/forwarding via get_results_for_run."""

    def test_dict_filter_forwarded(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.return_value = [_result(1)]

        wrapper.get_results_for_run("10", result_filter={"status_id": [5], "defects_filter": "TR-1"})

        wrapper._client.results.get_results_for_run_bulk.assert_called_once_with(
            run_id=10, status_id=[5], defects_filter="TR-1"
        )

    def test_string_filter_forwarded(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.return_value = [_result(1)]

        wrapper.get_results_for_run("10", result_filter='{"status_id": 5}')

        wrapper._client.results.get_results_for_run_bulk.assert_called_once_with(run_id=10, status_id=5)

    def test_limit_and_offset_are_stripped(self, wrapper):
        """The *_bulk endpoints page internally; limit/offset must not be forwarded."""
        wrapper._client.results.get_results_for_run_bulk.return_value = [_result(1)]

        wrapper.get_results_for_run("10", result_filter={"status_id": 5, "limit": 999, "offset": 10})

        wrapper._client.results.get_results_for_run_bulk.assert_called_once_with(run_id=10, status_id=5)

    def test_none_filter_sends_no_extra_kwargs(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.return_value = [_result(1)]

        wrapper.get_results_for_run("10")

        wrapper._client.results.get_results_for_run_bulk.assert_called_once_with(run_id=10)

    def test_invalid_json_filter_returns_tool_exception(self, wrapper):
        result = wrapper.get_results_for_run("10", result_filter="{not-json}")

        assert isinstance(result, ToolException)
        assert "Invalid parameter for result_filter" in str(result)
        wrapper._client.results.get_results_for_run_bulk.assert_not_called()

    def test_non_str_non_dict_filter_returns_tool_exception(self, wrapper):
        result = wrapper.get_results_for_run("10", result_filter=123)

        assert isinstance(result, ToolException)
        assert "must be a JSON string or dictionary" in str(result)

    @pytest.mark.parametrize("bad", ["[1, 2]", "5", '"hi"', "null"])
    def test_json_string_parsing_to_non_object_returns_tool_exception(self, wrapper, bad):
        """A result_filter that is valid JSON but not an object must not crash with TypeError."""
        result = wrapper.get_results_for_run("10", result_filter=bad)

        assert isinstance(result, ToolException)
        assert "must be a JSON object" in str(result)
        wrapper._client.results.get_results_for_run_bulk.assert_not_called()


class TestResponseShapeAndFormats:
    def test_handles_bare_list(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.return_value = [_result(1)]
        assert "result-1" in wrapper.get_results_for_run("10")

    def test_handles_dict_with_results_key(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.return_value = {"results": [_result(1)]}
        assert "result-1" in wrapper.get_results_for_run("10")

    def test_json_default(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.return_value = [_result(1)]
        assert wrapper.get_results_for_run("10").startswith("Extracted data:")

    def test_csv(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.return_value = [_result(1)]
        out = wrapper.get_results_for_run("10", output_format="csv")
        assert "id,test_id,status_id" in out and "result-1" in out

    def test_markdown(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.return_value = [_result(1)]
        out = wrapper.get_results_for_run("10", output_format="markdown")
        assert "|" in out and "result-1" in out

    def test_invalid_format_returns_tool_exception(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.return_value = [_result(1)]
        out = wrapper.get_results_for_run("10", output_format="xml")
        assert isinstance(out, ToolException) and "Invalid format" in str(out)


class TestEmptyAndErrorsAndProjection:
    def test_empty_returns_empty_markup(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.return_value = []
        out = wrapper.get_results_for_run("10")
        assert not isinstance(out, ToolException)
        assert out == "Extracted data:\n[]"

    def test_status_code_error_is_formatted(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.side_effect = _status_code_error(
            code=400, body=b'{"error": "Field :run_id is not a valid test run."}'
        )
        out = wrapper.get_results_for_run("999")
        assert isinstance(out, ToolException)
        assert "TestRail API error 400" in str(out)

    def test_non_allowlisted_non_custom_fields_dropped(self, wrapper):
        result = _result(1)
        result["created_by_avatar"] = "http://x"  # not in allowlist, not custom -> dropped
        wrapper._client.results.get_results_for_run_bulk.return_value = [result]
        out = wrapper.get_results_for_run("10")
        assert "created_by_avatar" not in out
        assert "result-1" in out

    def test_custom_fields_pass_through(self, wrapper):
        result = _result(1)
        result["custom_severity"] = "high"
        wrapper._client.results.get_results_for_run_bulk.return_value = [result]
        out = wrapper.get_results_for_run("10")
        assert "custom_severity" in out and "high" in out

    def test_created_on_rendered_as_iso(self, wrapper):
        wrapper._client.results.get_results_for_run_bulk.return_value = [_result(1, created_on=1700000000)]
        out = wrapper.get_results_for_run("10")
        assert "2023-11-14T22:13:20+00:00" in out
        assert "1700000000" not in out


class TestGetResultsForCase:
    def test_happy_path_forwards_ids(self, wrapper):
        wrapper._client.results.get_results_for_case_bulk.return_value = [_result(1)]

        wrapper.get_results_for_case("10", "55", result_filter={"status_id": 1})

        wrapper._client.results.get_results_for_case_bulk.assert_called_once_with(
            run_id=10, case_id=55, status_id=1
        )

    def test_non_numeric_ids_return_tool_exception(self, wrapper):
        out = wrapper.get_results_for_case("10", "abc")
        assert isinstance(out, ToolException)
        assert "run_id and case_id must be numeric" in str(out)
        wrapper._client.results.get_results_for_case_bulk.assert_not_called()


class TestGetResults:
    def test_happy_path_forwards_test_id(self, wrapper):
        wrapper._client.results.get_results_bulk.return_value = [_result(1)]

        wrapper.get_results("7", result_filter={"status_id": 5})

        wrapper._client.results.get_results_bulk.assert_called_once_with(test_id=7, status_id=5)

    def test_non_numeric_test_id_returns_tool_exception(self, wrapper):
        out = wrapper.get_results("abc")
        assert isinstance(out, ToolException)
        assert "test_id must be numeric" in str(out)
        wrapper._client.results.get_results_bulk.assert_not_called()
