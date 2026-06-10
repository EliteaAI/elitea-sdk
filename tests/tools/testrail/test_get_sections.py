"""
Tests for TestRail toolkit get_sections.

Covers the suite-mode branches of `_fetch_sections_with_suite_handling`
(single-suite, multi-suite scoped, multi-suite aggregated), the response-shape
handling (old list response vs new {'sections': [...]} dict), all three output
formats, and the error paths surfaced by `_to_markup` and `_format_status_error`.

Test pattern mirrors `tests/tools/jira/test_jira_auth_error_handling.py`:
bypass the heavy pydantic/indexer-base validator chain by constructing the
wrapper via `object.__new__` and injecting a mocked TestRail client.
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


def _section(section_id: int, suite_id: int = 1, parent_id=None) -> dict:
    """Build a minimal TestRail section dict matching api_wrapper.py field allowlist."""
    return {
        "id": section_id,
        "suite_id": suite_id,
        "name": f"section-{section_id}",
        "description": None,
        "parent_id": parent_id,
        "display_order": section_id,
        "depth": 0,
    }


def _status_code_error(code: int = 400, body: bytes = b'{"error": "oops"}') -> StatusCodeError:
    """Build a StatusCodeError whose args layout matches what _format_status_error expects."""
    # _format_status_error reads args[0] as status code and args[3] as body bytes.
    return StatusCodeError(code, "GET", "fake_url", body)


class TestGetSectionsSuiteModeBranches:
    """Branches in _fetch_sections_with_suite_handling driven by project suite_mode."""

    def test_single_suite_mode_fetches_at_project_level(self, wrapper):
        """suite_mode 1: one call to sections.get_sections without suite_id."""
        wrapper._client.projects.get_project.return_value = {"suite_mode": 1}
        wrapper._client.sections.get_sections.return_value = [_section(1)]

        result = wrapper.get_sections(project_id="10")

        wrapper._client.sections.get_sections.assert_called_once_with(project_id="10")
        wrapper._client.suites.get_suites.assert_not_called()
        assert "section-1" in result

    def test_multi_suite_mode_with_suite_id_scopes_call(self, wrapper):
        """suite_mode 3 + suite_id: single scoped call, no suite enumeration."""
        wrapper._client.projects.get_project.return_value = {"suite_mode": 3}
        wrapper._client.sections.get_sections.return_value = [_section(42, suite_id=6)]

        result = wrapper.get_sections(project_id="10", suite_id="6")

        wrapper._client.sections.get_sections.assert_called_once_with(project_id="10", suite_id=6)
        wrapper._client.suites.get_suites.assert_not_called()
        assert "section-42" in result

    def test_multi_suite_mode_without_suite_id_aggregates_across_suites(self, wrapper):
        """suite_mode 3 + no suite_id: iterate all suites, concatenate results."""
        wrapper._client.projects.get_project.return_value = {"suite_mode": 3}
        wrapper._client.suites.get_suites.return_value = [{"id": 1}, {"id": 6}]
        wrapper._client.sections.get_sections.side_effect = [
            [_section(1, suite_id=1)],
            [_section(42, suite_id=6)],
        ]

        result = wrapper.get_sections(project_id="10")

        assert wrapper._client.sections.get_sections.call_count == 2
        assert "section-1" in result and "section-42" in result

    def test_suite_mode_2_treated_as_requiring_suite_id(self, wrapper):
        """suite_mode 2 (single + baselines) takes the multi-suite aggregation path."""
        wrapper._client.projects.get_project.return_value = {"suite_mode": 2}
        wrapper._client.suites.get_suites.return_value = [{"id": 1}]
        wrapper._client.sections.get_sections.return_value = [_section(1)]

        wrapper.get_sections(project_id="10")

        wrapper._client.suites.get_suites.assert_called_once()


class TestGetSectionsResponseShape:
    """testrail_api versions return either a bare list or {'sections': [...]}; both must work."""

    def test_handles_dict_response_with_sections_key(self, wrapper):
        """Newer testrail_api wraps results in {'sections': [...]} — extract correctly."""
        wrapper._client.projects.get_project.return_value = {"suite_mode": 1}
        wrapper._client.sections.get_sections.return_value = {"sections": [_section(1)]}

        result = wrapper.get_sections(project_id="10")

        assert "section-1" in result

    def test_handles_bare_list_response(self, wrapper):
        """Older testrail_api returns a bare list."""
        wrapper._client.projects.get_project.return_value = {"suite_mode": 1}
        wrapper._client.sections.get_sections.return_value = [_section(1)]

        result = wrapper.get_sections(project_id="10")

        assert "section-1" in result


class TestGetSectionsOutputFormats:
    """`output_format` dispatches through _to_markup."""

    def _arrange_one_section(self, wrapper):
        wrapper._client.projects.get_project.return_value = {"suite_mode": 1}
        wrapper._client.sections.get_sections.return_value = [_section(1)]

    def test_json_default(self, wrapper):
        self._arrange_one_section(wrapper)

        result = wrapper.get_sections(project_id="10")

        assert result.startswith("Extracted data:")

    def test_csv(self, wrapper):
        self._arrange_one_section(wrapper)

        result = wrapper.get_sections(project_id="10", output_format="csv")

        assert "id,suite_id,name" in result
        assert "section-1" in result

    def test_markdown(self, wrapper):
        self._arrange_one_section(wrapper)

        result = wrapper.get_sections(project_id="10", output_format="markdown")

        assert "|" in result and "name" in result and "section-1" in result

    def test_invalid_format_returns_tool_exception(self, wrapper):
        """_to_markup rejects unsupported format with a ToolException."""
        self._arrange_one_section(wrapper)

        result = wrapper.get_sections(project_id="10", output_format="xml")

        assert isinstance(result, ToolException)
        assert "Invalid format" in str(result)


class TestGetSectionsErrorPaths:
    """Error and edge-case handling."""

    def test_empty_result_returns_tool_exception(self, wrapper):
        """No sections returned for project → ToolException (current behaviour; see review #2)."""
        wrapper._client.projects.get_project.return_value = {"suite_mode": 1}
        wrapper._client.sections.get_sections.return_value = []

        result = wrapper.get_sections(project_id="10")

        assert isinstance(result, ToolException)
        assert "No sections found" in str(result)

    def test_status_code_error_is_formatted(self, wrapper):
        """StatusCodeError on the scoped call surfaces via _format_status_error."""
        wrapper._client.projects.get_project.return_value = {"suite_mode": 3}
        wrapper._client.sections.get_sections.side_effect = _status_code_error(
            code=400, body=b'{"error": "Field :project_id is not a valid or accessible project."}'
        )

        result = wrapper.get_sections(project_id="999", suite_id="6")

        assert isinstance(result, ToolException)
        assert "TestRail API error 400" in str(result)
        assert "Field :project_id" in str(result)

    def test_per_suite_errors_swallowed_during_aggregation(self, wrapper):
        """Aggregation loop continues past per-suite StatusCodeError — review #3."""
        wrapper._client.projects.get_project.return_value = {"suite_mode": 3}
        wrapper._client.suites.get_suites.return_value = [{"id": 1}, {"id": 6}]
        wrapper._client.sections.get_sections.side_effect = [
            _status_code_error(),
            [_section(42, suite_id=6)],
        ]

        result = wrapper.get_sections(project_id="10")

        assert "section-42" in result

    def test_non_numeric_suite_id_raises_value_error(self, wrapper):
        """Pre-existing latent bug (review #1): int(suite_id) crashes the tool on non-numeric input."""
        wrapper._client.projects.get_project.return_value = {"suite_mode": 3}

        with pytest.raises(ValueError, match=r"invalid literal for int\(\) with base 10: 'abc'"):
            wrapper.get_sections(project_id="10", suite_id="abc")


class TestGetSectionsFieldProjection:
    """The wrapper projects each section onto a fixed field allowlist."""

    def test_only_allowlisted_fields_are_returned(self, wrapper):
        """Extra fields from the TestRail response are dropped; allowlisted ones pass through."""
        wrapper._client.projects.get_project.return_value = {"suite_mode": 1}
        raw_section = _section(1)
        raw_section["unexpected_field"] = "should-be-dropped"
        wrapper._client.sections.get_sections.return_value = [raw_section]

        result = wrapper.get_sections(project_id="10")

        assert "should-be-dropped" not in result
        assert "section-1" in result
