"""
Tests for TestRail toolkit delete_section.

Covers the hard-delete default (soft=0), the soft dry-run path (soft=1), the
confirmation messages, and the two error paths that surface as ToolException
(StatusCodeError from the API and any other unexpected exception).

Test pattern mirrors `tests/tools/testrail/test_add_section.py`: bypass the heavy
pydantic/indexer-base validator chain by constructing the wrapper via
`object.__new__` and injecting a mocked TestRail client. `_log_tool_event` is
inherited and swallows its own exceptions, so it is safe to call here.
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


def _status_code_error(code: int = 400, body: bytes = b'{"error": "oops"}') -> StatusCodeError:
    """Build a StatusCodeError whose args layout matches what _format_status_error expects."""
    return StatusCodeError(code, "POST", "fake_url", body)


class TestDeleteSectionHappyPath:
    """Default (hard) delete and the soft dry-run variant."""

    def test_default_is_hard_delete(self, wrapper):
        """Default soft_delete=False maps to soft=0 (actual deletion)."""
        result = wrapper.delete_section(section_id="42")

        wrapper._client.sections.delete_section.assert_called_once_with(section_id="42", soft=0)
        assert "permanently deleted" in result
        assert "#42" in result

    def test_soft_delete_is_dry_run(self, wrapper):
        """soft_delete=True maps to soft=1 and reports a dry run."""
        result = wrapper.delete_section(section_id="42", soft_delete=True)

        wrapper._client.sections.delete_section.assert_called_once_with(section_id="42", soft=1)
        assert "soft dry run" in result
        assert "#42" in result


class TestDeleteSectionErrorPaths:
    """Both failure modes raise ToolException."""

    def test_status_code_error_raises_tool_exception(self, wrapper):
        """A StatusCodeError from the API is wrapped in a ToolException."""
        wrapper._client.sections.delete_section.side_effect = _status_code_error(
            code=400, body=b'{"error": "Field :section_id is not a valid section."}'
        )

        with pytest.raises(ToolException, match=r"Unable to delete section #999"):
            wrapper.delete_section(section_id="999")

    def test_unexpected_error_raises_tool_exception(self, wrapper):
        """Any other exception is also wrapped in a ToolException."""
        wrapper._client.sections.delete_section.side_effect = RuntimeError("boom")

        with pytest.raises(ToolException, match=r"Error deleting section #7"):
            wrapper.delete_section(section_id="7")
