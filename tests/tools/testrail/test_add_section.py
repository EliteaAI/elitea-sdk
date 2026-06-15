"""
Tests for TestRail toolkit add_section.

Covers the happy path (formatted success string), property pass-through
(description / suite_id / parent_id forwarded as kwargs to the client),
the JSON-parsing path for `section_properties` (string and dict inputs),
and the two error paths that surface as ToolException (invalid JSON and
StatusCodeError from the API).

Test pattern mirrors `tests/tools/testrail/test_get_sections.py`:
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


def _status_code_error(code: int = 400, body: bytes = b'{"error": "oops"}') -> StatusCodeError:
    """Build a StatusCodeError whose args layout matches what _format_status_error expects."""
    return StatusCodeError(code, "POST", "fake_url", body)


class TestAddSectionHappyPath:
    """Basic creation and the formatted success message."""

    def test_minimal_call_creates_section(self, wrapper):
        """project_id + name only: client called with no extra kwargs, success string returned."""
        wrapper._client.sections.add_section.return_value = {"id": 5, "name": "Smoke tests"}

        result = wrapper.add_section(project_id="10", name="Smoke tests")

        wrapper._client.sections.add_section.assert_called_once_with(
            project_id="10", name="Smoke tests"
        )
        assert result == "New section has been created: id - 5 - 'Smoke tests'"

    def test_success_message_includes_id_and_name(self, wrapper):
        """The returned string surfaces both the new id and the name echoed by TestRail."""
        wrapper._client.sections.add_section.return_value = {"id": 99, "name": "Regression"}

        result = wrapper.add_section(project_id="10", name="Regression")

        assert "id - 99" in result and "'Regression'" in result


class TestAddSectionProperties:
    """`section_properties` is parsed and forwarded to the client as keyword args."""

    def test_properties_forwarded_as_kwargs(self, wrapper):
        """description / suite_id / parent_id pass through to sections.add_section."""
        wrapper._client.sections.add_section.return_value = {"id": 7, "name": "Child"}

        wrapper.add_section(
            project_id="10",
            name="Child",
            section_properties='{"description": "desc", "suite_id": 6, "parent_id": 3}',
        )

        wrapper._client.sections.add_section.assert_called_once_with(
            project_id="10", name="Child", description="desc", suite_id=6, parent_id=3
        )

    def test_default_properties_send_no_extra_kwargs(self, wrapper):
        """The default '{}' results in a call with only project_id and name."""
        wrapper._client.sections.add_section.return_value = {"id": 1, "name": "Top"}

        wrapper.add_section(project_id="10", name="Top")

        wrapper._client.sections.add_section.assert_called_once_with(
            project_id="10", name="Top"
        )

    def test_accepts_dict_section_properties(self, wrapper):
        """A non-string section_properties (already a dict) is forwarded without json.loads."""
        wrapper._client.sections.add_section.return_value = {"id": 8, "name": "Dict"}

        wrapper.add_section(
            project_id="10", name="Dict", section_properties={"parent_id": 4}
        )

        wrapper._client.sections.add_section.assert_called_once_with(
            project_id="10", name="Dict", parent_id=4
        )

    def test_none_section_properties_send_no_extra_kwargs(self, wrapper):
        """A None section_properties falls back to an empty dict."""
        wrapper._client.sections.add_section.return_value = {"id": 2, "name": "Top"}

        wrapper.add_section(project_id="10", name="Top", section_properties=None)

        wrapper._client.sections.add_section.assert_called_once_with(
            project_id="10", name="Top"
        )


class TestAddSectionErrorPaths:
    """Both failure modes raise ToolException."""

    def test_invalid_json_raises_tool_exception(self, wrapper):
        """Malformed section_properties JSON surfaces as a ToolException, not a raw error."""
        with pytest.raises(ToolException, match="Invalid JSON in section_properties"):
            wrapper.add_section(project_id="10", name="Bad", section_properties="{not-json}")

        wrapper._client.sections.add_section.assert_not_called()

    def test_status_code_error_raises_tool_exception(self, wrapper):
        """A StatusCodeError from the API is wrapped in a ToolException."""
        wrapper._client.sections.add_section.side_effect = _status_code_error(
            code=400, body=b'{"error": "Field :name is required."}'
        )

        with pytest.raises(ToolException, match="Unable to add new section"):
            wrapper.add_section(project_id="10", name="Whatever")
