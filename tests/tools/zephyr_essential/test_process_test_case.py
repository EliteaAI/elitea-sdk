"""
Regression tests for ZephyrEssentialApiWrapper._process_test_case.

Covers the fallback chain (steps -> script -> empty) now that
get_test_case_test_steps and get_test_case_test_script raise ToolException
instead of returning it, proving the try/except conversion preserves the
original returned-exception fallback behavior.
"""
import pytest
from unittest.mock import patch

from langchain_core.tools import ToolException

from elitea_sdk.tools.zephyr_essential.api_wrapper import ZephyrEssentialApiWrapper


@pytest.fixture
def wrapper():
    return object.__new__(ZephyrEssentialApiWrapper)


def test_returns_steps_when_available(wrapper):
    with patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_steps", return_value=[{"description": "step 1"}]), \
         patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_script") as get_script:
        result = wrapper._process_test_case("PROJ-T1")

    assert result == {"steps": [{"description": "step 1"}]}
    get_script.assert_not_called()


def test_falls_back_to_script_when_steps_missing(wrapper):
    with patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_steps", return_value=None), \
         patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_script", return_value="print('hello')"):
        result = wrapper._process_test_case("PROJ-T2")

    assert result == {"script": "print('hello')"}


def test_falls_back_to_script_when_steps_raise(wrapper):
    with patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_steps",
                       side_effect=ToolException("Unable to extract test case steps")), \
         patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_script", return_value="print('hello')"):
        result = wrapper._process_test_case("PROJ-T3")

    assert result == {"script": "print('hello')"}


def test_raises_when_both_raise(wrapper):
    with patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_steps",
                       side_effect=ToolException("Unable to extract test case steps")), \
         patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_script",
                       side_effect=ToolException("Unable to get test script")):
        with pytest.raises(ToolException, match="Unable to get test script"):
            wrapper._process_test_case("PROJ-T4")


def test_returns_empty_when_both_missing(wrapper):
    with patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_steps", return_value=None), \
         patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_script", return_value=None):
        result = wrapper._process_test_case("PROJ-T5")

    assert result == {"empty": ""}


def essential_http_error(status, resource):
    return ToolException(
        f"Unexpected status code {status} for GET /testcases/PROJ-T6/{resource}: b'{{}}'")


def test_raises_a_server_error_after_empty_steps(wrapper):
    with patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_steps", return_value=[]), \
         patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_script",
                      side_effect=essential_http_error(500, "testscript")):
        with pytest.raises(ToolException, match="status code 500"):
            wrapper._process_test_case("PROJ-T7")


def test_returns_empty_for_a_step_case_without_steps(wrapper):
    with patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_steps", return_value=[]), \
         patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_script",
                      side_effect=essential_http_error(404, "testscript")):
        result = wrapper._process_test_case("PROJ-T8")

    assert result == {"empty": ""}


def test_returns_empty_for_a_plain_case_without_a_script(wrapper):
    with patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_steps",
                      side_effect=essential_http_error(422, "teststeps")), \
         patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_script",
                      side_effect=essential_http_error(404, "testscript")):
        result = wrapper._process_test_case("PROJ-T9")

    assert result == {"empty": ""}


def test_raises_when_the_test_case_itself_is_gone(wrapper):
    with patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_steps",
                      side_effect=essential_http_error(404, "teststeps")), \
         patch.object(ZephyrEssentialApiWrapper, "get_test_case_test_script",
                      side_effect=essential_http_error(404, "testscript")):
        with pytest.raises(ToolException, match="status code 404"):
            wrapper._process_test_case("PROJ-T10")
