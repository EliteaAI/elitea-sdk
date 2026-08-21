"""
Regression tests for ZephyrScaleApiWrapper._process_test_case.

Covers the fallback chain (steps -> script -> empty) now that get_test_steps
and get_test_script raise ToolException instead of returning it, proving the
try/except conversion preserves the original returned-exception behavior.
"""
import pytest
from unittest.mock import patch

from langchain_core.tools import ToolException

from elitea_sdk.tools.zephyr_scale.api_wrapper import ZephyrScaleApiWrapper


@pytest.fixture
def wrapper():
    return object.__new__(ZephyrScaleApiWrapper)


def test_returns_steps_when_available(wrapper):
    with patch.object(ZephyrScaleApiWrapper, "get_test_steps", return_value=[{"description": "step 1"}]), \
         patch.object(ZephyrScaleApiWrapper, "get_test_script") as get_script:
        result = wrapper._process_test_case("PROJ-T1")

    assert result == {"steps": [{"description": "step 1"}]}
    get_script.assert_not_called()


def test_falls_back_to_script_when_steps_missing(wrapper):
    with patch.object(ZephyrScaleApiWrapper, "get_test_steps", return_value=None), \
         patch.object(ZephyrScaleApiWrapper, "get_test_script", return_value="print('hello')"):
        result = wrapper._process_test_case("PROJ-T2")

    assert result == {"script": "print('hello')"}


def test_falls_back_to_script_when_steps_raise(wrapper):
    with patch.object(ZephyrScaleApiWrapper, "get_test_steps",
                       side_effect=ToolException("Unable to extract test case steps")), \
         patch.object(ZephyrScaleApiWrapper, "get_test_script", return_value="print('hello')"):
        result = wrapper._process_test_case("PROJ-T3")

    assert result == {"script": "print('hello')"}


def test_returns_empty_when_both_raise(wrapper):
    with patch.object(ZephyrScaleApiWrapper, "get_test_steps",
                       side_effect=ToolException("Unable to extract test case steps")), \
         patch.object(ZephyrScaleApiWrapper, "get_test_script",
                       side_effect=ToolException("Unable to get test script")):
        result = wrapper._process_test_case("PROJ-T4")

    assert result == {"empty": ""}


def test_returns_empty_when_both_missing(wrapper):
    with patch.object(ZephyrScaleApiWrapper, "get_test_steps", return_value=None), \
         patch.object(ZephyrScaleApiWrapper, "get_test_script", return_value=None):
        result = wrapper._process_test_case("PROJ-T5")

    assert result == {"empty": ""}
