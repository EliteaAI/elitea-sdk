"""Regression tests for TestPlanApiWrapper.create_test_case's consumer of
self._work_item_wrapper.create_work_item.

No test previously existed for test_plan_wrapper.py. create_work_item now
raises ToolException instead of returning it; create_test_case must catch it,
enhance validation-style errors with field-discovery guidance, and re-raise
other errors unchanged.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.ado.test_plan.test_plan_wrapper import TestPlanApiWrapper


def _make_wrapper():
    wrapper = TestPlanApiWrapper.model_construct(
        organization_url="https://dev.azure.com/org", project="proj", token="tok", limit=5
    )
    wrapper._client = MagicMock()
    wrapper._work_item_wrapper = MagicMock()
    return wrapper


def _steps():
    return '[{"stepNumber": 1, "action": "do it", "expectedResult": "works"}]'


def test_validation_error_is_enhanced_with_field_discovery_hint():
    wrapper = _make_wrapper()
    wrapper._work_item_wrapper.create_work_item.side_effect = ToolException(
        "TF401320: Rule Validation Error for field 'Custom.SDLC'."
    )

    with pytest.raises(ToolException, match="get_all_test_case_fields_for_project"):
        wrapper.create_test_case(1, 2, "title", "desc", _steps())


def test_non_validation_error_propagates_unchanged():
    wrapper = _make_wrapper()
    err = ToolException("Connection to Azure DevOps timed out")
    wrapper._work_item_wrapper.create_work_item.side_effect = err

    with pytest.raises(ToolException, match="Connection to Azure DevOps timed out") as excinfo:
        wrapper.create_test_case(1, 2, "title", "desc", _steps())

    assert "get_all_test_case_fields_for_project" not in str(excinfo.value)


def test_successful_creation_adds_the_new_work_item_to_the_suite():
    wrapper = _make_wrapper()
    wrapper._work_item_wrapper.create_work_item.return_value = {"id": 42}

    with patch.object(TestPlanApiWrapper, "add_test_case", return_value=["ok"]) as add_test_case:
        result = wrapper.create_test_case(1, 2, "title", "desc", _steps())

    assert result == ["ok"]
    add_test_case.assert_called_once_with([{"work_item": {"id": 42}}], 1, 2)
