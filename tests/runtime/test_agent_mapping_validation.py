import pytest
from langchain_core.tools import ToolException

from elitea_sdk.runtime.langchain.langraph_agent import validate_agent_mappings


def test_validate_agent_mappings_accepts_task_without_chat_history():
    """Agent pipeline nodes no longer need to map chat_history."""
    input_mapping = {
        'task': {'type': 'fixed', 'value': 'Run child agent'},
    }
    node = {'id': 'child_agent', 'input_mapping': input_mapping}

    assert validate_agent_mappings(node) == input_mapping


def test_validate_agent_mappings_accepts_legacy_chat_history_mapping():
    """Existing persisted pipelines may still include chat_history mappings."""
    input_mapping = {
        'task': {'type': 'fixed', 'value': 'Run child agent'},
        'chat_history': {'type': 'variable', 'value': 'messages'},
    }
    node = {'id': 'child_agent', 'input_mapping': input_mapping}

    assert validate_agent_mappings(node) == input_mapping


def test_validate_agent_mappings_still_requires_task():
    """Task remains the required sub-agent input contract."""
    node = {
        'id': 'child_agent',
        'input_mapping': {'chat_history': {'type': 'variable', 'value': 'messages'}},
    }

    with pytest.raises(ToolException, match='task'):
        validate_agent_mappings(node)
