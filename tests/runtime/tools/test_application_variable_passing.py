"""
Tests for Application tool variable passing in nested pipelines.

This module tests the fix for the issue where parent pipeline variable values
were not being passed to child pipelines when both define the same variable.

The root cause was that BaseTool._parse_input() validates against args_schema
and strips keys not in the schema. Since Application's args_schema only has
'task' and 'chat_history', extra variables (like pipeline state variables)
were lost before reaching _run().

The fix passes extras through config['configurable']['_application_extras']
which survives the BaseTool validation pipeline.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from langchain_core.messages import HumanMessage
from langchain_core.tools import ToolException

# Import the Application class and related functions
from elitea_sdk.runtime.toolkits.application import build_dynamic_application_schema
from elitea_sdk.runtime.tools.application import (
    Application,
    applicationToolSchema,
    formulate_query,
)


class TestApplicationVariablePassing:
    """Test cases for Application tool variable passing through nested invocations."""

    def test_extras_stored_in_config_during_invoke(self):
        """Test that extra variables (beyond task/chat_history) are stored in config."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"output": "test result"}
        mock_client = MagicMock()
        mock_client.application.return_value = mock_app

        app_tool = Application(
            name="TestApp",
            description="Test application",
            application=mock_app,
            client=mock_client,
            args_runnable={},
        )

        # Invoke with extra variables beyond task/chat_history
        input_with_extras = {
            "task": "Do something",
            "chat_history": [],
            "my_test_var": "parent_value",
            "another_var": 123,
        }

        # Patch super().invoke to capture what config is passed
        captured_config = {}

        def capture_invoke(self_arg, input_dict, config=None, **kwargs):
            captured_config['config'] = config
            # Return a mock result instead of calling _run to avoid side effects
            return "mock result"

        with patch.object(Application.__bases__[0], 'invoke', capture_invoke):
            app_tool.invoke(input_with_extras)

        # Verify extras were stored in config
        assert captured_config['config'] is not None
        assert 'configurable' in captured_config['config']
        assert '_application_extras' in captured_config['config']['configurable']
        extras = captured_config['config']['configurable']['_application_extras']
        assert extras['my_test_var'] == 'parent_value'
        assert extras['another_var'] == 123

    def test_extras_retrieved_in_run(self):
        """Test that _run() retrieves extras from config and uses them."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"output": "test result"}
        mock_client = MagicMock()
        mock_client.application.return_value = mock_app

        app_tool = Application(
            name="TestApp",
            description="Test application",
            application=mock_app,
            client=mock_client,
            args_runnable={"agent_id": 123},
            variable_defaults={"my_test_var": "default_value"},
        )

        # Simulate config with extras as it would be passed from invoke()
        config_with_extras = {
            'configurable': {
                '_application_extras': {
                    'my_test_var': 'parent_value',
                    'new_var': 'extra_value',
                }
            },
            'metadata': {},
        }

        # Call _run directly with the config
        app_tool._run(
            config=config_with_extras,
            task="Do something",
            chat_history=[],
        )

        # Verify client.application was called with merged variables
        call_kwargs = mock_client.application.call_args.kwargs
        app_vars = call_kwargs['application_variables']

        # my_test_var should be 'parent_value' (from extras), not 'default_value'
        assert app_vars['my_test_var']['value'] == 'parent_value'
        # new_var should be present from extras
        assert app_vars['new_var']['value'] == 'extra_value'

    def test_parent_value_overrides_child_default(self):
        """Test that parent pipeline's variable value overrides child's default."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"output": "result"}
        mock_client = MagicMock()
        mock_client.application.return_value = mock_app

        # Child has default value for my_var
        app_tool = Application(
            name="ChildApp",
            description="Child application",
            application=mock_app,
            client=mock_client,
            args_runnable={"agent_id": 456},
            variable_defaults={"my_var": "child_default", "child_only": "stays"},
        )

        # Parent passes value for my_var
        config_with_parent_value = {
            'configurable': {
                '_application_extras': {
                    'my_var': 'parent_value',
                }
            },
            'metadata': {},
        }

        app_tool._run(
            config=config_with_parent_value,
            task="Execute child",
            chat_history=[],
        )

        call_kwargs = mock_client.application.call_args.kwargs
        app_vars = call_kwargs['application_variables']

        # Parent value should override child default
        assert app_vars['my_var']['value'] == 'parent_value'
        # Child-only variable should keep its default
        assert app_vars['child_only']['value'] == 'stays'

    def test_formulate_query_passes_extras(self):
        """Test that formulate_query includes extra variables in result."""
        kwargs = {
            'task': 'Do the thing',
            'chat_history': [HumanMessage(content='Previous context must stay parent-side')],
            'custom_var': 'custom_value',
            'another_var': 42,
        }

        result = formulate_query(kwargs, is_subgraph=False)

        # Should have input and extras, but no chat_history in the sub-agent payload.
        assert 'input' in result
        assert 'chat_history' not in result
        assert result['custom_var'] == 'custom_value'
        assert result['another_var'] == 42

    def test_formulate_query_omits_chat_history_for_sub_agent_payload(self):
        """Test that chat_history is accepted but never forwarded to sub-agents."""
        result = formulate_query(
            {
                'task': 'Summarize all required context from the task only',
                'chat_history': [HumanMessage(content='Do not forward me')],
            },
            is_subgraph=False,
        )

        assert 'chat_history' not in result
        assert len(result['input']) == 1
        assert result['input'][0].content == 'Summarize all required context from the task only'

    def test_empty_string_parent_value_overrides_empty_child_default(self):
        """Test that non-None parent values (including empty string) override defaults."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"output": "result"}
        mock_client = MagicMock()
        mock_client.application.return_value = mock_app

        # Child has empty default
        app_tool = Application(
            name="ChildApp",
            description="Child application",
            application=mock_app,
            client=mock_client,
            args_runnable={"agent_id": 789},
            variable_defaults={"var": ""},
        )

        # Parent passes actual value
        config_with_parent_value = {
            'configurable': {
                '_application_extras': {
                    'var': 'filled_value',
                }
            },
            'metadata': {},
        }

        app_tool._run(
            config=config_with_parent_value,
            task="Execute",
            chat_history=[],
        )

        call_kwargs = mock_client.application.call_args.kwargs
        app_vars = call_kwargs['application_variables']

        # Parent's filled value should override child's empty default
        assert app_vars['var']['value'] == 'filled_value'


class TestApplicationSchemaHandling:
    """Test that args_schema handling preserves extra variables."""

    def test_args_schema_only_has_task_and_chat_history(self):
        """Verify applicationToolSchema only defines task and chat_history."""
        schema_fields = applicationToolSchema.model_fields
        assert 'task' in schema_fields
        assert 'chat_history' in schema_fields
        # Only these two fields
        assert len(schema_fields) == 2

    def test_dynamic_application_schema_does_not_expose_chat_history(self):
        """Verify parent LLM-facing application schemas only expose task and variables."""
        schema = build_dynamic_application_schema(
            [{'name': 'region', 'description': 'Deployment region', 'value': 'us'}],
            app_name='Child Agent',
        )

        assert 'task' in schema.model_fields
        assert 'region' in schema.model_fields
        assert 'chat_history' not in schema.model_fields

    def test_extras_calculation_in_invoke(self):
        """Test that extras correctly identifies non-schema fields."""
        mock_app = MagicMock()
        mock_client = MagicMock()

        app_tool = Application(
            name="TestApp",
            description="Test",
            application=mock_app,
            client=mock_client,
            args_runnable={},
        )

        input_dict = {
            'task': 'Do task',
            'chat_history': [],
            'extra1': 'val1',
            'extra2': 'val2',
        }

        # Calculate what should be extras (fields not in args_schema)
        schema_values = applicationToolSchema(**input_dict).model_dump()
        extras = {k: v for k, v in input_dict.items() if k not in schema_values}

        assert 'task' not in extras
        assert 'chat_history' not in extras
        assert extras['extra1'] == 'val1'
        assert extras['extra2'] == 'val2'

    def test_dynamic_schema_ignores_legacy_chat_history_as_extra(self):
        """Old callers may still pass chat_history; it must not become an app variable."""
        mock_app = MagicMock()
        mock_client = MagicMock()
        dynamic_schema = build_dynamic_application_schema([], app_name='Child Agent')

        app_tool = Application(
            name='TestApp',
            description='Test',
            application=mock_app,
            client=mock_client,
            args_runnable={},
            args_schema=dynamic_schema,
        )

        captured_config = {}

        def capture_invoke(self_arg, input_dict, config=None, **kwargs):
            captured_config['config'] = config
            return 'mock result'

        with patch.object(Application.__bases__[0], 'invoke', capture_invoke):
            app_tool.invoke({
                'task': 'Do task',
                'chat_history': [HumanMessage(content='legacy history')],
                'extra1': 'val1',
            })

        extras = captured_config['config']['configurable']['_application_extras']
        assert 'chat_history' not in extras
        assert extras['extra1'] == 'val1'


class TestNestedPipelineScenario:
    """
    Integration-style tests simulating the nested pipeline scenario.

    Scenario: Parent pipeline sets variable "foo", then calls child pipeline
    that also has variable "foo". Child should see parent's value.
    """

    def test_parent_to_child_variable_flow(self):
        """Simulate the full flow of variable passing from parent to child pipeline."""
        # Child application mock
        child_app = MagicMock()
        child_app.invoke.return_value = {"output": "child executed with foo=parent_foo_value"}
        child_client = MagicMock()
        child_client.application.return_value = child_app

        # Create child Application tool (as it would be in parent's toolkit)
        child_tool = Application(
            name="ChildPipeline",
            description="Child pipeline with foo variable",
            application=child_app,
            client=child_client,
            args_runnable={"agent_id": 100},
            variable_defaults={"foo": "child_default_foo"},  # Child's default
        )

        # Simulate FunctionTool passing state to child Application
        # This is what happens in FunctionTool.invoke() when calling an Application
        func_args = {
            "task": "Execute child pipeline",
            "chat_history": [],
            "foo": "parent_foo_value",  # Parent's value for foo
        }

        # Call invoke (which stores extras in config) then _run retrieves them
        # Simulate the actual flow through super().invoke()
        schema_values = applicationToolSchema(**func_args).model_dump()
        extras = {k: v for k, v in func_args.items() if k not in schema_values}

        config = {
            'configurable': {'_application_extras': extras},
            'metadata': {},
        }

        # Call _run with the config (as BaseTool would)
        child_tool._run(
            config=config,
            task="Execute child pipeline",
            chat_history=[],
        )

        # Verify child received parent's foo value
        call_kwargs = child_client.application.call_args.kwargs
        app_vars = call_kwargs['application_variables']

        assert 'foo' in app_vars
        assert app_vars['foo']['value'] == 'parent_foo_value'  # NOT 'child_default_foo'

    def test_child_application_payload_omits_chat_history(self):
        """Sub-agent invocation payload uses task only, even when parent passes history."""
        child_app = MagicMock()
        child_app.invoke.return_value = {'output': 'child executed'}
        child_tool = Application(
            name='ChildAgent',
            description='Child agent',
            application=child_app,
            client=None,
            args_runnable={},
        )

        child_tool._run(
            config={'configurable': {}, 'metadata': {}},
            task='Complete this task with all context here',
            chat_history=[HumanMessage(content='parent history')],
        )

        child_payload = child_app.invoke.call_args.args[0]
        assert 'input' in child_payload
        assert 'chat_history' not in child_payload


class TestChildToParentVariablePropagation:
    """
    Tests for propagating state variables from child pipeline back to parent.

    When a child pipeline modifies a state variable, the modified value should
    be available in the parent pipeline's state after the child completes.
    """

    def test_child_state_variables_returned_in_result(self):
        """Test that child's state variables are included in _run() result."""
        # Child application returns response with state variables
        child_app = MagicMock()
        child_app.invoke.return_value = {
            "messages": [{"role": "assistant", "content": "Done"}],
            "output": "Child completed",
            "my_var": "child_modified_value",
            "another_var": 42,
        }
        child_client = MagicMock()
        child_client.application.return_value = child_app

        child_tool = Application(
            name="ChildPipeline",
            description="Child pipeline",
            application=child_app,
            client=child_client,
            args_runnable={"agent_id": 200},
            is_subgraph=True,  # Mark as subgraph to get full dict result
        )

        config = {'configurable': {}, 'metadata': {}}
        result = child_tool._run(
            config=config,
            task="Execute",
            chat_history=[],
        )

        # Result should be a dict containing state variables from child
        assert isinstance(result, dict)
        assert 'messages' in result
        assert result['my_var'] == 'child_modified_value'
        assert result['another_var'] == 42

    def test_run_always_returns_dict_with_state_variables(self):
        """Test that _run() always returns dict with state variables (invoke handles conversion)."""
        child_app = MagicMock()
        child_app.invoke.return_value = {
            "output": "Child output text",
            "my_var": "should_be_returned",
        }
        child_client = MagicMock()
        child_client.application.return_value = child_app

        child_tool = Application(
            name="ChildTool",
            description="Standalone child tool",
            application=child_app,
            client=child_client,
            args_runnable={"agent_id": 300},
            return_type="str",
            is_subgraph=False,  # Even when not subgraph, _run returns full dict
        )

        config = {'configurable': {}, 'metadata': {}}
        result = child_tool._run(
            config=config,
            task="Execute",
            chat_history=[],
        )

        # _run() now always returns a dict with state variables
        # The invoke() method handles converting to string for LLM agent calls
        assert isinstance(result, dict)
        assert 'messages' in result
        assert result['my_var'] == 'should_be_returned'

    def test_excluded_keys_not_propagated(self):
        """Test that internal keys (messages, output, input, etc.) are not duplicated."""
        child_app = MagicMock()
        child_app.invoke.return_value = {
            "messages": [{"role": "assistant", "content": "Done"}],
            "output": "Output text",
            "input": "Should not propagate",
            "chat_history": [{"role": "user", "content": "test"}],
            "user_var": "should_propagate",
        }
        child_client = MagicMock()
        child_client.application.return_value = child_app

        child_tool = Application(
            name="ChildPipeline",
            description="Child pipeline",
            application=child_app,
            client=child_client,
            args_runnable={"agent_id": 400},
            is_subgraph=True,
        )

        config = {'configurable': {}, 'metadata': {}}
        result = child_tool._run(
            config=config,
            task="Execute",
            chat_history=[],
        )

        # user_var should propagate
        assert result['user_var'] == 'should_propagate'
        # These should not be duplicated/propagated as state vars
        assert 'input' not in result
        assert 'output' not in result
        # messages is the result format, should exist
        assert 'messages' in result

    def test_bidirectional_variable_flow(self):
        """Test full round-trip: parent sends value to child, child modifies and returns."""
        # Child modifies the variable and returns it
        child_app = MagicMock()
        child_app.invoke.return_value = {
            "output": "Processed",
            "shared_var": "modified_by_child",  # Child modified the value
        }
        child_client = MagicMock()
        child_client.application.return_value = child_app

        child_tool = Application(
            name="ChildPipeline",
            description="Child that modifies shared_var",
            application=child_app,
            client=child_client,
            args_runnable={"agent_id": 500},
            variable_defaults={"shared_var": "child_default"},
            is_subgraph=True,
        )

        # Parent passes its value of shared_var
        config = {
            'configurable': {'_application_extras': {'shared_var': 'parent_value'}},
            'metadata': {},
        }
        result = child_tool._run(
            config=config,
            task="Process and modify shared_var",
            chat_history=[],
        )

        # Verify child received parent's value
        call_kwargs = child_client.application.call_args.kwargs
        app_vars = call_kwargs['application_variables']
        assert app_vars['shared_var']['value'] == 'parent_value'

        # Verify child's modified value is returned to parent
        assert result['shared_var'] == 'modified_by_child'


class TestFunctionToolNestedAppDetection:
    """Test that FunctionTool properly detects nested Application tools."""

    def test_is_nested_app_detection_by_is_subgraph(self):
        """Test that is_nested_app is True when tool has is_subgraph=True."""
        # This tests the detection logic in isolation
        class MockTool:
            is_subgraph = True

        tool = MockTool()
        is_nested_app = (
            (hasattr(tool, 'is_subgraph') and tool.is_subgraph) or
            type(tool).__name__ == 'Application'
        )
        assert is_nested_app is True

    def test_is_nested_app_detection_by_type_name(self):
        """Test that is_nested_app is True when tool class name is 'Application'."""

        class Application:
            pass

        tool = Application()
        is_nested_app = (
            (hasattr(tool, 'is_subgraph') and tool.is_subgraph) or
            type(tool).__name__ == 'Application'
        )
        assert is_nested_app is True

    def test_is_nested_app_false_for_regular_tool(self):
        """Test that is_nested_app is False for regular tools."""

        class RegularTool:
            is_subgraph = False

        tool = RegularTool()
        is_nested_app = (
            (hasattr(tool, 'is_subgraph') and tool.is_subgraph) or
            type(tool).__name__ == 'Application'
        )
        assert is_nested_app is False

    def test_is_nested_app_false_when_no_is_subgraph(self):
        """Test that is_nested_app is False when tool has no is_subgraph attr."""

        class RegularTool:
            pass

        tool = RegularTool()
        is_nested_app = (
            (hasattr(tool, 'is_subgraph') and tool.is_subgraph) or
            type(tool).__name__ == 'Application'
        )
        assert is_nested_app is False
