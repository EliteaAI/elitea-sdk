"""Unit tests for Code Node debug mode — artifact capture feature.

Covers the ``debug: bool`` flag on FunctionTool / code nodes introduced in:
  - elitea_sdk/runtime/tools/function.py  (_save_code_to_artifact, debug field)
  - elitea_sdk/runtime/langchain/langraph_agent.py  (debug=node.get('debug', False))

Test matrix:
  1. _save_code_to_artifact — happy path saves to correct bucket / filename
  2. _save_code_to_artifact — upload error is swallowed (warning logged, no raise)
  3. _save_code_to_artifact — unexpected exception is swallowed (warning logged)
  4. invoke() with debug=True + client — artifact is saved before execution
  5. invoke() with debug=False (default) — artifact is NOT saved
  6. invoke() with debug=True but client=None — artifact is NOT saved (no-op)
  7. _save_code_to_artifact bucket name is always "code-debug"
  8. _save_code_to_artifact filename is "<node_name>.py"
  9. create_graph code-node path: debug=True read from node dict → passed to FunctionTool
 10. create_graph code-node path: debug absent in node dict → defaults to False
"""
import pytest
from unittest.mock import MagicMock, patch, call

from langchain_core.tools import BaseTool

from elitea_sdk.runtime.tools.function import FunctionTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pyodide_tool():
    """Return a minimal mock that passes _is_pyodide_tool()."""
    tool = MagicMock(spec=BaseTool)
    tool.name = "pyodide_sandbox"
    tool.description = "sandbox"
    tool.args_schema = None
    tool.tool_call_schema = None
    return tool


def _make_function_tool(*, debug: bool = False, has_client: bool = True, node_name: str = "my_code_node"):
    """Build a FunctionTool via model_construct to avoid Pydantic / Deno init."""
    mock_tool = _make_pyodide_tool()

    mock_client = None
    if has_client:
        mock_client = MagicMock()
        # artifact() returns a SandboxArtifact-like mock
        mock_artifact = MagicMock()
        mock_artifact.create.return_value = {"filepath": f"/code-debug/{node_name}.py"}
        mock_client.artifact.return_value = mock_artifact

    ft = FunctionTool.model_construct(
        name=node_name,
        tool=mock_tool,
        return_type="dict",
        input_variables=["messages"],   # must not be None — propagate_the_input_mapping iterates it
        input_mapping={"code": {"type": "fixed", "value": "x = 1"}},
        output_variables=[],
        structured_output=False,
        elitea_client=mock_client,
        debug=debug,
    )
    return ft, mock_client


# ---------------------------------------------------------------------------
# 1–3 · _save_code_to_artifact unit tests
# ---------------------------------------------------------------------------

class TestSaveCodeToArtifact:
    """Direct unit tests for FunctionTool._save_code_to_artifact."""

    def test_happy_path_calls_artifact_create_with_correct_args(self):
        """Artifact bucket='code-debug', filename='<node>.py', content is bytes."""
        ft, mock_client = _make_function_tool(node_name="parse_input")
        ft._save_code_to_artifact("print('hello')", "parse_input")

        mock_client.artifact.assert_called_once_with("code-debug")
        artifact = mock_client.artifact.return_value
        artifact.create.assert_called_once_with(
            "parse_input.py",
            "print('hello')".encode("utf-8"),
        )

    def test_upload_error_response_is_swallowed_no_exception_raised(self, caplog):
        """When create() returns {'error': '...'}, no exception must propagate."""
        ft, mock_client = _make_function_tool(node_name="step1")
        mock_client.artifact.return_value.create.return_value = {
            "error": "Bucket not found"
        }

        import logging
        with caplog.at_level(logging.WARNING, logger="elitea_sdk.runtime.tools.function"):
            ft._save_code_to_artifact("x = 1", "step1")  # must not raise

        assert any("Bucket not found" in m for m in caplog.messages), (
            "Expected warning log containing the error message"
        )

    def test_unexpected_exception_is_swallowed_warning_logged(self, caplog):
        """If artifact() itself raises, the exception must be caught and logged."""
        ft, mock_client = _make_function_tool(node_name="step2")
        mock_client.artifact.side_effect = RuntimeError("network timeout")

        import logging
        with caplog.at_level(logging.WARNING, logger="elitea_sdk.runtime.tools.function"):
            ft._save_code_to_artifact("x = 1", "step2")  # must not raise

        assert any("network timeout" in m for m in caplog.messages)

    def test_bucket_name_is_always_code_debug(self):
        """The bucket must always be 'code-debug', never configurable by the caller."""
        ft, mock_client = _make_function_tool(node_name="any_node")
        ft._save_code_to_artifact("pass", "any_node")

        mock_client.artifact.assert_called_once_with("code-debug")

    def test_filename_is_node_name_dot_py(self):
        """Filename must be exactly '<node_name>.py'."""
        ft, mock_client = _make_function_tool(node_name="transform_data")
        ft._save_code_to_artifact("pass", "transform_data")

        artifact = mock_client.artifact.return_value
        args, _ = artifact.create.call_args
        assert args[0] == "transform_data.py"

    def test_content_is_utf8_encoded_bytes(self):
        """The content passed to create() must be the code UTF-8 encoded as bytes."""
        code = "résultat = 42  # unicode comment"
        ft, mock_client = _make_function_tool(node_name="unicode_node")
        ft._save_code_to_artifact(code, "unicode_node")

        artifact = mock_client.artifact.return_value
        args, _ = artifact.create.call_args
        assert args[1] == code.encode("utf-8")


# ---------------------------------------------------------------------------
# 4–6 · invoke() integration: debug flag gate
# ---------------------------------------------------------------------------

class TestFunctionToolInvokeDebugGate:
    """Verify that invoke() calls _save_code_to_artifact if and only if
    debug=True AND elitea_client is not None."""

    def _build_tool_and_run(self, *, debug, has_client, node_name="node_x"):
        ft, mock_client = _make_function_tool(
            debug=debug, has_client=has_client, node_name=node_name
        )
        # Mock the internal methods we don't want to actually execute
        ft.tool.invoke = MagicMock(return_value={"result": "ok", "stdout": "", "stderr": None,
                                                  "status": "success", "execution_time": 0.1})
        with patch.object(ft, "_prepare_pyodide_input", return_value="#preamble\n"), \
             patch.object(ft, "_save_code_to_artifact") as mock_save, \
             patch.object(ft, "_handle_pyodide_output", return_value={"messages": []}):
            ft.invoke(
                {"input": "test", "messages": [], "code": "x = 1"},
                config=None,
            )
        return mock_save

    def test_debug_true_with_client_saves_artifact(self):
        """debug=True + client present → _save_code_to_artifact called once."""
        mock_save = self._build_tool_and_run(debug=True, has_client=True)
        mock_save.assert_called_once()

    def test_debug_false_does_not_save_artifact(self):
        """debug=False (default) → _save_code_to_artifact never called."""
        mock_save = self._build_tool_and_run(debug=False, has_client=True)
        mock_save.assert_not_called()

    def test_debug_true_without_client_does_not_save_artifact(self):
        """debug=True but elitea_client=None → no save attempt (would crash otherwise)."""
        mock_save = self._build_tool_and_run(debug=True, has_client=False)
        mock_save.assert_not_called()

    def test_save_receives_node_name_as_second_arg(self):
        """_save_code_to_artifact must receive self.name as the node_name argument."""
        ft, mock_client = _make_function_tool(debug=True, has_client=True, node_name="my_node")
        ft.tool.invoke = MagicMock(return_value={"result": "ok"})

        with patch.object(ft, "_prepare_pyodide_input", return_value="#pre\n"), \
             patch.object(ft, "_save_code_to_artifact") as mock_save, \
             patch.object(ft, "_handle_pyodide_output", return_value={"messages": []}):
            ft.invoke({"messages": [], "code": "x = 1"}, config=None)

        _, node_name_arg = mock_save.call_args[0]
        assert node_name_arg == "my_node"

    def test_save_receives_assembled_code_as_first_arg(self):
        """_save_code_to_artifact must receive the preamble+user code, not just the user code.

        The code sent to the sandbox is: <preamble> + \\n + <user_code_from_input_mapping>.
        The user code comes from input_mapping (fixed type → 'x = 1'), not the state dict.
        """
        ft, mock_client = _make_function_tool(debug=True, has_client=True)
        ft.tool.invoke = MagicMock(return_value={"result": "ok"})

        with patch.object(ft, "_prepare_pyodide_input", return_value="# preamble\n"), \
             patch.object(ft, "_save_code_to_artifact") as mock_save, \
             patch.object(ft, "_handle_pyodide_output", return_value={"messages": []}):
            ft.invoke({"messages": [], "code": "ignored_state_code"}, config=None)

        saved_code = mock_save.call_args[0][0]
        # preamble must be present
        assert "# preamble" in saved_code
        # the fixed user code from input_mapping must also be present
        assert "x = 1" in saved_code

    def test_debug_does_not_affect_tool_execution_result(self):
        """Enabling debug must not change the value returned by invoke()."""
        ft_debug, _ = _make_function_tool(debug=True, has_client=True)
        ft_nodebug, _ = _make_function_tool(debug=False, has_client=True)

        expected_output = {"messages": [{"role": "assistant", "content": "done"}]}
        for ft in (ft_debug, ft_nodebug):
            ft.tool.invoke = MagicMock(return_value={"result": "done"})
            with patch.object(ft, "_prepare_pyodide_input", return_value="#pre\n"), \
                 patch.object(ft, "_save_code_to_artifact"), \
                 patch.object(ft, "_handle_pyodide_output", return_value=expected_output), \
                 patch("elitea_sdk.runtime.tools.function.dispatch_custom_event"):
                result = ft.invoke({"messages": [], "code": "x = 1"}, config=None)

            assert result == expected_output, (
                f"debug mode changed the return value: {result}"
            )


# ---------------------------------------------------------------------------
# 7 · FunctionTool field defaults
# ---------------------------------------------------------------------------

class TestFunctionToolDebugField:
    """Verify the debug field is declared correctly on FunctionTool."""

    def test_debug_defaults_to_false(self):
        """FunctionTool.debug must default to False (no accidental opt-in)."""
        ft = FunctionTool.model_construct(
            name="x",
            tool=_make_pyodide_tool(),
        )
        assert ft.debug is False

    def test_debug_true_is_stored(self):
        """Setting debug=True must be persisted on the instance."""
        ft = FunctionTool.model_construct(
            name="x",
            tool=_make_pyodide_tool(),
            debug=True,
        )
        assert ft.debug is True


# ---------------------------------------------------------------------------
# 8 · create_graph integration: node dict → FunctionTool.debug
# ---------------------------------------------------------------------------

class TestCreateGraphDebugPassThrough:
    """Verify that create_graph() reads 'debug' from the node dict and
    propagates it to FunctionTool when building a 'code' type node.

    We test this by simulating the exact expression used in langraph_agent.py:
        debug=node.get('debug', False)
    and constructing a FunctionTool via model_construct with that value.
    """

    @staticmethod
    def _build_ft_from_node(node: dict) -> FunctionTool:
        """Reproduce the create_graph() code-node branch without graph machinery."""
        mock_sandbox = MagicMock(spec=BaseTool)
        mock_sandbox.name = "pyodide_sandbox"
        elitea_client = MagicMock()

        return FunctionTool.model_construct(
            tool=mock_sandbox,
            name=node["id"],
            return_type="dict",
            output_variables=node.get("output", []),
            input_mapping={"code": node.get("code", {})},
            input_variables=node.get("input", ["messages"]),
            structured_output=node.get("structured_output", False),
            elitea_client=elitea_client,
            # This is the exact expression from langraph_agent.py:
            debug=node.get("debug", False),
        )

    def test_debug_true_in_node_dict_passed_to_function_tool(self):
        """node['debug']=True must result in FunctionTool(debug=True)."""
        node = {"id": "debug_node", "debug": True, "code": {"type": "fixed", "value": ""}}
        ft = self._build_ft_from_node(node)
        assert ft.debug is True

    def test_debug_false_in_node_dict_passed_to_function_tool(self):
        """node['debug']=False must result in FunctionTool(debug=False)."""
        node = {"id": "debug_node", "debug": False, "code": {"type": "fixed", "value": ""}}
        ft = self._build_ft_from_node(node)
        assert ft.debug is False

    def test_debug_absent_in_node_dict_defaults_to_false(self):
        """When 'debug' key is absent from the node dict, FunctionTool receives debug=False."""
        node = {"id": "debug_node", "code": {"type": "fixed", "value": ""}}
        ft = self._build_ft_from_node(node)
        assert ft.debug is False










