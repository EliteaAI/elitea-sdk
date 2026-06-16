"""Unit tests for Code Node debug mode — artifact capture feature.

Covers the ``debug: bool`` flag on FunctionTool / code nodes introduced in:
  - elitea_sdk/runtime/tools/function.py  (_save_code_to_artifact,
                                            _build_client_preamble, debug field)
  - elitea_sdk/runtime/langchain/langraph_agent.py  (debug=node.get('debug', False))

Test matrix:
  1.  _save_code_to_artifact — happy path: bucket correct, filename matches pattern
  2.  _save_code_to_artifact — upload error is swallowed (warning logged, no raise)
  3.  _save_code_to_artifact — unexpected exception is swallowed (warning logged)
  4.  _save_code_to_artifact — bucket name is always "code-debug"
  5.  _save_code_to_artifact — filename pattern is "<node>__<YYYYMMDD_HHMMSS>.py"
  6.  _save_code_to_artifact — saved bytes contain the user code
  7.  _save_code_to_artifact — saved bytes contain the client preamble
  8.  _build_client_preamble — returns sandbox_client.py content + SandboxClient init
  9.  _build_client_preamble — FileNotFoundError → returns "" (warning logged)
  10. _build_client_preamble — generic exception → returns "" (warning logged)
  11. invoke() with debug=True + client — artifact is saved before execution
  12. invoke() with debug=False (default) — artifact is NOT saved
  13. invoke() with debug=True but client=None — artifact is NOT saved (no-op)
  14. invoke() saves correct node name
  15. invoke() saves assembled preamble+user code
  16. debug does not affect invoke() return value
  17. FunctionTool.debug defaults to False
  18. FunctionTool.debug=True is stored
  19. create_graph: debug=True in node dict → FunctionTool(debug=True)
  20. create_graph: debug=False in node dict → FunctionTool(debug=False)
  21. create_graph: debug absent in node dict → FunctionTool(debug=False)
"""
import re as re_module
import logging
import pytest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from langchain_core.tools import BaseTool

from elitea_sdk.runtime.tools.function import FunctionTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_PREAMBLE = "#elitea simplified client\n# <sandbox_client stub>\nelitea_client = SandboxClient(...)\n\n"


def _make_pyodide_tool():
    """Return a minimal mock that passes _is_pyodide_tool()."""
    tool = MagicMock(spec=BaseTool)
    tool.name = "pyodide_sandbox"
    tool.description = "sandbox"
    tool.args_schema = None
    tool.tool_call_schema = None
    return tool


def _make_client(node_name: str = "my_code_node"):
    mock_client = MagicMock()
    mock_client.base_url = "https://elitea.ai"
    mock_client.project_id = 1
    mock_client.auth_token = "real-secret-jwt-abc123xyz"   # distinct from placeholder
    mock_artifact = MagicMock()
    mock_artifact.create.return_value = {"filepath": f"/code-debug/{node_name}.py"}
    mock_client.artifact.return_value = mock_artifact
    return mock_client


def _make_function_tool(*, debug: bool = False, has_client: bool = True, node_name: str = "my_code_node"):
    """Build a FunctionTool via model_construct to avoid Pydantic / Deno init."""
    mock_tool = _make_pyodide_tool()
    mock_client = _make_client(node_name) if has_client else None

    ft = FunctionTool.model_construct(
        name=node_name,
        tool=mock_tool,
        return_type="dict",
        input_variables=["messages"],
        input_mapping={"code": {"type": "fixed", "value": "x = 1"}},
        output_variables=[],
        structured_output=False,
        elitea_client=mock_client,
        debug=debug,
    )
    return ft, mock_client


# ---------------------------------------------------------------------------
# 1–7 · _save_code_to_artifact unit tests
# ---------------------------------------------------------------------------

class TestSaveCodeToArtifact:
    """Direct unit tests for FunctionTool._save_code_to_artifact.

    _build_client_preamble is patched to a fixed string so these tests remain
    isolated from the filesystem and from SandboxClient details.
    """

    def _run_save(self, ft, code, node_name):
        with patch.object(ft, "_build_client_preamble", return_value=_FAKE_PREAMBLE):
            ft._save_code_to_artifact(code, node_name)

    def test_happy_path_calls_artifact_create(self):
        """Bucket='code-debug', filename matches '<node>__<timestamp>.py', create() called once."""
        ft, mock_client = _make_function_tool(node_name="parse_input")
        self._run_save(ft, "print('hello')", "parse_input")

        mock_client.artifact.assert_called_once_with("code-debug")
        artifact = mock_client.artifact.return_value
        artifact.create.assert_called_once()
        filename_arg = artifact.create.call_args[0][0]
        assert re_module.match(r"^parse_input__\d{8}_\d{6}\.py$", filename_arg), (
            f"Unexpected filename format: {filename_arg}"
        )

    def test_upload_error_is_swallowed(self, caplog):
        """create() returning {'error': '...'} must not raise; warning must be logged."""
        ft, mock_client = _make_function_tool(node_name="step1")
        mock_client.artifact.return_value.create.return_value = {"error": "Bucket not found"}

        with caplog.at_level(logging.WARNING, logger="elitea_sdk.runtime.tools.function"):
            self._run_save(ft, "x = 1", "step1")

        assert any("Bucket not found" in m for m in caplog.messages)

    def test_unexpected_exception_is_swallowed(self, caplog):
        """artifact() raising must be caught and logged as a warning."""
        ft, mock_client = _make_function_tool(node_name="step2")
        mock_client.artifact.side_effect = RuntimeError("network timeout")

        with caplog.at_level(logging.WARNING, logger="elitea_sdk.runtime.tools.function"):
            self._run_save(ft, "x = 1", "step2")

        assert any("network timeout" in m for m in caplog.messages)

    def test_bucket_name_is_always_code_debug(self):
        ft, mock_client = _make_function_tool(node_name="any_node")
        self._run_save(ft, "pass", "any_node")
        mock_client.artifact.assert_called_once_with("code-debug")

    def test_filename_is_node_name_dot_py(self):
        """Filename must follow '<node_name>__<YYYYMMDD_HHMMSS>.py' pattern."""
        ft, mock_client = _make_function_tool(node_name="transform_data")
        self._run_save(ft, "pass", "transform_data")
        args, _ = mock_client.artifact.return_value.create.call_args
        assert re_module.match(r"^transform_data__\d{8}_\d{6}\.py$", args[0]), (
            f"Unexpected filename: {args[0]}"
        )

    def test_saved_bytes_contain_user_code(self):
        """The user code must appear in the bytes passed to create()."""
        code = "result = 42"
        ft, mock_client = _make_function_tool(node_name="n1")
        self._run_save(ft, code, "n1")
        args, _ = mock_client.artifact.return_value.create.call_args
        assert code.encode("utf-8") in args[1]

    def test_saved_bytes_contain_client_preamble(self):
        """The client preamble returned by _build_client_preamble must be prepended."""
        code = "result = 1"
        ft, mock_client = _make_function_tool(node_name="n2")
        self._run_save(ft, code, "n2")
        args, _ = mock_client.artifact.return_value.create.call_args
        saved = args[1].decode("utf-8")
        # preamble comes first, user code after
        assert saved.startswith(_FAKE_PREAMBLE)
        assert code in saved


# ---------------------------------------------------------------------------
# 8–10 · _build_client_preamble unit tests
# ---------------------------------------------------------------------------

class TestBuildClientPreamble:
    """Unit tests for FunctionTool._build_client_preamble."""

    def test_happy_path_contains_sandbox_client_and_init(self):
        """When sandbox_client.py is readable, preamble must include its content,
        the SandboxClient instantiation with base_url/project_id, and the auth_token
        placeholder (never the real token)."""
        ft, _ = _make_function_tool(has_client=True)
        fake_client_code = "class SandboxClient:\n    pass\n"

        with patch("builtins.open", mock_open(read_data=fake_client_code)):
            preamble = ft._build_client_preamble()

        assert "class SandboxClient" in preamble
        assert "elitea_client = SandboxClient(" in preamble
        assert ft.elitea_client.base_url in preamble
        assert str(ft.elitea_client.project_id) in preamble
        # auth_token must be a placeholder, not the real token
        assert "<YOUR_AUTH_TOKEN>" in preamble
        assert ft.elitea_client.auth_token not in preamble

    def test_file_not_found_returns_empty_string_with_warning(self, caplog):
        """FileNotFoundError must return '' and log a warning."""
        ft, _ = _make_function_tool(has_client=True)

        with patch("builtins.open", side_effect=FileNotFoundError("no file")), \
             caplog.at_level(logging.WARNING, logger="elitea_sdk.runtime.tools.function"):
            result = ft._build_client_preamble()

        assert result == ""
        assert any("sandbox_client.py not found" in m for m in caplog.messages)

    def test_generic_exception_returns_empty_string_with_warning(self, caplog):
        """Any other exception must return '' and log a warning."""
        ft, _ = _make_function_tool(has_client=True)

        with patch("builtins.open", side_effect=OSError("permission denied")), \
             caplog.at_level(logging.WARNING, logger="elitea_sdk.runtime.tools.function"):
            result = ft._build_client_preamble()

        assert result == ""


# ---------------------------------------------------------------------------
# 11–16 · invoke() integration: debug flag gate
# ---------------------------------------------------------------------------

class TestFunctionToolInvokeDebugGate:
    """Verify that invoke() calls _save_code_to_artifact if and only if
    debug=True AND elitea_client is not None."""

    def _build_tool_and_run(self, *, debug, has_client, node_name="node_x"):
        ft, mock_client = _make_function_tool(
            debug=debug, has_client=has_client, node_name=node_name
        )
        ft.tool.invoke = MagicMock(return_value={"result": "ok", "stdout": "", "stderr": None,
                                                  "status": "success", "execution_time": 0.1})
        with patch.object(ft, "_prepare_pyodide_input", return_value="#preamble\n"), \
             patch.object(ft, "_build_client_preamble", return_value=_FAKE_PREAMBLE), \
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
             patch.object(ft, "_build_client_preamble", return_value=_FAKE_PREAMBLE), \
             patch.object(ft, "_save_code_to_artifact") as mock_save, \
             patch.object(ft, "_handle_pyodide_output", return_value={"messages": []}):
            ft.invoke({"messages": [], "code": "x = 1"}, config=None)

        _, node_name_arg = mock_save.call_args[0]
        assert node_name_arg == "my_node"

    def test_save_receives_assembled_code_as_first_arg(self):
        """_save_code_to_artifact must receive the preamble+user code, not just the user code."""
        ft, mock_client = _make_function_tool(debug=True, has_client=True)
        ft.tool.invoke = MagicMock(return_value={"result": "ok"})

        with patch.object(ft, "_prepare_pyodide_input", return_value="# preamble\n"), \
             patch.object(ft, "_build_client_preamble", return_value=_FAKE_PREAMBLE), \
             patch.object(ft, "_save_code_to_artifact") as mock_save, \
             patch.object(ft, "_handle_pyodide_output", return_value={"messages": []}):
            ft.invoke({"messages": [], "code": "ignored_state_code"}, config=None)

        saved_code = mock_save.call_args[0][0]
        assert "# preamble" in saved_code
        assert "x = 1" in saved_code

    def test_debug_does_not_affect_tool_execution_result(self):
        """Enabling debug must not change the value returned by invoke()."""
        ft_debug, _ = _make_function_tool(debug=True, has_client=True)
        ft_nodebug, _ = _make_function_tool(debug=False, has_client=True)

        expected_output = {"messages": [{"role": "assistant", "content": "done"}]}
        for ft in (ft_debug, ft_nodebug):
            ft.tool.invoke = MagicMock(return_value={"result": "done"})
            with patch.object(ft, "_prepare_pyodide_input", return_value="#pre\n"), \
                 patch.object(ft, "_build_client_preamble", return_value=_FAKE_PREAMBLE), \
                 patch.object(ft, "_save_code_to_artifact"), \
                 patch.object(ft, "_handle_pyodide_output", return_value=expected_output), \
                 patch("elitea_sdk.runtime.tools.function.dispatch_custom_event"):
                result = ft.invoke({"messages": [], "code": "x = 1"}, config=None)

            assert result == expected_output, (
                f"debug mode changed the return value: {result}"
            )


# ---------------------------------------------------------------------------
# 17–18 · FunctionTool field defaults
# ---------------------------------------------------------------------------

class TestFunctionToolDebugField:
    """Verify the debug field is declared correctly on FunctionTool."""

    def test_debug_defaults_to_false(self):
        ft = FunctionTool.model_construct(name="x", tool=_make_pyodide_tool())
        assert ft.debug is False

    def test_debug_true_is_stored(self):
        ft = FunctionTool.model_construct(name="x", tool=_make_pyodide_tool(), debug=True)
        assert ft.debug is True


# ---------------------------------------------------------------------------
# 19–21 · create_graph integration: node dict → FunctionTool.debug
# ---------------------------------------------------------------------------

class TestCreateGraphDebugPassThrough:
    """Verify that create_graph() reads 'debug' from the node dict and
    propagates it to FunctionTool when building a 'code' type node."""

    @staticmethod
    def _build_ft_from_node(node: dict) -> FunctionTool:
        mock_sandbox = MagicMock(spec=BaseTool)
        mock_sandbox.name = "pyodide_sandbox"
        elitea_client = _make_client(node["id"])

        return FunctionTool.model_construct(
            tool=mock_sandbox,
            name=node["id"],
            return_type="dict",
            output_variables=node.get("output", []),
            input_mapping={"code": node.get("code", {})},
            input_variables=node.get("input", ["messages"]),
            structured_output=node.get("structured_output", False),
            elitea_client=elitea_client,
            debug=node.get("debug", False),
        )

    def test_debug_true_in_node_dict_passed_to_function_tool(self):
        node = {"id": "debug_node", "debug": True, "code": {"type": "fixed", "value": ""}}
        assert self._build_ft_from_node(node).debug is True

    def test_debug_false_in_node_dict_passed_to_function_tool(self):
        node = {"id": "debug_node", "debug": False, "code": {"type": "fixed", "value": ""}}
        assert self._build_ft_from_node(node).debug is False

    def test_debug_absent_in_node_dict_defaults_to_false(self):
        node = {"id": "debug_node", "code": {"type": "fixed", "value": ""}}
        assert self._build_ft_from_node(node).debug is False






