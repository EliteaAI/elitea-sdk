"""Tests for #7720: deno probe OSError handling and string results in code nodes."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.runtime.tools.function import FunctionTool
from elitea_sdk.runtime.tools.sandbox import _is_deno_available


class TestIsDenoAvailableOSError:
    @pytest.mark.parametrize("exc", [
        PermissionError(13, "Permission denied: 'deno'"),
        OSError(8, "Exec format error"),
    ])
    def test_oserror_returns_false(self, exc):
        with patch('elitea_sdk.runtime.tools.sandbox.subprocess.run', side_effect=exc):
            assert _is_deno_available() is False

    def test_oserror_is_logged_as_warning(self):
        with patch('elitea_sdk.runtime.tools.sandbox.subprocess.run',
                   side_effect=PermissionError(13, "denied")), \
             patch('elitea_sdk.runtime.tools.sandbox.logger') as log:
            _is_deno_available()
        log.warning.assert_called_once()

    def test_file_not_found_still_false(self):
        with patch('elitea_sdk.runtime.tools.sandbox.subprocess.run', side_effect=FileNotFoundError):
            assert _is_deno_available() is False


def _ft(**kw):
    return FunctionTool.model_construct(
        name="n", tool=MagicMock(), return_type="dict", input_variables=[],
        input_mapping={}, output_variables=kw.get("output_variables", []),
        structured_output=kw.get("structured_output", False), elitea_client=None, debug=False,
    )


class TestHandlePyodideOutputStringResult:
    @pytest.mark.parametrize("kw", [
        {},
        {"output_variables": ["x"]},
        {"structured_output": True},
    ])
    def test_string_result_raises_readable_tool_error(self, kw):
        with pytest.raises(ToolException, match="Deno is required"):
            _ft(**kw)._handle_pyodide_output("Deno is required for PyodideSandbox")

    def test_dict_result_unchanged(self):
        out = _ft(output_variables=["x"])._handle_pyodide_output({"x": 1})
        assert out == {"x": 1}
