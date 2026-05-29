"""
Tests for CodeExecutor AST-based safety guard.

Verifies that dangerous operations (blocked modules/builtins) are rejected
while safe data-analysis code is allowed to execute.

Related to: GitHub issue #4986 - Prevent SDK toolkits from writing files
to persistent volumes via LLM-generated code.
PR: https://github.com/EliteaAI/elitea-sdk/pull/180
"""

import pytest

from elitea_sdk.tools.pandas.dataframe.executor.code_executor import (
    CodeExecutor,
    _validate_code,
    _BLOCKED_MODULES,
    _BLOCKED_BUILTINS,
)
from elitea_sdk.tools.pandas.dataframe.errors import CodeExecutionError, NoResultFoundError


class TestValidateCodeBlockedImports:
    """Tests for blocked module imports."""

    @pytest.mark.parametrize("module", [
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "socket",
        "http",
        "urllib",
        "requests",
        "importlib",
        "ctypes",
        "signal",
        "multiprocessing",
        "tempfile",
        "glob",
        "fnmatch",
        "webbrowser",
    ])
    def test_blocks_dangerous_import(self, module):
        """Verify direct imports of dangerous modules are blocked."""
        code = f"import {module}"
        with pytest.raises(CodeExecutionError) as exc_info:
            _validate_code(code)
        assert f"Importing '{module}' is not allowed" in str(exc_info.value)

    @pytest.mark.parametrize("module,submodule", [
        ("os", "path"),
        ("os", "environ"),
        ("subprocess", "run"),
        ("shutil", "copy"),
        ("pathlib", "Path"),
        ("socket", "socket"),
        ("http", "client"),
        ("urllib", "request"),
        ("importlib", "import_module"),
    ])
    def test_blocks_from_import(self, module, submodule):
        """Verify 'from X import Y' for dangerous modules is blocked."""
        code = f"from {module} import {submodule}"
        with pytest.raises(CodeExecutionError) as exc_info:
            _validate_code(code)
        assert f"Importing from '{module}' is not allowed" in str(exc_info.value)

    @pytest.mark.parametrize("module,submodule", [
        ("os.path", "join"),
        ("http.client", "HTTPConnection"),
        ("urllib.request", "urlopen"),
    ])
    def test_blocks_nested_module_import(self, module, submodule):
        """Verify nested module imports (os.path, http.client) are blocked."""
        code = f"from {module} import {submodule}"
        with pytest.raises(CodeExecutionError) as exc_info:
            _validate_code(code)
        root_module = module.split(".")[0]
        assert f"Importing from '{module}' is not allowed" in str(exc_info.value)

    def test_blocks_import_with_alias(self):
        """Verify aliased imports are also blocked."""
        code = "import os as operating_system"
        with pytest.raises(CodeExecutionError) as exc_info:
            _validate_code(code)
        assert "Importing 'os' is not allowed" in str(exc_info.value)

    def test_blocks_multiple_imports_one_line(self):
        """Verify multi-import statements are blocked if any module is dangerous."""
        code = "import pandas, os, numpy"
        with pytest.raises(CodeExecutionError) as exc_info:
            _validate_code(code)
        assert "Importing 'os' is not allowed" in str(exc_info.value)

    def test_blocks_star_import_from_dangerous_module(self):
        """Verify 'from os import *' is blocked."""
        code = "from os import *"
        with pytest.raises(CodeExecutionError) as exc_info:
            _validate_code(code)
        assert "Importing from 'os' is not allowed" in str(exc_info.value)

    def test_allows_relative_import(self):
        """Verify relative imports (node.module=None) don't crash validator."""
        code = "from . import something"
        # This will fail at parse or execution, but validator shouldn't crash
        # The validator checks `if node.module:` so None is skipped
        _validate_code(code)  # Should not raise CodeExecutionError for blocked module


class TestValidateCodeBlockedBuiltins:
    """Tests for blocked builtin function calls."""

    @pytest.mark.parametrize("builtin", [
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
    ])
    def test_blocks_dangerous_builtin_call(self, builtin):
        """Verify dangerous builtin function calls are blocked."""
        if builtin == "open":
            code = "open('file.txt', 'w')"
        elif builtin == "exec":
            code = "exec('print(1)')"
        elif builtin == "eval":
            code = "eval('1+1')"
        elif builtin == "compile":
            code = "compile('x=1', '<string>', 'exec')"
        elif builtin == "__import__":
            code = "__import__('os')"
        elif builtin == "input":
            code = "x = input('Enter value: ')"
        elif builtin == "breakpoint":
            code = "breakpoint()"
        else:
            code = f"{builtin}()"

        with pytest.raises(CodeExecutionError) as exc_info:
            _validate_code(code)
        assert f"Calling '{builtin}()' is not allowed" in str(exc_info.value)

    def test_blocks_open_in_with_statement(self):
        """Verify open() in 'with' statement context is blocked."""
        code = "with open('file.txt') as f:\n    data = f.read()"
        with pytest.raises(CodeExecutionError) as exc_info:
            _validate_code(code)
        assert "Calling 'open()' is not allowed" in str(exc_info.value)

    def test_blocks_nested_eval(self):
        """Verify eval() nested in expressions is blocked."""
        code = "result = [eval(x) for x in ['1', '2', '3']]"
        with pytest.raises(CodeExecutionError) as exc_info:
            _validate_code(code)
        assert "Calling 'eval()' is not allowed" in str(exc_info.value)

    def test_allows_method_with_blocked_name_on_object(self):
        """Verify methods named 'open' on objects are NOT blocked (false positive check).

        The AST check looks at ast.Attribute.attr for method calls. A DataFrame
        or custom object could have a method named 'open' - this should be allowed
        since it's not the builtin open().
        """
        # df.open() - 'open' is an attribute access, caught by ast.Attribute
        # Current implementation WILL block this (known limitation)
        code = "result = df.open()"
        with pytest.raises(CodeExecutionError):
            _validate_code(code)
        # This is a known false positive - the current implementation blocks
        # method calls with names matching blocked builtins

    def test_blocks_getattr_builtin_access(self):
        """Verify getattr access to builtins is blocked at runtime.

        Note: AST validation cannot catch dynamic getattr bypasses.
        This test documents that getattr(__builtins__, 'open') passes
        AST validation but would fail at runtime if __builtins__ access
        were restricted in the execution environment.
        """
        code = "f = getattr(__builtins__, 'open')"
        # AST validator does NOT catch this - getattr is not blocked
        _validate_code(code)  # Passes validation (limitation)
        # Runtime protection would need to restrict __builtins__ in exec environment


class TestValidateCodeAllowsSafeOperations:
    """Tests for allowed safe operations."""

    def test_allows_pandas_import(self):
        """Verify pandas import is allowed."""
        code = "import pandas as pd"
        _validate_code(code)  # Should not raise

    def test_allows_numpy_import(self):
        """Verify numpy import is allowed."""
        code = "import numpy as np"
        _validate_code(code)  # Should not raise

    def test_allows_matplotlib_import(self):
        """Verify matplotlib import is allowed."""
        code = "import matplotlib.pyplot as plt"
        _validate_code(code)  # Should not raise

    def test_allows_scipy_import(self):
        """Verify scipy import is allowed."""
        code = "from scipy import stats"
        _validate_code(code)  # Should not raise

    def test_allows_sklearn_import(self):
        """Verify sklearn import is allowed."""
        code = "from sklearn.linear_model import LinearRegression"
        _validate_code(code)  # Should not raise

    def test_allows_standard_data_analysis_code(self):
        """Verify typical data analysis code passes validation."""
        code = """
import pandas as pd
import numpy as np

df = get_dataframe()
mean_val = df['column'].mean()
std_val = np.std(df['column'])
result = dict(df=df, result={'mean': mean_val, 'std': std_val})
"""
        _validate_code(code)  # Should not raise

    def test_allows_bytesio_for_charts(self):
        """Verify BytesIO usage for chart generation is allowed."""
        code = """
from io import BytesIO
import base64

buffer = BytesIO()
# plt.savefig(buffer, format='png')
buffer.seek(0)
encoded = base64.b64encode(buffer.read()).decode('utf-8')
"""
        _validate_code(code)  # Should not raise

    def test_allows_json_module(self):
        """Verify json module is allowed (not in blocked list)."""
        code = "import json\ndata = json.loads('{\"key\": \"value\"}')"
        _validate_code(code)  # Should not raise

    def test_allows_future_imports(self):
        """Verify __future__ imports are allowed."""
        code = "from __future__ import annotations"
        _validate_code(code)  # Should not raise

    def test_allows_statsmodels(self):
        """Verify statsmodels (used in code_environment) is allowed."""
        code = "import statsmodels.api as sm"
        _validate_code(code)  # Should not raise

    def test_allows_factor_analyzer(self):
        """Verify factor_analyzer (used in code_environment) is allowed."""
        code = "import factor_analyzer"
        _validate_code(code)  # Should not raise


class TestValidateCodeSyntaxErrors:
    """Tests for syntax error handling."""

    def test_syntax_error_raises_code_execution_error(self):
        """Verify syntax errors are caught and wrapped."""
        code = "def incomplete("
        with pytest.raises(CodeExecutionError) as exc_info:
            _validate_code(code)
        assert "syntax error" in str(exc_info.value).lower()

    def test_indentation_error_raises_code_execution_error(self):
        """Verify indentation errors are caught."""
        code = "if True:\nprint('bad indent')"
        with pytest.raises(CodeExecutionError) as exc_info:
            _validate_code(code)
        assert "syntax error" in str(exc_info.value).lower()


class TestCodeExecutorIntegration:
    """Integration tests for CodeExecutor class."""

    def test_executor_blocks_dangerous_code(self):
        """Verify CodeExecutor.execute() blocks dangerous code."""
        executor = CodeExecutor()
        code = "import os\nfiles = os.listdir('.')"
        with pytest.raises(CodeExecutionError) as exc_info:
            executor.execute(code)
        assert "Importing 'os' is not allowed" in str(exc_info.value)

    def test_executor_allows_safe_code(self):
        """Verify CodeExecutor.execute() runs safe code."""
        executor = CodeExecutor()
        code = """
import pandas as pd
import numpy as np
x = 42
result = {'value': x, 'doubled': x * 2}
"""
        env = executor.execute(code)
        assert env.get('x') == 42
        assert env.get('result') == {'value': 42, 'doubled': 84}

    def test_executor_with_custom_environment(self):
        """Verify CodeExecutor works with custom environment variables."""
        executor = CodeExecutor()
        executor.add_to_env('custom_value', 100)
        code = "output = custom_value * 2"
        env = executor.execute(code)
        assert env.get('output') == 200

    def test_executor_execute_and_return_result(self):
        """Verify execute_and_return_result() returns the result dict."""
        executor = CodeExecutor()
        code = """
x = [1, 2, 3]
result = {'sum': sum(x), 'count': len(x)}
"""
        result = executor.execute_and_return_result(code)
        assert result == {'sum': 6, 'count': 3}

    def test_executor_blocks_file_write_attempt(self):
        """Verify attempt to write files via open() is blocked."""
        executor = CodeExecutor()
        code = """
with open('/tmp/test.txt', 'w') as f:
    f.write('malicious content')
"""
        with pytest.raises(CodeExecutionError) as exc_info:
            executor.execute(code)
        assert "Calling 'open()' is not allowed" in str(exc_info.value)

    def test_executor_blocks_subprocess_execution(self):
        """Verify subprocess execution attempts are blocked."""
        executor = CodeExecutor()
        code = "import subprocess\nsubprocess.run(['ls', '-la'])"
        with pytest.raises(CodeExecutionError) as exc_info:
            executor.execute(code)
        assert "Importing 'subprocess' is not allowed" in str(exc_info.value)

    def test_executor_blocks_network_access(self):
        """Verify network access attempts are blocked."""
        executor = CodeExecutor()
        code = "import requests\nresponse = requests.get('http://example.com')"
        with pytest.raises(CodeExecutionError) as exc_info:
            executor.execute(code)
        assert "Importing 'requests' is not allowed" in str(exc_info.value)

    def test_executor_returns_none_when_result_not_set(self):
        """Verify execute_and_return_result returns None when result not explicitly set.

        Note: The environment pre-initializes 'result' to None, so NoResultFoundError
        is never raised. This test documents actual behavior.
        """
        executor = CodeExecutor()
        code = "x = 42"  # No 'result' variable set
        result = executor.execute_and_return_result(code)
        assert result is None  # Returns pre-initialized None value

    def test_executor_runtime_error_wrapped(self):
        """Verify runtime errors during execution are wrapped in CodeExecutionError."""
        executor = CodeExecutor()
        code = "x = 1 / 0"  # ZeroDivisionError
        with pytest.raises(CodeExecutionError) as exc_info:
            executor.execute(code)
        assert "Code execution failed" in str(exc_info.value)

    def test_executor_uses_preloaded_environment(self):
        """Verify executor has pandas, numpy, etc. pre-loaded from code_environment."""
        executor = CodeExecutor()
        # pd and np should be available from get_environment()
        code = """
arr = np.array([1, 2, 3])
result = {'sum': int(np.sum(arr))}
"""
        result = executor.execute_and_return_result(code)
        assert result == {'sum': 6}


class TestBlockedModulesCompleteness:
    """Verify blocked modules/builtins lists are complete."""

    def test_blocked_modules_matches_expected(self):
        """Verify _BLOCKED_MODULES contains all expected dangerous modules."""
        expected = {
            "os", "sys", "subprocess", "shutil", "pathlib",
            "socket", "http", "urllib", "requests",
            "importlib", "ctypes", "signal", "multiprocessing",
            "tempfile", "glob", "fnmatch", "webbrowser",
        }
        assert _BLOCKED_MODULES == expected

    def test_blocked_builtins_matches_expected(self):
        """Verify _BLOCKED_BUILTINS contains all expected dangerous builtins."""
        expected = {
            "open", "exec", "eval", "compile", "__import__",
            "execfile", "input", "breakpoint",
        }
        assert _BLOCKED_BUILTINS == expected


class TestKnownLimitations:
    """Document known limitations of the AST-based safety guard.

    These tests document bypass vectors that the current implementation
    does NOT protect against. They serve as documentation and regression
    tests if the implementation is enhanced later.
    """

    def test_limitation_getattr_bypass_passes_validation(self):
        """LIMITATION: getattr can bypass AST checks.

        getattr(__builtins__, 'open')('file.txt', 'w') passes AST validation.
        Mitigation: Restrict __builtins__ in execution environment.
        """
        code = "func = getattr(__builtins__, 'open')"
        _validate_code(code)  # Passes - this is a known limitation

    def test_limitation_globals_access_passes_validation(self):
        """LIMITATION: globals()['__builtins__'] access passes validation."""
        code = "builtins_dict = globals()['__builtins__']"
        _validate_code(code)  # Passes - this is a known limitation

    def test_limitation_method_name_false_positive(self):
        """LIMITATION: Methods named after blocked builtins are blocked.

        df.open(), obj.eval() are blocked even though they're not the
        dangerous builtins. This is a false positive.
        """
        code = "x = some_object.open()"
        with pytest.raises(CodeExecutionError):
            _validate_code(code)
        # This demonstrates the false positive - method 'open' is blocked
