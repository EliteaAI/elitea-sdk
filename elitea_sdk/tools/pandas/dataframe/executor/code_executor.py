import ast
import logging
from typing import Any
from traceback import format_exc
from .code_environment import get_environment
from ..errors import CodeExecutionError, NoResultFoundError

logger = logging.getLogger(__name__)

# Modules/functions that must never appear in generated code
_BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
    "importlib", "ctypes", "signal", "multiprocessing",
    "tempfile", "glob", "fnmatch", "webbrowser",
})

_BLOCKED_BUILTINS = frozenset({
    "open", "exec", "eval", "compile", "__import__",
    "execfile", "input", "breakpoint",
})


def _validate_code(code: str) -> None:
    """AST-scan generated code and reject dangerous operations."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeExecutionError(f"Generated code has a syntax error: {e}")

    for node in ast.walk(tree):
        # Block: import os / import subprocess / ...
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_MODULES:
                    raise CodeExecutionError(
                        f"Importing '{alias.name}' is not allowed in data-analysis code"
                    )
        # Block: from os import ... / from subprocess import ...
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _BLOCKED_MODULES:
                    raise CodeExecutionError(
                        f"Importing from '{node.module}' is not allowed in data-analysis code"
                    )
        # Block: open(...), eval(...), exec(...), __import__(...)
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _BLOCKED_BUILTINS:
                raise CodeExecutionError(
                    f"Calling '{name}()' is not allowed in data-analysis code"
                )


class CodeExecutor:
    """Handle the logic on how to handle different lines of code."""

    _environment: dict

    def __init__(self) -> None:
        self._environment = get_environment()

    def add_to_env(self, key: str, value: Any) -> None:
        """Expose extra variables in the code to be used."""
        self._environment[key] = value

    def execute(self, code: str) -> dict:
        """Execute code after safety validation."""
        _validate_code(code)
        try:
            exec(code, self._environment)
        except Exception as e:
            raise CodeExecutionError(f"Code execution failed: {format_exc()}")
        return self._environment

    def execute_and_return_result(self, code: str) -> Any:
        """
        Executes the return updated environment
        """
        self.execute(code)

        # Get the result
        if "result" not in self._environment:
            raise NoResultFoundError("No result returned")

        return self._environment.get("result", None)

    @property
    def environment(self) -> dict:
        return self._environment