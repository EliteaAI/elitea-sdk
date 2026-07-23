"""Python wrapper that calls pyodide & deno for code execution.

This is a standalone implementation compatible with langchain-core 1.x,
based on langchain-sandbox but without the version constraints.
"""

import asyncio
import dataclasses
import json
import logging
import os
import shutil
import subprocess
import time
from typing import Annotated, Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


Status = Literal["success", "error"]


@dataclasses.dataclass(kw_only=True)
class CodeExecutionResult:
    """Container for code execution results."""

    result: Any = None
    stdout: str | None = None
    stderr: str | None = None
    status: Status
    execution_time: float
    session_metadata: dict | None = None
    session_bytes: bytes | None = None


def get_default_pkg_name() -> str | None:
    """Get the Pyodide sandbox package/script path from environment.

    Set PYODIDE_SANDBOX_PKG environment variable to specify a custom path
    to the Deno entrypoint script (e.g., local main.js file).
    """
    return os.environ.get("PYODIDE_SANDBOX_PKG")


def build_permission_flag(
    flag: str,
    *,
    value: bool | list[str],
) -> str | None:
    """Build a permission flag string based on the provided setting."""
    if value is True:
        return flag
    if isinstance(value, list) and value:
        return f"{flag}={','.join(value)}"
    return None


class BasePyodideSandbox:
    """Base class for PyodideSandbox implementations.

    The sandbox leverages Deno's security model to create a secure runtime for
    executing untrusted Python code via Pyodide (Python compiled to WebAssembly).
    """

    def __init__(
        self,
        *,
        stateful: bool = False,
        allow_env: list[str] | bool = False,
        allow_read: list[str] | bool = False,
        allow_write: list[str] | bool = False,
        allow_net: list[str] | bool = False,
        allow_run: list[str] | bool = False,
        allow_ffi: list[str] | bool = False,
        node_modules_dir: str = "auto",
        skip_deno_check: bool = False,
        pkg_name: str | None = None,
    ) -> None:
        """Initialize the sandbox with specific Deno permissions.

        Args:
            stateful: Whether to maintain state between executions.
            allow_env: Environment variable access (False/True/list of vars).
            allow_read: File system read access (False/True/list of paths).
            allow_write: File system write access (False/True/list of paths).
            allow_net: Network access (False/True/list of domains).
            allow_run: Subprocess execution (False/True/list of commands).
            allow_ffi: FFI access (False/True/list of libraries).
            node_modules_dir: Node modules directory setting.
            skip_deno_check: Skip Deno installation check.
            pkg_name: Path to the Deno entrypoint script. If not provided,
                reads from PYODIDE_SANDBOX_PKG environment variable.
        """
        self.stateful = stateful
        self.allow_env = allow_env
        self.allow_read = allow_read
        self.allow_write = allow_write
        self.allow_net = allow_net
        self.allow_run = allow_run
        self.allow_ffi = allow_ffi
        self.node_modules_dir = node_modules_dir
        self.pkg_name = pkg_name or get_default_pkg_name()

        if not self.pkg_name:
            raise RuntimeError(
                "Pyodide sandbox entrypoint not configured. "
                "Set PYODIDE_SANDBOX_PKG environment variable or pass pkg_name parameter."
            )

        if not skip_deno_check and not shutil.which("deno"):
            raise RuntimeError(
                "Deno is required for PyodideSandbox but is not installed. "
                "Install from: https://docs.deno.com/runtime/getting_started/installation/"
            )

    def _build_command(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        memory_limit_mb: int | None = None,
        root_ca_path: str | None = None,
        insecure_tls_domains: list[str] | None = None,
    ) -> list[str]:
        """Build the Deno command with appropriate flags."""
        cmd = [
            "deno",
            "run",
            f"--node-modules-dir={self.node_modules_dir}",
            "--allow-sys=cpus",  # Required for Pyodide
        ]

        # Add permission flags
        permission_configs = [
            ("--allow-env", self.allow_env),
            ("--allow-read", self.allow_read),
            ("--allow-write", self.allow_write),
            ("--allow-net", self.allow_net),
            ("--allow-run", self.allow_run),
            ("--allow-ffi", self.allow_ffi),
        ]

        for flag, value in permission_configs:
            if permission_flag := build_permission_flag(flag, value=value):
                cmd.append(permission_flag)

        # WASM memory cap — bounds Pyodide's heap (Python objects, numpy arrays,
        # etc.), which lives in WASM linear memory. This REPLACES the old
        # --max-old-space-size flag, which only capped the V8 JS heap and had no
        # effect on Pyodide memory (the bomb that hung the VM was Python data in
        # WASM, not JS). Pages are 64 KiB each, so 1 MB = 16 pages.
        if memory_limit_mb:
            wasm_pages = int(memory_limit_mb) * 16
            cmd.append(f"--v8-flags=--wasm-max-mem-pages={wasm_pages}")

        # TLS trust override for outbound sandbox HTTPS calls (both default off,
        # so when neither is configured this block appends nothing and the
        # command is byte-for-byte unchanged). --cert (a custom root CA) is the
        # secure option and always wins when set; Deno reads the PEM in its own
        # runtime, so it needs NO --allow-read on the CA file. The per-domain
        # bypass is the fallback used only when no CA path is set, and only for
        # the whitelisted hosts — it never disables validation globally. The two
        # are mutually exclusive by construction.
        if root_ca_path:
            cmd.append(f"--cert={root_ca_path}")
        elif insecure_tls_domains:
            cmd.append(
                f"--unsafely-ignore-certificate-errors={','.join(insecure_tls_domains)}"
            )

        # Add the package and code
        cmd.extend([self.pkg_name, "-c", code])

        # Add stateful flag
        if self.stateful:
            cmd.append("-s")

        # Add session state if provided
        if session_bytes:
            bytes_array = list(session_bytes)
            cmd.extend(["-b", json.dumps(bytes_array)])

        if session_metadata:
            cmd.extend(["-m", json.dumps(session_metadata)])

        return cmd


class PyodideSandbox(BasePyodideSandbox):
    """Asynchronous implementation of PyodideSandbox."""

    async def execute(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        timeout_seconds: float | None = None,
        memory_limit_mb: int | None = None,
        root_ca_path: str | None = None,
        insecure_tls_domains: list[str] | None = None,
    ) -> CodeExecutionResult:
        """Execute Python code asynchronously in a sandboxed Deno subprocess."""
        start_time = time.time()
        stdout = ""
        stderr = ""
        result = None
        status: Literal["success", "error"] = "success"

        cmd = self._build_command(
            code,
            session_bytes=session_bytes,
            session_metadata=session_metadata,
            memory_limit_mb=memory_limit_mb,
            root_ca_path=root_ca_path,
            insecure_tls_domains=insecure_tls_domains,
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")

            if stdout:
                full_result = json.loads(stdout)
                stdout = full_result.get("stdout", None)
                stderr = full_result.get("stderr", None)
                result = full_result.get("result", None)
                status = "success" if full_result.get("success", False) else "error"
                session_metadata = full_result.get("sessionMetadata", None)
                session_bytes_array = full_result.get("sessionBytes", None)
                session_bytes = (
                    bytes(session_bytes_array) if session_bytes_array else None
                )
            else:
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                status = "error"
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            status = "error"
            stderr = f"Execution timed out after {timeout_seconds} seconds"
        except asyncio.CancelledError:
            pass

        end_time = time.time()

        return CodeExecutionResult(
            status=status,
            execution_time=end_time - start_time,
            stdout=stdout or None,
            stderr=stderr or None,
            result=result,
            session_metadata=session_metadata,
            session_bytes=session_bytes,
        )


class SyncPyodideSandbox(BasePyodideSandbox):
    """Synchronous version of PyodideSandbox."""

    def execute(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        timeout_seconds: float | None = None,
        memory_limit_mb: int | None = None,
        root_ca_path: str | None = None,
        insecure_tls_domains: list[str] | None = None,
    ) -> CodeExecutionResult:
        """Execute Python code synchronously in a sandboxed Deno subprocess."""
        start_time = time.time()
        stdout = ""
        result = None
        stderr: str
        status: Literal["success", "error"]

        cmd = self._build_command(
            code,
            session_bytes=session_bytes,
            session_metadata=session_metadata,
            memory_limit_mb=memory_limit_mb,
            root_ca_path=root_ca_path,
            insecure_tls_domains=insecure_tls_domains,
        )

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=False,
                timeout=timeout_seconds,
                check=False,
            )

            stdout_bytes = process.stdout
            stderr_bytes = process.stderr
            stdout = stdout_bytes.decode("utf-8", errors="replace")

            if stdout:
                full_result = json.loads(stdout)
                stdout = full_result.get("stdout", None)
                stderr = full_result.get("stderr", None)
                result = full_result.get("result", None)
                status = "success" if full_result.get("success", False) else "error"
                session_metadata = full_result.get("sessionMetadata", None)
                session_bytes_array = full_result.get("sessionBytes", None)
                session_bytes = (
                    bytes(session_bytes_array) if session_bytes_array else None
                )
            else:
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                status = "error"

        except subprocess.TimeoutExpired:
            status = "error"
            stderr = f"Execution timed out after {timeout_seconds} seconds"

        end_time = time.time()

        return CodeExecutionResult(
            status=status,
            execution_time=end_time - start_time,
            stdout=stdout or None,
            stderr=stderr or None,
            result=result,
            session_metadata=session_metadata,
            session_bytes=session_bytes,
        )


class PyodideSandboxTool(BaseTool):
    """Tool for running python code in a PyodideSandbox.

    If you use a stateful sandbox (stateful=True), the state between executions
    (variables, imports, definitions) will be persisted using LangGraph checkpointer.

    When using stateful sandbox, this tool must be used inside a LangGraph graph
    with a checkpointer, and with create_react_agent or ToolNode.
    """

    name: str = "python_code_sandbox"
    description: str = (
        "A secure Python code sandbox. Use this to execute python commands.\n"
        "- Input should be valid python code.\n"
        "- To return output, use print(...)\n"
        "- For web requests, use httpx.AsyncClient"
    )

    stateful: bool = False
    allow_env: list[str] | bool = False
    allow_read: list[str] | bool = False
    allow_write: list[str] | bool = False
    allow_net: list[str] | bool = False
    allow_run: list[str] | bool = False
    allow_ffi: list[str] | bool = False
    timeout_seconds: float | None = 60
    node_modules_dir: str = "auto"
    pkg_name: str | None = None

    _sandbox: PyodideSandbox = None
    _sync_sandbox: SyncPyodideSandbox = None

    def __init__(
        self,
        *,
        stateful: bool = False,
        timeout_seconds: float | None = 60,
        allow_net: list[str] | bool = False,
        pkg_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the tool."""
        super().__init__(
            stateful=stateful,
            timeout_seconds=timeout_seconds,
            allow_net=allow_net,
            pkg_name=pkg_name,
            **kwargs,
        )

        if self.stateful:
            try:
                from langgraph.prebuilt import InjectedState
            except ImportError as e:
                raise ImportError(
                    "langgraph is required for stateful sandbox. "
                    "Install with: pip install langgraph"
                ) from e

            class PyodideSandboxToolInput(BaseModel):
                """Python code to execute in the sandbox."""
                code: str = Field(description="Code to execute.")
                state: Annotated[dict[str, Any] | BaseModel, InjectedState]
                tool_call_id: Annotated[str, InjectedToolCallId]
        else:
            class PyodideSandboxToolInput(BaseModel):
                """Python code to execute in the sandbox."""
                code: str = Field(description="Code to execute.")

        self.args_schema = PyodideSandboxToolInput
        self._sandbox = PyodideSandbox(
            stateful=self.stateful,
            allow_env=self.allow_env,
            allow_read=self.allow_read,
            allow_write=self.allow_write,
            allow_net=self.allow_net,
            allow_run=self.allow_run,
            allow_ffi=self.allow_ffi,
            node_modules_dir=self.node_modules_dir,
            pkg_name=self.pkg_name,
        )
        self._sync_sandbox = SyncPyodideSandbox(
            stateful=self.stateful,
            allow_env=self.allow_env,
            allow_read=self.allow_read,
            allow_write=self.allow_write,
            allow_net=self.allow_net,
            allow_run=self.allow_run,
            allow_ffi=self.allow_ffi,
            node_modules_dir=self.node_modules_dir,
            pkg_name=self.pkg_name,
            skip_deno_check=True,
        )

    def _run(
        self,
        code: str,
        state: dict[str, Any] | BaseModel | None = None,
        tool_call_id: str | None = None,
        config: RunnableConfig | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """Use the tool synchronously."""
        if self.stateful:
            required_keys = {"session_bytes", "session_metadata", "messages"}
            actual_keys = set(state) if isinstance(state, dict) else set(state.__dict__)
            if missing_keys := required_keys - actual_keys:
                raise ValueError(
                    f"Input state missing required keys: {missing_keys}"
                )

            if isinstance(state, dict):
                session_bytes = state["session_bytes"]
                session_metadata = state["session_metadata"]
            else:
                session_bytes = state.session_bytes
                session_metadata = state.session_metadata

            result = self._sync_sandbox.execute(
                code,
                session_bytes=session_bytes,
                session_metadata=session_metadata,
                timeout_seconds=self.timeout_seconds,
            )
        else:
            result = self._sync_sandbox.execute(
                code, timeout_seconds=self.timeout_seconds
            )

        tool_result = (
            f"Error during execution: {result.stderr}"
            if result.stderr and result.status == 'error'
            else result.stdout
        )

        if self.stateful:
            from langgraph.types import Command
            return Command(
                update={
                    "session_bytes": result.session_bytes,
                    "session_metadata": result.session_metadata,
                    "messages": [
                        ToolMessage(content=tool_result, tool_call_id=tool_call_id)
                    ],
                }
            )

        return tool_result

    async def _arun(
        self,
        code: str,
        state: dict[str, Any] | BaseModel | None = None,
        tool_call_id: str | None = None,
        config: RunnableConfig | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        """Use the tool asynchronously."""
        if self.stateful:
            required_keys = {"session_bytes", "session_metadata", "messages"}
            actual_keys = set(state) if isinstance(state, dict) else set(state.__dict__)
            if missing_keys := required_keys - actual_keys:
                raise ValueError(
                    f"Input state missing required keys: {missing_keys}"
                )

            if isinstance(state, dict):
                session_bytes = state["session_bytes"]
                session_metadata = state["session_metadata"]
            else:
                session_bytes = state.session_bytes
                session_metadata = state.session_metadata

            result = await self._sandbox.execute(
                code,
                session_bytes=session_bytes,
                session_metadata=session_metadata,
                timeout_seconds=self.timeout_seconds,
            )
        else:
            result = await self._sandbox.execute(
                code, timeout_seconds=self.timeout_seconds
            )

        tool_result = (
            f"Error during execution: {result.stderr}"
            if result.stderr and result.status == 'error'
            else result.stdout
        )

        if self.stateful:
            from langgraph.types import Command
            return Command(
                update={
                    "session_bytes": result.session_bytes,
                    "session_metadata": result.session_metadata,
                    "messages": [
                        ToolMessage(content=tool_result, tool_call_id=tool_call_id)
                    ],
                }
            )

        return tool_result
