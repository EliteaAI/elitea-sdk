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
import tempfile
import time
from typing import Annotated, Any, Awaitable, Callable, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


from .sandbox_types import BridgeExecutionResult, CodeExecutionResult, Status  # noqa: E402

BRIDGE_MAX_LINE_BYTES = 4 * 1024 * 1024
BRIDGE_MAX_CONCURRENT_CALLS = 8
_BRIDGE_OUTPUT_KEEP_CHARS = 1024 * 1024
_BRIDGE_STDERR_TAIL_BYTES = 64 * 1024


def get_default_pkg_name() -> str | None:
    """Get the Pyodide sandbox package/script path from environment.

    Set PYODIDE_SANDBOX_PKG environment variable to specify a custom path
    to the Deno entrypoint script (e.g., local main.js file).
    """
    return os.environ.get("PYODIDE_SANDBOX_PKG")


def _usable_snapshot_path() -> str | None:
    # Set by indexer_worker when the snapshot flag is on (#6718); a granted read of a vanished file is harmless.
    path = os.environ.get("SANDBOX_SNAPSHOT_PATH")
    return path if path and os.path.isfile(path) else None


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
        allow_remote_modules: bool = True,
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
            allow_remote_modules: Whether code may import remote, JSR or npm modules.
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
        self.allow_remote_modules = allow_remote_modules
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
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        memory_limit_mb: int | None = None,
        root_ca_path: str | None = None,
        insecure_tls_domains: list[str] | None = None,
        bridge_max_line_bytes: int | None = None,
    ) -> list[str]:
        """Build the Deno command with appropriate flags."""
        cmd = [
            "deno",
            "run",
            f"--node-modules-dir={self.node_modules_dir}",
            "--allow-sys=cpus",  # Required for Pyodide
        ]

        # Add permission flags
        snapshot = _usable_snapshot_path()
        allow_read = self.allow_read
        if snapshot and isinstance(allow_read, list):
            allow_read = [*allow_read, snapshot]
        permission_configs = [
            ("--allow-env", self.allow_env),
            ("--allow-read", allow_read),
            ("--allow-write", self.allow_write),
            ("--allow-net", self.allow_net),
            ("--allow-run", self.allow_run),
            ("--allow-ffi", self.allow_ffi),
        ]

        for flag, value in permission_configs:
            if permission_flag := build_permission_flag(flag, value=value):
                cmd.append(permission_flag)

        if not self.allow_remote_modules:
            cmd.extend(["--no-remote", "--no-npm"])

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

        # Add the package and stdin flag (code will be passed via stdin to avoid
        # Linux kernel's MAX_ARG_STRLEN 128KB limit on individual CLI arguments)
        snapshot_flag = [f"--snapshot={snapshot}"] if snapshot else []
        if bridge_max_line_bytes:
            cmd.extend([self.pkg_name, "--bridge", f"--max-line-bytes={int(bridge_max_line_bytes)}", *snapshot_flag])
            return cmd
        cmd.extend([self.pkg_name, "--stdin", *snapshot_flag])

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

    @staticmethod
    def _run_directory() -> tempfile.TemporaryDirectory:
        """A temporary directory of one run's own, removed after it: Deno writes what a run's code puts in its caches
        and web storage under TMPDIR and DENO_DIR, which would otherwise be shared by every run on the worker."""
        return tempfile.TemporaryDirectory(prefix="pyodide-run-", ignore_cleanup_errors=True)

    @staticmethod
    def _run_environment(run_directory: tempfile.TemporaryDirectory) -> dict:
        return {**os.environ, "TMPDIR": run_directory.name, "DENO_DIR": os.path.join(run_directory.name, "deno_dir")}


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
        timed_out = False

        cmd = self._build_command(
            session_bytes=session_bytes,
            session_metadata=session_metadata,
            memory_limit_mb=memory_limit_mb,
            root_ca_path=root_ca_path,
            insecure_tls_domains=insecure_tls_domains,
        )

        try:
            code_bytes = code.encode('utf-8')
        except UnicodeEncodeError as e:
            return CodeExecutionResult(
                status="error",
                execution_time=time.time() - start_time,
                stderr=f"Failed to encode code as UTF-8: {e}",
            )

        run_directory = self._run_directory()
        spawn_start = time.time()
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._run_environment(run_directory),
        )
        spawn_s = time.time() - spawn_start
        sandbox_timings = None

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=code_bytes),
                timeout=timeout_seconds,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")

            if stdout:
                full_result = json.loads(stdout)
                sandbox_timings = full_result.get("timings")
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
            timed_out = True
        except asyncio.CancelledError:
            pass
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            run_directory.cleanup()

        end_time = time.time()
        logger.debug(
            "[sandbox-timing] deno run: total=%.3fs spawn=%.3fs status=%s code_bytes=%d in_sandbox=%s",
            end_time - start_time, spawn_s, status, len(code_bytes), sandbox_timings,
        )

        return CodeExecutionResult(
            status=status,
            execution_time=end_time - start_time,
            stdout=stdout or None,
            stderr=stderr or None,
            result=result,
            session_metadata=session_metadata,
            session_bytes=session_bytes,
            timed_out=timed_out,
        )

    async def execute_with_bridge(
        self,
        code: str,
        dispatcher: Callable[[str, str], Awaitable[str]],
        *,
        timeout_seconds: float,
        max_tool_calls: int,
        memory_limit_mb: int | None = None,
        max_line_bytes: int = BRIDGE_MAX_LINE_BYTES,
        max_concurrent_calls: int = BRIDGE_MAX_CONCURRENT_CALLS,
        on_output: Callable[[str, str], None] | None = None,
    ) -> BridgeExecutionResult:
        """Run `code` with `elitea_bridge.call(name, args_json)` routed to `dispatcher(name, args_json) -> value_json`.

        A dispatcher exception becomes a catchable error in the program; tool time counts against the timeout."""
        start_time = time.time()
        stdout_parts: list[str] = []
        stdout_len = 0
        stderr_tail = bytearray()
        outcome: dict[str, Any] = {}
        calls_completed = 0
        calls_started = 0
        call_tasks: set[asyncio.Task] = set()
        call_slots = asyncio.Semaphore(max(1, max_concurrent_calls))
        write_lock = asyncio.Lock()

        cmd = self._build_command(memory_limit_mb=memory_limit_mb, bridge_max_line_bytes=max_line_bytes)
        run_directory = self._run_directory()
        # The reader limit must exceed the line cap, or asyncio rejects lines over its 64 KB default.
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._run_environment(run_directory),
                limit=max_line_bytes + 1024,
            )
        except BaseException:
            run_directory.cleanup()
            raise

        def fail(error_type: str, error: str) -> None:
            outcome.setdefault("final", {"success": False, "error": error, "error_type": error_type})

        async def send(message: dict) -> None:
            data = (json.dumps(message) + "\n").encode("utf-8")
            async with write_lock:
                try:
                    process.stdin.write(data)
                    await process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError, RuntimeError):
                    pass

        async def drain_stderr() -> None:
            while chunk := await process.stderr.read(65536):
                stderr_tail.extend(chunk)
                del stderr_tail[:-_BRIDGE_STDERR_TAIL_BYTES]

        async def run_call(call_id: Any, name: str, args_json: str) -> None:
            nonlocal calls_completed
            try:
                async with call_slots:
                    value = await dispatcher(name, args_json)
                reply = {"type": "result", "id": call_id, "ok": True, "value": value}
            except Exception as exc:  # pylint: disable=broad-except
                reply = {"type": "result", "id": call_id, "ok": False, "error": str(exc) or type(exc).__name__}
            calls_completed += 1
            if len(reply.get("value") or "") > max_line_bytes:
                reply = {"type": "result", "id": call_id, "ok": False,
                         "error": f"Tool result is over the {max_line_bytes}-byte bridge limit."}
            await send(reply)

        async def pump() -> None:
            nonlocal stdout_len, calls_started
            await send({"type": "code", "code": code})
            while True:
                try:
                    line = await process.stdout.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    fail("result_too_large", f"A sandbox message is over the {max_line_bytes}-byte limit.")
                    return
                if not line:
                    return
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                kind = message.get("type")
                if kind == "final":
                    outcome.setdefault("final", message)
                    return
                if kind in ("stdout", "stderr"):
                    text = str(message.get("text", ""))
                    if kind == "stdout" and stdout_len < _BRIDGE_OUTPUT_KEEP_CHARS:
                        stdout_parts.append(text)
                        stdout_len += len(text)
                    elif kind == "stderr":
                        stderr_tail.extend(text.encode("utf-8", "replace"))
                        del stderr_tail[:-_BRIDGE_STDERR_TAIL_BYTES]
                    if on_output:
                        on_output(kind, text)
                elif kind == "call":
                    if calls_started >= max_tool_calls:
                        fail("call_limit", f"The program made more than {max_tool_calls} tool calls.")
                        return
                    calls_started += 1
                    task = asyncio.create_task(run_call(message.get("id"), str(message.get("name")), str(message.get("args") or "{}")))
                    call_tasks.add(task)
                    task.add_done_callback(call_tasks.discard)

        stderr_task = asyncio.create_task(drain_stderr())
        try:
            await asyncio.wait_for(pump(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            fail("timeout", f"Execution timed out after {timeout_seconds} seconds.")
        finally:
            # Calls still running when the program ends or times out are cancelled and reported, not hidden.
            calls_cancelled = len(call_tasks)
            for task in list(call_tasks):
                task.cancel()
            if process.returncode is None:
                process.kill()
            await process.wait()
            # The pipe closes with the process, so this finishes promptly and keeps the crash message.
            try:
                await asyncio.wait_for(stderr_task, timeout=2)
            except (asyncio.TimeoutError, Exception):  # pylint: disable=broad-except
                stderr_task.cancel()
            run_directory.cleanup()

        final = outcome.get("final")
        stderr = stderr_tail.decode("utf-8", "replace")
        if final is None:
            final = {"success": False, "error_type": "sandbox_crashed",
                     "error": f"The sandbox exited (code {process.returncode}) without a result. {stderr[-2000:]}".strip()}
        success = bool(final.get("success"))
        return BridgeExecutionResult(
            status="success" if success else "error",
            execution_time=time.time() - start_time,
            result=final.get("result") if success else None,
            stdout="".join(stdout_parts),
            stderr=stderr,
            error=None if success else (final.get("error") or "unknown sandbox error"),
            error_type=None if success else (final.get("error_type") or "sandbox_crashed"),
            calls_completed=calls_completed,
            calls_cancelled=calls_cancelled,
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
        timed_out = False

        cmd = self._build_command(
            session_bytes=session_bytes,
            session_metadata=session_metadata,
            memory_limit_mb=memory_limit_mb,
            root_ca_path=root_ca_path,
            insecure_tls_domains=insecure_tls_domains,
        )

        try:
            code_bytes = code.encode('utf-8')
        except UnicodeEncodeError as e:
            return CodeExecutionResult(
                status="error",
                execution_time=time.time() - start_time,
                stderr=f"Failed to encode code as UTF-8: {e}",
            )

        run_directory = self._run_directory()
        try:
            process = subprocess.run(
                cmd,
                input=code_bytes,
                capture_output=True,
                text=False,
                timeout=timeout_seconds,
                check=False,
                env=self._run_environment(run_directory),
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
            timed_out = True
        finally:
            run_directory.cleanup()

        end_time = time.time()

        return CodeExecutionResult(
            status=status,
            execution_time=end_time - start_time,
            stdout=stdout or None,
            stderr=stderr or None,
            result=result,
            session_metadata=session_metadata,
            session_bytes=session_bytes,
            timed_out=timed_out,
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
