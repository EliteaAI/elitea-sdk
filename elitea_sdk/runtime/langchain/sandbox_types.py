"""Shared types for sandbox implementations (local and remote)."""

import dataclasses
from typing import Any, Literal

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
    # Typed infra-refusal signal, so callers don't match stderr prose.
    timed_out: bool = False
    infra_category: str | None = None


BridgeErrorType = Literal[
    "out_of_memory", "result_too_large", "timeout", "call_limit", "python_exception", "sandbox_crashed",
]


@dataclasses.dataclass(kw_only=True)
class BridgeExecutionResult:
    """Outcome of a `--bridge` run; on error `error_type` is always set, never an empty success."""

    status: Status
    execution_time: float
    result: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    error_type: BridgeErrorType | None = None
    calls_completed: int = 0
    # Calls still in flight when the run ended; their side effects may be partial or missing.
    calls_cancelled: int = 0
