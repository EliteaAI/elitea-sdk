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
