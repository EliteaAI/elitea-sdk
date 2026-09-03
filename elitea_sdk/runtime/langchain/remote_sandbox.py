"""Remote sandbox backend that delegates execution to the pylon_sandbox service."""

import base64
import json
import logging
import time

import aiohttp

from ..utils.trace_limits import cap_trace_json, cap_trace_text
from .sandbox_types import CodeExecutionResult

logger = logging.getLogger(__name__)

_MAX_TIMEOUT_SECONDS = 55.0
_HTTP_TIMEOUT_HEADROOM = 15
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024  # 4 MB cap on session_bytes (binary, not trace-capped)


class RemoteSandbox:
    """HTTP client that delegates code execution to the sandbox_runner service.

    Implements the same async execute() interface as PyodideSandbox so it can
    be used as a drop-in replacement in the sandbox tool.
    """

    def __init__(self, url: str, auth_token: str, tenant_id: str = "unknown"):
        self.url = url.rstrip("/")
        self.auth_token = auth_token
        self.tenant_id = tenant_id
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self, timeout: aiohttp.ClientTimeout) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

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
        timeout = min(timeout_seconds or _MAX_TIMEOUT_SECONDS, _MAX_TIMEOUT_SECONDS)
        http_timeout = aiohttp.ClientTimeout(total=timeout + _HTTP_TIMEOUT_HEADROOM)

        body = {
            "code": code,
            "session_bytes": base64.b64encode(session_bytes).decode("ascii") if session_bytes else None,
            "session_metadata": session_metadata,
            "timeout_seconds": int(timeout),
        }

        headers = {
            "X-Sandbox-Token": self.auth_token,
            "X-Tenant-ID": self.tenant_id,
            "Content-Type": "application/json",
        }

        start = time.monotonic()
        try:
            session = await self._get_session(http_timeout)
            async with session.post(self.url, json=body, headers=headers) as resp:
                elapsed = time.monotonic() - start

                if resp.status == 401:
                    raise ValueError("Sandbox auth token invalid")

                if resp.status == 503:
                    try:
                        data = await resp.json()
                        detail = data.get("stderr") or data.get("detail", "")
                    except (json.JSONDecodeError, aiohttp.ContentTypeError):
                        detail = ""
                    return CodeExecutionResult(
                        status="error",
                        stderr=detail or "Sandbox service unavailable, retry shortly",
                        execution_time=elapsed,
                        infra_category="service_busy",
                    )

                if resp.status != 200:
                    text = await resp.text()
                    return CodeExecutionResult(
                        status="error",
                        stderr=f"Sandbox service returned HTTP {resp.status}: {text[:200]}",
                        execution_time=elapsed,
                        infra_category="service_busy",
                    )

                data = await resp.json()

        except aiohttp.ClientError as exc:
            elapsed = time.monotonic() - start
            logger.warning("Sandbox service request failed: %s", exc)
            return CodeExecutionResult(
                status="error",
                stderr=f"Sandbox service unreachable: {exc}",
                execution_time=elapsed,
                infra_category="service_busy",
            )
        except TimeoutError:
            elapsed = time.monotonic() - start
            return CodeExecutionResult(
                status="error",
                stderr=f"Sandbox service timed out after {elapsed:.1f}s",
                execution_time=elapsed,
                timed_out=True,
            )

        response_session_bytes = None
        raw_session = data.get("session_bytes")
        if raw_session:
            decoded = base64.b64decode(raw_session)
            if len(decoded) <= _MAX_RESPONSE_BYTES:
                response_session_bytes = decoded
            else:
                logger.warning("session_bytes response truncated (%d bytes)", len(decoded))

        return CodeExecutionResult(
            result=cap_trace_json(data.get("result")),
            stdout=cap_trace_text(data.get("stdout")),
            stderr=cap_trace_text(data.get("stderr")),
            status="success" if data.get("success") else "error",
            execution_time=elapsed,
            session_metadata=data.get("session_metadata"),
            session_bytes=response_session_bytes,
        )
