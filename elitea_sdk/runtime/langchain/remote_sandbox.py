"""Remote sandbox backend that delegates execution to the pylon_sandbox service."""

import base64
import logging
import time

import aiohttp

from .sandbox_types import CodeExecutionResult

logger = logging.getLogger(__name__)


class RemoteSandbox:
    """HTTP client that delegates code execution to the sandbox_runner service.

    Implements the same async execute() interface as PyodideSandbox so it can
    be used as a drop-in replacement in the sandbox tool.
    """

    def __init__(self, url: str, auth_token: str, tenant_id: str = "unknown"):
        self.url = url.rstrip("/")
        self.auth_token = auth_token
        self.tenant_id = tenant_id

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
        timeout = timeout_seconds or 55
        http_timeout = aiohttp.ClientTimeout(total=timeout + 15)

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
            async with aiohttp.ClientSession(timeout=http_timeout) as session:
                async with session.post(self.url, json=body, headers=headers) as resp:
                    elapsed = time.monotonic() - start

                    if resp.status == 401:
                        raise ValueError("Sandbox auth token invalid")

                    if resp.status == 503:
                        data = await resp.json()
                        return CodeExecutionResult(
                            status="error",
                            stderr=data.get("stderr") or data.get("detail", "Service unavailable, retry shortly"),
                            execution_time=elapsed,
                        )

                    if resp.status != 200:
                        text = await resp.text()
                        return CodeExecutionResult(
                            status="error",
                            stderr=f"Sandbox service returned HTTP {resp.status}: {text[:200]}",
                            execution_time=elapsed,
                        )

                    data = await resp.json()

        except aiohttp.ClientError as exc:
            elapsed = time.monotonic() - start
            logger.warning("Sandbox service request failed: %s", exc)
            return CodeExecutionResult(
                status="error",
                stderr=f"Sandbox service unreachable: {exc}",
                execution_time=elapsed,
            )
        except TimeoutError:
            elapsed = time.monotonic() - start
            return CodeExecutionResult(
                status="error",
                stderr=f"Sandbox service timed out after {elapsed:.1f}s",
                execution_time=elapsed,
            )

        response_session_bytes = None
        if data.get("session_bytes"):
            response_session_bytes = base64.b64decode(data["session_bytes"])

        return CodeExecutionResult(
            result=data.get("result"),
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
            status="success" if data.get("success") else "error",
            execution_time=elapsed,
            session_metadata=data.get("session_metadata"),
            session_bytes=response_session_bytes,
        )
