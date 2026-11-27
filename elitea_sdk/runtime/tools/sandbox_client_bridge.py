"""SandboxClient over the Code Mode bridge (#7719): sandbox code gets a stub, the worker makes the calls."""

import asyncio
import base64
import inspect
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CODE_NODE = "code_node"
CLIENT_METHODS = (
    "get_mcp_toolkits", "mcp_tool_call", "get_app_details", "get_list_of_apps", "get_app_version_details",
    "get_user_data", "unsecret", "get_private_project_secret", "bucket_exists", "create_bucket", "list_artifacts",
    "create_artifact", "download_artifact", "delete_artifact", "list_artifacts_s3", "upload_artifact_s3",
    "download_artifact_s3", "delete_artifact_s3", "head_artifact_s3",
)
ARTIFACT_METHODS = ("create", "get", "delete", "list", "append", "get_content_bytes")
SECRET_METHODS = frozenset(("unsecret", "get_private_project_secret"))
ALLOWED_CALLS = frozenset(
    [f"client.{m}" for m in CLIENT_METHODS] + ["client.artifact"] + [f"artifact.{m}" for m in ARTIFACT_METHODS]
)
# Exception classes the stub re-raises by name; anything else becomes ApiDetailsRequestError.
ERROR_CLASSES = frozenset((
    "ApiDetailsRequestError", "PrivateSecretError", "NoPersonalProject", "PrivateSecretNotShared",
    "PrivateSecretNotFound", "PermissionError", "ValueError", "TypeError", "KeyError",
))
_SECRET_OUTCOMES = {"NoPersonalProject": "no_personal_project", "PrivateSecretNotShared": "not_shared",
                    "PrivateSecretNotFound": "not_found"}
_MIN_MASKED_LEN = 4
AUDIT_TRACER_NAME = "audit-trail"
GOVERNED_REFUSAL = ("The platform client is not available to code in governed runs yet: the run's identity never "
                    "leaves its worker. Use the agent's toolkits (for example the artifact toolkit) instead.")
LLM_CODE_REFUSAL = ("Secrets are not available to code written by the model. Read them in a pipeline code node, "
                    "or use a toolkit that holds the credential.")


def bridge_enabled() -> bool:
    return os.environ.get("SANDBOX_CLIENT_BRIDGE", "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def bridge_limits() -> Dict[str, int]:
    return {
        # The run timeout is the real bound; this only stops a runaway loop.
        "max_calls": _env_int("SANDBOX_CLIENT_BRIDGE_MAX_CALLS", 10000),
        # Room for a base64 artifact download of ~18 MB in one reply.
        "max_line_bytes": _env_int("SANDBOX_CLIENT_BRIDGE_MAX_LINE_BYTES", 24 * 1024 * 1024),
    }


BYTES_KEY = "__elitea_bridge_bytes__"


def encode_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {BYTES_KEY: base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, dict):
        return {k: encode_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(v) for v in value]
    return value


def decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {BYTES_KEY}:
            return base64.b64decode(value[BYTES_KEY])
        return {k: decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    return value


class SecretMasker:
    """Values handed out as secrets during one run, replaced with *** in what the run returns."""

    def __init__(self) -> None:
        self._values: set = set()

    def add(self, value: Any) -> None:
        if value is None:
            return
        text = value if isinstance(value, str) else str(value)
        if len(text) >= _MIN_MASKED_LEN:
            self._values.add(text)

    def mask(self, value: Any) -> Any:
        if not self._values:
            return value
        if isinstance(value, str):
            for secret in sorted(self._values, key=len, reverse=True):
                value = value.replace(secret, "***")
            return value
        if isinstance(value, dict):
            return {k: self.mask(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.mask(v) for v in value]
        return value


def emit_secret_audit(*, method: str, secret_name: Any, outcome: str, caller: str,
                      project_id: Any = None, user_id: Any = None) -> None:
    """One zero-length audit span per secret read (name and outcome, never the value); never raises."""
    try:
        from opentelemetry import trace  # pylint: disable=C0415
        attrs = {
            # "tool" is the audit row type the tracing plugin already stores.
            "audit.observation.type": "tool",
            "audit.tool.name": f"sandbox_secret:{str(secret_name)[:200]}",
            "audit.is_error": outcome != "success",
            "audit.secret.method": method,
            "audit.secret.outcome": outcome,
            "audit.secret.caller": caller,
        }
        for key, value in (("project.id", project_id), ("user.id", user_id)):
            if value is not None:
                attrs[key] = int(value)
        trace.get_tracer(AUDIT_TRACER_NAME).start_span(
            name=f"sandbox_secret_read:{method}:{outcome}", attributes=attrs).end()
    except Exception as exc:  # pylint: disable=W0718
        logger.debug("sandbox secret audit failed: %s", exc)


class ClientBridgeDispatcher:
    """Answers the stub's `client.*` / `artifact.*` calls with a real SandboxClient that lives on the worker."""

    def __init__(self, client_factory: Callable[[Optional[int]], Any], *, caller: str, project_id: Any = None,
                 user_id: Any = None, governed: bool = False, max_result_bytes: int = 24 * 1024 * 1024,
                 audit: Callable[..., None] = emit_secret_audit, deadline: Optional[float] = None) -> None:
        self._client_factory = client_factory
        self._clients: Dict[Optional[int], Any] = {}
        self._artifacts: Dict[Tuple[Optional[int], str], Any] = {}
        # time.monotonic() value after which no platform request starts or waits (None = unbounded)
        self.deadline = deadline
        self._lock = threading.RLock()
        self._inflight = 0
        self._closing = False
        self.caller = caller
        self.project_id = project_id
        self.user_id = user_id
        self.governed = governed
        self.max_result_bytes = max_result_bytes
        self.audit = audit
        self.masker = SecretMasker()
        self.call_log: List[Dict[str, Any]] = []

    async def __call__(self, name: str, args_json: str) -> str:
        started = time.perf_counter()
        reply = await self._dispatch(name, args_json)
        self.call_log.append({"tool": name, "ok": reply.get("ok", False), "error_class": reply.get("error_class"),
                              "ms": round((time.perf_counter() - started) * 1000)})
        try:
            encoded = json.dumps(reply, default=str)
        except (TypeError, ValueError) as exc:
            encoded = json.dumps(_error("TypeError", f"The result of {name} cannot be sent to the sandbox: {exc}"))
        if len(encoded) > self.max_result_bytes:
            encoded = json.dumps(_error("ValueError", f"The result of {name} is over the "
                                                      f"{self.max_result_bytes}-byte sandbox limit."))
        return encoded

    async def _dispatch(self, name: str, args_json: str) -> Dict[str, Any]:
        if name not in ALLOWED_CALLS:
            return _error("PermissionError", f"'{name}' is not available to sandbox code.")
        try:
            payload = json.loads(args_json or "{}")
            args = decode_value(payload.get("args") or [])
            kwargs = decode_value(payload.get("kwargs") or {})
            project_id = payload.get("project_id")
            project_id = None if project_id in (None, self.project_id) else int(project_id)
        except (TypeError, ValueError, AttributeError) as exc:
            return _error("ValueError", f"Malformed client call: {exc}")
        kind, method = name.split(".", 1)
        secret = kind == "client" and method in SECRET_METHODS
        secret_name = (args[0] if args else kwargs.get("secret_name")) if secret else None
        if self.governed:
            if secret:
                self._audit(method, secret_name, "denied")
            return _error("PermissionError", GOVERNED_REFUSAL)
        if secret and self.caller != CODE_NODE:
            self._audit(method, secret_name, "denied")
            return _error("PermissionError", LLM_CODE_REFUSAL)
        try:
            value = await asyncio.to_thread(self._run_in_thread, kind, method, project_id, args, kwargs)
        except Exception as exc:  # pylint: disable=W0718
            error_class = type(exc).__name__
            if secret:
                self._audit(method, secret_name, _SECRET_OUTCOMES.get(error_class, "failed"))
            return _error(error_class if error_class in ERROR_CLASSES else "ApiDetailsRequestError", str(exc))
        if secret:
            self._audit(method, secret_name, "success" if value is not None else "not_found")
            self.masker.add(value)
        if _payload_size(value) > self.max_result_bytes:
            return _error("ValueError", f"The result of {name} is over the {self.max_result_bytes}-byte sandbox limit.")
        return {"ok": True, "value": encode_value(value)}

    def close(self) -> None:
        """Ends the run's platform access. A request still running in a thread keeps its session; the last one out closes."""
        with self._lock:
            self._closing = True
            if self._inflight == 0:
                self._close_sessions()

    def _close_sessions(self) -> None:
        for client in self._clients.values():
            try:
                session = getattr(client, "_session", None)
                if session is not None:
                    session.close()
            except Exception:  # pylint: disable=W0718
                pass
        self._clients.clear()
        self._artifacts.clear()

    def _client(self, project_id: Optional[int]) -> Any:
        with self._lock:
            if project_id not in self._clients:
                client = self._client_factory(project_id)
                self._bound_to_deadline(client)
                self._clients[project_id] = client
            return self._clients[project_id]

    def _bound_to_deadline(self, client: Any) -> None:
        """Every request the client makes gets timeouts no longer than what is left of the run."""
        if self.deadline is None or not hasattr(client, "_request"):
            return
        original = client._request  # pylint: disable=W0212

        def request(method: str, url: str, **kwargs: Any) -> Any:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The sandbox run is out of time; the platform request was not sent.")
            timeout = kwargs.get("timeout", getattr(client, "timeout", None))
            connect, read = timeout if isinstance(timeout, tuple) else (timeout, timeout)
            kwargs["timeout"] = tuple(min(t, remaining) if t else remaining for t in (connect, read))
            return original(method, url, **kwargs)

        client._request = request  # pylint: disable=W0212

    def _artifact(self, project_id: Optional[int], bucket_name: str) -> Any:
        key = (project_id, str(bucket_name))
        with self._lock:
            if key not in self._artifacts:
                self._artifacts[key] = self._client(project_id).artifact(bucket_name)
            return self._artifacts[key]

    def _run_in_thread(self, kind: str, method: str, project_id: Optional[int], args: list, kwargs: dict) -> Any:
        # Counted inside the thread: awaiting to_thread can be cancelled after the call has already started.
        with self._lock:
            if self._closing:
                raise RuntimeError("The sandbox run has ended; the platform request was not sent.")
            self._inflight += 1
        try:
            return self._invoke(kind, method, project_id, args, kwargs)
        finally:
            with self._lock:
                self._inflight -= 1
                if self._closing and self._inflight == 0:
                    self._close_sessions()

    def _invoke(self, kind: str, method: str, project_id: Optional[int], args: list, kwargs: dict) -> Any:
        if kind == "artifact":
            if not args:
                raise ValueError("artifact call without a bucket")
            return getattr(self._artifact(project_id, args[0]), method)(*args[1:], **kwargs)
        if method == "artifact":
            self._artifact(project_id, args[0] if args else kwargs.get("bucket_name"))
            return None
        return getattr(self._client(project_id), method)(*args, **kwargs)

    def _audit(self, method: str, secret_name: Any, outcome: str) -> None:
        try:
            self.audit(method=method, secret_name=secret_name, outcome=outcome, caller=self.caller,
                       project_id=self.project_id, user_id=self.user_id)
        except Exception as exc:  # pylint: disable=W0718
            logger.debug("sandbox secret audit failed: %s", exc)


def _payload_size(value: Any) -> int:
    """Rough size of a result as the stub receives it (bytes grow by a third in base64), before anything is copied."""
    if isinstance(value, (bytes, bytearray)):
        return len(value) * 4 // 3
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_payload_size(v) + len(str(k)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_payload_size(v) for v in value)
    return 8


def _error(error_class: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "error_class": error_class, "error": message}


def sandbox_client_factory(elitea_client: Any) -> Callable[[Optional[int]], Any]:
    """Builds the worker-side SandboxClient, the same one sandbox code used to get, per project."""
    from ..clients.sandbox_client import SandboxClient  # pylint: disable=C0415

    def factory(project_id: Optional[int]) -> Any:
        return SandboxClient(
            base_url=elitea_client.base_url,
            project_id=elitea_client.project_id if project_id is None else project_id,
            auth_token=elitea_client.auth_token,
            auth_session=getattr(elitea_client, "auth_session", None),
            session_cookie_name=getattr(elitea_client, "session_cookie_name", None),
        )
    return factory


_STUB_TEMPLATE = '''
import json as _eb_json
import base64 as _eb_b64
_EB_BYTES_KEY = __BYTES_KEY__
import elitea_bridge as _eb_bridge
from pyodide.ffi import run_sync as _eb_run_sync


class ApiDetailsRequestError(Exception):
    ...


class PrivateSecretError(Exception):
    """ Base error for personal-project secret lookups """


class NoPersonalProject(PrivateSecretError):
    ...


class PrivateSecretNotShared(PrivateSecretError):
    ...


class PrivateSecretNotFound(PrivateSecretError):
    ...


_EB_ERRORS = {c.__name__: c for c in (ApiDetailsRequestError, PrivateSecretError, NoPersonalProject,
                                      PrivateSecretNotShared, PrivateSecretNotFound, PermissionError,
                                      ValueError, TypeError, KeyError)}


__CODEC__


def _eb_call(_eb_name, _eb_project, *args, **kwargs):
    payload = _eb_json.dumps({"args": _eb_enc(list(args)), "kwargs": _eb_enc(kwargs), "project_id": _eb_project},
                             default=str)
    reply = _eb_json.loads(_eb_run_sync(_eb_bridge.call(_eb_name, payload)))
    if reply.get("ok"):
        return _eb_dec(reply.get("value"))
    raise _EB_ERRORS.get(reply.get("error_class"), ApiDetailsRequestError)(reply.get("error"))


class SandboxArtifact:
    def __init__(self, client, bucket_name):
        self.client = client
        self.bucket_name = bucket_name
        _eb_call("client.artifact", client.project_id, bucket_name)


def _eb_artifact_method(name):
    def method(self, *args, **kwargs):
        return _eb_call("artifact." + name, self.client.project_id, self.bucket_name, *args, **kwargs)
    method.__name__ = name
    return method


for _eb_m in __ARTIFACT_METHODS__:
    setattr(SandboxArtifact, _eb_m, _eb_artifact_method(_eb_m))

_EB_REMOVED = ("auth_token", "auth_session", "session_cookie_name", "headers", "_session")


class SandboxClient:
    def __init__(self, base_url=None, project_id=None, *args, **kwargs):
        self.base_url = (base_url or __BASE_URL__).rstrip("/")
        self.api_v2_path = "/api/v2"
        self.project_id = __PROJECT_ID__ if project_id is None else int(project_id)

    def __getattr__(self, name):
        if name in _EB_REMOVED:
            raise AttributeError(
                f"'{name}' is not available in the sandbox: the platform credential stays on the worker (#7719). "
                "Use the elitea_client methods, or an Elitea MCP toolkit for other platform calls.")
        raise AttributeError(f"'SandboxClient' object has no attribute '{name}'")

    def artifact(self, bucket_name):
        return SandboxArtifact(self, bucket_name)


def _eb_client_method(name):
    def method(self, *args, **kwargs):
        return _eb_call("client." + name, self.project_id, *args, **kwargs)
    method.__name__ = name
    return method


for _eb_m in __CLIENT_METHODS__:
    setattr(SandboxClient, _eb_m, _eb_client_method(_eb_m))
del _eb_m

elitea_client = SandboxClient()
'''


def _stub_codec() -> str:
    """The stub's encode/decode are the worker's own functions renamed, so the two sides cannot drift apart."""
    source = "\n\n".join(inspect.getsource(fn).rstrip() for fn in (encode_value, decode_value))
    # The stub namespace has no typing names, so the annotations go.
    for old, new in ((": Any", ""), (" -> Any", ""), ("encode_value", "_eb_enc"), ("decode_value", "_eb_dec"),
                     ("base64.", "_eb_b64."), ("BYTES_KEY", "_EB_BYTES_KEY")):
        source = source.replace(old, new)
    return source


def build_client_stub(base_url: str, project_id: Any) -> str:
    """The sandbox-side client: same public names as SandboxClient, no credential, every call bridged."""
    return (_STUB_TEMPLATE
            .replace("__CODEC__", _stub_codec())
            .replace("__BYTES_KEY__", repr(BYTES_KEY))
            .replace("__ARTIFACT_METHODS__", repr(ARTIFACT_METHODS))
            .replace("__CLIENT_METHODS__", repr(CLIENT_METHODS))
            .replace("__BASE_URL__", repr(str(base_url)))
            .replace("__PROJECT_ID__", repr(int(project_id)) if project_id is not None else "None"))
