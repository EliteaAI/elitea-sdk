"""The run token of a governed Elitea run, on the internal Elitea MCP requests (C-1009, EliteaAI/elitea_issues#7214).

Elitea Main mints one token per admitted run; the worker hangs it on the run's client as ``internal_run_auth``
(``token``, ``origins``, ``identity``). An MCP toolkit whose server is an internal Elitea MCP endpoint
(``/app/<project>/mcp...``) on one of those origins authenticates with it, whatever ``Authorization`` its settings
carried; any other MCP server on those origins gets no credentials of its own (Authorization, Cookie and
Proxy-Authorization are dropped), and servers on other origins are left alone, so the token never reaches another host.
The path is matched as the HTTP client sends it (``client_path``: decoded, dot segments resolved, slashes collapsed),
the same way Main matches it. Without a run token (a legacy run) nothing changes. The discovery cache is keyed by
``identity`` (who the run acts for), not by the token, which changes every run.
"""
import posixpath
import re
from typing import Any, Mapping, Optional, Tuple
from urllib.parse import unquote, urlsplit

# The same shape Main matches, on the path as sent: /app/<project>/mcp, then / or nothing
_INTERNAL_PATH = re.compile(r"^/app/\d+/mcp(?:/|$)")
_AUTH_HEADERS = frozenset({"authorization", "cookie", "proxy-authorization"})


def client_path(path: str) -> str:
    """A URL path as the HTTP client ends up sending it: percent-decoded (at most 3 times), backslashes as slashes,
    repeated slashes collapsed, ``.`` and ``..`` resolved."""
    text = str(path or "")
    for _ in range(3):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded
    text = re.sub(r"/{2,}", "/", "/" + text.replace("\\", "/").lstrip("/"))
    resolved = "/" + posixpath.normpath(text).lstrip("/")
    return resolved + "/" if text.endswith("/") and resolved != "/" else resolved


def internal_run_headers(client: Any, url: str, headers: Optional[Mapping[str, str]]) -> Tuple[Optional[dict], Optional[str]]:
    """``(headers, cache identity)``: the headers to send to ``url`` and, when the run token applies, the identity to key
    the discovery cache with (else ``None``)."""
    auth = getattr(client, "internal_run_auth", None) if client is not None else None
    token = getattr(auth, "token", None)
    origins = getattr(auth, "origins", None) or ()
    if not isinstance(token, str) or not token or not isinstance(url, str):
        return headers, None
    parts = urlsplit(url)
    if f"{parts.scheme}://{parts.netloc}".lower() not in origins:
        return headers, None
    stripped = {key: value for key, value in dict(headers or {}).items()
                if not (isinstance(key, str) and key.lower() in _AUTH_HEADERS)}
    if not _INTERNAL_PATH.match(client_path(parts.path)):
        return stripped, None   # the platform's own origin, another route: no credentials of the toolkit's own
    stripped["Authorization"] = f"Bearer {token}"
    return stripped, getattr(auth, "identity", None) or None


def is_run_token_bearer(client: Any) -> bool:
    """Is the client's own bearer the run token (#7229: a governed run whose only platform identity is that token)?"""
    auth = getattr(client, "internal_run_auth", None) if client is not None else None
    token = getattr(auth, "token", None)
    return isinstance(token, str) and bool(token) and getattr(client, "auth_token", None) == token
