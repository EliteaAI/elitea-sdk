"""Which Microsoft Graph requests a toolkit may send: allowed HTTP methods and path prefixes.

Rules narrow what the LLM can call; the Graph permissions of the token stay the real limit.
A path prefix matches whole segments (``/me/messages`` matches ``/me/messages/123``, not
``/me/messagesX``) and ``*`` matches exactly one segment (``/users/*/events``). Denied
prefixes win over allowed ones; an empty allow list allows every path.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from langchain_core.tools import ToolException

METHODS = ("GET", "POST", "PATCH", "PUT", "DELETE")
DEFAULT_METHODS = ["GET"]
API_VERSIONS = ("v1.0", "beta")
GRAPH_HOST = "graph.microsoft.com"

_SLASHES_RE = re.compile(r"/{2,}")


class RuleViolation(ToolException):
    """A request the toolkit's settings do not allow."""


def split_graph_url(path: str) -> Tuple[Optional[str], str]:
    """(api version or None, path without host, version and query) for a relative path or a Graph URL.

    Raises ToolException for a URL on any other host: the token is sent only to Microsoft Graph.
    """
    text = (path or "").strip()
    if not text:
        raise ToolException("path is required, e.g. /me/messages")
    if "://" in text or text.startswith("//"):
        parsed = urlparse(text)
        if parsed.scheme != "https" or (parsed.hostname or "").lower() != GRAPH_HOST or parsed.username \
                or parsed.port not in (None, 443):
            raise ToolException(f"Only https://{GRAPH_HOST} URLs are allowed, got: {parsed.scheme}://{parsed.hostname}")
        text = parsed.path
    else:
        text = text.split("?", 1)[0].split("#", 1)[0]
    if not text.startswith("/"):
        text = "/" + text
    text = _SLASHES_RE.sub("/", text)
    version = None
    head, _, rest = text[1:].partition("/")
    if head.lower() in API_VERSIONS:
        version = head.lower()
        text = "/" + rest
    return version, text


def normalize_path(path: str) -> str:
    """Canonical path for rule matching: decoded once, lowercase, no trailing slash, no dot segments."""
    _, text = split_graph_url(path)
    decoded = unquote(text)
    if "%" in decoded and unquote(decoded) != decoded:
        raise ToolException("Double-encoded paths are not allowed")
    segments = [s for s in _SLASHES_RE.sub("/", decoded).split("/") if s]
    if any(s in (".", "..") for s in segments):
        raise ToolException("Paths with '.' or '..' segments are not allowed")
    return "/" + "/".join(s.lower() for s in segments)


def _prefix_segments(prefix: str) -> List[str]:
    text = (prefix or "").strip()
    if not text:
        return []
    _, text = split_graph_url(text)
    return [s.lower() for s in unquote(text).split("/") if s]


def _matches(path_segments: List[str], prefix_segments: List[str]) -> bool:
    if not prefix_segments or len(prefix_segments) > len(path_segments):
        return False
    return all(p == "*" or p == s for p, s in zip(prefix_segments, path_segments))


class RequestRules:
    """Allowed methods plus allowed / denied path prefixes of one toolkit."""

    def __init__(self, allowed_methods: Optional[Iterable[str]] = None,
                 allowed_paths: Optional[Iterable[str]] = None,
                 denied_paths: Optional[Iterable[str]] = None):
        methods = [str(m).strip().upper() for m in (allowed_methods if allowed_methods is not None else DEFAULT_METHODS)]
        self.allowed_methods = [m for m in METHODS if m in methods]
        self.allowed_paths = [p.strip() for p in (allowed_paths or []) if p and p.strip()]
        self.denied_paths = [p.strip() for p in (denied_paths or []) if p and p.strip()]
        self._allowed = [s for s in (_prefix_segments(p) for p in self.allowed_paths) if s]
        self._denied = [s for s in (_prefix_segments(p) for p in self.denied_paths) if s]

    def check(self, method: str, path: str) -> None:
        """Raise RuleViolation naming the rule when the request is not allowed."""
        method = (method or "GET").upper()
        if method not in METHODS:
            raise RuleViolation(f"Unsupported HTTP method {method}; use one of {', '.join(METHODS)}")
        if method not in self.allowed_methods:
            raise RuleViolation(
                f"{method} is not allowed by this toolkit (allowed: {', '.join(self.allowed_methods) or 'none'})"
            )
        segments = [s for s in normalize_path(path).split("/") if s]
        for prefix, segs in zip(self.denied_paths, self._denied):
            if _matches(segments, segs):
                raise RuleViolation(f"Path {path} is denied by this toolkit (rule: {prefix})")
        if self._allowed and not any(_matches(segments, segs) for segs in self._allowed):
            raise RuleViolation(
                f"Path {path} is not in the toolkit's allowed paths: {', '.join(self.allowed_paths)}"
            )

    def describe(self) -> str:
        """One line for tool descriptions, so the LLM does not try requests that will be refused."""
        parts = [f"Allowed methods: {', '.join(self.allowed_methods) or 'none'}."]
        if self.allowed_paths:
            parts.append(f"Allowed paths: {', '.join(self.allowed_paths)}.")
        if self.denied_paths:
            parts.append(f"Denied paths: {', '.join(self.denied_paths)}.")
        return " ".join(parts)
