"""The run token of a governed Elitea run, on the internal Elitea MCP requests (C-1009, EliteaAI/elitea_issues#7214).

Elitea Main mints one token per admitted run; the worker hangs it on the run's client as ``internal_run_auth``
(``token``, ``origins``, ``identity``). An MCP toolkit whose server is an internal Elitea MCP endpoint
(``.../app/<project>/mcp...``) on one of those origins authenticates with it, whatever ``Authorization`` its settings
carried; every other server is left alone, so the token never reaches another host. Without a run token (a legacy run)
nothing changes. The discovery cache is keyed by ``identity`` (who the run acts for), not by the token, which changes
every run.
"""
import re
from typing import Any, Mapping, Optional, Tuple
from urllib.parse import urlsplit

_INTERNAL_PATH = re.compile(r"(?:^|/)app/\d+/mcp(?:/|$)")


def internal_run_headers(client: Any, url: str, headers: Optional[Mapping[str, str]]) -> Tuple[Optional[dict], Optional[str]]:
    """``(headers, cache identity)``: the headers to send to ``url`` and, when the run token applies, the identity to key
    the discovery cache with (else ``None`` and the headers unchanged)."""
    auth = getattr(client, "internal_run_auth", None) if client is not None else None
    token = getattr(auth, "token", None)
    origins = getattr(auth, "origins", None) or ()
    if not isinstance(token, str) or not token or not isinstance(url, str):
        return headers, None
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}".lower()
    if origin not in origins or not _INTERNAL_PATH.search(parts.path):
        return headers, None
    merged = {key: value for key, value in dict(headers or {}).items()
              if not (isinstance(key, str) and key.lower() == "authorization")}
    merged["Authorization"] = f"Bearer {token}"
    return merged, getattr(auth, "identity", None) or None
