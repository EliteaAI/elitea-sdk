"""Cross-run cache for remote MCP tool discovery.

The platform runs every agent in a short-lived process, so the tool list a run discovers
is gone before the next run starts. A backend registered by the host (Redis on the
platform) keeps the unfiltered ``tools/list`` result for the toolkit's Cache TTL. Without
a registered backend every run discovers live.

Entries are keyed by a digest of the server URL, the request headers and the TLS setting:
a different credential is a different entry, matching the ``cacheScope: private`` rule of
the MCP caching utility and VS Code's per-config cache nonce. A per-server generation is
part of every key, so Load Tools can drop every credential's entry for a server at once
without knowing the other users' keys.

A generation is a random value, never a counter: once its key expires a counter restarts
and would re-address entries it had already retired. It outlives the longest entry TTL
plus an in-flight discovery, so no entry written under an earlier generation can still be
alive when reads fall back to the empty generation.
"""

import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Protocol

from .mcp_oauth import canonical_resource

logger = logging.getLogger(__name__)

CACHE_TTL_DEFAULT = 300
CACHE_TTL_MIN = 60
CACHE_TTL_MAX = 3600
CACHE_FORMAT_VERSION = 1
GENERATION_KEY_PREFIX = "gen:"
GENERATION_TTL = 2 * CACHE_TTL_MAX


class McpDiscoveryCacheBackend(Protocol):
    def get(self, key: str) -> Optional[str]: ...

    def set(self, key: str, value: str, ttl: int) -> None: ...


_backend: Optional[McpDiscoveryCacheBackend] = None


def register_discovery_cache_backend(backend: Optional[McpDiscoveryCacheBackend]) -> None:
    global _backend
    _backend = backend
    logger.info("MCP discovery cache backend: %s", type(backend).__name__ if backend else "none")


def get_discovery_cache_backend() -> Optional[McpDiscoveryCacheBackend]:
    return _backend


def clamp_cache_ttl(value: Any, default: int = CACHE_TTL_DEFAULT) -> int:
    try:
        ttl = int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid cache_ttl '%s', using %d", value, default)
        return default
    if ttl == 0:
        return 0
    if ttl < 0:
        logger.warning("Negative cache_ttl '%s', using %d", value, default)
        return default
    clamped = min(max(ttl, CACHE_TTL_MIN), CACHE_TTL_MAX)
    if clamped != ttl:
        logger.warning("cache_ttl %d outside %d-%d, using %d", ttl, CACHE_TTL_MIN, CACHE_TTL_MAX, clamped)
    return clamped


def _digest(identity: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_server_scope_key(url: str) -> str:
    return _digest({"v": CACHE_FORMAT_VERSION, "url": canonical_resource(url)})


def build_discovery_cache_key(
    url: str, headers: Optional[Dict[str, Any]], ssl_verify: bool, generation: str = ""
) -> str:
    return _digest({
        "v": CACHE_FORMAT_VERSION,
        "url": canonical_resource(url),
        "headers": sorted((str(key).lower(), str(value)) for key, value in (headers or {}).items()),
        "ssl_verify": bool(ssl_verify),
        "generation": str(generation),
    })


def _read_generation(url: str) -> Optional[str]:
    try:
        raw = _backend.get(GENERATION_KEY_PREFIX + build_server_scope_key(url))
    except Exception:
        logger.warning("MCP discovery cache generation read failed, discovering live", exc_info=True)
        return None
    return raw or ""


def resolve_discovery_cache_key(url: str, headers: Optional[Dict[str, Any]], ssl_verify: bool) -> Optional[str]:
    """Pin the entry key once per discovery, so a Load Tools bump mid-discovery retires what it writes."""
    if _backend is None:
        return None
    generation = _read_generation(url)
    if generation is None:
        return None
    return build_discovery_cache_key(url, headers, ssl_verify, generation)


def _is_tool_list(tools: Any) -> bool:
    return isinstance(tools, list) and all(isinstance(tool, dict) and tool.get("name") for tool in tools)


def read_cached_discovery(key: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if _backend is None or key is None:
        return None
    try:
        raw = _backend.get(key)
    except Exception:
        logger.warning("MCP discovery cache read failed, discovering live", exc_info=True)
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        tools = payload["tools"]
        version = payload.get("v")
    except (ValueError, TypeError, KeyError, AttributeError):
        logger.warning("MCP discovery cache entry unreadable, discovering live")
        return None
    if version != CACHE_FORMAT_VERSION or not _is_tool_list(tools):
        logger.warning("MCP discovery cache entry has an unexpected shape, discovering live")
        return None
    return tools


def write_cached_discovery(key: Optional[str], tools: List[Dict[str, Any]], ttl: int) -> None:
    if _backend is None or key is None or ttl <= 0 or not tools:
        return
    try:
        _backend.set(key, json.dumps({"v": CACHE_FORMAT_VERSION, "tools": tools}), ttl)
    except Exception:
        logger.warning("MCP discovery cache write failed", exc_info=True)


RETIREMENT_TTL = 60


def retire_cached_discovery(url: str, headers: Optional[Dict[str, Any]], ssl_verify: bool) -> None:
    """Retire the one entry written under this credential, leaving every other credential's
    entry for the same server alone: a 401 says this credential is dead, nothing more.
    An empty payload reads back as a miss, so no delete is needed on the backend."""
    if _backend is None:
        return
    key = resolve_discovery_cache_key(url, headers, ssl_verify)
    if key is None:
        return
    try:
        _backend.set(key, "", RETIREMENT_TTL)
    except Exception:
        logger.warning("MCP discovery cache retirement failed", exc_info=True)


def invalidate_server_discovery(url: str) -> bool:
    """Retire every credential's cached entry for a server under either TLS setting, whoever wrote it.

    Returns False when a backend is registered but could not be told, so the caller can say
    that cached lists may outlive this refresh.
    """
    if _backend is None:
        return True
    try:
        _backend.set(GENERATION_KEY_PREFIX + build_server_scope_key(url), uuid.uuid4().hex, GENERATION_TTL)
    except Exception:
        logger.warning("MCP discovery cache invalidation failed", exc_info=True)
        return False
    return True
