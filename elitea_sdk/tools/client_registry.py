"""
Client Registry for Shared API Client Management

This module provides a centralized registry for API clients to avoid
creating duplicate HTTP connections when multiple toolkits connect
to the same service.

Key Features:
- Clients are cached by an explicit, hashable ClientKey (service, target,
  toolkit instance, credential reference/generation, auth-config digest)
- Bounded cache: LRU eviction at max_entries and a TTL checked on access
- Entries can be invalidated by credential reference or toolkit instance
- Thread-safe access with locking
- Automatic cleanup of stale connections
- Connection pooling across toolkits

Usage:
    from elitea_sdk.tools.client_registry import ClientRegistry

    # Get or create a client
    client = ClientRegistry.get_client(
        service_type="jira",
        base_url="https://jira.company.com",
        auth_config={"token": "..."},
        factory=lambda: Jira(url=url, token=token)
    )

    # Client is shared across all toolkits with same connection params
"""

import hashlib
import json
import logging
import threading
import weakref
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, TypeVar
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


def _normalize(config: Dict[str, Any]) -> Dict[str, str]:
    """Reduce a config dict to hashable strings; secrets are hashed, not kept."""
    normalized = {}
    for key, value in sorted(config.items()):
        if hasattr(value, 'get_secret_value'):
            normalized[key] = hashlib.sha256(
                value.get_secret_value().encode()
            ).hexdigest()[:16]
        elif isinstance(value, str):
            normalized[key] = value
        elif isinstance(value, dict):
            normalized[key] = json.dumps(value, sort_keys=True, default=str)
        elif value is not None:
            normalized[key] = str(value)
    return normalized


def _partition_digest(auth_config: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> str:
    """sha256 over the auth config and any extra settings that change client behaviour."""
    payload = str((sorted(_normalize(auth_config).items()),
                   sorted(_normalize(extra or {}).items())))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ClientKey:
    """Identity of a cached client.

    The instance/credential fields are populated by callers that know them;
    legacy callers leave them None and are partitioned by ``partition_digest``.
    """
    service_type: str
    base_url: str
    toolkit_instance_id: Optional[str] = None
    credential_ref_id: Optional[str] = None
    credential_generation: Optional[int] = None
    auth_mode: Optional[str] = None
    canonical_target: Optional[str] = None
    partition_digest: str = ""

    @classmethod
    def from_legacy(
        cls,
        service_type: str,
        base_url: str,
        auth_config: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> 'ClientKey':
        """Key derived from the auth config plus extra settings (e.g. custom headers)."""
        return cls(
            service_type=service_type,
            base_url=base_url.rstrip('/'),
            partition_digest=_partition_digest(auth_config, extra),
        )


@dataclass
class ClientEntry:
    """Entry in the client registry."""
    client: Any
    service_type: str
    base_url: str
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    reference_count: int = 1

    def touch(self):
        """Update last accessed time and increment reference count."""
        self.last_accessed = datetime.now()
        self.reference_count += 1


class ClientRegistry:
    """
    Centralized registry for sharing API clients across toolkits.

    This prevents resource duplication when multiple toolkits connect
    to the same service (e.g., 3 Jira toolkits for same server).

    Thread-safe singleton implementation.
    """

    _instance: Optional['ClientRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._clients: 'OrderedDict[ClientKey, ClientEntry]' = OrderedDict()
        self.max_entries: int = 256
        self.ttl_seconds: Optional[float] = 3600
        self._client_lock = threading.RLock()
        self._initialized = True
        logger.info("[ClientRegistry] Initialized")

    @classmethod
    def get_instance(cls) -> 'ClientRegistry':
        """Get the singleton instance."""
        return cls()

    def get_client(
        self,
        service_type: str,
        base_url: str,
        auth_config: Dict[str, Any],
        factory: Callable[[], T],
        force_new: bool = False,
        key: Optional[ClientKey] = None
    ) -> T:
        """
        Get or create a client for the given service.

        Args:
            service_type: Type of service (e.g., 'jira', 'github', 'confluence')
            base_url: Base URL of the service
            auth_config: Authentication configuration (token, username, etc.)
            factory: Callable that creates a new client if not cached
            force_new: If True, always create a new client (bypass cache)
            key: Explicit cache key; derived from the other arguments when omitted

        Returns:
            The API client (shared or new)
        """
        cache_key = key or ClientKey.from_legacy(service_type, base_url, auth_config)

        with self._client_lock:
            entry = self._clients.get(cache_key)
            if entry is not None and self._is_expired(entry):
                del self._clients[cache_key]
                entry = None
            if not force_new and entry is not None:
                entry.touch()
                self._clients.move_to_end(cache_key)
                logger.info(
                    f"[ClientRegistry] Reusing {service_type} client for {base_url} "
                    f"(refs: {entry.reference_count})"
                )
                return entry.client

            # Create new client
            logger.info(f"[ClientRegistry] Creating new {service_type} client for {base_url}")
            try:
                client = factory()
                self._clients[cache_key] = ClientEntry(
                    client=client,
                    service_type=service_type,
                    base_url=base_url
                )
                self._clients.move_to_end(cache_key)
                while len(self._clients) > self.max_entries:
                    self._clients.popitem(last=False)
                return client
            except Exception as e:
                logger.error(f"[ClientRegistry] Failed to create {service_type} client: {e}")
                raise

    def _is_expired(self, entry: ClientEntry) -> bool:
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - entry.created_at).total_seconds() > self.ttl_seconds

    def invalidate(self, credential_ref_id: str, generation_lt: Optional[int] = None) -> int:
        """
        Drop clients built from a credential, e.g. after it was rotated or revoked.

        Args:
            credential_ref_id: Credential reference to invalidate
            generation_lt: If given, only drop entries whose credential generation
                is older (a missing generation counts as 0)

        Returns:
            Number of entries removed
        """
        return self._remove_where(
            lambda k: k.credential_ref_id == credential_ref_id
            and (generation_lt is None or (k.credential_generation or 0) < generation_lt)
        )

    def invalidate_instance(self, toolkit_instance_id: str) -> int:
        """Drop clients owned by a toolkit instance. Returns the number removed."""
        return self._remove_where(lambda k: k.toolkit_instance_id == toolkit_instance_id)

    def _remove_where(self, predicate: Callable[[ClientKey], bool]) -> int:
        with self._client_lock:
            doomed = [k for k in self._clients if predicate(k)]
            for k in doomed:
                del self._clients[k]
        if doomed:
            logger.info(f"[ClientRegistry] Invalidated {len(doomed)} client(s)")
        return len(doomed)

    def release_client(
        self,
        service_type: str,
        base_url: str,
        auth_config: Dict[str, Any],
        key: Optional[ClientKey] = None
    ) -> bool:
        """
        Release a reference to a client.

        Decrements reference count. Client is kept for potential reuse.

        Args:
            service_type: Type of service
            base_url: Base URL of the service
            auth_config: Authentication configuration
            key: Explicit cache key; derived from the other arguments when omitted

        Returns:
            True if client was found and released
        """
        cache_key = key or ClientKey.from_legacy(service_type, base_url, auth_config)

        with self._client_lock:
            if cache_key in self._clients:
                entry = self._clients[cache_key]
                entry.reference_count = max(0, entry.reference_count - 1)
                logger.debug(
                    f"[ClientRegistry] Released {service_type} client "
                    f"(refs: {entry.reference_count})"
                )
                return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cached clients."""
        with self._client_lock:
            stats = {
                'total_clients': len(self._clients),
                'by_service': {},
                'clients': []
            }

            for key, entry in self._clients.items():
                # Count by service type
                svc = entry.service_type
                stats['by_service'][svc] = stats['by_service'].get(svc, 0) + 1

                # Client details
                stats['clients'].append({
                    'service_type': entry.service_type,
                    'base_url': entry.base_url,
                    'reference_count': entry.reference_count,
                    'created_at': entry.created_at.isoformat(),
                    'last_accessed': entry.last_accessed.isoformat()
                })

            return stats

    def clear(self, service_type: Optional[str] = None):
        """
        Clear cached clients.

        Args:
            service_type: If provided, only clear clients of this type
        """
        with self._client_lock:
            if service_type:
                keys_to_remove = [
                    k for k, v in self._clients.items()
                    if v.service_type == service_type
                ]
                for key in keys_to_remove:
                    del self._clients[key]
                logger.info(f"[ClientRegistry] Cleared {len(keys_to_remove)} {service_type} clients")
            else:
                count = len(self._clients)
                self._clients.clear()
                logger.info(f"[ClientRegistry] Cleared all {count} clients")


# Global singleton instance
_registry = None


def get_client_registry() -> ClientRegistry:
    """Get the global ClientRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ClientRegistry()
    return _registry
