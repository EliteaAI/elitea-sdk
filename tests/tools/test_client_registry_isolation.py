"""ClientRegistry keying, eviction and invalidation (no network)."""


import pytest
from pydantic import SecretStr

from elitea_sdk.tools.client_registry import ClientKey, ClientRegistry

URL = "https://jira.example.com"


class CountingFactory:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return object()


@pytest.fixture
def registry(monkeypatch):
    reg = ClientRegistry()
    reg.clear()
    monkeypatch.setattr(reg, "max_entries", 256)
    monkeypatch.setattr(reg, "ttl_seconds", 3600)
    yield reg
    reg.clear()


def _get(reg, factory, token="t", url=URL, **kwargs):
    return reg.get_client("jira", url, {"token": SecretStr(token)}, factory, **kwargs)


def test_different_credentials_get_different_clients(registry):
    f = CountingFactory()
    assert _get(registry, f, token="a") is not _get(registry, f, token="b")
    assert f.calls == 2


def test_same_credentials_hit_cache(registry):
    f = CountingFactory()
    assert _get(registry, f) is _get(registry, f)
    assert f.calls == 1


def test_trailing_slash_in_base_url_is_same_target(registry):
    f = CountingFactory()
    assert _get(registry, f, url=URL) is _get(registry, f, url=URL + "/")
    assert f.calls == 1


def test_extra_headers_partition_the_key():
    auth = {"token": "t"}
    k_a = ClientKey.from_legacy("jira", URL, auth, extra={"custom_headers": {"X": "a"}})
    k_b = ClientKey.from_legacy("jira", URL, auth, extra={"custom_headers": {"X": "b"}})
    k_a2 = ClientKey.from_legacy("jira", URL, auth, extra={"custom_headers": {"X": "a"}})
    k_none = ClientKey.from_legacy("jira", URL, auth)

    assert k_a != k_b and k_a != k_none
    assert k_a == k_a2 and hash(k_a) == hash(k_a2)


def test_extra_dict_order_does_not_change_key():
    k1 = ClientKey.from_legacy("jira", URL, {}, extra={"h": {"a": "1", "b": "2"}})
    k2 = ClientKey.from_legacy("jira", URL, {}, extra={"h": {"b": "2", "a": "1"}})
    assert k1 == k2


def test_registry_separates_clients_by_extra(registry):
    f = CountingFactory()
    auth = {"token": "t"}
    a = registry.get_client("jira", URL, auth, f, key=ClientKey.from_legacy("jira", URL, auth, {"h": "a"}))
    b = registry.get_client("jira", URL, auth, f, key=ClientKey.from_legacy("jira", URL, auth, {"h": "b"}))
    assert a is not b and f.calls == 2


def test_lru_eviction_at_max_entries(registry):
    registry.max_entries = 2
    f = CountingFactory()
    c1 = _get(registry, f, token="1")
    _get(registry, f, token="2")
    _get(registry, f, token="1")  # touch 1 -> 2 becomes least recently used
    _get(registry, f, token="3")  # evicts 2

    assert registry.get_stats()["total_clients"] == 2
    assert _get(registry, f, token="1") is c1
    calls = f.calls
    _get(registry, f, token="2")  # was evicted -> rebuilt
    assert f.calls == calls + 1


def test_ttl_expiry_rebuilds(registry):
    f = CountingFactory()
    first = _get(registry, f)
    next(iter(registry._clients.values())).created_mono -= 7200
    second = _get(registry, f)

    assert second is not first
    assert f.calls == 2
    assert registry.get_stats()["total_clients"] == 1


def test_ttl_none_disables_expiry(registry):
    registry.ttl_seconds = None
    f = CountingFactory()
    first = _get(registry, f)
    next(iter(registry._clients.values())).created_mono -= 365 * 86400
    assert _get(registry, f) is first


def _explicit(ref, generation, instance="inst-1"):
    return ClientKey(
        service_type="jira", base_url=URL, toolkit_instance_id=instance,
        credential_ref_id=ref, credential_generation=generation,
        auth_mode="token", canonical_target=URL, partition_digest="d",
    )


def test_invalidate_by_credential_ref(registry):
    f = CountingFactory()
    keep = registry.get_client("jira", URL, {}, f, key=_explicit("cred-2", 1))
    registry.get_client("jira", URL, {}, f, key=_explicit("cred-1", 1))
    registry.get_client("jira", URL, {}, f, key=_explicit("cred-1", 2, instance="inst-2"))

    assert registry.invalidate("cred-1") == 2
    assert registry.get_stats()["total_clients"] == 1
    assert registry.get_client("jira", URL, {}, f, key=_explicit("cred-2", 1)) is keep


def test_invalidate_generation_lt(registry):
    f = CountingFactory()
    registry.get_client("jira", URL, {}, f, key=_explicit("cred-1", 1))
    current = registry.get_client("jira", URL, {}, f, key=_explicit("cred-1", 2, instance="inst-2"))

    assert registry.invalidate("cred-1", generation_lt=2) == 1
    assert registry.get_client("jira", URL, {}, f, key=_explicit("cred-1", 2, instance="inst-2")) is current


def test_invalidate_instance(registry):
    f = CountingFactory()
    registry.get_client("jira", URL, {}, f, key=_explicit("cred-1", 1, instance="inst-1"))
    registry.get_client("jira", URL, {}, f, key=_explicit("cred-1", 1, instance="inst-2"))

    assert registry.invalidate_instance("inst-1") == 1
    assert registry.get_stats()["total_clients"] == 1
    assert registry.invalidate_instance("unknown") == 0


def test_legacy_positional_call_returns_same_object(registry):
    f = CountingFactory()
    auth = {"token": "t", "username": None}
    a = registry.get_client("github", "https://api.github.com", auth, f)
    b = registry.get_client("github", "https://api.github.com", dict(auth), f)
    assert a is b and f.calls == 1


def test_force_new_bypasses_cache_and_replaces_entry(registry):
    f = CountingFactory()
    first = _get(registry, f)
    forced = _get(registry, f, force_new=True)
    assert forced is not first
    assert _get(registry, f) is forced


def test_release_client_finds_entry(registry):
    f = CountingFactory()
    _get(registry, f)
    assert registry.release_client("jira", URL, {"token": SecretStr("t")}) is True
    assert registry.release_client("jira", URL, {"token": SecretStr("other")}) is False
