"""Tests for elitea_sdk._system_ca."""
import os
import sys
from unittest.mock import MagicMock

import pytest

from elitea_sdk import _system_ca

# Env vars the module may inject; cleaned between tests to keep cases independent.
_CA_ENV_VARS = (
    "DENO_TLS_CA_STORE",
    "DENO_CERT",
    "NODE_EXTRA_CA_CERTS",
    "NODE_OPTIONS",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
)


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset module-level guards/caches and CA env vars around each test."""
    _system_ca._injected = False
    _system_ca._child_env_cache = None
    _system_ca._node_gte_22_15_cache = None
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("ELITEA_DISABLE_SYSTEM_CA", raising=False)
    yield
    _system_ca._injected = False
    _system_ca._child_env_cache = None
    _system_ca._node_gte_22_15_cache = None


@pytest.fixture
def fake_truststore(monkeypatch):
    module = MagicMock()
    monkeypatch.setitem(sys.modules, "truststore", module)
    return module


# --------------------------------------------------------------------------- #
# In-process truststore injection
# --------------------------------------------------------------------------- #

def test_injects_via_truststore(fake_truststore):
    assert _system_ca.enable_system_ca() is True
    fake_truststore.inject_into_ssl.assert_called_once()
    assert _system_ca._injected is True


def test_is_idempotent(fake_truststore):
    assert _system_ca.enable_system_ca() is True
    assert _system_ca.enable_system_ca() is True
    fake_truststore.inject_into_ssl.assert_called_once()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
def test_disabled_via_env(monkeypatch, fake_truststore, value):
    monkeypatch.setenv("ELITEA_DISABLE_SYSTEM_CA", value)
    assert _system_ca.enable_system_ca() is False
    fake_truststore.inject_into_ssl.assert_not_called()
    assert _system_ca._injected is False


def test_import_error_is_swallowed(monkeypatch):
    monkeypatch.setitem(sys.modules, "truststore", None)
    assert _system_ca.enable_system_ca() is False
    assert _system_ca._injected is False


def test_injection_exception_is_swallowed(fake_truststore):
    fake_truststore.inject_into_ssl.side_effect = RuntimeError("boom")
    assert _system_ca.enable_system_ca() is False
    assert _system_ca._injected is False


# --------------------------------------------------------------------------- #
# get_child_process_ca_env — tiers
# --------------------------------------------------------------------------- #

def test_child_env_tier0_ssl_cert_file(monkeypatch, tmp_path):
    pem = tmp_path / "corp-ca.pem"
    pem.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setenv("SSL_CERT_FILE", str(pem))

    env = _system_ca.get_child_process_ca_env()
    assert env["DENO_CERT"] == str(pem)
    assert env["NODE_EXTRA_CA_CERTS"] == str(pem)
    # Tier 0 short-circuits before platform tiers.
    assert "DENO_TLS_CA_STORE" not in env


def test_child_env_tier0_nonexistent_file_falls_through(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/does/not/exist.pem")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(_system_ca, "_node_version_gte_22_15", lambda: False)

    env = _system_ca.get_child_process_ca_env()
    # Falls through to the platform tier rather than trusting a missing file.
    assert env.get("NODE_EXTRA_CA_CERTS") != "/does/not/exist.pem"
    assert env["DENO_TLS_CA_STORE"] == "system"


def test_child_env_macos_new_node(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(_system_ca, "_node_version_gte_22_15", lambda: True)

    env = _system_ca.get_child_process_ca_env()
    assert env["DENO_TLS_CA_STORE"] == "system"
    assert "--use-system-ca" in env["NODE_OPTIONS"].split()


def test_child_env_macos_old_node(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(_system_ca, "_node_version_gte_22_15", lambda: False)

    env = _system_ca.get_child_process_ca_env()
    assert env["DENO_TLS_CA_STORE"] == "system"
    assert "NODE_OPTIONS" not in env


def test_child_env_linux_system_pem(monkeypatch, tmp_path):
    pem = tmp_path / "ca-certificates.crt"
    pem.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_system_ca, "_SSL_CAFILE", str(pem))

    env = _system_ca.get_child_process_ca_env()
    assert env["DENO_CERT"] == str(pem)
    assert env["NODE_EXTRA_CA_CERTS"] == str(pem)
    assert env["DENO_TLS_CA_STORE"] == "system"


def test_child_env_linux_certifi_fallback(monkeypatch, tmp_path, caplog):
    certifi_pem = tmp_path / "cacert.pem"
    certifi_pem.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_system_ca, "_SSL_CAFILE", None)
    monkeypatch.setattr(_system_ca, "_LINUX_CA_CANDIDATES", ())

    fake_certifi = MagicMock()
    fake_certifi.where.return_value = str(certifi_pem)
    monkeypatch.setitem(sys.modules, "certifi", fake_certifi)

    with caplog.at_level("WARNING"):
        env = _system_ca.get_child_process_ca_env()
    assert env["DENO_CERT"] == str(certifi_pem)
    assert any("certifi" in r.message for r in caplog.records)


def test_child_env_disabled(monkeypatch):
    monkeypatch.setenv("ELITEA_DISABLE_SYSTEM_CA", "1")
    assert _system_ca.get_child_process_ca_env() == {}


def test_child_env_is_cached(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    calls = []
    monkeypatch.setattr(
        _system_ca, "_node_version_gte_22_15", lambda: calls.append(1) or False
    )
    first = _system_ca.get_child_process_ca_env()
    second = _system_ca.get_child_process_ca_env()
    assert first is second
    assert len(calls) == 1  # not recomputed


# --------------------------------------------------------------------------- #
# _merge_node_options
# --------------------------------------------------------------------------- #

def test_merge_node_options_appends(monkeypatch):
    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=512")
    merged = _system_ca._merge_node_options("--use-system-ca")
    assert "--max-old-space-size=512" in merged.split()
    assert "--use-system-ca" in merged.split()


def test_merge_node_options_no_duplicate(monkeypatch):
    monkeypatch.setenv("NODE_OPTIONS", "--use-system-ca")
    merged = _system_ca._merge_node_options("--use-system-ca")
    assert merged.split().count("--use-system-ca") == 1


def test_merge_node_options_substring_safe(monkeypatch):
    monkeypatch.setenv("NODE_OPTIONS", "--no-use-system-ca")
    merged = _system_ca._merge_node_options("--use-system-ca")
    # Token-level check: the lookalike must not suppress the real flag.
    assert "--use-system-ca" in merged.split()


def test_merge_node_options_empty(monkeypatch):
    monkeypatch.delenv("NODE_OPTIONS", raising=False)
    assert _system_ca._merge_node_options("--use-system-ca") == "--use-system-ca"


# --------------------------------------------------------------------------- #
# os.environ injection
# --------------------------------------------------------------------------- #

def test_inject_uses_setdefault(monkeypatch):
    monkeypatch.setattr(
        _system_ca, "get_child_process_ca_env", lambda: {"DENO_TLS_CA_STORE": "system"}
    )
    _system_ca._inject_child_process_ca_env()
    assert os.environ.get("DENO_TLS_CA_STORE") == "system"


def test_inject_does_not_overwrite(monkeypatch):
    monkeypatch.setenv("DENO_TLS_CA_STORE", "user-value")
    monkeypatch.setattr(
        _system_ca, "get_child_process_ca_env", lambda: {"DENO_TLS_CA_STORE": "system"}
    )
    _system_ca._inject_child_process_ca_env()
    assert os.environ["DENO_TLS_CA_STORE"] == "user-value"


def test_enable_injects_even_when_truststore_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "truststore", None)  # ImportError path
    monkeypatch.setattr(
        _system_ca, "get_child_process_ca_env", lambda: {"DENO_TLS_CA_STORE": "system"}
    )
    assert _system_ca.enable_system_ca() is False  # in-process injection failed
    assert os.environ.get("DENO_TLS_CA_STORE") == "system"  # child env still set


# --------------------------------------------------------------------------- #
# _node_version_gte_22_15
# --------------------------------------------------------------------------- #

def test_node_probe_caches(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(1)
        return MagicMock(stdout="22.15.0")

    monkeypatch.setattr(_system_ca.subprocess, "run", fake_run)
    assert _system_ca._node_version_gte_22_15() is True
    assert _system_ca._node_version_gte_22_15() is True
    assert len(calls) == 1


def test_node_probe_false_on_error(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("node not found")

    monkeypatch.setattr(_system_ca.subprocess, "run", boom)
    assert _system_ca._node_version_gte_22_15() is False


@pytest.mark.parametrize(
    "version,expected",
    [("22.15.0", True), ("22.14.0", False), ("20.11.1", False), ("24.0.0", True)],
)
def test_node_probe_version_boundaries(monkeypatch, version, expected):
    monkeypatch.setattr(
        _system_ca.subprocess, "run", lambda *a, **k: MagicMock(stdout=version)
    )
    assert _system_ca._node_version_gte_22_15() is expected
