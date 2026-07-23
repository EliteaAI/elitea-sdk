"""Tests for elitea_sdk._system_ca.enable_system_ca()."""
import sys
from unittest.mock import MagicMock

import pytest

from elitea_sdk import _system_ca


@pytest.fixture(autouse=True)
def reset_injection_guard():
    """Ensure each test starts with a clean injection state."""
    _system_ca._injected = False
    yield
    _system_ca._injected = False


@pytest.fixture
def fake_truststore(monkeypatch):
    """Provide a stub ``truststore`` module with a mocked inject_into_ssl."""
    module = MagicMock()
    monkeypatch.setitem(sys.modules, "truststore", module)
    return module


def test_injects_via_truststore(monkeypatch, fake_truststore):
    monkeypatch.delenv("ELITEA_DISABLE_SYSTEM_CA", raising=False)

    assert _system_ca.enable_system_ca() is True
    fake_truststore.inject_into_ssl.assert_called_once()
    assert _system_ca._injected is True


def test_is_idempotent(monkeypatch, fake_truststore):
    monkeypatch.delenv("ELITEA_DISABLE_SYSTEM_CA", raising=False)

    assert _system_ca.enable_system_ca() is True
    assert _system_ca.enable_system_ca() is True
    # Second call must not inject again.
    fake_truststore.inject_into_ssl.assert_called_once()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
def test_disabled_via_env(monkeypatch, fake_truststore, value):
    monkeypatch.setenv("ELITEA_DISABLE_SYSTEM_CA", value)

    assert _system_ca.enable_system_ca() is False
    fake_truststore.inject_into_ssl.assert_not_called()
    assert _system_ca._injected is False


def test_import_error_is_swallowed(monkeypatch):
    monkeypatch.delenv("ELITEA_DISABLE_SYSTEM_CA", raising=False)
    # Simulate truststore not being installed.
    monkeypatch.setitem(sys.modules, "truststore", None)

    # Must not raise and must not mark itself as injected.
    assert _system_ca.enable_system_ca() is False
    assert _system_ca._injected is False


def test_injection_exception_is_swallowed(monkeypatch, fake_truststore):
    monkeypatch.delenv("ELITEA_DISABLE_SYSTEM_CA", raising=False)
    fake_truststore.inject_into_ssl.side_effect = RuntimeError("boom")

    assert _system_ca.enable_system_ca() is False
    assert _system_ca._injected is False
