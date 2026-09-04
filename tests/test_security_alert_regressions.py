"""Regression contracts for dependency security alerts."""

from __future__ import annotations

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_critical_and_high_dependency_pins_stay_at_patched_versions():
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    optional_dependencies = project["project"]["optional-dependencies"]
    requirements = {
        canonicalize_name(Requirement(value).name): str(
            Requirement(value).specifier
        )
        for group in ("runtime", "tools")
        for value in optional_dependencies[group]
    }

    assert requirements["unstructured"] == "==0.24.0"
    assert requirements["unstructured-inference"] == "==1.6.12"
    assert requirements["psutil"] == "==7.2.2"
    assert requirements["langchain-unstructured"] == "==1.0.1"
    assert requirements["statsmodels"] == "==0.14.5"
    assert requirements["pdfminer-six"] == "==20251230"
    assert requirements["pdf2image"] == "==1.17.0"
    assert requirements["pypdf"] == "==6.16.1"
    assert requirements["gitpython"] == "==3.1.59"
    assert requirements["azure-core"] == "==1.38.0"
    assert requirements["lxml"] == "==6.1.0"


def test_unstructured_url_partitioning_rejects_loopback_addresses():
    """Keep URL-based document parsing from becoming an SSRF path again."""
    # The generic PR job overlays this checkout onto a fixed dependency image
    # without installing ``.[runtime]``. Keep the pin assertion above
    # unconditional, and exercise the runtime guard whenever the patched
    # Unstructured dependency is actually installed.
    safe_http = pytest.importorskip(
        "unstructured.safe_http",
        reason="requires the patched Unstructured runtime dependency",
    )
    from unstructured.partition.html import partition_html

    with pytest.raises(safe_http.UnsafeURLError, match="blocked IP address"):
        partition_html(url="http://127.0.0.1/private")


def test_unstructured_nlp_preload_uses_upstream_model_contract(monkeypatch):
    from elitea_sdk.runtime.langchain.tools.utils import (
        preload_unstructured_nlp_model,
    )

    inputs = []
    monkeypatch.setattr(
        "unstructured.nlp.tokenize.sent_tokenize",
        inputs.append,
    )

    preload_unstructured_nlp_model()

    assert inputs == ["Elitea NLP model preload."]
