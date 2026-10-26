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

    assert requirements["unstructured"] == "==0.24.1"
    assert requirements["nh3"] == "==0.3.7"
    assert requirements["unstructured-inference"] == "==1.6.12"
    assert requirements["psutil"] == "==7.2.2"
    assert requirements["zstandard"] == "==0.25.0"
    assert requirements["xxhash"] == "==4.0.1"
    assert requirements["xlsxwriter"] == "==3.2.9"
    assert requirements["websocket-client"] == "==1.9.2"
    assert requirements["sse-starlette"] == "==3.0.3"
    assert requirements["unstructured-client"] == "==0.39.1"
    assert requirements["langchain-unstructured"] == "==1.0.1"
    assert requirements["statsmodels"] == "==0.14.5"
    assert requirements["pdfminer-six"] == "==20251230"
    assert requirements["pdf2image"] == "==1.17.0"
    assert requirements["pypdf"] == "==6.21.2"
    assert requirements["gitpython"] == "==3.2.1"
    assert requirements["dulwich"] == "==1.2.9"
    assert requirements["langgraph-checkpoint"] == "==4.3.0"
    assert requirements["langgraph-checkpoint-sqlite"] == "==3.1.2"
    assert requirements["langgraph-checkpoint-postgres"] == "==3.1.3"
    assert requirements["azure-core"] == "==1.38.0"
    assert requirements["lxml"] == "==6.1.0"
    assert "textract-py3" not in requirements


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


def test_unstructured_html_output_neutralizes_document_markup():
    """#7216: markup from an untrusted document must not survive into HTML output."""
    pytest.importorskip(
        "unstructured.documents.html_sanitization",
        reason="requires unstructured >= 0.24.1 (HTML output sanitization)",
    )
    from bs4 import BeautifulSoup
    from unstructured.documents.ontology import Hyperlink, Image, Paragraph

    rendered = "".join(
        element.to_html()
        for element in (
            Paragraph(
                text='hi<img src=x onerror="alert(1)">',
                additional_attributes={
                    "onmouseover": "alert(2)",
                    "title": '"><svg onload=alert(3)>',
                },
            ),
            Hyperlink(
                text="click",
                additional_attributes={"href": "javascript:alert(document.cookie)"},
            ),
            Image(text="", additional_attributes={"src": "x", "onerror": "alert(4)"}),
        )
    )

    # Escaped text such as ``&lt;img onerror=...&gt;`` is inert, so inspect the parsed
    # DOM for live elements and attributes rather than substring-matching the markup.
    soup = BeautifulSoup(rendered, "html.parser")
    assert not soup.find_all(["script", "svg"])
    assert [img.attrs for img in soup.find_all("img")] == [{"class": ["Image"], "src": "x"}]
    for tag in soup.find_all(True):
        assert not [name for name in tag.attrs if name.lower().startswith("on")], tag
        for name in ("href", "src"):
            assert not str(tag.get(name, "")).lower().startswith("javascript:"), tag
    assert "click" in soup.get_text()


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
