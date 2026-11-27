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
    assert requirements["pypdf"] == "==6.21.4"
    assert requirements["gitpython"] == "==3.2.1"
    assert requirements["dulwich"] == "==1.2.9"
    assert requirements["langgraph-checkpoint"] == "==4.3.0"
    assert requirements["langgraph-checkpoint-sqlite"] == "==3.1.2"
    assert requirements["langgraph-checkpoint-postgres"] == "==3.1.3"
    assert requirements["azure-core"] == "==1.38.0"
    assert requirements["lxml"] == "==6.1.0"
    assert requirements["svglib"] == "==2.3.1"
    assert requirements["reportlab"] == "==4.4.10"
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


def _nested_use_svg(levels: int, fan: int = 10) -> bytes:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'
        ' width="10" height="10"><defs><rect id="l0" width="1" height="1"/>'
    ]
    for i in range(1, levels + 1):
        parts.append(f'<g id="l{i}">' + f'<use xlink:href="#l{i - 1}"/>' * fan + "</g>")
    parts.append(f'</defs><use xlink:href="#l{levels}"/></svg>')
    return "".join(parts).encode()


def test_svglib_bounds_nested_use_expansion():
    # GHSA-555j-qj3j-c8jv: 1e6 <use> expansions would take minutes unpatched;
    # the patched svglib stops expanding after its fixed limit.
    import time
    from io import BytesIO

    from svglib.svglib import svg2rlg

    started = time.monotonic()
    svg2rlg(BytesIO(_nested_use_svg(levels=6)))

    assert time.monotonic() - started < 15


def test_loader_svg_pipeline_renders_clip_path_and_gradient_on_svglib_2():
    # svglib 2.x clip-path and gradient nodes are PDF-only and crash renderPM;
    # the loader pipeline must strip them and keep px sizing like svglib 1.x.
    from reportlab.graphics import renderPM
    from elitea_sdk.runtime.langchain.document_loaders.utils import (
        preprocess_svg_for_rendering,
        svg_to_px_drawing,
    )

    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
        b'<defs><clipPath id="c"><rect width="20" height="20"/></clipPath></defs>'
        b'<linearGradient id="g"><stop offset="0" stop-color="red"/>'
        b'<stop offset="1" stop-color="blue"/></linearGradient>'
        b'<g clip-path="url(#c)"><rect width="40" height="40" fill="currentColor"/></g>'
        b'<rect width="10" height="10" fill="url(#g)"/>'
        b"</svg>"
    )

    drawing = svg_to_px_drawing(preprocess_svg_for_rendering(svg))
    image = renderPM.drawToPIL(drawing)

    assert (drawing.width, drawing.height) == (40, 40)
    assert image.size == (40, 40)


def test_svg_to_px_drawing_rejects_unparseable_svg():
    from elitea_sdk.runtime.langchain.document_loaders.utils import svg_to_px_drawing

    with pytest.raises(ValueError):
        svg_to_px_drawing(b"")
