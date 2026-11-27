"""SVG preprocessing for browser-exported diagrams (Mermaid "save diagram" from ADO wiki)."""

import pytest

from elitea_sdk.runtime.langchain.document_loaders.utils import preprocess_svg_for_rendering

# Trimmed shape of a real ADO wiki attachment: HTML wrapper with a CSS ``width: auto``,
# Mermaid dash arrays, and node labels as HTML inside <foreignObject>.
MERMAID_EXPORT = (
    '<div xmlns="http://www.w3.org/1999/xhtml" style="width: auto; height: auto; display: flex;">'
    '<svg id="g" width="100%" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 120">'
    '<style>#g .edge-pattern-solid{stroke-dasharray:0;}#g .dashed{stroke-dasharray:3;}</style>'
    '<path d="M10 10L190 110" stroke="black" style="stroke-dasharray: 0, 0;"/>'
    '<path d="M10 110L190 10" stroke="black" style="stroke-dasharray: 1, 0;"/>'
    '<g transform="translate(100, 60)"><rect x="-40" y="-12" width="80" height="24" fill="#333"/>'
    '<g class="label" transform="translate(-30, -12)">'
    '<foreignObject width="60" height="24"><div xmlns="http://www.w3.org/1999/xhtml">'
    '<span class="nodeLabel"><p>Do &amp; Check</p></span></div></foreignObject></g></g>'
    '<g class="edgeLabel"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml">'
    '<span class="edgeLabel"></span></div></foreignObject></g>'
    '</svg></div>'
).encode()


def _render(svg: bytes):
    pytest.importorskip("svglib")
    from reportlab.graphics import renderPM
    from elitea_sdk.runtime.langchain.document_loaders.utils import svg_to_px_drawing

    drawing = svg_to_px_drawing(preprocess_svg_for_rendering(svg))
    return drawing, renderPM.drawToPIL(drawing)


def test_html_wrapper_is_dropped_down_to_the_svg_element():
    out = preprocess_svg_for_rendering(MERMAID_EXPORT).decode()

    assert out.startswith("<svg ")
    assert out.endswith("</svg>")
    assert "<div" not in out.split("<style>")[0]


def test_foreign_object_labels_become_centered_text():
    out = preprocess_svg_for_rendering(MERMAID_EXPORT).decode()

    assert "foreignObject" not in out
    assert out.count("<text ") == 1
    assert '<text x="30" ' in out
    assert ">Do &amp; Check</text>" in out


def test_all_zero_dasharray_becomes_none_and_others_are_kept():
    out = preprocess_svg_for_rendering(MERMAID_EXPORT).decode()

    assert "stroke-dasharray:none;" in out
    assert "stroke-dasharray:none;\"" in out
    assert "stroke-dasharray:3;" in out
    assert "stroke-dasharray: 1, 0;" in out


def test_zero_dasharray_attribute_form():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1" stroke-dasharray="0 0"/></svg>'

    assert b'stroke-dasharray="none"' in preprocess_svg_for_rendering(svg)


@pytest.mark.parametrize("prolog", ["", '<?xml version="1.0"?>\n', "<!-- exported -->\n<!DOCTYPE svg>\n"])
def test_plain_svg_with_prolog_is_left_alone(prolog):
    svg = f'{prolog}<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>'

    assert preprocess_svg_for_rendering(svg.encode()).decode() == svg


def test_mermaid_export_renders_with_its_viewbox_size():
    drawing, image = _render(MERMAID_EXPORT)

    assert (round(drawing.width), round(drawing.height)) == (200, 120)
    assert image.size == (200, 120)
