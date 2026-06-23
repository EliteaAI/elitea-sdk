"""
Unit tests for FigmaApiWrapper.extract_design_tokens and its helper methods.

Tests cover:
- _extract_node_styles_recursive: recursive style walking
- _dedup_colors:      fill color deduplication (solid + gradient)
- _dedup_strokes:     stroke color deduplication
- _dedup_typography:  font combination deduplication
- _dedup_effects:     effect signature deduplication
- extract_design_tokens: end-to-end integration with mocked API
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.tools import ToolException

from elitea_sdk.tools.figma.api_wrapper import FigmaApiWrapper


# ---------------------------------------------------------------------------
# Fixtures & factories
# ---------------------------------------------------------------------------

def make_color(r: float, g: float, b: float, a: float = 1.0) -> dict:
    return {"r": r, "g": g, "b": b, "a": a}


def make_solid_fill(r, g, b, a=1.0, opacity=1.0) -> dict:
    return {"type": "SOLID", "color": make_color(r, g, b, a), "opacity": opacity}


def make_solid_stroke(r, g, b, a=1.0) -> dict:
    return {"type": "SOLID", "color": make_color(r, g, b, a)}


def make_drop_shadow(offset_y=4.0, radius=6.0, spread=0.0, color_a=0.1, visible=True) -> dict:
    return {
        "type": "DROP_SHADOW",
        "visible": visible,
        "color": make_color(0.0, 0.0, 0.0, color_a),
        "offset": {"x": 0.0, "y": offset_y},
        "radius": radius,
        "spread": spread,
        "blendMode": "NORMAL",
        "showShadowBehindNode": False,
    }


def make_text_style(family="Noto Sans", size=14.0, weight=400, line_height=20.0) -> dict:
    return {
        "fontFamily": family,
        "fontSize": size,
        "fontWeight": weight,
        "lineHeightPx": line_height,
        "letterSpacing": 0.0,
        "textAlignHorizontal": "LEFT",
    }


def make_node(
    name="Node",
    ntype="FRAME",
    fills=None,
    strokes=None,
    effects=None,
    style=None,
    styles=None,
    children=None,
) -> dict:
    node = {"name": name, "type": ntype}
    if fills is not None:
        node["fills"] = fills
    if strokes is not None:
        node["strokes"] = strokes
    if effects is not None:
        node["effects"] = effects
    if style is not None:
        node["style"] = style
    if styles is not None:
        node["styles"] = styles
    if children is not None:
        node["children"] = children
    return node


def make_wrapper() -> FigmaApiWrapper:
    """Create a FigmaApiWrapper instance with a mocked client (no real auth).

    Uses model_construct() to skip Pydantic validation and avoid real Figma
    authentication, then injects a MagicMock for the private _client attribute.
    """
    mock_client = MagicMock()
    wrapper = FigmaApiWrapper.model_construct(
        token=None,
        oauth2=None,
        global_limit=1_000_000,
        global_regexp=None,
        global_fields_retain=["id", "name", "type", "document", "children"],
        global_fields_remove=[],
        global_depth_start=1,
        global_depth_end=6,
        llm=None,
        apply_images_prompt=False,
        images_prompt={},
        apply_summary_prompt=False,
        summary_prompt={},
        number_of_threads=1,
    )
    # Inject the mock into the private Pydantic attribute slot
    object.__setattr__(wrapper, '_client', mock_client)
    return wrapper


# ===========================================================================
# _extract_node_styles_recursive
# ===========================================================================

class TestExtractNodeStylesRecursive:

    def test_empty_node_returns_empty(self):
        node = make_node()
        result = FigmaApiWrapper._extract_node_styles_recursive(node)
        assert result == []

    def test_solid_fill_captured(self):
        node = make_node(fills=[make_solid_fill(0.11, 0.46, 0.42)])
        result = FigmaApiWrapper._extract_node_styles_recursive(node)
        assert len(result) == 1
        assert result[0]["fills"][0]["type"] == "SOLID"

    def test_image_fill_excluded(self):
        node = make_node(fills=[{"type": "IMAGE", "imageRef": "abc"}])
        result = FigmaApiWrapper._extract_node_styles_recursive(node)
        assert result == []

    def test_invisible_alpha_zero_fill_excluded(self):
        # alpha=0 solid fill should be excluded
        node = make_node(fills=[make_solid_fill(1.0, 0.0, 0.0, a=0.0)])
        result = FigmaApiWrapper._extract_node_styles_recursive(node)
        assert result == []

    def test_invisible_effect_excluded(self):
        shadow = make_drop_shadow(visible=False)
        node = make_node(effects=[shadow])
        result = FigmaApiWrapper._extract_node_styles_recursive(node)
        assert result == []

    def test_visible_effect_captured(self):
        shadow = make_drop_shadow(visible=True)
        node = make_node(effects=[shadow])
        result = FigmaApiWrapper._extract_node_styles_recursive(node)
        assert len(result) == 1
        assert result[0]["effects"][0]["type"] == "DROP_SHADOW"

    def test_typography_captured(self):
        node = make_node(ntype="TEXT", style=make_text_style("Inter", 16.0, 700))
        result = FigmaApiWrapper._extract_node_styles_recursive(node)
        assert len(result) == 1
        assert result[0]["typography"]["fontFamily"] == "Inter"
        assert result[0]["typography"]["fontSize"] == 16.0
        assert result[0]["typography"]["fontWeight"] == 700

    def test_style_refs_captured(self):
        node = make_node(styles={"fill": "S:abc123", "text": "S:def456"})
        result = FigmaApiWrapper._extract_node_styles_recursive(node)
        assert len(result) == 1
        assert result[0]["style_refs"]["fill"] == "S:abc123"

    def test_recursive_children(self):
        child = make_node(name="Child", fills=[make_solid_fill(1.0, 1.0, 1.0)])
        parent = make_node(name="Parent", fills=[make_solid_fill(0.1, 0.1, 0.1)], children=[child])
        result = FigmaApiWrapper._extract_node_styles_recursive(parent)
        assert len(result) == 2
        paths = [r["path"] for r in result]
        assert "Parent" in paths
        assert "Parent/Child" in paths

    def test_deep_nesting_path_built_correctly(self):
        grandchild = make_node(name="GC", fills=[make_solid_fill(0.5, 0.5, 0.5)])
        child = make_node(name="C", children=[grandchild])
        parent = make_node(name="P", children=[child])
        result = FigmaApiWrapper._extract_node_styles_recursive(parent)
        assert result[0]["path"] == "P/C/GC"

    def test_node_with_no_style_no_entry(self):
        node = make_node(children=[
            make_node(name="Styled", fills=[make_solid_fill(0.2, 0.2, 0.2)]),
            make_node(name="Unstyled"),
        ])
        result = FigmaApiWrapper._extract_node_styles_recursive(node)
        paths = [r["path"] for r in result]
        assert any("Styled" in p for p in paths)
        assert not any("Unstyled" in p for p in paths)

    def test_stroke_captured(self):
        node = make_node(strokes=[make_solid_stroke(0.0, 0.0, 0.0)])
        result = FigmaApiWrapper._extract_node_styles_recursive(node)
        assert len(result) == 1
        assert result[0]["strokes"][0]["type"] == "SOLID"


# ===========================================================================
# _dedup_colors
# ===========================================================================

class TestDedupColors:

    def test_same_hex_deduped(self):
        """Same hex AND same alpha → one token."""
        entries = [
            {"path": "A", "fills": [make_solid_fill(1.0, 0.0, 0.0)], "strokes": [], "effects": [], "typography": {}, "style_refs": {}},
            {"path": "B", "fills": [make_solid_fill(1.0, 0.0, 0.0)], "strokes": [], "effects": [], "typography": {}, "style_refs": {}},
        ]
        colors = FigmaApiWrapper._dedup_colors(entries)
        assert len(colors) == 1
        assert colors[0]["hex"] == "#FF0000"

    def test_same_hex_different_alpha_kept_separately(self):
        """Same hex at different alpha values must produce two distinct tokens."""
        entries = [
            {"path": "A", "fills": [make_solid_fill(1.0, 0.0, 0.0, a=1.0)], "strokes": [], "effects": [], "typography": {}, "style_refs": {}},
            {"path": "B", "fills": [make_solid_fill(1.0, 0.0, 0.0, a=0.3)], "strokes": [], "effects": [], "typography": {}, "style_refs": {}},
        ]
        colors = FigmaApiWrapper._dedup_colors(entries)
        assert len(colors) == 2
        alphas = {c["alpha"] for c in colors}
        assert 1.0 in alphas
        assert 0.3 in alphas

    def test_different_hex_both_kept(self):
        entries = [
            {"path": "A", "fills": [make_solid_fill(1.0, 0.0, 0.0)], "strokes": [], "effects": [], "typography": {}, "style_refs": {}},
            {"path": "B", "fills": [make_solid_fill(0.0, 0.0, 1.0)], "strokes": [], "effects": [], "typography": {}, "style_refs": {}},
        ]
        colors = FigmaApiWrapper._dedup_colors(entries)
        hexes = {c["hex"] for c in colors}
        assert "#FF0000" in hexes
        assert "#0000FF" in hexes

    def test_alpha_zero_fill_excluded(self):
        entries = [
            {"path": "A", "fills": [make_solid_fill(1.0, 0.0, 0.0, a=0.0)], "strokes": [], "effects": [], "typography": {}, "style_refs": {}},
        ]
        colors = FigmaApiWrapper._dedup_colors(entries)
        assert colors == []

    def test_gradient_stops_extracted(self):
        gradient_fill = {
            "type": "GRADIENT_LINEAR",
            "gradientStops": [
                {"color": make_color(1.0, 0.0, 0.0)},
                {"color": make_color(0.0, 0.0, 1.0)},
            ],
        }
        entries = [{"path": "G", "fills": [gradient_fill], "strokes": [], "effects": [], "typography": {}, "style_refs": {}}]
        colors = FigmaApiWrapper._dedup_colors(entries)
        hexes = {c["hex"] for c in colors}
        assert "#FF0000" in hexes
        assert "#0000FF" in hexes

    def test_gradient_alpha_zero_stop_excluded(self):
        """Gradient stops with alpha=0 must be excluded."""
        gradient_fill = {
            "type": "GRADIENT_LINEAR",
            "gradientStops": [
                {"color": make_color(1.0, 0.0, 0.0, a=0.0)},  # transparent — skip
                {"color": make_color(0.0, 0.0, 1.0, a=1.0)},  # opaque — keep
            ],
        }
        entries = [{"path": "G", "fills": [gradient_fill], "strokes": [], "effects": [], "typography": {}, "style_refs": {}}]
        colors = FigmaApiWrapper._dedup_colors(entries)
        assert len(colors) == 1
        assert colors[0]["hex"] == "#0000FF"

    def test_opacity_preserved(self):
        entries = [
            {"path": "A", "fills": [make_solid_fill(0.28, 0.23, 0.25, opacity=0.3)], "strokes": [], "effects": [], "typography": {}, "style_refs": {}},
        ]
        colors = FigmaApiWrapper._dedup_colors(entries)
        assert colors[0]["opacity"] == 0.3

    def test_source_path_is_first_occurrence(self):
        entries = [
            {"path": "First", "fills": [make_solid_fill(0.0, 1.0, 0.0)], "strokes": [], "effects": [], "typography": {}, "style_refs": {}},
            {"path": "Second", "fills": [make_solid_fill(0.0, 1.0, 0.0)], "strokes": [], "effects": [], "typography": {}, "style_refs": {}},
        ]
        colors = FigmaApiWrapper._dedup_colors(entries)
        assert colors[0]["source_path"] == "First"

    def test_empty_entries(self):
        assert FigmaApiWrapper._dedup_colors([]) == []


# ===========================================================================
# _dedup_strokes
# ===========================================================================

class TestDedupStrokes:

    def test_same_stroke_hex_deduped(self):
        entries = [
            {"path": "A", "strokes": [make_solid_stroke(0.0, 0.0, 0.0)], "fills": [], "effects": [], "typography": {}, "style_refs": {}},
            {"path": "B", "strokes": [make_solid_stroke(0.0, 0.0, 0.0)], "fills": [], "effects": [], "typography": {}, "style_refs": {}},
        ]
        strokes = FigmaApiWrapper._dedup_strokes(entries)
        assert len(strokes) == 1
        assert strokes[0]["hex"] == "#000000"

    def test_different_stroke_hex_kept(self):
        entries = [
            {"path": "A", "strokes": [make_solid_stroke(0.0, 0.0, 0.0)], "fills": [], "effects": [], "typography": {}, "style_refs": {}},
            {"path": "B", "strokes": [make_solid_stroke(1.0, 1.0, 1.0)], "fills": [], "effects": [], "typography": {}, "style_refs": {}},
        ]
        strokes = FigmaApiWrapper._dedup_strokes(entries)
        assert len(strokes) == 2

    def test_same_hex_different_alpha_stroke_kept_separately(self):
        """Same hex at different alpha values must produce two distinct stroke tokens."""
        entries = [
            {"path": "A", "strokes": [make_solid_stroke(0.0, 0.0, 0.0, a=1.0)], "fills": [], "effects": [], "typography": {}, "style_refs": {}},
            {"path": "B", "strokes": [make_solid_stroke(0.0, 0.0, 0.0, a=0.5)], "fills": [], "effects": [], "typography": {}, "style_refs": {}},
        ]
        strokes = FigmaApiWrapper._dedup_strokes(entries)
        assert len(strokes) == 2
        alphas = {s["alpha"] for s in strokes}
        assert 1.0 in alphas
        assert 0.5 in alphas

    def test_gradient_stroke_stops_extracted(self):
        """Gradient strokes should have their stops extracted as color tokens."""
        gradient_stroke = {
            "type": "GRADIENT_LINEAR",
            "gradientStops": [
                {"color": make_color(1.0, 0.0, 0.0)},
                {"color": make_color(0.0, 1.0, 0.0)},
            ],
        }
        entries = [{"path": "A", "strokes": [gradient_stroke], "fills": [], "effects": [], "typography": {}, "style_refs": {}}]
        strokes = FigmaApiWrapper._dedup_strokes(entries)
        hexes = {s["hex"] for s in strokes}
        assert "#FF0000" in hexes
        assert "#00FF00" in hexes

    def test_alpha_zero_stroke_excluded(self):
        entries = [
            {"path": "A", "strokes": [make_solid_stroke(0.0, 0.0, 0.0, a=0.0)], "fills": [], "effects": [], "typography": {}, "style_refs": {}},
        ]
        strokes = FigmaApiWrapper._dedup_strokes(entries)
        assert strokes == []

    def test_empty_entries(self):
        assert FigmaApiWrapper._dedup_strokes([]) == []


# ===========================================================================
# _dedup_typography
# ===========================================================================

class TestDedupTypography:

    def test_same_combination_deduped(self):
        entries = [
            {"path": "A", "typography": make_text_style("Noto Sans", 14.0, 400), "fills": [], "strokes": [], "effects": [], "style_refs": {}},
            {"path": "B", "typography": make_text_style("Noto Sans", 14.0, 400), "fills": [], "strokes": [], "effects": [], "style_refs": {}},
        ]
        typo = FigmaApiWrapper._dedup_typography(entries)
        assert len(typo) == 1
        assert typo[0]["fontFamily"] == "Noto Sans"

    def test_different_size_kept_separately(self):
        entries = [
            {"path": "A", "typography": make_text_style("Noto Sans", 12.0, 400), "fills": [], "strokes": [], "effects": [], "style_refs": {}},
            {"path": "B", "typography": make_text_style("Noto Sans", 14.0, 400), "fills": [], "strokes": [], "effects": [], "style_refs": {}},
        ]
        typo = FigmaApiWrapper._dedup_typography(entries)
        assert len(typo) == 2

    def test_different_weight_kept_separately(self):
        entries = [
            {"path": "A", "typography": make_text_style("Inter", 14.0, 400), "fills": [], "strokes": [], "effects": [], "style_refs": {}},
            {"path": "B", "typography": make_text_style("Inter", 14.0, 700), "fills": [], "strokes": [], "effects": [], "style_refs": {}},
        ]
        typo = FigmaApiWrapper._dedup_typography(entries)
        assert len(typo) == 2

    def test_different_family_kept_separately(self):
        entries = [
            {"path": "A", "typography": make_text_style("Noto Sans", 14.0, 400), "fills": [], "strokes": [], "effects": [], "style_refs": {}},
            {"path": "B", "typography": make_text_style("Inter", 14.0, 400), "fills": [], "strokes": [], "effects": [], "style_refs": {}},
        ]
        typo = FigmaApiWrapper._dedup_typography(entries)
        assert len(typo) == 2

    def test_entry_without_font_family_skipped(self):
        entries = [
            {"path": "A", "typography": {}, "fills": [], "strokes": [], "effects": [], "style_refs": {}},
        ]
        typo = FigmaApiWrapper._dedup_typography(entries)
        assert typo == []

    def test_source_path_preserved(self):
        entries = [
            {"path": "MyPath", "typography": make_text_style("Noto Sans", 14.0, 400), "fills": [], "strokes": [], "effects": [], "style_refs": {}},
        ]
        typo = FigmaApiWrapper._dedup_typography(entries)
        assert typo[0]["source_path"] == "MyPath"


# ===========================================================================
# _dedup_effects
# ===========================================================================

class TestDedupEffects:

    def test_same_signature_deduped(self):
        shadow = make_drop_shadow(offset_y=4.0, radius=6.0, spread=0.0, color_a=0.1)
        entries = [
            {"path": "A", "effects": [shadow], "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
            {"path": "B", "effects": [shadow], "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
        ]
        effects = FigmaApiWrapper._dedup_effects(entries)
        assert len(effects) == 1

    def test_different_radius_kept_separately(self):
        entries = [
            {"path": "A", "effects": [make_drop_shadow(radius=6.0)], "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
            {"path": "B", "effects": [make_drop_shadow(radius=12.0)], "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
        ]
        effects = FigmaApiWrapper._dedup_effects(entries)
        assert len(effects) == 2

    def test_different_offset_y_kept_separately(self):
        entries = [
            {"path": "A", "effects": [make_drop_shadow(offset_y=4.0)], "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
            {"path": "B", "effects": [make_drop_shadow(offset_y=10.0)], "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
        ]
        effects = FigmaApiWrapper._dedup_effects(entries)
        assert len(effects) == 2

    def test_different_offset_x_kept_separately(self):
        """Two shadows with different offsetX must NOT be collapsed."""
        s1 = {**make_drop_shadow(), "offset": {"x": 0.0, "y": 4.0}}
        s2 = {**make_drop_shadow(), "offset": {"x": 4.0, "y": 4.0}}
        entries = [
            {"path": "A", "effects": [s1], "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
            {"path": "B", "effects": [s2], "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
        ]
        effects = FigmaApiWrapper._dedup_effects(entries)
        assert len(effects) == 2

    def test_different_color_rgb_kept_separately(self):
        """Two shadows identical except color RGB must NOT be collapsed."""
        black_shadow = make_drop_shadow(offset_y=4, radius=6, color_a=0.1)
        black_shadow["color"] = make_color(0.0, 0.0, 0.0, 0.1)
        red_shadow = make_drop_shadow(offset_y=4, radius=6, color_a=0.1)
        red_shadow["color"] = make_color(1.0, 0.0, 0.0, 0.1)
        entries = [
            {"path": "A", "effects": [black_shadow], "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
            {"path": "B", "effects": [red_shadow],   "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
        ]
        effects = FigmaApiWrapper._dedup_effects(entries)
        assert len(effects) == 2

    def test_multi_layer_shadows_kept(self):
        """Three layered shadows (tooltip pattern) should all be kept."""
        entries = [
            {"path": "Tooltip", "effects": [
                make_drop_shadow(offset_y=4.0,  radius=6.0,  color_a=0.05),
                make_drop_shadow(offset_y=10.0, radius=15.0, color_a=0.10),
                make_drop_shadow(offset_y=0.0,  radius=8.0,  color_a=0.10),
            ], "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
        ]
        effects = FigmaApiWrapper._dedup_effects(entries)
        assert len(effects) == 3

    def test_empty_entries(self):
        assert FigmaApiWrapper._dedup_effects([]) == []

    def test_source_path_preserved(self):
        entries = [
            {"path": "SomePath", "effects": [make_drop_shadow()], "fills": [], "strokes": [], "typography": {}, "style_refs": {}},
        ]
        effects = FigmaApiWrapper._dedup_effects(entries)
        assert effects[0]["source_path"] == "SomePath"


# ===========================================================================
# extract_design_tokens (end-to-end with mocked API)
# ===========================================================================

class TestExtractDesignTokens:

    def _make_api_response(self, node_id: str, node_doc: dict) -> dict:
        return {"nodes": {node_id: {"document": node_doc}}}

    def test_happy_path_returns_json(self):
        wrapper = make_wrapper()
        node_doc = make_node(
            name="Button",
            ntype="COMPONENT",
            fills=[make_solid_fill(0.11, 0.46, 0.42)],
            strokes=[make_solid_stroke(0.0, 0.0, 0.0)],
            effects=[make_drop_shadow()],
            style=make_text_style("Noto Sans", 14.0, 400),
        )
        wrapper._client.api_request.return_value = self._make_api_response("10:20", node_doc)

        raw = wrapper.extract_design_tokens("FILE", "10:20", depth=4)
        result = json.loads(raw)

        assert result["node_id"] == "10:20"
        assert result["node_name"] == "Button"
        assert result["node_type"] == "COMPONENT"
        assert len(result["colors"]) == 1
        assert len(result["strokes"]) == 1
        assert len(result["typography"]) == 1
        assert len(result["effects"]) == 1
        assert result["summary"]["unique_colors"] == 1

    def test_hyphen_node_id_normalised(self):
        wrapper = make_wrapper()
        node_doc = make_node(name="Test", fills=[make_solid_fill(1.0, 0.0, 0.0)])
        wrapper._client.api_request.return_value = self._make_api_response("10:20", node_doc)

        wrapper.extract_design_tokens("FILE", "10-20")  # hyphens

        call_url = wrapper._client.api_request.call_args[0][0]
        assert "10:20" in call_url
        assert "10-20" not in call_url

    def test_null_node_raises_tool_exception(self):
        wrapper = make_wrapper()
        wrapper._client.api_request.return_value = {"nodes": {"10:20": None}}

        with pytest.raises(ToolException, match="null"):
            wrapper.extract_design_tokens("FILE", "10:20")

    def test_api_error_raises_tool_exception(self):
        wrapper = make_wrapper()
        wrapper._client.api_request.side_effect = ToolException("Figma API error 403")

        with pytest.raises(ToolException, match="Failed to fetch node"):
            wrapper.extract_design_tokens("FILE", "10:20")

    def test_depth_passed_to_api(self):
        wrapper = make_wrapper()
        node_doc = make_node(name="N", fills=[make_solid_fill(1.0, 1.0, 1.0)])
        wrapper._client.api_request.return_value = self._make_api_response("1:1", node_doc)

        wrapper.extract_design_tokens("FILE", "1:1", depth=6)

        call_url = wrapper._client.api_request.call_args[0][0]
        assert "depth=6" in call_url

    def test_nested_children_colors_aggregated(self):
        child = make_node(name="Child", fills=[make_solid_fill(0.0, 0.0, 1.0)])
        parent = make_node(
            name="Parent",
            fills=[make_solid_fill(1.0, 0.0, 0.0)],
            children=[child],
        )
        wrapper = make_wrapper()
        wrapper._client.api_request.return_value = self._make_api_response("1:1", parent)

        result = json.loads(wrapper.extract_design_tokens("FILE", "1:1"))
        hexes = {c["hex"] for c in result["colors"]}
        assert "#FF0000" in hexes
        assert "#0000FF" in hexes

    def test_duplicate_colors_across_children_deduped(self):
        red_fill = make_solid_fill(1.0, 0.0, 0.0)
        child1 = make_node(name="C1", fills=[red_fill])
        child2 = make_node(name="C2", fills=[red_fill])
        parent = make_node(name="P", children=[child1, child2])

        wrapper = make_wrapper()
        wrapper._client.api_request.return_value = self._make_api_response("1:1", parent)

        result = json.loads(wrapper.extract_design_tokens("FILE", "1:1"))
        assert result["summary"]["unique_colors"] == 1

    def test_no_styles_returns_empty_tokens(self):
        node_doc = make_node(name="Empty")
        wrapper = make_wrapper()
        wrapper._client.api_request.return_value = self._make_api_response("1:1", node_doc)

        result = json.loads(wrapper.extract_design_tokens("FILE", "1:1"))
        assert result["colors"] == []
        assert result["strokes"] == []
        assert result["typography"] == []
        assert result["effects"] == []
        assert result["summary"]["total_style_entries"] == 0

    def test_summary_counts_correct(self):
        node_doc = make_node(
            name="Root",
            fills=[make_solid_fill(1.0, 0.0, 0.0), make_solid_fill(0.0, 1.0, 0.0)],
            strokes=[make_solid_stroke(0.0, 0.0, 0.0)],
            effects=[make_drop_shadow(offset_y=4.0), make_drop_shadow(offset_y=10.0)],
            style=make_text_style("Inter", 14.0, 700),
        )
        wrapper = make_wrapper()
        wrapper._client.api_request.return_value = self._make_api_response("1:1", node_doc)

        result = json.loads(wrapper.extract_design_tokens("FILE", "1:1"))
        s = result["summary"]
        assert s["unique_colors"] == 2
        assert s["unique_strokes"] == 1
        assert s["unique_effects"] == 2
        assert s["unique_fonts"] == 1

    def test_invisible_effects_excluded(self):
        node_doc = make_node(
            name="N",
            effects=[make_drop_shadow(visible=False)],
        )
        wrapper = make_wrapper()
        wrapper._client.api_request.return_value = self._make_api_response("1:1", node_doc)

        result = json.loads(wrapper.extract_design_tokens("FILE", "1:1"))
        assert result["effects"] == []

    def test_image_fills_excluded(self):
        node_doc = make_node(
            name="N",
            fills=[{"type": "IMAGE", "imageRef": "abc"}],
        )
        wrapper = make_wrapper()
        wrapper._client.api_request.return_value = self._make_api_response("1:1", node_doc)

        result = json.loads(wrapper.extract_design_tokens("FILE", "1:1"))
        assert result["colors"] == []

    def test_tool_registered_in_available_tools(self):
        wrapper = make_wrapper()
        tool_names = [t["name"] for t in wrapper.get_available_tools()]
        assert "extract_design_tokens" in tool_names

    def test_tool_schema_has_required_fields(self):
        wrapper = make_wrapper()
        tool = next(t for t in wrapper.get_available_tools() if t["name"] == "extract_design_tokens")
        schema = tool["args_schema"].schema()
        props = schema.get("properties", {})
        assert "file_key" in props
        assert "node_id" in props
        assert "depth" in props
        required = schema.get("required", [])
        assert "file_key" in required
        assert "node_id" in required
        # depth is optional (has default)
        assert "depth" not in required

    def test_raw_entries_present_in_result(self):
        """extract_design_tokens must include raw_entries in JSON output."""
        wrapper = make_wrapper()
        node_doc = make_node(name="N", fills=[make_solid_fill(1.0, 0.0, 0.0)])
        wrapper._client.api_request.return_value = self._make_api_response("1:1", node_doc)
        result = json.loads(wrapper.extract_design_tokens("FILE", "1:1"))
        assert "raw_entries" in result
        assert isinstance(result["raw_entries"], list)
        assert len(result["raw_entries"]) >= 1
        first = result["raw_entries"][0]
        assert "path" in first
        assert "fills" in first
        assert "strokes" in first
        assert "effects" in first
        assert "typography" in first
        assert "style_refs" in first

    def test_raw_entries_contain_all_styled_nodes(self):
        """raw_entries must have one entry per styled node, including children."""
        child = make_node(name="Child", fills=[make_solid_fill(0.0, 1.0, 0.0)])
        parent = make_node(name="Parent", fills=[make_solid_fill(1.0, 0.0, 0.0)], children=[child])
        wrapper = make_wrapper()
        wrapper._client.api_request.return_value = self._make_api_response("1:1", parent)
        result = json.loads(wrapper.extract_design_tokens("FILE", "1:1"))
        paths = [e["path"] for e in result["raw_entries"]]
        assert "Parent" in paths
        assert "Parent/Child" in paths


