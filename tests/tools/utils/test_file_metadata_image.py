"""Tests for file_metadata registry with image files (PRE-12 #5443).

Covers:
  * Raster images report image_width/image_height + bytes
  * Output conforms to the PRE-1 chunked-read schema (#5432)
  * unit is None (images have no chunk unit) and notes say multimodal/no chunking
  * No first_class_params (no line/page/row selectors for images)
  * Oversized image trips the byte-size guard (full_read_allowed=False)
  * Corrupt bytes degrade gracefully (no dimensions, still conformant)
  * SVG dimensions are reported
"""
import io

import pytest
from PIL import Image

from elitea_sdk.runtime.langchain.document_loaders.EliteAImageLoader import (
    MAX_IMAGE_READ_BYTES,
)
from elitea_sdk.tools.utils.file_metadata import get_file_metadata


def _make_png_bytes(width: int = 120, height: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buf, "PNG")
    return buf.getvalue()


_SVG = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="48">'
    b'<rect width="64" height="48" fill="blue"/></svg>'
)


def test_png_metadata_reports_dimensions_and_schema():
    data = _make_png_bytes(120, 80)

    meta = get_file_metadata("shot.png", file_content=data, file_size=len(data))

    assert meta["__result_status__"] == "file_metadata"
    assert meta["filename"] == "shot.png"
    assert meta["extension"] == ".png"
    assert meta["filesize"] == len(data)
    assert meta["read_limits"]["max_output_chars"] == 200000
    assert meta["read_limits"]["full_read_allowed"] is True

    assert meta.get("unit") is None
    assert meta["image_width"] == 120
    assert meta["image_height"] == 80


def test_png_metadata_notes_multimodal_no_chunking():
    data = _make_png_bytes()

    meta = get_file_metadata("shot.png", file_content=data, file_size=len(data))

    instr = meta["instruction_for_readFile"]
    # Images have no line/page/row selectors.
    assert instr["first_class_params"] == {}
    notes = instr["notes"].lower()
    assert "multimodal" in notes
    assert "no chunking" in notes


def test_jpeg_metadata_reports_dimensions():
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), "green").save(buf, "JPEG")
    data = buf.getvalue()

    meta = get_file_metadata("pic.jpeg", file_content=data, file_size=len(data))

    assert meta["image_width"] == 200
    assert meta["image_height"] == 100
    assert meta.get("unit") is None


def test_oversized_image_trips_byte_guard():
    # file_size claims over the limit; guard is size-driven, not pixel-driven.
    data = _make_png_bytes()

    meta = get_file_metadata("huge.png", file_content=data,
                             file_size=MAX_IMAGE_READ_BYTES + 1)

    assert meta["__result_status__"] == "file_metadata"
    assert meta["read_limits"]["full_read_allowed"] is False
    assert "refused" in meta["instruction_for_readFile"]["notes"].lower()


def test_image_under_limit_allows_full_read():
    data = _make_png_bytes()

    meta = get_file_metadata("ok.png", file_content=data,
                             file_size=MAX_IMAGE_READ_BYTES - 1)

    assert meta["read_limits"]["full_read_allowed"] is True


def test_corrupt_image_bytes_degrade_gracefully():
    meta = get_file_metadata("broken.png", file_content=b"not a real png",
                             file_size=14)

    assert meta["__result_status__"] == "file_metadata"
    assert meta.get("unit") is None
    # No dimensions when the image can't be parsed.
    assert "image_width" not in meta
    assert "image_height" not in meta


def test_image_metadata_no_content():
    meta = get_file_metadata("shot.png", file_content=None, file_size=4096)

    assert meta.get("unit") is None
    assert meta["filesize"] == 4096
    assert "image_width" not in meta


def test_svg_metadata_reports_dimensions():
    meta = get_file_metadata("icon.svg", file_content=_SVG, file_size=len(_SVG))

    assert meta["extension"] == ".svg"
    assert meta.get("unit") is None
    assert meta["image_width"] == 64
    assert meta["image_height"] == 48


@pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"])
def test_all_image_extensions_registered(ext):
    from elitea_sdk.runtime.langchain.document_loaders.constants import loaders_map
    from elitea_sdk.runtime.langchain.document_loaders.EliteAImageLoader import (
        EliteAImageLoader,
    )
    entry = loaders_map.get(ext)
    assert entry is not None
    assert entry["class"] is EliteAImageLoader
    assert hasattr(EliteAImageLoader, "get_file_metadata")
