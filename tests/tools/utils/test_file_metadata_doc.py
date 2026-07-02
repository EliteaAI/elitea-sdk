"""Tests for legacy Word (.doc) handling after PRE-11 (#5442).

Covers:
  * .doc is no longer in loaders_map — reading it via unstructured requires
    LibreOffice (soffice), which is not installed in the pylon image, so it was
    never readable. Users convert to .docx. Mirrors the .ppt removal in PRE-8.
  * get_file_metadata for a .doc still returns a schema-conformant generic base
    (no loader delegation, no crash) — unit is None because no loader claims it.
"""
from elitea_sdk.tools.utils.file_metadata import get_file_metadata


def test_doc_not_in_loaders_map():
    from elitea_sdk.runtime.langchain.document_loaders.constants import document_loaders_map
    assert ".doc" not in document_loaders_map, (
        ".doc requires LibreOffice (not installed); users should convert to .docx"
    )


def test_doc_metadata_generic_base_no_crash():
    # OLE2 magic — a real binary .doc starts with this signature.
    data = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1" + b"\x00" * 64

    meta = get_file_metadata("legacy.doc", file_content=data, file_size=len(data))

    assert meta["__result_status__"] == "file_metadata"
    # No loader claims .doc, so no unit is set (dropped by exclude_none).
    assert meta.get("unit") is None
    # Universal read_limits baseline is always present on a guidance response.
    assert "read_limits" in meta
    assert meta["read_limits"]["max_output_chars"] > 0
