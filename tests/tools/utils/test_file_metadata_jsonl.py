"""Tests for file_metadata registry with EliteAJSONLinesLoader (PRE-6 #5437).

Covers:
  * EliteAJSONLinesLoader (.jsonl) overrides the inherited JSON get_file_metadata
  * total_lines + start_line/end_line advertised; output conforms to PRE-1 schema
  * JSONL-specific note: record-aligned / valid JSONL (NOT the "not valid JSON" JSON note)
  * No-content degrades gracefully to total_lines == 0
  * Single-line honesty: large single-line JSONL (no line breaks) refuses full read
  * apply_line_slice round-trip: each sliced line is a parseable JSON record
  * loaders_map routes .jsonl to EliteAJSONLinesLoader, not EliteAJSONLoader
"""

import json

from elitea_sdk.runtime.langchain.document_loaders.EliteAJSONLinesLoader import (
    EliteAJSONLinesLoader,
)
from elitea_sdk.runtime.langchain.document_loaders.EliteAJSONLoader import EliteAJSONLoader
from elitea_sdk.tools.utils.file_metadata import (
    DEFAULT_MAX_OUTPUT_CHARS,
    get_file_metadata,
)
from elitea_sdk.tools.utils.text_operations import apply_line_slice


# Multi-line JSONL: 3 records, one per line.
JSONL_RECORDS = [
    {"id": 1, "name": "Alice", "score": 9.5},
    {"id": 2, "name": "Bob", "score": 7.0},
    {"id": 3, "name": "Carol", "score": 8.2},
]
JSONL_CONTENT = "\n".join(json.dumps(r) for r in JSONL_RECORDS) + "\n"
JSONL_BYTES = JSONL_CONTENT.encode("utf-8")
JSONL_LINES = JSONL_CONTENT.count("\n")  # trailing newline: lines == records here


class TestJSONLLoaderMetadata:

    def test_basic_contract(self):
        meta = get_file_metadata("data.jsonl", file_content=JSONL_BYTES,
                                 file_size=len(JSONL_BYTES))

        assert meta["__result_status__"] == "file_metadata"
        assert meta["extension"] == ".jsonl"
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == JSONL_LINES
        assert meta["read_limits"]["max_output_chars"] == DEFAULT_MAX_OUTPUT_CHARS
        assert meta["read_limits"]["full_read_allowed"] is True

        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        assert "end_line" in instr["first_class_params"]
        assert "1-indexed" in instr["first_class_params"]["start_line"]
        assert f"1..{JSONL_LINES}" in instr["first_class_params"]["start_line"]

    def test_jsonl_note_record_aligned(self):
        """JSONL note must mention record-alignment / valid JSONL."""
        meta = get_file_metadata("data.jsonl", file_content=JSONL_BYTES,
                                 file_size=len(JSONL_BYTES))
        notes = meta["instruction_for_readFile"]["notes"].lower()
        assert "record" in notes
        assert "valid jsonl" in notes

    def test_json_note_absent(self):
        """The inherited JSON note ('not valid json') must NOT appear for JSONL."""
        meta = get_file_metadata("data.jsonl", file_content=JSONL_BYTES,
                                 file_size=len(JSONL_BYTES))
        notes = meta["instruction_for_readFile"]["notes"].lower()
        assert "not necessarily valid json" not in notes

    def test_no_content_degrades(self):
        meta = get_file_metadata("empty.jsonl", file_content=None, file_size=0)
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 0
        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        assert "1.." not in instr["first_class_params"]["start_line"]

    def test_direct_classmethod(self):
        result = EliteAJSONLinesLoader.get_file_metadata(
            filename="data.jsonl", file_content=JSONL_BYTES
        )
        assert result["unit"] == "lines"
        assert result["total_lines"] == JSONL_LINES

    def test_override_not_inherited_json_method(self):
        """EliteAJSONLinesLoader.get_file_metadata must be its own method, not JSON's."""
        assert EliteAJSONLinesLoader.get_file_metadata is not EliteAJSONLoader.get_file_metadata

    def test_small_single_line_still_readable(self):
        """A single-record JSONL under the cap is fine — full read allowed."""
        content = b'{"id":1,"name":"Alice"}'
        meta = get_file_metadata("small.jsonl", file_content=content,
                                 file_size=len(content))
        assert meta["total_lines"] == 1
        assert meta["read_limits"]["full_read_allowed"] is True
        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]

    def test_large_single_line_refused(self):
        """A huge single-line JSONL (no newlines) refuses full read."""
        big = b'{"data":"' + b"x" * (DEFAULT_MAX_OUTPUT_CHARS + 1000) + b'"}'
        meta = get_file_metadata("big.jsonl", file_content=big,
                                 file_size=len(big))
        assert meta["__result_status__"] == "file_metadata"
        assert meta["total_lines"] == 1
        assert meta["read_limits"]["full_read_allowed"] is False
        instr = meta["instruction_for_readFile"]
        assert "start_line" not in instr["first_class_params"]
        assert "end_line" not in instr["first_class_params"]
        notes = instr["notes"].lower()
        assert "no usable line breaks" in notes
        assert "refused" in notes

    def test_line_slice_roundtrip(self):
        """Each line of JSONL is a parseable JSON record after apply_line_slice."""
        text = JSONL_BYTES.decode("utf-8")
        for i, expected in enumerate(JSONL_RECORDS, start=1):
            line = apply_line_slice(text, offset=i, limit=1).strip()
            assert json.loads(line) == expected
