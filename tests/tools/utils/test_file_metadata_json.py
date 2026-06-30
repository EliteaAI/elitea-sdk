"""Tests for file_metadata registry with EliteAJSONLoader (PRE-5 #5436).

Covers:
  * EliteAJSONLoader (.json) — total_lines + start_line/end_line, JSON read as text
  * Output conforms to the PRE-1 chunked-read schema (#5432)
  * The JSON-specific "read as text / slice not valid JSON" note is present
  * No-content degrades gracefully to total_lines == 0
  * Single-line honesty (#5436): a large minified (single-line) JSON has no
    usable line breaks, so start_line/end_line are NOT advertised and the full
    read is refused; a *small* single-line JSON still reads fully and fine.
  * Line-slice round-trip on pretty-printed JSON mirrors read_file behaviour.
"""

import json

from elitea_sdk.runtime.langchain.document_loaders.EliteAJSONLoader import EliteAJSONLoader
from elitea_sdk.tools.utils.file_metadata import (
    DEFAULT_MAX_OUTPUT_CHARS,
    get_file_metadata,
)
from elitea_sdk.tools.utils.text_operations import apply_line_slice


# Pretty-printed object => multiple physical lines.
JSON_PRETTY = json.dumps(
    {"id": 1, "name": "Alice", "tags": ["a", "b", "c"]}, indent=2
).encode("utf-8")
JSON_PRETTY_LINES = JSON_PRETTY.decode("utf-8").count("\n") + 1


class TestJSONLoaderMetadata:

    def test_basic_contract(self):
        meta = get_file_metadata("data.json", file_content=JSON_PRETTY,
                                 file_size=len(JSON_PRETTY))

        assert meta["__result_status__"] == "file_metadata"
        assert meta["extension"] == ".json"
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == JSON_PRETTY_LINES
        assert meta["read_limits"]["max_output_chars"] == DEFAULT_MAX_OUTPUT_CHARS

        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        assert "end_line" in instr["first_class_params"]
        assert "1-indexed" in instr["first_class_params"]["start_line"]
        assert f"1..{JSON_PRETTY_LINES}" in instr["first_class_params"]["start_line"]

    def test_json_note_present(self):
        meta = get_file_metadata("data.json", file_content=JSON_PRETTY,
                                 file_size=len(JSON_PRETTY))
        notes = meta["instruction_for_readFile"]["notes"].lower()
        assert "read as text" in notes
        assert "valid json" in notes

    def test_no_content_degrades(self):
        meta = get_file_metadata("empty.json", file_content=None, file_size=0)
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 0
        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        assert "1.." not in instr["first_class_params"]["start_line"]

    def test_direct_classmethod(self):
        result = EliteAJSONLoader.get_file_metadata(
            filename="data.json", file_content=JSON_PRETTY
        )
        assert result["unit"] == "lines"
        assert result["total_lines"] == JSON_PRETTY_LINES

    def test_small_single_line_still_readable(self):
        """A minified JSON under the cap is one line but reads fine — params kept."""
        content = b'{"id":1,"name":"Alice","tags":["a","b","c"]}'
        meta = get_file_metadata("small.json", file_content=content,
                                 file_size=len(content))
        assert meta["total_lines"] == 1
        instr = meta["instruction_for_readFile"]
        # start_line/end_line still advertised — full read is fine for a small file.
        assert "start_line" in instr["first_class_params"]
        # full_read_allowed remains the universal baseline (True).
        assert meta["read_limits"]["full_read_allowed"] is True

    def test_large_single_line_refused(self):
        """A large minified JSON has no usable line breaks => refuse, don't lie."""
        # One physical line, well over the output cap.
        big = b'{"data":"' + b"x" * (DEFAULT_MAX_OUTPUT_CHARS + 1000) + b'"}'
        meta = get_file_metadata("big.json", file_content=big,
                                 file_size=len(big))
        assert meta["__result_status__"] == "file_metadata"
        assert meta["total_lines"] == 1
        assert meta["read_limits"]["full_read_allowed"] is False
        instr = meta["instruction_for_readFile"]
        # start_line/end_line must NOT be advertised — they would do nothing.
        assert "start_line" not in instr["first_class_params"]
        assert "end_line" not in instr["first_class_params"]
        notes = instr["notes"].lower()
        assert "no usable line breaks" in notes
        assert "refused" in notes

    def test_line_slice_roundtrip(self):
        """total_lines matches what apply_line_slice can address on pretty JSON."""
        text = JSON_PRETTY.decode("utf-8")
        meta = EliteAJSONLoader.get_file_metadata(
            filename="data.json", file_content=JSON_PRETTY
        )
        total = meta["total_lines"]

        # First line of a pretty-printed object is the opening brace.
        first = apply_line_slice(text, offset=1, limit=1)
        assert first.strip() == "{"

        # Last addressable line equals total_lines (closing brace).
        last = apply_line_slice(text, offset=total, limit=1)
        assert last.strip() == "}"
