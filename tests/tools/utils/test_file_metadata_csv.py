"""Tests for file_metadata registry with EliteACSVLoader (PRE-4 #5435).

Covers:
  * EliteACSVLoader (.csv) — total_lines + start_line/end_line
  * Output conforms to the PRE-1 chunked-read schema (#5432)
  * Exact total_lines matches line count (incl. header row)
  * No-content degrades gracefully to total_lines == 0
  * The CSV-specific header note is present
  * Line-slice round-trip: total_lines matches what apply_line_slice addresses,
    mirroring how read_file slices CSV text by physical line.
"""

import pytest

from elitea_sdk.runtime.langchain.document_loaders.EliteACSVLoader import EliteACSVLoader
from elitea_sdk.tools.utils.file_metadata import get_file_metadata
from elitea_sdk.tools.utils.text_operations import apply_line_slice


# Header row + 4 data rows = 5 lines total.
CSV_5_LINES = (
    b"id,name,score\n"
    b"1,Alice,90\n"
    b"2,Bob,85\n"
    b"3,Carol,77\n"
    b"4,Dave,68\n"
)


class TestCSVLoaderMetadata:

    def test_basic_contract(self):
        meta = get_file_metadata("data.csv", file_content=CSV_5_LINES,
                                 file_size=len(CSV_5_LINES))

        assert meta["__result_status__"] == "file_metadata"
        assert meta["extension"] == ".csv"
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 5
        assert meta["read_limits"]["max_output_chars"] == 200000

        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        assert "end_line" in instr["first_class_params"]
        assert "1-indexed" in instr["first_class_params"]["start_line"]
        assert "1..5" in instr["first_class_params"]["start_line"]

    def test_header_note_present(self):
        meta = get_file_metadata("data.csv", file_content=CSV_5_LINES,
                                 file_size=len(CSV_5_LINES))
        notes = meta["instruction_for_readFile"]["notes"]
        assert "header" in notes.lower()
        # Mentions keeping line 1 for column context
        assert "Line 1" in notes

    def test_no_content_degrades(self):
        meta = get_file_metadata("empty.csv", file_content=None, file_size=0)
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 0
        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        # No range hint when count is 0
        assert "1.." not in instr["first_class_params"]["start_line"]

    def test_header_only(self):
        content = b"id,name,score\n"
        meta = get_file_metadata("header_only.csv", file_content=content,
                                 file_size=len(content))
        assert meta["total_lines"] == 1
        assert "1..1" in meta["instruction_for_readFile"]["first_class_params"]["start_line"]

    def test_no_trailing_newline(self):
        content = b"id,name\n1,Alice\n2,Bob"  # last line has no \n
        meta = get_file_metadata("data.csv", file_content=content,
                                 file_size=len(content))
        assert meta["total_lines"] == 3

    def test_undecodable_bytes_degrade(self):
        meta = get_file_metadata("latin.csv", file_content=b"\xff\xfe",
                                 file_size=2)
        # Must not raise; line count is computed on bytes (newline-counting).
        assert meta["__result_status__"] == "file_metadata"
        assert isinstance(meta["total_lines"], int)

    def test_direct_classmethod(self):
        result = EliteACSVLoader.get_file_metadata(
            filename="data.csv", file_content=CSV_5_LINES
        )
        assert result["unit"] == "lines"
        assert result["total_lines"] == 5

    def test_line_slice_roundtrip(self):
        """Confirm total_lines matches what apply_line_slice can address.

        Mirrors read_file: CSV text is sliced by physical line. Reading line 1
        (header) plus a body range is the documented pattern for column context.
        """
        text = CSV_5_LINES.decode("utf-8")
        meta = EliteACSVLoader.get_file_metadata(
            filename="data.csv", file_content=CSV_5_LINES
        )
        total = meta["total_lines"]

        # Data rows 2..3 (Alice, Bob) => 2 lines.
        sliced = apply_line_slice(text, offset=2, limit=2)
        assert len(sliced.splitlines()) == 2
        assert "Alice" in sliced and "Bob" in sliced

        # Header is line 1.
        header = apply_line_slice(text, offset=1, limit=1)
        assert header.strip() == "id,name,score"

        # Last addressable line equals total_lines.
        last = apply_line_slice(text, offset=total, limit=1)
        assert last.strip() == "4,Dave,68"
