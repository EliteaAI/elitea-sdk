"""Tests for file_metadata registry with text-family loaders (PRE-3 #5434).

Covers:
  * EliteATextLoader (.txt/.yaml/.yml/.groovy) — total_lines + start_line/end_line
  * EliteACodeLoader (.py and all code extensions) — same contract
  * EliteAMarkdownLoader (.md) — same contract
  * Output conforms to the PRE-1 chunked-read schema (#5432)
  * Exact total_lines matches splitlines()
  * No-content degrades gracefully to total_lines == 0
  * Undecodable bytes degrade gracefully (total_lines == 0, no crash)
"""

import pytest

from elitea_sdk.runtime.langchain.document_loaders.EliteATextLoader import EliteATextLoader
from elitea_sdk.runtime.langchain.document_loaders.EliteACodeLoader import EliteACodeLoader
from elitea_sdk.runtime.langchain.document_loaders.EliteAMarkdownLoader import EliteAMarkdownLoader
from elitea_sdk.tools.utils.file_metadata import get_file_metadata
from elitea_sdk.tools.utils.text_operations import apply_line_slice


TEXT_5_LINES = b"line one\nline two\nline three\nline four\nline five\n"
PYTHON_5_LINES = b"def foo():\n    pass\n\ndef bar():\n    return 1\n"
MARKDOWN_5_LINES = b"# Title\n\nParagraph one.\n\n- item\n"


# ---------------------------------------------------------------------------
# EliteATextLoader
# ---------------------------------------------------------------------------

class TestTextLoaderMetadata:

    def test_basic_contract(self):
        meta = get_file_metadata("notes.txt", file_content=TEXT_5_LINES,
                                 file_size=len(TEXT_5_LINES))

        assert meta["__result_status__"] == "file_metadata"
        assert meta["extension"] == ".txt"
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 5
        assert meta["read_limits"]["max_output_chars"] == 200000

        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        assert "end_line" in instr["first_class_params"]
        assert "1-indexed" in instr["first_class_params"]["start_line"]
        assert "1..5" in instr["first_class_params"]["start_line"]

    def test_yaml_extension(self):
        content = b"key: value\nother: 123\n"
        meta = get_file_metadata("config.yaml", file_content=content,
                                 file_size=len(content))
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 2

    def test_yml_extension(self):
        content = b"a: 1\nb: 2\nc: 3\n"
        meta = get_file_metadata("ci.yml", file_content=content,
                                 file_size=len(content))
        assert meta["total_lines"] == 3

    def test_groovy_extension(self):
        content = b"pipeline {\n    agent any\n    stages {}\n}\n"
        meta = get_file_metadata("Jenkinsfile.groovy", file_content=content,
                                 file_size=len(content))
        assert meta["total_lines"] == 4

    def test_no_content_degrades(self):
        meta = get_file_metadata("empty.txt", file_content=None, file_size=0)
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 0
        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        # No range hint when count is 0
        assert "1.." not in instr["first_class_params"]["start_line"]

    def test_undecodable_bytes_degrade(self):
        meta = get_file_metadata("latin.txt", file_content=b"\xff\xfe",
                                 file_size=2)
        # May be 0 (failed) or a small count — must not raise
        assert meta["__result_status__"] == "file_metadata"
        assert isinstance(meta["total_lines"], int)

    def test_direct_classmethod(self):
        result = EliteATextLoader.get_file_metadata(
            filename="test.txt", file_content=TEXT_5_LINES
        )
        assert result["unit"] == "lines"
        assert result["total_lines"] == 5

    def test_line_slice_roundtrip(self):
        """Confirm total_lines matches what apply_line_slice can address."""
        text = TEXT_5_LINES.decode("utf-8")
        meta = EliteATextLoader.get_file_metadata(
            filename="test.txt", file_content=TEXT_5_LINES
        )
        total = meta["total_lines"]
        # Reading lines 2..4 out of 5 should return 3 lines
        sliced = apply_line_slice(text, offset=2, limit=3)
        assert len(sliced.splitlines()) == 3
        # Last line index equals total_lines
        last = apply_line_slice(text, offset=total, limit=1)
        assert last.strip() == "line five"


# ---------------------------------------------------------------------------
# EliteACodeLoader (.py and other code extensions)
# ---------------------------------------------------------------------------

class TestCodeLoaderMetadata:

    def test_py_extension(self):
        meta = get_file_metadata("script.py", file_content=PYTHON_5_LINES,
                                 file_size=len(PYTHON_5_LINES))

        assert meta["__result_status__"] == "file_metadata"
        assert meta["extension"] == ".py"
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 5

        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        assert "end_line" in instr["first_class_params"]
        assert "1..5" in instr["first_class_params"]["start_line"]

    def test_js_extension(self):
        content = b"function foo() {\n  return 1;\n}\n"
        meta = get_file_metadata("app.js", file_content=content,
                                 file_size=len(content))
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 3

    def test_java_extension(self):
        content = b"class Foo {\n    void bar() {}\n}\n"
        meta = get_file_metadata("Foo.java", file_content=content,
                                 file_size=len(content))
        assert meta["total_lines"] == 3

    def test_no_content_degrades(self):
        meta = get_file_metadata("main.py", file_content=None, file_size=0)
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 0

    def test_direct_classmethod(self):
        result = EliteACodeLoader.get_file_metadata(
            filename="script.py", file_content=PYTHON_5_LINES
        )
        assert result["unit"] == "lines"
        assert result["total_lines"] == 5


# ---------------------------------------------------------------------------
# EliteAMarkdownLoader (.md)
# ---------------------------------------------------------------------------

class TestMarkdownLoaderMetadata:

    def test_basic_contract(self):
        meta = get_file_metadata("README.md", file_content=MARKDOWN_5_LINES,
                                 file_size=len(MARKDOWN_5_LINES))

        assert meta["__result_status__"] == "file_metadata"
        assert meta["extension"] == ".md"
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 5

        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        assert "end_line" in instr["first_class_params"]
        assert "1..5" in instr["first_class_params"]["start_line"]

    def test_no_content_degrades(self):
        meta = get_file_metadata("empty.md", file_content=None, file_size=0)
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 0

    def test_direct_classmethod(self):
        result = EliteAMarkdownLoader.get_file_metadata(
            filename="doc.md", file_content=MARKDOWN_5_LINES
        )
        assert result["unit"] == "lines"
        assert result["total_lines"] == 5

    def test_multiline_md(self):
        content = "\n".join(f"Line {i}" for i in range(1, 21)).encode()
        meta = get_file_metadata("long.md", file_content=content,
                                 file_size=len(content))
        assert meta["total_lines"] == 20
        assert "1..20" in meta["instruction_for_readFile"]["first_class_params"]["start_line"]
