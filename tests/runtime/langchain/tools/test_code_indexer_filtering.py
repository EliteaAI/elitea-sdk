"""
Unit tests for CodeIndexerToolkit.loader() — whitelist/blacklist/extension filtering
and chunking config forwarding.

CIL_WL  — whitelist filtering
CIL_BL  — blacklist filtering
CIL_EXT — skip_unsupported_extensions filtering
CIL_STAT — IndexingStats population
CIL_CHK  — chunked flag and chunking_config forwarding

All tests are pure unit tests: no network, no vectorstore, no DB.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from elitea_sdk.tools.code_indexer_toolkit import CodeIndexerToolkit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_toolkit(files: list[str], contents: dict | None = None, **_ignored) -> CodeIndexerToolkit:
    """
    Return a CodeIndexerToolkit instance with mocked I/O, bypassing __init__.

    files    — list returned by _get_files (the repo file listing)
    contents — dict mapping file_path -> content string.
               If a value is an Exception, _read_file raises it.
               Any file not present in contents gets "default content".
    """
    instance = object.__new__(CodeIndexerToolkit)
    instance._log_tool_event = MagicMock()
    instance._get_files = MagicMock(return_value=files)

    if contents is None:
        instance._read_file = MagicMock(return_value="default content")
    else:
        def _read(f, branch):
            val = contents.get(f, "default content")
            if isinstance(val, Exception):
                raise val
            return val
        instance._read_file = MagicMock(side_effect=_read)

    return instance


def load(toolkit: CodeIndexerToolkit, **kwargs) -> list[Document]:
    """Exhaust loader() into a list."""
    return list(toolkit.loader(**kwargs))


# ---------------------------------------------------------------------------
# CIL_WL — Whitelist filtering
# ---------------------------------------------------------------------------

class TestCIL_WL_Whitelist:

    def test_WL01_no_whitelist_passes_all_files(self):
        """whitelist=None → every file reaches the document yield stage."""
        tk = make_toolkit(["a.py", "b.md", "c.txt"])
        docs = load(tk, chunked=False, whitelist=None, skip_unsupported_extensions=False)
        assert len(docs) == 3
        assert tk._indexing_stats.files_skipped_whitelist == []

    def test_WL02_glob_pattern_filters_non_matching(self):
        """whitelist=['*.py'] → only .py files pass; others in files_skipped_whitelist."""
        tk = make_toolkit(["a.py", "b.md", "c.txt"])
        docs = load(tk, chunked=False, whitelist=["*.py"], skip_unsupported_extensions=False)
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "a.py"
        assert sorted(tk._indexing_stats.files_skipped_whitelist) == ["b.md", "c.txt"]

    def test_WL03_bare_extension_without_glob_matches_via_endswith(self):
        """whitelist=['py'] (no dot, no star) matches via file_path.endswith('.py')."""
        tk = make_toolkit(["a.py", "b.md"])
        docs = load(tk, chunked=False, whitelist=["py"], skip_unsupported_extensions=False)
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "a.py"
        assert tk._indexing_stats.files_skipped_whitelist == ["b.md"]

    def test_WL04_multiple_patterns_any_match_includes_file(self):
        """whitelist=['*.py', '*.md'] → files matching either pattern are included."""
        tk = make_toolkit(["a.py", "b.md", "c.java"])
        docs = load(tk, chunked=False, whitelist=["*.py", "*.md"],
                    skip_unsupported_extensions=False)
        filenames = {d.metadata["filename"] for d in docs}
        assert filenames == {"a.py", "b.md"}
        assert tk._indexing_stats.files_skipped_whitelist == ["c.java"]

    def test_WL05_glob_matches_path_with_directory_prefix(self):
        """*.py glob matches paths that include directory separators."""
        tk = make_toolkit(["src/main/app.py", "tests/test_app.py", "README.md"])
        docs = load(tk, chunked=False, whitelist=["*.py"],
                    skip_unsupported_extensions=False)
        filenames = {d.metadata["filename"] for d in docs}
        assert filenames == {"src/main/app.py", "tests/test_app.py"}


# ---------------------------------------------------------------------------
# CIL_BL — Blacklist filtering
# ---------------------------------------------------------------------------

class TestCIL_BL_Blacklist:

    def test_BL01_no_blacklist_excludes_nothing(self):
        """blacklist=None → no files excluded."""
        tk = make_toolkit(["a.py", "b.md"])
        docs = load(tk, chunked=False, blacklist=None, skip_unsupported_extensions=False)
        assert len(docs) == 2
        assert tk._indexing_stats.files_skipped_blacklist == []

    def test_BL02_glob_pattern_excludes_matching_files(self):
        """blacklist=['*.test.py'] → test files go to files_skipped_blacklist."""
        tk = make_toolkit(["app.py", "app.test.py", "utils.py"])
        docs = load(tk, chunked=False, blacklist=["*.test.py"],
                    skip_unsupported_extensions=False)
        filenames = {d.metadata["filename"] for d in docs}
        assert filenames == {"app.py", "utils.py"}
        assert tk._indexing_stats.files_skipped_blacklist == ["app.test.py"]

    def test_BL03_bare_extension_excludes_via_endswith(self):
        """blacklist=['py'] excludes .py files via endswith('.py')."""
        tk = make_toolkit(["a.py", "b.md"])
        docs = load(tk, chunked=False, blacklist=["py"],
                    skip_unsupported_extensions=False)
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "b.md"
        assert tk._indexing_stats.files_skipped_blacklist == ["a.py"]

    def test_BL04_blacklist_applied_after_whitelist_check(self):
        """A file that passes the whitelist but matches the blacklist is excluded."""
        tk = make_toolkit(["a.py", "test_a.py", "b.md"])
        docs = load(tk, chunked=False, whitelist=["*.py"], blacklist=["test_*"],
                    skip_unsupported_extensions=False)
        filenames = {d.metadata["filename"] for d in docs}
        assert filenames == {"a.py"}
        assert tk._indexing_stats.files_skipped_whitelist == ["b.md"]
        assert tk._indexing_stats.files_skipped_blacklist == ["test_a.py"]

    def test_BL05_file_matching_both_whitelist_and_blacklist_is_excluded(self):
        """Blacklist wins: a .py file on both whitelist and blacklist is not yielded."""
        tk = make_toolkit(["a.py"])
        docs = load(tk, chunked=False, whitelist=["*.py"], blacklist=["*.py"],
                    skip_unsupported_extensions=False)
        assert len(docs) == 0
        assert tk._indexing_stats.files_skipped_blacklist == ["a.py"]


# ---------------------------------------------------------------------------
# CIL_EXT — Extension filter (skip_unsupported_extensions)
# ---------------------------------------------------------------------------

class TestCIL_EXT_ExtensionFilter:

    def test_EXT01_unsupported_extension_skipped_by_default(self):
        """Files with .xyz extension are excluded when skip_unsupported_extensions=True."""
        tk = make_toolkit(["a.py", "b.xyz", "c.abc"])
        docs = load(tk, chunked=False)
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "a.py"
        assert sorted(tk._indexing_stats.files_unsupported_extension) == ["b.xyz", "c.abc"]

    def test_EXT02_unsupported_extension_included_when_flag_false(self):
        """skip_unsupported_extensions=False → .xyz files are not filtered."""
        tk = make_toolkit(["a.xyz"])
        docs = load(tk, chunked=False, skip_unsupported_extensions=False)
        assert len(docs) == 1
        assert tk._indexing_stats.files_unsupported_extension == []

    def test_EXT03_all_common_supported_extensions_pass(self):
        """Spot-check that .py .md .json .txt .yml are all considered supported."""
        files = ["a.py", "b.md", "c.json", "d.txt", "e.yml"]
        tk = make_toolkit(files)
        docs = load(tk, chunked=False)
        assert len(docs) == 5
        assert tk._indexing_stats.files_unsupported_extension == []

    def test_EXT04_extension_check_is_case_insensitive(self):
        """Upper-case extensions like .PY and .MD are recognised as supported."""
        tk = make_toolkit(["A.PY", "B.MD"])
        docs = load(tk, chunked=False)
        assert len(docs) == 2
        assert tk._indexing_stats.files_unsupported_extension == []


# ---------------------------------------------------------------------------
# CIL_STAT — IndexingStats population
# ---------------------------------------------------------------------------

class TestCIL_STAT_IndexingStats:

    def test_STAT01_total_fetched_counts_all_input_files(self):
        """total_fetched == number of files returned by _get_files."""
        tk = make_toolkit(["a.py", "b.xyz", "c.md"])
        load(tk, chunked=False)
        assert tk._indexing_stats.total_fetched == 3

    def test_STAT02_items_processed_counts_yielded_files(self):
        """items_processed == files that were successfully yielded."""
        tk = make_toolkit(["a.py", "b.md"])
        docs = load(tk, chunked=False)
        assert tk._indexing_stats.items_processed == 2
        assert tk._indexing_stats.items_processed == len(docs)

    def test_STAT03_read_error_populates_files_skipped_read_error(self):
        """Files where _read_file raises are recorded in files_skipped_read_error."""
        tk = make_toolkit(
            ["a.py", "b.py"],
            contents={"a.py": "content", "b.py": IOError("disk error")}
        )
        docs = load(tk, chunked=False)
        assert len(docs) == 1
        assert tk._indexing_stats.files_skipped_read_error == ["b.py"]

    def test_STAT04_empty_content_populates_files_skipped_empty(self):
        """Files where _read_file returns empty string land in files_skipped_empty."""
        tk = make_toolkit(
            ["a.py", "b.py"],
            contents={"a.py": "content", "b.py": ""}
        )
        docs = load(tk, chunked=False)
        assert len(docs) == 1
        assert tk._indexing_stats.files_skipped_empty == ["b.py"]

    def test_STAT05_stats_reset_on_each_loader_call(self):
        """Calling loader() twice resets stats — previous run does not bleed through."""
        tk = make_toolkit(["a.py"])
        load(tk, chunked=False)
        # Second call with different file list
        tk._get_files = MagicMock(return_value=["b.md", "c.xyz"])
        load(tk, chunked=False)
        # Stats should reflect only the second call
        assert tk._indexing_stats.total_fetched == 2
        assert tk._indexing_stats.files_unsupported_extension == ["c.xyz"]

    def test_STAT06_whitelist_skip_increments_correct_counter(self):
        """files_skipped_whitelist is populated only for whitelist misses, not for blacklist."""
        tk = make_toolkit(["a.py", "b.md"], skip_unsupported_extensions=False)
        load(tk, chunked=False, whitelist=["*.py"], skip_unsupported_extensions=False)
        assert tk._indexing_stats.files_skipped_whitelist == ["b.md"]
        assert tk._indexing_stats.files_skipped_blacklist == []


# ---------------------------------------------------------------------------
# CIL_CHK — chunked flag and chunking_config forwarding
# ---------------------------------------------------------------------------

LONG_TEXT = "Word " * 60  # 300 chars, well above default chunk_size=1000 if small config used


class TestCIL_CHK_ChunkingConfig:

    def test_CHK01_chunked_false_returns_one_doc_per_file(self):
        """chunked=False bypasses universal_chunker; exactly one Document per file."""
        tk = make_toolkit(["a.md", "b.py"])
        docs = load(tk, chunked=False)
        assert len(docs) == 2
        # Raw docs have no chunk_id injected by the chunker
        for doc in docs:
            assert "chunk_id" not in doc.metadata

    def test_CHK02_chunked_true_applies_universal_chunker_to_txt_file(self):
        """chunked=True (default) routes .txt files through the text chunker."""
        tk = make_toolkit(
            ["notes.txt"],
            contents={"notes.txt": LONG_TEXT}
        )
        docs = load(tk, chunked=True)
        assert len(docs) >= 1
        for doc in docs:
            assert doc.metadata.get("chunk_type") == "text"

    def test_CHK03_chunked_true_routes_markdown_to_markdown_chunker(self):
        """chunked=True routes .md to the markdown chunker (chunks have 'headers' key)."""
        md_content = "# Title\n\nSome content.\n\n## Section\n\nMore content here.\n"
        tk = make_toolkit(
            ["readme.md"],
            contents={"readme.md": md_content}
        )
        docs = load(tk, chunked=True)
        assert len(docs) >= 1
        assert any("headers" in d.metadata for d in docs)

    def test_CHK04_chunking_config_none_does_not_crash(self):
        """chunking_config=None must not raise — defaults are applied inside universal_chunker."""
        tk = make_toolkit(["notes.txt"], contents={"notes.txt": LONG_TEXT})
        docs = load(tk, chunked=True, chunking_config=None)
        assert len(docs) >= 1

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "EliteaAI/elitea_issues#4722 — "
            "chunking_config is accepted by loader() (line 88) but never forwarded "
            "to universal_chunker (line 235). "
            "Fix: change `return universal_chunker(raw_document_generator())` "
            "to `return universal_chunker(raw_document_generator(), config=chunking_config)`."
        )
    )
    def test_CHK05_text_chunking_config_chunk_size_is_honoured(self):
        """
        When chunking_config={'text_config': {'chunk_size': 30, 'chunk_overlap': 0}}
        is passed, text chunks must be ≤ 30 characters.

        Currently FAILS because loader() does not forward chunking_config to
        universal_chunker, so the default chunk_size=1000 is used instead.
        """
        content = "A" * 300  # 300-char string; default chunk_size=1000 produces 1 chunk
        tk = make_toolkit(["data.txt"], contents={"data.txt": content})
        docs = load(
            tk,
            chunked=True,
            chunking_config={"text_config": {"chunk_size": 30, "chunk_overlap": 0}}
        )
        assert len(docs) >= 2, "Expected content split into multiple small chunks"
        for doc in docs:
            assert len(doc.page_content) <= 30, (
                f"Chunk too large ({len(doc.page_content)} chars); config was not forwarded"
            )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "EliteaAI/elitea_issues#4722 — "
            "same root cause as CHK05: chunking_config not forwarded to universal_chunker. "
            "markdown_config max_tokens override has no effect."
        )
    )
    def test_CHK06_markdown_chunking_config_max_tokens_is_honoured(self):
        """
        When chunking_config={'markdown_config': {'max_tokens': 5}} is passed,
        markdown chunks must be very small (split aggressively).

        Currently FAILS because loader() does not forward chunking_config.
        """
        md_content = (
            "# Section One\n\n" + "word " * 50 +
            "\n\n## Section Two\n\n" + "word " * 50
        )
        tk = make_toolkit(["doc.md"], contents={"doc.md": md_content})
        docs_default = load(make_toolkit(["doc.md"], contents={"doc.md": md_content}),
                            chunked=True)
        docs_small = load(tk, chunked=True,
                          chunking_config={"markdown_config": {"max_tokens": 5,
                                                               "token_overlap": 0}})
        assert len(docs_small) > len(docs_default), (
            "Custom max_tokens=5 should produce more chunks than default max_tokens=1024"
        )
