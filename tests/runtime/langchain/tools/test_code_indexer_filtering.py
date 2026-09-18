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

import hashlib
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from elitea_sdk.tools.code_indexer_toolkit import (
    PROGRESS_EVENTS_MAXIMUM_STEP,
    CodeIndexerToolkit,
    progress_step_at,
)


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
        assert tk._indexing_stats.files_skipped_whitelist == set()

    def test_WL02_glob_pattern_filters_non_matching(self):
        """whitelist=['*.py'] → only .py files pass; others in files_skipped_whitelist."""
        tk = make_toolkit(["a.py", "b.md", "c.txt"])
        docs = load(tk, chunked=False, whitelist=["*.py"], skip_unsupported_extensions=False)
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "a.py"
        assert tk._indexing_stats.files_skipped_whitelist == {"b.md", "c.txt"}

    def test_WL03_bare_extension_without_glob_matches_via_endswith(self):
        """whitelist=['py'] (no dot, no star) matches via file_path.endswith('.py')."""
        tk = make_toolkit(["a.py", "b.md"])
        docs = load(tk, chunked=False, whitelist=["py"], skip_unsupported_extensions=False)
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "a.py"
        assert tk._indexing_stats.files_skipped_whitelist == {"b.md"}

    def test_WL04_multiple_patterns_any_match_includes_file(self):
        """whitelist=['*.py', '*.md'] → files matching either pattern are included."""
        tk = make_toolkit(["a.py", "b.md", "c.java"])
        docs = load(tk, chunked=False, whitelist=["*.py", "*.md"],
                    skip_unsupported_extensions=False)
        filenames = {d.metadata["filename"] for d in docs}
        assert filenames == {"a.py", "b.md"}
        assert tk._indexing_stats.files_skipped_whitelist == {"c.java"}

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
        assert tk._indexing_stats.files_skipped_blacklist == set()

    def test_BL02_glob_pattern_excludes_matching_files(self):
        """blacklist=['*.test.py'] → test files go to files_skipped_blacklist."""
        tk = make_toolkit(["app.py", "app.test.py", "utils.py"])
        docs = load(tk, chunked=False, blacklist=["*.test.py"],
                    skip_unsupported_extensions=False)
        filenames = {d.metadata["filename"] for d in docs}
        assert filenames == {"app.py", "utils.py"}
        assert tk._indexing_stats.files_skipped_blacklist == {"app.test.py"}

    def test_BL03_bare_extension_excludes_via_endswith(self):
        """blacklist=['py'] excludes .py files via endswith('.py')."""
        tk = make_toolkit(["a.py", "b.md"])
        docs = load(tk, chunked=False, blacklist=["py"],
                    skip_unsupported_extensions=False)
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "b.md"
        assert tk._indexing_stats.files_skipped_blacklist == {"a.py"}

    def test_BL04_blacklist_applied_after_whitelist_check(self):
        """A file that passes the whitelist but matches the blacklist is excluded."""
        tk = make_toolkit(["a.py", "test_a.py", "b.md"])
        docs = load(tk, chunked=False, whitelist=["*.py"], blacklist=["test_*"],
                    skip_unsupported_extensions=False)
        filenames = {d.metadata["filename"] for d in docs}
        assert filenames == {"a.py"}
        assert tk._indexing_stats.files_skipped_whitelist == {"b.md"}
        assert tk._indexing_stats.files_skipped_blacklist == {"test_a.py"}

    def test_BL05_file_matching_both_whitelist_and_blacklist_is_excluded(self):
        """Blacklist wins: a .py file on both whitelist and blacklist is not yielded."""
        tk = make_toolkit(["a.py"])
        docs = load(tk, chunked=False, whitelist=["*.py"], blacklist=["*.py"],
                    skip_unsupported_extensions=False)
        assert len(docs) == 0
        assert tk._indexing_stats.files_skipped_blacklist == {"a.py"}


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
        assert tk._indexing_stats.files_unsupported_extension == {"b.xyz", "c.abc"}

    def test_EXT02_unsupported_extension_included_when_flag_false(self):
        """skip_unsupported_extensions=False → .xyz files are not filtered."""
        tk = make_toolkit(["a.xyz"])
        docs = load(tk, chunked=False, skip_unsupported_extensions=False)
        assert len(docs) == 1
        assert tk._indexing_stats.files_unsupported_extension == set()

    def test_EXT03_all_common_supported_extensions_pass(self):
        """Spot-check that .py .md .json .txt .yml are all considered supported."""
        files = ["a.py", "b.md", "c.json", "d.txt", "e.yml"]
        tk = make_toolkit(files)
        docs = load(tk, chunked=False)
        assert len(docs) == 5
        assert tk._indexing_stats.files_unsupported_extension == set()

    def test_EXT04_extension_check_is_case_insensitive(self):
        """Upper-case extensions like .PY and .MD are recognised as supported."""
        tk = make_toolkit(["A.PY", "B.MD"])
        docs = load(tk, chunked=False)
        assert len(docs) == 2
        assert tk._indexing_stats.files_unsupported_extension == set()


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
        assert tk._indexing_stats.files_skipped_read_error == {"b.py"}

    def test_STAT04_empty_content_populates_files_skipped_empty(self):
        """Files where _read_file returns empty string land in files_skipped_empty."""
        tk = make_toolkit(
            ["a.py", "b.py"],
            contents={"a.py": "content", "b.py": ""}
        )
        docs = load(tk, chunked=False)
        assert len(docs) == 1
        assert tk._indexing_stats.files_skipped_empty == {"b.py"}

    def test_STAT05_stats_reset_on_each_loader_call(self):
        """Calling loader() twice resets stats — previous run does not bleed through."""
        tk = make_toolkit(["a.py"])
        load(tk, chunked=False)
        # Second call with different file list
        tk._get_files = MagicMock(return_value=["b.md", "c.xyz"])
        load(tk, chunked=False)
        # Stats should reflect only the second call
        assert tk._indexing_stats.total_fetched == 2
        assert tk._indexing_stats.files_unsupported_extension == {"c.xyz"}

    def test_STAT06_whitelist_skip_increments_correct_counter(self):
        """files_skipped_whitelist is populated only for whitelist misses, not for blacklist."""
        tk = make_toolkit(["a.py", "b.md"], skip_unsupported_extensions=False)
        load(tk, chunked=False, whitelist=["*.py"], skip_unsupported_extensions=False)
        assert tk._indexing_stats.files_skipped_whitelist == {"b.md"}
        assert tk._indexing_stats.files_skipped_blacklist == set()

    def test_STAT07_skipped_files_are_deduplicated(self):
        """Issue #4720: Sets deduplicate entries - same file tracked multiple times appears once."""
        from elitea_sdk.tools.base_indexer_toolkit import IndexingStats
        stats = IndexingStats()
        # Simulate tracking the same file multiple times (as happens with multi-page PDFs)
        stats.files_skipped_empty.add("document.pdf")
        stats.files_skipped_empty.add("document.pdf")
        stats.files_skipped_empty.add("document.pdf")
        stats.files_skipped_empty.add("document.pdf")
        # Should only have 1 entry, not 4
        assert len(stats.files_skipped_empty) == 1
        assert stats.files_skipped_empty == {"document.pdf"}
        # Summary should show count as 1
        summary = stats.get_summary()
        assert "Files with empty content (1)" in summary
        assert summary.count("document.pdf") == 1


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


# ---------------------------------------------------------------------------
# CIL_ID — Provider-supplied content identity (#6645)
# ---------------------------------------------------------------------------

DEFAULT_CONTENT = "default content"


def git_blob_sha(content):
    body = content.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()


def stored(blob_sha):
    return {"metadata": {"collection": "x"}, "commit_hashes": ["h"],
            "blob_shas": [blob_sha], "ids": ["row-1"]}


def make_identity_toolkit(identities, indexed=None, contents=None) -> CodeIndexerToolkit:
    """Toolkit whose listing carries a blob SHA per file."""
    tk = make_toolkit(list(identities), contents=contents)
    tk._get_files_with_identity = MagicMock(return_value=identities)
    tk._get_indexed_data = MagicMock(return_value=indexed or {})
    return tk


def identity_load(toolkit, **kwargs):
    preskipped = set()
    docs = list(toolkit.loader(index_name="x", preskipped_keys=preskipped, **kwargs))
    return docs, preskipped


class TestCIL_ID_ListingIdentity:

    def test_ID01_one_listing_call_serves_both_paths(self):
        tk = make_identity_toolkit({"a.py": "sha-a", "b.py": "sha-b"})
        docs = load(tk, chunked=False)
        tk._get_files.assert_not_called()
        assert {d.metadata["filename"] for d in docs} == {"a.py", "b.py"}

    def test_ID02_an_identity_matching_the_bytes_read_is_stamped(self):
        tk = make_identity_toolkit({"a.py": "8823529375b6c3a81c816ff25e94d88e4b66e2e2"})
        docs = load(tk, chunked=False)
        assert docs[0].metadata["blob_sha"] == "8823529375b6c3a81c816ff25e94d88e4b66e2e2"

    def test_ID02b_an_identity_that_does_not_describe_the_bytes_read_is_not_stamped(self):
        tk = make_identity_toolkit({"a.py": "sha-the-listing-claimed"})
        docs = load(tk, chunked=False)
        assert "blob_sha" not in docs[0].metadata

    def test_ID03_no_identity_hook_falls_back_to_the_plain_listing(self):
        tk = make_toolkit(["a.py", "b.py"])
        docs = load(tk, chunked=False)
        tk._get_files.assert_called_once()
        assert {d.metadata["filename"] for d in docs} == {"a.py", "b.py"}
        assert all("blob_sha" not in d.metadata for d in docs)

    def test_ID04_an_error_string_is_not_iterated_and_is_not_refetched(self):
        tk = make_toolkit(["a.py"])
        tk._get_files_with_identity = MagicMock(return_value="Error: status code 403, forbidden")
        with pytest.raises(ValueError):
            load(tk, chunked=False)
        tk._get_files.assert_not_called()

    def test_ID05_a_raising_hook_surfaces_instead_of_relisting(self):
        tk = make_toolkit(["a.py"])
        tk._get_files_with_identity = MagicMock(side_effect=RuntimeError("tree unavailable"))
        with pytest.raises(RuntimeError, match="tree unavailable"):
            load(tk, chunked=False)
        tk._get_files.assert_not_called()

    def test_ID06_a_partially_typed_map_keeps_the_paths_but_drops_the_identity(self):
        tk = make_toolkit(["a.py", "b.py"])
        tk._get_files_with_identity = MagicMock(return_value={"a.py": "sha-a", "b.py": None})
        docs = load(tk, chunked=False)
        tk._get_files.assert_not_called()
        assert {d.metadata["filename"] for d in docs} == {"a.py", "b.py"}
        assert all("blob_sha" not in d.metadata for d in docs)


class TestCIL_ID_UnchangedSkip:

    def test_ID07_a_matching_identity_is_never_read(self):
        tk = make_identity_toolkit({"a.py": "sha-a", "b.py": "sha-b"},
                                   indexed={"a.py": stored("sha-a")},
                                   contents={"b.py": "def beta():\n    return 1\n"})
        docs, preskipped = identity_load(tk)
        assert tk._read_file.call_count == 1
        assert preskipped == {"a.py"}
        assert {d.metadata["filename"] for d in docs} == {"b.py"}

    def test_ID08_a_stale_identity_is_read(self):
        tk = make_identity_toolkit({"a.py": "sha-new"}, indexed={"a.py": stored("sha-old")})
        _, preskipped = identity_load(tk)
        assert tk._read_file.call_count == 1
        assert preskipped == set()

    def test_ID09_a_blacklisted_file_is_excluded_not_preskipped(self):
        tk = make_identity_toolkit({"a.py": "sha-a", "skip_me.py": "sha-s"},
                                   indexed={"a.py": stored("sha-a"),
                                            "skip_me.py": stored("sha-s")})
        _, preskipped = identity_load(tk, blacklist=["*skip_me*"])
        assert preskipped == {"a.py"}
        assert tk._indexing_stats.files_skipped_blacklist == {"skip_me.py"}

    def test_ID10_a_whitelist_filtered_file_is_excluded_not_preskipped(self):
        tk = make_identity_toolkit({"a.py": "sha-a", "notes.md": "sha-n"},
                                   indexed={"a.py": stored("sha-a"),
                                            "notes.md": stored("sha-n")})
        _, preskipped = identity_load(tk, whitelist=["*.py"])
        assert preskipped == {"a.py"}
        assert tk._indexing_stats.files_skipped_whitelist == {"notes.md"}

    def test_ID11_the_raw_loader_never_arms_the_skip(self):
        tk = make_identity_toolkit({"a.py": "sha-a"}, indexed={"a.py": stored("sha-a")})
        docs, preskipped = identity_load(tk, chunked=False)
        assert preskipped == set()
        assert [d.metadata["filename"] for d in docs] == ["a.py"]

    def test_ID12_without_an_index_name_every_file_is_read(self):
        tk = make_identity_toolkit({"a.py": "sha-a"}, indexed={"a.py": stored("sha-a")})
        list(tk.loader(preskipped_keys=set()))
        assert tk._read_file.call_count == 1

    def test_ID13_without_a_preskip_sink_every_file_is_read(self):
        tk = make_identity_toolkit({"a.py": "sha-a"}, indexed={"a.py": stored("sha-a")})
        list(tk.loader(index_name="x"))
        assert tk._read_file.call_count == 1

    def test_ID14_a_row_with_no_stored_identity_is_read(self):
        entry = stored("sha-a")
        entry["blob_shas"] = [None]
        tk = make_identity_toolkit({"a.py": "sha-a"}, indexed={"a.py": entry})
        _, preskipped = identity_load(tk)
        assert tk._read_file.call_count == 1
        assert preskipped == set()


class TestCIL_ID_ProgressSurvivesTheSkip:
    """The better the skip works the fewer files are read, so progress must come from
    files considered, not files downloaded - otherwise the success case looks like a hang."""

    def _progress_events(self, toolkit):
        return [call.kwargs.get("message") for call in toolkit._log_tool_event.call_args_list
                if call.kwargs.get("tool_name") == "loader"]

    def test_ID15_an_all_unchanged_pass_still_reports_progress(self):
        identities = {f"f{n}.py": f"sha-{n}" for n in range(12)}
        indexed = {name: stored(sha) for name, sha in identities.items()}
        tk = make_identity_toolkit(identities, indexed=indexed)

        identity_load(tk)

        assert tk._read_file.call_count == 0
        assert "10 files processed" in self._progress_events(tk)

    def test_ID16_the_tenth_considered_file_reports_even_when_it_is_a_skip(self):
        identities = {f"f{n}.py": f"sha-{n}" for n in range(12)}
        indexed = {name: stored(sha) for name, sha in list(identities.items())[:10]}
        tk = make_identity_toolkit(
            identities, indexed=indexed,
            contents={name: "def beta():\n    return 1\n" for name in identities})

        identity_load(tk)

        assert tk._read_file.call_count == 2
        assert "10 files processed" in self._progress_events(tk)

    def test_ID17_the_closing_line_does_not_claim_undownloaded_files(self):
        identities = {f"f{n}.py": f"sha-{n}" for n in range(3)}
        indexed = {name: stored(sha) for name, sha in identities.items()}
        tk = make_identity_toolkit(identities, indexed=indexed)

        identity_load(tk)

        assert "3 files processed, 0 downloaded" in self._progress_events(tk)

    def test_ID18_the_event_rate_is_bounded_on_a_large_repository(self):
        identities = {f"f{n}.py": f"sha-{n}" for n in range(5000)}
        indexed = {name: stored(sha) for name, sha in identities.items()}
        tk = make_identity_toolkit(identities, indexed=indexed)

        identity_load(tk)

        progress_lines = [event for event in self._progress_events(tk)
                          if event and event.endswith("files processed")]
        assert len(progress_lines) <= 100


class TestCIL_ID_TheStampedIdentityDescribesTheBytesIndexed:
    """The listing and the per-file read are separate calls against a moving branch, so a
    push between them would otherwise store an identity for content never indexed - and a
    revert back to the listed tree would then pin the wrong content forever."""

    def test_ID19_a_push_between_listing_and_read_leaves_no_identity(self):
        listed = "0000000000000000000000000000000000000000"
        tk = make_identity_toolkit({"a.py": listed},
                                   contents={"a.py": "def pushed_after_the_listing():\n    pass\n"})

        docs = load(tk, chunked=False)

        assert "blob_sha" not in docs[0].metadata

    def test_ID20_one_unusable_entry_does_not_disarm_the_whole_listing(self):
        identities = {name: git_blob_sha(DEFAULT_CONTENT) for name in ["a.py", "b.py"]}
        identities["c.py"] = None
        indexed = {name: stored(sha) for name, sha in identities.items() if sha}
        tk = make_identity_toolkit(identities)
        tk._get_indexed_data = MagicMock(return_value=indexed)

        _, preskipped = identity_load(tk)

        assert preskipped == {"a.py", "b.py"}
        assert tk._read_file.call_count == 1


class TestCIL_ID_TheClosingCountIsHonest:

    def _progress_events(self, toolkit):
        return [call.kwargs.get("message") for call in toolkit._log_tool_event.call_args_list
                if call.kwargs.get("tool_name") == "loader"]

    def test_ID21_a_request_that_failed_still_counts_as_downloaded(self):
        tk = make_toolkit(["a.py", "b.py"], contents={"a.py": OSError("boom"), "b.py": ""})

        load(tk, chunked=False)

        assert "0 files processed, 2 downloaded" in self._progress_events(tk)

    def test_ID22_progress_still_reports_under_a_narrow_filter(self):
        listing = {f"vendor/v{n}.js": f"sha-v{n}" for n in range(2000)}
        listing.update({f"src/s{n}.py": f"sha-s{n}" for n in range(30)})
        tk = make_identity_toolkit(listing)

        load(tk, whitelist=["*.py"], chunked=False)

        assert "10 files processed" in self._progress_events(tk)


class TestCIL_ID_TheBannerNeverGoesQuietForLong:
    """Decade scaling alone would leave a first run silent for ten thousand consecutive
    downloads, which is the apparent hang the progress events exist to prevent."""

    def _progress_counts(self, toolkit):
        events = [call.kwargs.get("message") for call in toolkit._log_tool_event.call_args_list
                  if call.kwargs.get("tool_name") == "loader"]
        return [int(event.split()[0]) for event in events
                if event and event.endswith("files processed")]

    def test_ID23_no_gap_exceeds_the_maximum_step(self):
        tk = make_toolkit([f"f{n}.py" for n in range(25000)])

        load(tk, chunked=False)

        counts = self._progress_counts(tk)
        gaps = [later - earlier for earlier, later in zip(counts, counts[1:])]
        assert counts and max(gaps) <= PROGRESS_EVENTS_MAXIMUM_STEP

    def test_ID24_the_first_report_still_lands_early(self):
        tk = make_toolkit([f"f{n}.py" for n in range(25000)])

        load(tk, chunked=False)

        assert self._progress_counts(tk)[0] == 10


class TestCIL_ID_TheProgressStepScalesByDecade:

    def test_ID25_the_step_grows_only_once_a_decade_is_complete(self):
        assert progress_step_at(99) == 10
        assert progress_step_at(100) == 100
        assert progress_step_at(999) == 100
        assert progress_step_at(1000) == PROGRESS_EVENTS_MAXIMUM_STEP
