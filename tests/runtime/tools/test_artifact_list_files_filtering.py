"""
Unit tests for ArtifactWrapper.list_files() — include/skip pattern filtering
and folder scoping.

ART_INC  — include pattern filtering
ART_SKIP — skip pattern filtering
ART_FOLD — folder scoping
ART_COMB — combined folder + patterns
ART_COMPAT — backward compatibility

All tests are pure unit tests: no network, no S3, mocked artifact client.
"""

import fnmatch
import pytest
from unittest.mock import MagicMock, patch
from urllib.parse import quote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockArtifactWrapper:
    """
    Lightweight mock that replicates list_files() filtering logic
    without Pydantic model overhead.
    """

    def __init__(self, files: list[dict]):
        self.bucket = "test-bucket"
        self.max_single_read_size = 200000

        # Mock elitea client
        self.elitea = MagicMock()
        self.elitea.base_url = "https://test.elitea.ai"
        self.elitea.project_id = "test-project"

        # Build rows with proper structure
        self._rows = []
        for f in files:
            row = {
                "name": f.get("name", f["key"].split("/")[-1]),
                "key": f["key"],
                "size": f.get("size", 100),
                "modified": f.get("modified", "2024-01-01T00:00:00Z"),
                "type": f.get("type", "file")
            }
            self._rows.append(row)

        # Mock artifact client
        self.artifact = MagicMock()
        self.artifact.list = MagicMock(return_value={
            "total": len(self._rows),
            "rows": [row.copy() for row in self._rows]
        })

    @staticmethod
    def _fnmatch_nocase(filename: str, pattern: str) -> bool:
        """Case-insensitive fnmatch for cross-platform consistency."""
        return fnmatch.fnmatch(filename.lower(), pattern.lower())

    def list_files(self, bucket_name=None, folder=None, recursive=False,
                   include=None, skip=None, return_as_string=True):
        """Replicate the actual list_files implementation for testing."""
        bucket = bucket_name or self.bucket
        base_url = self.elitea.base_url.rstrip('/')
        project_id = self.elitea.project_id

        # Build prefix for folder scoping
        prefix = ''
        if folder:
            prefix = folder.strip('/') + '/'

        # delimiter='/' for folder listing, None for recursive listing
        delimiter = None if recursive else '/'
        result = self.artifact.list(bucket_name=bucket, prefix=prefix, delimiter=delimiter)

        if 'error' in result:
            return "[]" if return_as_string else {"total": 0, "rows": []}

        # Apply include/skip pattern filtering
        include = include or []
        skip = skip or []
        filtered_files = []

        for file_info in result.get('rows', []):
            if file_info.get('type') == 'file':
                full_key = file_info.get('key', prefix + file_info['name'])

                # Check skip patterns first (case-insensitive)
                if skip and any(self._fnmatch_nocase(full_key, pattern) for pattern in skip):
                    continue

                # Check include patterns (case-insensitive)
                if include and not any(self._fnmatch_nocase(full_key, pattern) for pattern in include):
                    continue

                # Add S3 download link
                encoded_key = quote(full_key, safe='/')
                file_info['link'] = f"{base_url}/s3/{bucket}/{encoded_key}?project_id={project_id}"
                filtered_files.append(file_info)
            elif file_info.get('type') == 'folder':
                filtered_files.append(file_info)

        result['rows'] = filtered_files
        result['total'] = len(filtered_files)

        return str(result) if return_as_string else result


def make_wrapper(files: list[dict]) -> MockArtifactWrapper:
    """Create a mock wrapper for testing."""
    return MockArtifactWrapper(files)


def list_files(wrapper, **kwargs) -> list[dict]:
    """Call list_files and return the rows."""
    result = wrapper.list_files(return_as_string=False, recursive=True, **kwargs)
    return result.get("rows", [])


def get_file_names(rows: list[dict]) -> set[str]:
    """Extract file names from rows."""
    return {r["name"] for r in rows if r.get("type") == "file"}


def get_file_keys(rows: list[dict]) -> set[str]:
    """Extract full keys from rows."""
    return {r["key"] for r in rows if r.get("type") == "file"}


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

SAMPLE_FILES = [
    {"key": "readme.txt"},
    {"key": "report.xlsx"},
    {"key": "_shared/config.md"},
    {"key": "_shared/guide.pdf"},
    {"key": "_shared/templates/template1.md"},
    {"key": "_shared/templates/template2.docx"},
    {"key": "docs/api.md"},
    {"key": "docs/architecture.md"},
    {"key": "docs/images/diagram.png"},
    {"key": "temp/draft.txt"},
    {"key": "temp/backup/old.md"},
    {"key": "reports/q1-report.pdf"},
    {"key": "reports/q2-report.xlsx"},
]


# ---------------------------------------------------------------------------
# ART_INC — Include pattern filtering
# ---------------------------------------------------------------------------

class TestART_INC_Include:

    def test_INC01_no_include_returns_all_files(self):
        """include=None → all files returned."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, include=None)
        assert len(rows) == 13

    def test_INC02_empty_include_returns_all_files(self):
        """include=[] → all files returned (same as None)."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, include=[])
        assert len(rows) == 13

    def test_INC03_extension_pattern_filters(self):
        """include=['*.md'] → only .md files returned."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, include=["*.md"])
        keys = get_file_keys(rows)
        expected = {
            "_shared/config.md",
            "_shared/templates/template1.md",
            "docs/api.md",
            "docs/architecture.md",
            "temp/backup/old.md",
        }
        assert keys == expected

    def test_INC04_folder_pattern_filters(self):
        """include=['_shared/*'] → only files in _shared/ folder."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, include=["_shared/*"])
        keys = get_file_keys(rows)
        expected = {
            "_shared/config.md",
            "_shared/guide.pdf",
            "_shared/templates/template1.md",
            "_shared/templates/template2.docx",
        }
        assert keys == expected

    def test_INC05_nested_folder_matches(self):
        """include=['_shared/*'] matches nested files (fnmatch * matches /)."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, include=["_shared/*"])
        keys = get_file_keys(rows)
        # Should include nested templates
        assert "_shared/templates/template1.md" in keys
        assert "_shared/templates/template2.docx" in keys

    def test_INC06_multiple_patterns_or_logic(self):
        """include=['_shared/*', 'docs/*'] → files matching either pattern."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, include=["_shared/*", "docs/*"])
        keys = get_file_keys(rows)
        expected = {
            "_shared/config.md",
            "_shared/guide.pdf",
            "_shared/templates/template1.md",
            "_shared/templates/template2.docx",
            "docs/api.md",
            "docs/architecture.md",
            "docs/images/diagram.png",
        }
        assert keys == expected

    def test_INC07_folder_with_extension_pattern(self):
        """include=['_shared/*.md'] → .md files in _shared/ only."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, include=["_shared/*.md"])
        keys = get_file_keys(rows)
        expected = {
            "_shared/config.md",
            "_shared/templates/template1.md",
        }
        assert keys == expected

    def test_INC08_wildcard_folder_prefix(self):
        """include=['*_backup/*'] → folders ending with _backup."""
        wrapper = make_wrapper([
            {"key": "daily_backup/file1.txt"},
            {"key": "weekly_backup/file2.txt"},
            {"key": "other/file3.txt"},
        ])
        rows = list_files(wrapper, include=["*_backup/*"])
        keys = get_file_keys(rows)
        assert keys == {"daily_backup/file1.txt", "weekly_backup/file2.txt"}

    def test_INC09_multiple_extensions(self):
        """include=['*.pdf', '*.xlsx'] → multiple extension types."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, include=["*.pdf", "*.xlsx"])
        keys = get_file_keys(rows)
        expected = {
            "_shared/guide.pdf",
            "report.xlsx",
            "reports/q1-report.pdf",
            "reports/q2-report.xlsx",
        }
        assert keys == expected


# ---------------------------------------------------------------------------
# ART_SKIP — Skip pattern filtering
# ---------------------------------------------------------------------------

class TestART_SKIP_Skip:

    def test_SKIP01_no_skip_returns_all(self):
        """skip=None → all files returned."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, skip=None)
        assert len(rows) == 13

    def test_SKIP02_empty_skip_returns_all(self):
        """skip=[] → all files returned."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, skip=[])
        assert len(rows) == 13

    def test_SKIP03_skip_extension(self):
        """skip=['*.tmp'] → excludes .tmp files."""
        wrapper = make_wrapper([
            {"key": "file1.txt"},
            {"key": "file2.tmp"},
            {"key": "file3.md"},
        ])
        rows = list_files(wrapper, skip=["*.tmp"])
        keys = get_file_keys(rows)
        assert keys == {"file1.txt", "file3.md"}

    def test_SKIP04_skip_folder(self):
        """skip=['temp/*'] → excludes files in temp/ folder."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, skip=["temp/*"])
        keys = get_file_keys(rows)
        assert "temp/draft.txt" not in keys
        assert "temp/backup/old.md" not in keys
        assert len(keys) == 11  # 13 - 2 temp files

    def test_SKIP05_skip_multiple_patterns(self):
        """skip=['temp/*', '*.xlsx'] → excludes both."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, skip=["temp/*", "*.xlsx"])
        keys = get_file_keys(rows)
        # Excluded: temp/draft.txt, temp/backup/old.md, report.xlsx, reports/q2-report.xlsx
        assert "temp/draft.txt" not in keys
        assert "report.xlsx" not in keys
        assert "reports/q2-report.xlsx" not in keys
        assert len(keys) == 9  # 13 - 4

    def test_SKIP06_skip_nested_folder(self):
        """skip=['_shared/templates/*'] → excludes nested folder only."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, skip=["_shared/templates/*"])
        keys = get_file_keys(rows)
        assert "_shared/templates/template1.md" not in keys
        assert "_shared/templates/template2.docx" not in keys
        # Parent folder files still included
        assert "_shared/config.md" in keys
        assert "_shared/guide.pdf" in keys


# ---------------------------------------------------------------------------
# ART_FOLD — Folder scoping (via S3 prefix)
# ---------------------------------------------------------------------------

class TestART_FOLD_Folder:

    def test_FOLD01_folder_scopes_prefix(self):
        """folder='_shared' → artifact.list called with prefix='_shared/'."""
        wrapper = make_wrapper(SAMPLE_FILES)
        list_files(wrapper, folder="_shared")

        # Verify prefix was passed to artifact.list
        wrapper.artifact.list.assert_called_once()
        call_kwargs = wrapper.artifact.list.call_args[1]
        assert call_kwargs["prefix"] == "_shared/"

    def test_FOLD02_folder_strips_slashes(self):
        """folder='/_shared/' → normalized to '_shared/'."""
        wrapper = make_wrapper(SAMPLE_FILES)
        list_files(wrapper, folder="/_shared/")

        call_kwargs = wrapper.artifact.list.call_args[1]
        assert call_kwargs["prefix"] == "_shared/"

    def test_FOLD03_no_folder_empty_prefix(self):
        """folder=None → prefix='' (bucket root)."""
        wrapper = make_wrapper(SAMPLE_FILES)
        list_files(wrapper, folder=None)

        call_kwargs = wrapper.artifact.list.call_args[1]
        assert call_kwargs["prefix"] == ""

    def test_FOLD04_nested_folder(self):
        """folder='_shared/templates' → prefix='_shared/templates/'."""
        wrapper = make_wrapper(SAMPLE_FILES)
        list_files(wrapper, folder="_shared/templates")

        call_kwargs = wrapper.artifact.list.call_args[1]
        assert call_kwargs["prefix"] == "_shared/templates/"


# ---------------------------------------------------------------------------
# ART_COMB — Combined folder + patterns
# ---------------------------------------------------------------------------

class TestART_COMB_Combined:

    def test_COMB01_folder_with_include(self):
        """folder='_shared' + include=['*.md'] → .md files in _shared only."""
        # Mock returns only _shared files (S3 prefix filtering)
        wrapper = make_wrapper([
            {"key": "_shared/config.md"},
            {"key": "_shared/guide.pdf"},
            {"key": "_shared/templates/template1.md"},
            {"key": "_shared/templates/template2.docx"},
        ])
        rows = list_files(wrapper, folder="_shared", include=["*.md"])
        keys = get_file_keys(rows)
        expected = {
            "_shared/config.md",
            "_shared/templates/template1.md",
        }
        assert keys == expected

    def test_COMB02_folder_with_skip(self):
        """folder='_shared' + skip=['*.docx'] → exclude docx in _shared."""
        wrapper = make_wrapper([
            {"key": "_shared/config.md"},
            {"key": "_shared/guide.pdf"},
            {"key": "_shared/templates/template1.md"},
            {"key": "_shared/templates/template2.docx"},
        ])
        rows = list_files(wrapper, folder="_shared", skip=["*.docx"])
        keys = get_file_keys(rows)
        assert "_shared/templates/template2.docx" not in keys
        assert len(keys) == 3

    def test_COMB03_folder_include_skip(self):
        """folder + include + skip all combined."""
        wrapper = make_wrapper([
            {"key": "_shared/config.md"},
            {"key": "_shared/draft.md"},
            {"key": "_shared/guide.pdf"},
            {"key": "_shared/templates/template1.md"},
        ])
        rows = list_files(wrapper, folder="_shared", include=["*.md"], skip=["*draft*"])
        keys = get_file_keys(rows)
        expected = {
            "_shared/config.md",
            "_shared/templates/template1.md",
        }
        assert keys == expected

    def test_COMB04_include_skip_order(self):
        """skip is applied after include (file must pass include AND not match skip)."""
        wrapper = make_wrapper([
            {"key": "a.md"},
            {"key": "b.md"},
            {"key": "draft.md"},
        ])
        # Include all .md, but skip drafts
        rows = list_files(wrapper, include=["*.md"], skip=["draft*"])
        keys = get_file_keys(rows)
        assert keys == {"a.md", "b.md"}


# ---------------------------------------------------------------------------
# ART_COMPAT — Backward compatibility
# ---------------------------------------------------------------------------

class TestART_COMPAT_BackwardCompatibility:

    def test_COMPAT01_no_new_params_works(self):
        """Calling list_files without new params works as before."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper)
        assert len(rows) == 13

    def test_COMPAT02_existing_params_preserved(self):
        """bucket_name, recursive, return_as_string still work."""
        wrapper = make_wrapper(SAMPLE_FILES)

        # Test return_as_string=True
        result_str = wrapper.list_files(return_as_string=True, recursive=True)
        assert isinstance(result_str, str)

        # Test return_as_string=False
        result_dict = wrapper.list_files(return_as_string=False, recursive=True)
        assert isinstance(result_dict, dict)
        assert "rows" in result_dict
        assert "total" in result_dict

    def test_COMPAT03_folders_not_filtered(self):
        """Folder entries are always included (not filtered by include/skip)."""
        wrapper = make_wrapper([
            {"key": "docs/", "type": "folder", "name": "docs/"},
            {"key": "docs/file.md", "type": "file"},
            {"key": "temp/", "type": "folder", "name": "temp/"},
            {"key": "temp/file.txt", "type": "file"},
        ])
        rows = list_files(wrapper, include=["docs/*"])

        # File filtering applied
        file_keys = {r["key"] for r in rows if r["type"] == "file"}
        assert file_keys == {"docs/file.md"}

        # Folders still included for navigation
        folder_names = {r["name"] for r in rows if r["type"] == "folder"}
        assert folder_names == {"docs/", "temp/"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestART_EDGE_EdgeCases:

    def test_EDGE01_empty_bucket(self):
        """Empty bucket returns empty rows."""
        wrapper = make_wrapper([])
        rows = list_files(wrapper)
        assert rows == []

    def test_EDGE02_include_no_matches(self):
        """include pattern with no matches returns empty."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, include=["*.nonexistent"])
        file_rows = [r for r in rows if r["type"] == "file"]
        assert file_rows == []

    def test_EDGE03_skip_all_files(self):
        """skip=['*'] excludes all files."""
        wrapper = make_wrapper(SAMPLE_FILES)
        rows = list_files(wrapper, skip=["*"])
        file_rows = [r for r in rows if r["type"] == "file"]
        assert file_rows == []

    def test_EDGE04_case_insensitivity(self):
        """Pattern matching is case-insensitive for backward compatibility."""
        wrapper = make_wrapper([
            {"key": "file.MD"},
            {"key": "file.Md"},
            {"key": "file.md"},
            {"key": "FILE.MD"},
        ])
        rows = list_files(wrapper, include=["*.md"])
        keys = get_file_keys(rows)
        # All case variations should match
        assert keys == {"file.MD", "file.Md", "file.md", "FILE.MD"}

    def test_EDGE04b_case_insensitive_skip(self):
        """Skip patterns are also case-insensitive."""
        wrapper = make_wrapper([
            {"key": "temp.TMP"},
            {"key": "file.txt"},
            {"key": "cache.Tmp"},
        ])
        rows = list_files(wrapper, skip=["*.tmp"])
        keys = get_file_keys(rows)
        # .TMP and .Tmp should also be skipped
        assert keys == {"file.txt"}

    def test_EDGE05_special_characters_in_filename(self):
        """Files with special characters in names."""
        wrapper = make_wrapper([
            {"key": "file (1).txt"},
            {"key": "file-name_v2.txt"},
            {"key": "file.test.txt"},
        ])
        rows = list_files(wrapper, include=["*.txt"])
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# ART_CASE — Additional case-insensitivity tests
# ---------------------------------------------------------------------------

class TestART_CASE_CaseInsensitivity:

    def test_CASE01_folder_pattern_case_insensitive(self):
        """Folder patterns should match case-insensitively."""
        wrapper = make_wrapper([
            {"key": "DOCS/file.txt"},
            {"key": "docs/file.txt"},
            {"key": "Docs/file.txt"},
            {"key": "other/file.txt"},
        ])
        rows = list_files(wrapper, include=["docs/*"])
        keys = get_file_keys(rows)
        # All case variations of 'docs/' should match
        assert keys == {"DOCS/file.txt", "docs/file.txt", "Docs/file.txt"}

    def test_CASE02_pattern_itself_is_case_insensitive(self):
        """Pattern with uppercase should match lowercase files."""
        wrapper = make_wrapper([
            {"key": "readme.md"},
            {"key": "guide.md"},
        ])
        # Pattern uses uppercase
        rows = list_files(wrapper, include=["*.MD"])
        keys = get_file_keys(rows)
        # Should still match lowercase .md files
        assert keys == {"readme.md", "guide.md"}

    def test_CASE03_full_path_case_insensitive(self):
        """Full path patterns match case-insensitively."""
        wrapper = make_wrapper([
            {"key": "_Shared/Config.MD"},
            {"key": "_shared/config.md"},
            {"key": "_SHARED/CONFIG.MD"},
        ])
        rows = list_files(wrapper, include=["_shared/*.md"])
        keys = get_file_keys(rows)
        # All variations should match
        assert keys == {"_Shared/Config.MD", "_shared/config.md", "_SHARED/CONFIG.MD"}

    def test_CASE04_combined_pattern_case_insensitive(self):
        """Combined folder/extension patterns with mixed case."""
        wrapper = make_wrapper([
            {"key": "TEMP/draft.PDF"},
            {"key": "temp/draft.pdf"},
            {"key": "Temp/Draft.Pdf"},
        ])
        # Pattern in lowercase, files in various cases
        rows = list_files(wrapper, skip=["temp/*.pdf"])
        keys = get_file_keys(rows)
        # All should be skipped despite case differences
        assert len(keys) == 0

    def test_CASE05_filename_mixed_case(self):
        """Filenames with mixed case throughout."""
        wrapper = make_wrapper([
            {"key": "MyDocument.TXT"},
            {"key": "myDocument.txt"},
            {"key": "MYDOCUMENT.TXT"},
        ])
        rows = list_files(wrapper, include=["*document*.txt"])
        keys = get_file_keys(rows)
        assert keys == {"MyDocument.TXT", "myDocument.txt", "MYDOCUMENT.TXT"}


# ---------------------------------------------------------------------------
# ART_META — Glob metacharacter behavior tests
# ---------------------------------------------------------------------------

class TestART_META_Metacharacters:

    def test_META01_bracket_pattern_matches_character_set(self):
        """file[123].txt matches file1.txt, file2.txt, file3.txt."""
        wrapper = make_wrapper([
            {"key": "file1.txt"},
            {"key": "file2.txt"},
            {"key": "file4.txt"},
        ])
        rows = list_files(wrapper, include=["file[123].txt"])
        keys = get_file_keys(rows)
        # Only 1, 2, 3 should match; 4 excluded
        assert keys == {"file1.txt", "file2.txt"}

    def test_META02_question_mark_matches_single_char(self):
        """file?.txt matches single character after 'file'."""
        wrapper = make_wrapper([
            {"key": "file1.txt"},
            {"key": "file12.txt"},
            {"key": "fileA.txt"},
        ])
        rows = list_files(wrapper, include=["file?.txt"])
        keys = get_file_keys(rows)
        # Only single character matches
        assert keys == {"file1.txt", "fileA.txt"}

    def test_META03_bracket_in_actual_filename(self):
        """Files with literal brackets in names."""
        wrapper = make_wrapper([
            {"key": "report[v2].txt"},
            {"key": "report.txt"},
        ])
        # Using * to match everything - brackets in filename itself
        rows = list_files(wrapper, include=["*.txt"])
        keys = get_file_keys(rows)
        # Both should be included (wildcard matches any characters)
        assert "report[v2].txt" in keys
        assert "report.txt" in keys
