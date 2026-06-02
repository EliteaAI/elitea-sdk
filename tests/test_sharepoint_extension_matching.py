"""Unit tests for SharePoint extension matching functionality.

Tests for fix of issue #4897: skip_extensions with full filename support.
"""

import pytest

from elitea_sdk.tools.sharepoint.file_filters import (
    matches_extension_filter,
    normalize_extension_filters,
)


class TestExtensionNormalization:
    """Test the normalize_extension_filters function."""

    def test_simple_extensions(self):
        """Test normalization of simple extensions."""
        assert normalize_extension_filters(["csv"]) == ["*.csv"]
        assert normalize_extension_filters([".csv"]) == ["*.csv"]
        assert normalize_extension_filters(["*.csv"]) == ["*.csv"]
        assert normalize_extension_filters(["*csv"]) == ["*.csv"]

    def test_multiple_extensions(self):
        """Test normalization of multiple extensions."""
        result = normalize_extension_filters(["csv", ".pdf", "*.docx"])
        assert result == ["*.csv", "*.pdf", "*.docx"]

    def test_full_filenames(self):
        """Test normalization preserves full filenames."""
        assert normalize_extension_filters(["testdata.csv"]) == ["testdata.csv"]
        assert normalize_extension_filters(["TestDataEJ.csv"]) == ["testdataej.csv"]

    def test_wildcard_patterns(self):
        """Test normalization of wildcard patterns."""
        assert normalize_extension_filters(["*.backup.csv"]) == ["*.backup.csv"]
        assert normalize_extension_filters(["*.test.pdf"]) == ["*.test.pdf"]

    def test_empty_input(self):
        """Test normalization with empty input."""
        assert normalize_extension_filters([]) == []
        assert normalize_extension_filters(None) == []
        assert normalize_extension_filters([""]) == []
        assert normalize_extension_filters(["  "]) == []

    def test_case_insensitive(self):
        """Test normalization converts to lowercase."""
        assert normalize_extension_filters(["CSV"]) == ["*.csv"]
        assert normalize_extension_filters(["TestFile.PDF"]) == ["testfile.pdf"]


class TestExtensionMatching:
    """Test the matches_extension_filter function."""

    def test_simple_extension_matching(self):
        """Test matching with simple extensions."""
        patterns = normalize_extension_filters(["*.csv"])
        assert matches_extension_filter("data.csv", patterns) is True
        assert matches_extension_filter("report.csv", patterns) is True
        assert matches_extension_filter("data.xlsx", patterns) is False

    def test_legacy_star_extension_matches_only_file_extensions(self):
        """Test legacy '*ext' input remains an extension filter."""
        patterns = normalize_extension_filters(["*pdf"])
        assert matches_extension_filter("document.pdf", patterns) is True
        assert matches_extension_filter("file.notpdf", patterns) is False
        assert matches_extension_filter("my-pdf", patterns) is False

    def test_full_filename_matching(self):
        """Test matching with full filenames (issue #4897)."""
        patterns = normalize_extension_filters(["testdataEJ.csv"])
        assert matches_extension_filter("testdataEJ.csv", patterns) is True
        assert matches_extension_filter("testdataej.csv", patterns) is True  # Case-insensitive
        assert matches_extension_filter("TESTDATAEJ.CSV", patterns) is True  # Case-insensitive
        assert matches_extension_filter("other.csv", patterns) is False
        assert matches_extension_filter("testdata.csv", patterns) is False

    def test_wildcard_pattern_matching(self):
        """Test matching with wildcard patterns."""
        patterns = normalize_extension_filters(["*.backup.csv"])
        assert matches_extension_filter("file.backup.csv", patterns) is True
        assert matches_extension_filter("data.backup.csv", patterns) is True
        assert matches_extension_filter("file.csv", patterns) is False
        assert matches_extension_filter("file.backup.xlsx", patterns) is False

    def test_multiple_patterns(self):
        """Test matching with multiple patterns."""
        patterns = normalize_extension_filters(["*.csv", "*.pdf", "exclude.xlsx"])
        assert matches_extension_filter("data.csv", patterns) is True
        assert matches_extension_filter("report.pdf", patterns) is True
        assert matches_extension_filter("exclude.xlsx", patterns) is True
        assert matches_extension_filter("other.xlsx", patterns) is False
        assert matches_extension_filter("data.docx", patterns) is False

    def test_case_insensitivity(self):
        """Test matching is case-insensitive."""
        patterns = normalize_extension_filters(["*.CSV"])
        assert matches_extension_filter("Data.csv", patterns) is True
        assert matches_extension_filter("DATA.CSV", patterns) is True

        patterns = normalize_extension_filters(["TestFile.PDF"])
        assert matches_extension_filter("testfile.pdf", patterns) is True
        assert matches_extension_filter("TESTFILE.PDF", patterns) is True

    def test_empty_patterns(self):
        """Test matching with empty pattern list."""
        assert matches_extension_filter("any.csv", []) is False
        assert matches_extension_filter("any.csv", None) is False

    def test_files_without_extension(self):
        """Test matching files without extensions."""
        patterns = normalize_extension_filters(["*.csv"])
        assert matches_extension_filter("README", patterns) is False
        assert matches_extension_filter("Makefile", patterns) is False


class TestIssue4897Scenario:
    """Test the exact scenario from issue #4897."""

    def test_skip_specific_file(self):
        """Test that skip_extensions with full filename works correctly."""
        # Setup: include all CSV files, but skip one specific file
        include_patterns = normalize_extension_filters(["*.csv"])
        skip_patterns = normalize_extension_filters(["testdataEJ.csv"])

        # Test file that should be skipped
        filename = "testdataEJ.csv"
        matches_include = matches_extension_filter(filename, include_patterns)
        matches_skip = matches_extension_filter(filename, skip_patterns)

        # File matches include but also matches skip
        assert matches_include is True
        assert matches_skip is True

        # Logic: file should be skipped (skip takes priority)
        should_process = matches_include and not matches_skip
        assert should_process is False

    def test_process_other_csv_files(self):
        """Test that other CSV files are still processed."""
        include_patterns = normalize_extension_filters(["*.csv"])
        skip_patterns = normalize_extension_filters(["testdataEJ.csv"])

        # Test other CSV files that should NOT be skipped
        other_files = ["data.csv", "report.csv", "summary.csv"]
        for filename in other_files:
            matches_include = matches_extension_filter(filename, include_patterns)
            matches_skip = matches_extension_filter(filename, skip_patterns)

            # File matches include but does NOT match skip
            assert matches_include is True
            assert matches_skip is False

            # Logic: file should be processed
            should_process = matches_include and not matches_skip
            assert should_process is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
