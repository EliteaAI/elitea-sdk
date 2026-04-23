"""Tests for GitHubClient._apply_patch_to_content hunk application logic."""

from elitea_sdk.tools.github.github_client import GitHubClient


def _make_client():
    return GitHubClient.model_construct(
        github_repository='owner/repo',
        active_branch='feature',
        github_base_branch='main',
        github_api=None,
        elitea=None,
    )


ORIGINAL = "line1\nline2\nline3\nline4\nline5"


class TestAdditionOnlyHunk:
    """Hunks with old_count=0 (pure insertions)."""

    def test_add_lines_after_middle(self):
        """@@ -4,0 +4,3 @$ — insert 3 lines after line 4."""
        client = _make_client()
        hunks = [{
            'old_start': 4,
            'old_count': 0,
            'new_start': 4,
            'new_count': 3,
            'lines': ['+added1', '+added2', '+added3'],
        }]
        result = client._apply_patch_to_content(ORIGINAL, hunks)
        assert result == "line1\nline2\nline3\nline4\nadded1\nadded2\nadded3\nline5"

    def test_add_lines_at_beginning(self):
        """@@ -0,0 +1,2 @$ — insert at very start of file."""
        client = _make_client()
        hunks = [{
            'old_start': 0,
            'old_count': 0,
            'new_start': 1,
            'new_count': 2,
            'lines': ['+header1', '+header2'],
        }]
        result = client._apply_patch_to_content(ORIGINAL, hunks)
        assert result == "header1\nheader2\nline1\nline2\nline3\nline4\nline5"

    def test_add_lines_at_end(self):
        """@@ -5,0 +6,2 $$ — insert after last line."""
        client = _make_client()
        hunks = [{
            'old_start': 5,
            'old_count': 0,
            'new_start': 6,
            'new_count': 2,
            'lines': ['+footer1', '+footer2'],
        }]
        result = client._apply_patch_to_content(ORIGINAL, hunks)
        assert result == "line1\nline2\nline3\nline4\nline5\nfooter1\nfooter2"

    def test_add_single_line(self):
        """@@ -2,0 +3,1 @$ — insert 1 line after line 2."""
        client = _make_client()
        hunks = [{
            'old_start': 2,
            'old_count': 0,
            'new_start': 3,
            'new_count': 1,
            'lines': ['+inserted'],
        }]
        result = client._apply_patch_to_content(ORIGINAL, hunks)
        assert result == "line1\nline2\ninserted\nline3\nline4\nline5"


class TestDeletionHunk:
    """Hunks that remove lines."""

    def test_delete_single_line(self):
        client = _make_client()
        hunks = [{
            'old_start': 3,
            'old_count': 1,
            'new_start': 3,
            'new_count': 0,
            'lines': ['-line3'],
        }]
        result = client._apply_patch_to_content(ORIGINAL, hunks)
        assert result == "line1\nline2\nline4\nline5"

    def test_delete_multiple_lines(self):
        client = _make_client()
        hunks = [{
            'old_start': 2,
            'old_count': 2,
            'new_start': 2,
            'new_count': 0,
            'lines': ['-line2', '-line3'],
        }]
        result = client._apply_patch_to_content(ORIGINAL, hunks)
        assert result == "line1\nline4\nline5"


class TestModifyHunk:
    """Hunks that replace lines (context + deletions + additions)."""

    def test_replace_line_with_context(self):
        client = _make_client()
        hunks = [{
            'old_start': 2,
            'old_count': 3,
            'new_start': 2,
            'new_count': 3,
            'lines': [' line2', '-line3', '+line3_modified', ' line4'],
        }]
        result = client._apply_patch_to_content(ORIGINAL, hunks)
        assert result == "line1\nline2\nline3_modified\nline4\nline5"

    def test_replace_with_more_lines(self):
        client = _make_client()
        hunks = [{
            'old_start': 3,
            'old_count': 1,
            'new_start': 3,
            'new_count': 3,
            'lines': ['-line3', '+line3a', '+line3b', '+line3c'],
        }]
        result = client._apply_patch_to_content(ORIGINAL, hunks)
        assert result == "line1\nline2\nline3a\nline3b\nline3c\nline4\nline5"


class TestMultipleHunks:
    """Multiple hunks in the same patch."""

    def test_two_addition_hunks(self):
        client = _make_client()
        hunks = [
            {
                'old_start': 1,
                'old_count': 0,
                'new_start': 2,
                'new_count': 1,
                'lines': ['+after1'],
            },
            {
                'old_start': 4,
                'old_count': 0,
                'new_start': 6,
                'new_count': 1,
                'lines': ['+after4'],
            },
        ]
        result = client._apply_patch_to_content(ORIGINAL, hunks)
        assert result == "line1\nafter1\nline2\nline3\nline4\nafter4\nline5"

    def test_mixed_addition_and_modify(self):
        client = _make_client()
        hunks = [
            {
                'old_start': 1,
                'old_count': 0,
                'new_start': 1,
                'new_count': 1,
                'lines': ['+prepended'],
            },
            {
                'old_start': 3,
                'old_count': 1,
                'new_start': 4,
                'new_count': 1,
                'lines': ['-line3', '+LINE3'],
            },
        ]
        result = client._apply_patch_to_content(ORIGINAL, hunks)
        # After hunk1: prepend before line1 -> "prepended\nline1\nline2..."
        # Wait, old_start=1, old_count=0 means insert after line 1
        assert result == "line1\nprepended\nline2\nLINE3\nline4\nline5"


class TestEdgeCases:
    """Edge cases for patch application."""

    def test_empty_hunks_returns_original(self):
        client = _make_client()
        result = client._apply_patch_to_content(ORIGINAL, [])
        assert result == ORIGINAL

    def test_empty_content_with_addition(self):
        client = _make_client()
        hunks = [{
            'old_start': 0,
            'old_count': 0,
            'new_start': 1,
            'new_count': 1,
            'lines': ['+first line'],
        }]
        result = client._apply_patch_to_content("", hunks)
        assert result == "first line\n"
