"""Tests for Confluence CQL query escaping and search_by_title CQL construction.

Covers fix for issue #4057: search_by_title fails with special characters in
page titles (brackets, tildes, hyphens, etc.) due to over-escaping.
"""

import pytest
from unittest.mock import MagicMock, patch

from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper


@pytest.fixture
def wrapper():
    """Create a ConfluenceAPIWrapper bypassing Pydantic validation."""
    w = ConfluenceAPIWrapper.model_construct(
        base_url="https://example.atlassian.net",
        space="AT",
        cloud=True,
        limit=25,
        max_pages=50,
        api_version="2",
    )
    w.client = MagicMock()
    return w


class TestEscapeCqlQuery:
    """Tests for _escape_cql_query — only CQL string delimiters should be escaped."""

    def test_plain_text_unchanged(self, wrapper):
        assert wrapper._escape_cql_query("hello world") == "hello world"

    def test_brackets_not_escaped(self, wrapper):
        """Brackets should pass through — they are tokenized by Lucene, not CQL operators."""
        assert wrapper._escape_cql_query("[DO_NOT_DELETE] TC-003") == "[DO_NOT_DELETE] TC-003"

    def test_parentheses_not_escaped(self, wrapper):
        assert wrapper._escape_cql_query("My (Cool) Page") == "My (Cool) Page"

    def test_tilde_not_escaped(self, wrapper):
        """Tilde is a Lucene fuzzy operator — should be left for the user to control."""
        assert wrapper._escape_cql_query("test~") == "test~"

    def test_asterisk_not_escaped(self, wrapper):
        """Asterisk is a Lucene wildcard — should be left for the user to control."""
        assert wrapper._escape_cql_query("win*") == "win*"

    def test_question_mark_not_escaped(self, wrapper):
        """Question mark is a Lucene wildcard — should be left for the user to control."""
        assert wrapper._escape_cql_query("te?t") == "te?t"

    def test_hash_colon_not_escaped(self, wrapper):
        assert wrapper._escape_cql_query("Page #1: Beginning") == "Page #1: Beginning"

    def test_double_quote_escaped(self, wrapper):
        """Double quotes MUST be escaped — they are CQL string delimiters."""
        assert wrapper._escape_cql_query('say "hello"') == 'say \\"hello\\"'

    def test_backslash_escaped(self, wrapper):
        """Backslashes MUST be escaped — they are CQL escape characters."""
        assert wrapper._escape_cql_query("path\\to") == "path\\\\to"

    def test_backslash_before_quote(self, wrapper):
        """Backslash must be escaped first, then quotes, to avoid double-escaping."""
        assert wrapper._escape_cql_query('a\\"b') == 'a\\\\\\"b'

    def test_empty_string(self, wrapper):
        assert wrapper._escape_cql_query("") == ""

    def test_original_bug_title(self, wrapper):
        """The exact title from issue #4057 should pass through unchanged."""
        title = "[DO_NOT_DELETE] TC-003 Page with label"
        assert wrapper._escape_cql_query(title) == title

    def test_mixed_special_chars(self, wrapper):
        """All Lucene operators should pass through, only quotes/backslash escaped."""
        assert wrapper._escape_cql_query('[]{}()^$~#:+*?|') == '[]{}()^$~#:+*?|'


class TestSearchByTitleCql:
    """Tests that search_by_title builds correct CQL queries."""

    def test_cql_with_special_chars_in_title(self, wrapper):
        """CQL should contain unescaped brackets — the original bug scenario."""
        wrapper.client.cql.return_value = {"results": []}

        wrapper.search_by_title("[DO_NOT_DELETE] TC-003 Page with label")

        cql_arg = wrapper.client.cql.call_args[0][0]
        assert 'title~"[DO_NOT_DELETE] TC-003 Page with label"' in cql_arg

    def test_cql_with_tilde_in_title(self, wrapper):
        """Tilde in query should NOT be escaped."""
        wrapper.client.cql.return_value = {"results": []}

        wrapper.search_by_title("test~")

        cql_arg = wrapper.client.cql.call_args[0][0]
        assert 'title~"test~"' in cql_arg

    def test_cql_with_quotes_in_title(self, wrapper):
        """Quotes in the query must be escaped for CQL safety."""
        wrapper.client.cql.return_value = {"results": []}

        wrapper.search_by_title('say "hello"')

        cql_arg = wrapper.client.cql.call_args[0][0]
        assert 'title~"say \\"hello\\""' in cql_arg

    def test_cql_includes_space_when_set(self, wrapper):
        wrapper.client.cql.return_value = {"results": []}

        wrapper.search_by_title("test page")

        cql_arg = wrapper.client.cql.call_args[0][0]
        assert 'space="AT"' in cql_arg
        assert 'title~"test page"' in cql_arg

    def test_cql_no_space_when_unset(self, wrapper):
        wrapper.space = None
        wrapper.client.cql.return_value = {"results": []}

        wrapper.search_by_title("test page")

        cql_arg = wrapper.client.cql.call_args[0][0]
        assert 'space=' not in cql_arg
        assert '(type=page) and (title~"test page")' == cql_arg


class TestSiteSearchCql:
    """Tests that site_search also uses the fixed escaping."""

    def test_site_search_brackets_not_escaped(self, wrapper):
        wrapper.client.cql.return_value = {"results": []}

        wrapper.site_search("[DO_NOT_DELETE] test")

        cql_arg = wrapper.client.cql.call_args[0][0]
        assert 'siteSearch~"[DO_NOT_DELETE] test"' in cql_arg

    def test_site_search_quotes_escaped(self, wrapper):
        wrapper.client.cql.return_value = {"results": []}

        wrapper.site_search('say "hi"')

        cql_arg = wrapper.client.cql.call_args[0][0]
        assert 'siteSearch~"say \\"hi\\""' in cql_arg
