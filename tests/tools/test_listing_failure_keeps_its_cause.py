"""A failed repository listing must surface the provider's own words.

Every code toolkit — GitHub, GitLab, ADO Repos, Bitbucket — reports a failed listing by
RETURNING a string instead of a file list, and that convention is deliberate (see
test_6645_provider_identities). The defect was downstream: __handle_get_files fed the
string to ast.literal_eval and, when that failed, replaced it with the fixed sentence
"Expected a list of strings, but got a string that cannot be converted".

So a rate limit, an expired credential and a missing branch all arrived identically, with
the one description of the cause thrown away. This is toolkit- and status-agnostic on
purpose: 403, 429 and anything else a provider invents all read through.
"""

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.code_indexer_toolkit import (
    LISTING_FAILURE_EXCERPT_LIMIT,
    CodeIndexerToolkit,
    summarize_listing_failure,
)


def listing(reported):
    """Drive the real seam with whatever a loader handed back."""
    toolkit = CodeIndexerToolkit.model_construct()
    handle = getattr(toolkit, "_CodeIndexerToolkit__handle_get_files")
    return handle("", "main", prefetched=reported)


class TestTheProvidersWordsSurvive:

    @pytest.mark.parametrize("reported", [
        "Error: status code 403, rate limit exceeded",
        "429 Too Many Requests",
        "Error: Could not resolve ref 'nope': No commit found",
        "Error fetching files: Bad credentials",
        "TF401019: The Git repository does not exist or you do not have permissions",
    ])
    def test_the_reported_cause_reaches_the_caller(self, reported):
        """Mutation: raise the fixed 'Expected a list of strings' sentence again."""
        with pytest.raises(ToolException) as raised:
            listing(reported)
        assert reported[:40] in str(raised.value)

    def test_the_old_fixed_sentence_is_gone(self):
        with pytest.raises(ToolException) as raised:
            listing("Error: status code 429, too many requests")
        assert "Expected a list of strings" not in str(raised.value)


class TestLegitimateListingsStillWork:

    def test_a_real_list_passes_through(self):
        assert listing(["a.py", "b.py"]) == ["a.py", "b.py"]

    def test_a_stringified_list_is_still_evaluated(self):
        """Some loaders hand back a repr of the list; that path predates this change."""
        assert listing("['a.py', 'b.py']") == ["a.py", "b.py"]

    def test_an_empty_list_is_not_a_failure(self):
        assert listing([]) == []

    def test_a_list_of_non_strings_is_still_rejected(self):
        with pytest.raises((ValueError, ToolException)):
            listing([1, 2, 3])


class TestTheMessageIsBounded:

    def test_a_page_of_html_is_truncated(self):
        """A loader may hand back a whole error page; the run's error field is not a log."""
        message = summarize_listing_failure("<html>" + "x" * 5000)
        assert len(message) < LISTING_FAILURE_EXCERPT_LIMIT + 120
        assert message.endswith("...")

    def test_a_short_message_is_not_truncated(self):
        assert summarize_listing_failure("429 Too Many Requests").endswith("Requests")

    def test_an_empty_report_still_says_something(self):
        """Mutation: return the empty string, leaving the user with a blank error."""
        assert summarize_listing_failure("").strip() != ""
        assert summarize_listing_failure(None).strip() != ""
