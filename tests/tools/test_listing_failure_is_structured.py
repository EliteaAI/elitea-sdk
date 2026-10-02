"""A failed listing must publish WHAT KIND of failure it was, not just prose.

Two defects sat behind an undiagnosable rate-limited run:

1. The GitHub listing RETURNED its errors as strings, so the provider's exception — and
   with it the type and status classify_tool_error reads — was destroyed before anything
   could classify it. __handle_get_files then fed the string to ast.literal_eval and
   replaced it with "Expected a list of strings", losing even the prose.
2. Nothing published the classification, so a consumer wanting to know whether waiting
   could help had to match substrings. Message text is not a contract: "rate limit
   exceeded", "quota exhausted" and a localised string all mean the same thing and only
   the first matches a naive filter.

The fix is structural rather than textual: raise with __cause__ intact, and publish
error_class/retriable as fields.
"""

import pytest
from github.GithubException import BadCredentialsException, GithubException, RateLimitExceededException
from langchain_core.tools import ToolException

from elitea_sdk.tools.base_indexer_toolkit import build_error_report, describe_failure_cause
from elitea_sdk.tools.github.github_client import GitHubClient


class StubRepo:
    def __init__(self, error): self._error = error
    def get_branch(self, ref): raise self._error
    def get_commit(self, ref): raise self._error


def client_failing_with(error):
    client = GitHubClient.model_construct()
    object.__setattr__(client, "_github_repo_instance", StubRepo(error))
    object.__setattr__(client, "active_branch", "main")
    return client


RATE_LIMIT = RateLimitExceededException(403, {"message": "API rate limit exceeded"},
                                       {"x-ratelimit-remaining": "0"})
NO_ACCESS = GithubException(403, {"message": "Resource not accessible"}, {})
BAD_TOKEN = BadCredentialsException(401, {"message": "Bad credentials"}, {})
MISSING_REF = GithubException(404, {"message": "No commit found"}, {})


class TestTheProviderExceptionSurvivesAsACause:
    """Mutation: return the error as a string again, or drop `from e` from the raise."""

    @pytest.mark.parametrize("inner", [RATE_LIMIT, NO_ACCESS, BAD_TOKEN])
    def test_the_listing_raises_rather_than_returning_text(self, inner):
        with pytest.raises(ToolException):
            client_failing_with(inner)._get_files_with_identity("", "main")

    @pytest.mark.parametrize("inner", [RATE_LIMIT, NO_ACCESS, BAD_TOKEN, MISSING_REF])
    def test_the_original_exception_is_reachable_from_the_raised_one(self, inner):
        with pytest.raises(ToolException) as raised:
            client_failing_with(inner)._get_files_with_identity("", "main")
        causes = []
        current = raised.value
        while current is not None:
            causes.append(type(current))
            current = current.__cause__ or current.__context__
        assert type(inner) in causes


class TestTheKindOfFailureIsClassifiedWithoutReadingText:

    def test_a_rate_limit_is_infrastructure_and_retriable(self):
        """PyGithub's rate limit carries a 403, which would otherwise read as policy."""
        with pytest.raises(ToolException) as raised:
            client_failing_with(RATE_LIMIT)._get_files_with_identity("", "main")
        assert describe_failure_cause(raised.value) == {
            "error_class": "infrastructure", "retriable": True}

    @pytest.mark.parametrize("inner", [NO_ACCESS, BAD_TOKEN])
    def test_a_credential_failure_is_policy_and_not_retriable(self, inner):
        """Waiting cannot fix a credential, and saying it is retriable invites a loop."""
        with pytest.raises(ToolException) as raised:
            client_failing_with(inner)._get_files_with_identity("", "main")
        assert describe_failure_cause(raised.value) == {
            "error_class": "policy", "retriable": False}

    def test_classification_does_not_depend_on_the_message(self):
        """The whole point: reword the provider's text and the class must not move."""
        reworded = RateLimitExceededException(
            403, {"message": "secondary quota exhausted, try later"},
            {"x-ratelimit-remaining": "0"})
        with pytest.raises(ToolException) as raised:
            client_failing_with(reworded)._get_files_with_identity("", "main")
        assert describe_failure_cause(raised.value)["error_class"] == "infrastructure"

    def test_an_unclassifiable_failure_publishes_nothing_rather_than_guessing(self):
        class Mystery(Exception):
            pass
        assert describe_failure_cause(Mystery("???")) == {}


class TestTheReportCarriesTheFields:

    def report_for(self, failure):
        return build_error_report("listing failed", item_labels=("file", "files"),
                                  dependent_labels=("dep", "deps"), failure=failure)

    def test_the_fields_reach_the_report(self):
        """Mutation: stop passing failure= from index_data, or stop merging it in."""
        with pytest.raises(ToolException) as raised:
            client_failing_with(RATE_LIMIT)._get_files_with_identity("", "main")
        report = self.report_for(raised.value)
        assert report["error_class"] == "infrastructure"
        assert report["retriable"] is True

    def test_a_report_without_a_failure_is_unchanged(self):
        report = build_error_report("boom", item_labels=("f", "fs"),
                                    dependent_labels=("d", "ds"))
        assert "error_class" not in report

    def test_the_prose_is_still_there_for_a_human(self):
        report = self.report_for(ToolException("listing failed"))
        assert "listing failed" in str(report.get("errors"))


class TestTheAgentFacingContractIsUnchanged:
    """get_files_from_directory documents 'List of file paths, or an error message', and
    agents read text rather than catching. Mutation: let the raise escape it."""

    def test_an_agent_still_receives_a_string(self):
        result = client_failing_with(RATE_LIMIT).get_files_from_directory("")
        assert isinstance(result, str) and "Error" in result


class TestTheLayerSplitIsWhatMakesBothContractsHold:
    """The client is the agent's tool surface and returns text; the api_wrapper is the
    indexer's adapter and raises. Collapsing the adapter back onto the client's
    _get_files is the obvious 'simplification' that silently restores the old bug, so
    it is pinned here.
    Mutation: make api_wrapper._get_files call client._get_files instead."""

    def wrapper_failing_with(self, inner):
        from elitea_sdk.tools.github.api_wrapper import EliteAGitHubAPIWrapper

        wrapper = EliteAGitHubAPIWrapper.model_construct()
        object.__setattr__(wrapper, "github_client_instance", client_failing_with(inner))
        object.__setattr__(wrapper, "active_branch", "main")
        return wrapper

    def test_the_indexer_adapter_raises(self):
        with pytest.raises(ToolException):
            self.wrapper_failing_with(RATE_LIMIT)._get_files("", "main")

    def test_the_adapter_never_hands_back_a_string(self):
        """A string here is what ast.literal_eval mangled into 'Expected a list of
        strings', destroying the cause for every code toolkit."""
        try:
            result = self.wrapper_failing_with(RATE_LIMIT)._get_files("", "main")
        except ToolException:
            return
        assert not isinstance(result, str)

    def test_the_cause_survives_the_adapter(self):
        with pytest.raises(ToolException) as raised:
            self.wrapper_failing_with(RATE_LIMIT)._get_files("", "main")
        assert describe_failure_cause(raised.value)["error_class"] == "infrastructure"

    def test_the_client_still_serves_agents_text(self):
        assert isinstance(client_failing_with(RATE_LIMIT)._get_files("", "main"), str)
