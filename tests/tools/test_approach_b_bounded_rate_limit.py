"""Approach B: a long rate-limit reset is reported, never slept through.

PyGithub's default retry sleeps until a primary rate limit resets — measured at 2140s
for an anonymous credential — which is why a rate-limited run went silent with no error
and no progress. The bound is on the WAIT, not the retry count, so transient failures
keep retrying normally.
"""

from github import GithubRetry
from github.GithubException import GithubException, RateLimitExceededException

from elitea_sdk.tools.github.github_client import (
    GITHUB_MAX_RATE_LIMIT_SLEEP_SECONDS,
    BoundedGithubRetry,
)


class FakeResponse:
    status = 403
    headers = {"x-ratelimit-remaining": "0", "x-ratelimit-limit": "60",
               "x-ratelimit-reset": "9999999999"}
    # The body is what createException reads to pick RateLimitExceededException over a
    # plain GithubException, so a fake without one cannot exercise the typing at all.
    data = b'{"message": "API rate limit exceeded"}' 


class StubRetry:
    def __init__(self, wait): self._wait = wait
    def get_backoff_time(self): return self._wait


def increment_with(monkeypatch, intended_wait, response=FakeResponse()):
    monkeypatch.setattr(GithubRetry, "increment",
                        lambda self, *a, **kw: StubRetry(intended_wait))
    return BoundedGithubRetry(total=10).increment(response=response)


class TestALongWaitIsReportedNotSlept:

    def test_an_hour_long_reset_raises(self, monkeypatch):
        """Mutation: return the retry instead of raising past the bound."""
        import pytest
        with pytest.raises(RateLimitExceededException):
            increment_with(monkeypatch, 2140.0)

    def test_the_raised_error_carries_the_quota_headers(self, monkeypatch):
        import pytest
        with pytest.raises(GithubException) as raised:
            increment_with(monkeypatch, 2140.0)
        headers = {k.lower(): v for k, v in (raised.value.headers or {}).items()}
        assert headers["x-ratelimit-remaining"] == "0"
        assert headers["x-ratelimit-limit"] == "60"


class TestShortWaitsAndTransientErrorsStillRetry:

    def test_a_wait_inside_the_bound_is_left_alone(self, monkeypatch):
        """A brief secondary-rate pause is a legitimate retry, not an outage."""
        retry = increment_with(monkeypatch, 5.0)
        assert retry.get_backoff_time() == 5.0

    def test_a_wait_exactly_at_the_bound_is_left_alone(self, monkeypatch):
        retry = increment_with(monkeypatch, GITHUB_MAX_RATE_LIMIT_SLEEP_SECONDS)
        assert retry.get_backoff_time() == GITHUB_MAX_RATE_LIMIT_SLEEP_SECONDS

    def test_a_zero_backoff_is_left_alone(self, monkeypatch):
        retry = increment_with(monkeypatch, 0)
        assert retry.get_backoff_time() == 0

    def test_a_connection_error_with_no_response_is_left_alone(self, monkeypatch):
        """error= paths carry no response; the bound must not touch them."""
        retry = increment_with(monkeypatch, 9999.0, response=None)
        assert retry.get_backoff_time() == 9999.0

    def test_an_unreadable_backoff_is_left_alone(self, monkeypatch):
        class Odd:
            def get_backoff_time(self): raise RuntimeError("no idea")
        monkeypatch.setattr(GithubRetry, "increment", lambda self, *a, **kw: Odd())
        assert BoundedGithubRetry(total=10).increment(response=FakeResponse()) is not None


class TestTheBoundIsSane:

    def test_it_is_short_enough_to_report_promptly(self):
        assert GITHUB_MAX_RATE_LIMIT_SLEEP_SECONDS <= 120

    def test_it_is_long_enough_to_absorb_a_secondary_rate_pause(self):
        """PyGithub's own secondary_rate_wait default is 60s."""
        assert GITHUB_MAX_RATE_LIMIT_SLEEP_SECONDS >= 60


class TestEveryClientActuallyGetsTheBound:
    """The class being correct is worthless if a constructor misses it, and a new auth
    branch is exactly where it would be missed. Checked by AST rather than by substring
    so a commented-out or differently-formatted call cannot fool it.
    Mutation: drop retry=BoundedGithubRetry(...) from any Github()/GithubIntegration()."""

    CLIENT_FACTORIES = {"Github", "GithubIntegration"}

    def client_constructions(self):
        import ast
        import inspect

        from elitea_sdk.tools.github import github_client

        tree = ast.parse(inspect.getsource(github_client))
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in self.CLIENT_FACTORIES:
                found.append({kw.arg for kw in node.keywords})
        return found

    def test_at_least_one_client_is_constructed(self):
        """Guards the guard: an empty scan would make every assertion below vacuous."""
        assert len(self.client_constructions()) >= 3

    def test_every_client_passes_a_retry_policy(self):
        for keywords in self.client_constructions():
            assert "retry" in keywords

    def test_every_client_is_paced(self):
        for keywords in self.client_constructions():
            assert "seconds_between_requests" in keywords
