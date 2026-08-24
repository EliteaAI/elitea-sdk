"""Unit tests for the ToolOutcome envelope and classify_tool_error (issue #6169).

No middleware here — these pin the classifier's contract in isolation.
"""
import ast
import inspect

import pytest
from langchain_core.tools import ToolException
from pydantic import BaseModel, ValidationError

from elitea_sdk.runtime.tool_outcome import (
    ToolErrorClass,
    ToolOutcome,
    ToolResultStatus,
    classify_tool_error,
    retriable_for,
)


class TestClassifyByType:
    @pytest.mark.parametrize("exc,expected", [
        (TimeoutError("took too long"), ToolErrorClass.INFRASTRUCTURE),
        (ConnectionError("connection refused"), ToolErrorClass.INFRASTRUCTURE),
        (ConnectionResetError("reset by peer"), ToolErrorClass.INFRASTRUCTURE),
        (ValueError("bad value"), ToolErrorClass.INPUT),
        (TypeError("bad type"), ToolErrorClass.INPUT),
        (AttributeError("no such attr"), ToolErrorClass.TOOL_INTERNAL),
        (KeyError("missing"), ToolErrorClass.TOOL_INTERNAL),
        (PermissionError("nope"), ToolErrorClass.POLICY),
    ])
    def test_builtin_types(self, exc, expected):
        assert classify_tool_error(exc) is expected

    def test_pydantic_validation_error_is_input(self):
        """ValidationError subclasses ValueError, so the INPUT mapping covers it for free."""

        class Args(BaseModel):
            n: int

        with pytest.raises(ValidationError) as caught:
            Args(n="not-an-int")

        assert classify_tool_error(caught.value) is ToolErrorClass.INPUT

    def test_third_party_type_matched_by_name_without_importing_it(self):
        """httpx/openai/requests need not be installed — matching is on MRO class names."""

        class ConnectTimeout(Exception):
            pass

        assert classify_tool_error(ConnectTimeout("api.example.com")) is ToolErrorClass.INFRASTRUCTURE

    def test_rate_limit_is_infrastructure_not_policy(self):
        """A 429 is transient: retrying identical input does help, so retriable must be True."""

        class RateLimitError(Exception):
            pass

        error_class = classify_tool_error(RateLimitError("slow down"))
        assert error_class is ToolErrorClass.INFRASTRUCTURE
        assert retriable_for(error_class) is True

    def test_unknown_exception_is_unclassified(self):
        assert classify_tool_error(Exception("something happened")) is None
        assert retriable_for(None) is False


class TestClassifyByHttpStatus:
    """Structured signal that per-library class names cannot cover.

    The name tables were written from the httpx/openai/requests vocabulary, so PyGithub's
    BadCredentialsException went unclassified in production — and extending them one
    library at a time does not scale. Any exception carrying an HTTP status is
    classifiable regardless of what its class happens to be called.
    """

    @pytest.mark.parametrize("status,expected", [
        (401, ToolErrorClass.POLICY),
        (403, ToolErrorClass.POLICY),
        (429, ToolErrorClass.INFRASTRUCTURE),
        (500, ToolErrorClass.INFRASTRUCTURE),
        (503, ToolErrorClass.INFRASTRUCTURE),
    ])
    def test_status_attribute_is_read(self, status, expected):
        class ApiError(Exception):
            def __init__(self, status):
                self.status = status

        assert classify_tool_error(ApiError(status)) is expected

    def test_status_code_on_a_wrapped_response(self):
        """The httpx/requests shape: the code lives on exc.response.status_code."""

        class _Response:
            status_code = 403

        class HTTPStatusError(Exception):
            response = _Response()

        assert classify_tool_error(HTTPStatusError("nope")) is ToolErrorClass.POLICY

    @pytest.mark.parametrize("status", [400, 404, 422])
    def test_ambiguous_client_statuses_stay_unclassified(self, status):
        """A 404 is as often a permissions artifact — GitHub hides private repos behind
        one — as it is a bad argument, so guessing INPUT here would be a wrong label."""

        class ApiError(Exception):
            def __init__(self, status):
                self.status = status

        assert classify_tool_error(ApiError(status)) is None

    def test_int_attribute_that_is_not_a_status_code_is_ignored(self):
        """`status` is a common attribute name; only plausible codes may be read as one."""

        class JobFailed(Exception):
            status = 3

        assert classify_tool_error(JobFailed("job did not finish")) is None

    def test_bool_status_is_not_a_code(self):
        """bool subclasses int, so a `status=True` flag would otherwise read as 1."""

        class Flagged(Exception):
            status = True

        assert classify_tool_error(Flagged("flagged")) is None

    def test_a_raising_property_does_not_escape_the_classifier(self):
        """The classifier runs on the error path over arbitrary third-party exceptions.
        If probing one for a status could raise, a handled tool error would become an
        unhandled crash — strictly worse than the unclassified result it replaces."""

        class Hostile(Exception):
            @property
            def status(self):
                raise RuntimeError("property exploded")

            @property
            def response(self):
                raise RuntimeError("property exploded")

        assert classify_tool_error(Hostile("boom")) is None

    def test_rate_limit_name_beats_its_403_status(self):
        """PyGithub reports rate limiting as a 403, which would otherwise land in POLICY
        and publish retriable=False for the one error class that genuinely is retriable."""

        class RateLimitExceededException(Exception):
            status = 403

        error_class = classify_tool_error(RateLimitExceededException("API rate limit exceeded"))
        assert error_class is ToolErrorClass.INFRASTRUCTURE
        assert retriable_for(error_class) is True


class TestGithubBadCredentialsRegression:
    """The live failure #6169's first cut missed, reproduced at the exact shape.

    A pipeline node calling github create_file with broken credentials published
    error_class=None. The chain walk did reach PyGithub's BadCredentialsException via
    __context__; the classifier simply could not recognise it. Note neither assertion
    below can be satisfied by the string fallback — GitHub words a 401 as "Bad
    credentials", and _POLICY_PHRASES has "invalid credentials", not that. So POLICY here
    can only come from the status carried on the chained exception.
    """

    @staticmethod
    def _wrap_like_create_file(inner):
        """github_client.create_file's verbatim shape: interpolated, and no `from e`."""
        try:
            try:
                raise inner
            except Exception as e:
                raise ToolException(f"Unable to create file due to error:\n{str(e)}")
        except ToolException as wrapped:
            return wrapped

    def test_hermetic_401_chain_is_policy(self):
        class BadCredentialsException(Exception):
            status = 401

        wrapped = self._wrap_like_create_file(
            BadCredentialsException('401 {"message": "Bad credentials", "status": "401"}')
        )

        error_class = classify_tool_error(wrapped)
        assert error_class is ToolErrorClass.POLICY
        assert retriable_for(error_class) is False

    def test_real_pygithub_exception_is_policy(self):
        """Pins the assumption the hermetic test rests on: the real library does carry
        .status, so the hermetic stand-in is not quietly testing a fiction."""
        github_exceptions = pytest.importorskip("github.GithubException")

        wrapped = self._wrap_like_create_file(
            github_exceptions.BadCredentialsException(401, {"message": "Bad credentials"}, None)
        )

        assert classify_tool_error(wrapped) is ToolErrorClass.POLICY


class TestClassifyByChain:
    """The reason the classifier is not decorative.

    ~800 sites across the SDK do `raise ToolException(f"... {e}")` from inside an except
    block. That loses the original type from the string, but Python still sets
    __context__ implicitly — only a couple of dozen sites use an explicit `from e`. Drop
    the chain walk and nearly every real failure degrades to unclassified.
    """

    def test_implicit_context_is_walked_without_from_e(self):
        try:
            try:
                raise ConnectionError("connection refused")
            except ConnectionError as e:
                # The dominant SDK idiom: no `from e`, type survives only via __context__.
                raise ToolException(f"ServiceNow tool exception. {e}")
        except ToolException as wrapped:
            assert classify_tool_error(wrapped) is ToolErrorClass.INFRASTRUCTURE

    def test_explicit_cause_is_walked(self):
        try:
            try:
                raise TimeoutError("deadline exceeded")
            except TimeoutError as e:
                raise ToolException("wrapped") from e
        except ToolException as wrapped:
            assert classify_tool_error(wrapped) is ToolErrorClass.INFRASTRUCTURE

    def test_outer_type_wins_over_chain(self):
        try:
            try:
                raise ConnectionError("connection refused")
            except ConnectionError as e:
                raise ValueError("caller passed a malformed id") from e
        except ValueError as outer:
            assert classify_tool_error(outer) is ToolErrorClass.INPUT

    def test_chain_walk_is_depth_capped(self):
        """A chain deeper than the cap is a re-wrap we do not model; stop rather than dig."""
        exc = ConnectionError("connection refused")
        for _ in range(5):
            wrapper = ToolException("rewrapped")
            wrapper.__cause__ = exc
            exc = wrapper

        # "connection refused" never reaches the string fallback: the message at each
        # level is a bare "rewrapped", so an uncapped walk is the only thing that could
        # classify this.
        assert classify_tool_error(exc) is None


class TestClassifyByStringFallback:
    @pytest.mark.parametrize("message,expected", [
        ("Request to https://api.example.com timed out", ToolErrorClass.INFRASTRUCTURE),
        ("Rate limit exceeded, retry shortly", ToolErrorClass.INFRASTRUCTURE),
        ("401 Unauthorized", ToolErrorClass.POLICY),
        ("Permission denied for this repository", ToolErrorClass.POLICY),
    ])
    def test_narrow_phrases(self, message, expected):
        assert classify_tool_error(ToolException(message)) is expected

    @pytest.mark.parametrize("message", [
        "Issue PROJ-400 could not be updated",
        "Payload was 400 bytes, expected 500",
        "400 Bad Request: field 'title' is required",
    ])
    def test_bare_status_codes_are_not_treated_as_signal(self, message):
        """tools/utils/retry.py returns retriable for any string containing "400"/"500".

        That is acceptable in a retry gate — a wasted retry costs one call — but wrong in
        a published field a consumer branches on, so the classifier must not copy it.
        """
        assert classify_tool_error(ToolException(message)) is None


class TestRetriableDerivation:
    @pytest.mark.parametrize("error_class,expected", [
        (ToolErrorClass.INFRASTRUCTURE, True),
        (ToolErrorClass.INPUT, False),
        (ToolErrorClass.POLICY, False),
        (ToolErrorClass.TOOL_INTERNAL, False),
        (None, False),
    ])
    def test_retriable_is_a_function_of_error_class(self, error_class, expected):
        assert retriable_for(error_class) is expected

    def test_module_does_not_depend_on_the_retry_predicates(self):
        """Documents the boundary: no behavioural coupling to tools/utils/retry.py.

        Those predicates return retriable for any exception whose string contains "400"
        or a 5xx code, which is fine as a retry gate and wrong as a contract field.
        """
        import elitea_sdk.runtime.tool_outcome as module

        tree = ast.parse(inspect.getsource(module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")

        assert not [name for name in imported if "retry" in name]


class TestToolOutcome:
    def test_only_status_and_message_are_required(self):
        outcome = ToolOutcome(status=ToolResultStatus.ERROR, message="boom")

        assert outcome.error_class is None
        assert outcome.retriable is False
        assert outcome.truncated is False
        assert outcome.retry_after is None

    def test_status_serialises_as_its_string_value(self):
        outcome = ToolOutcome(status=ToolResultStatus.ERROR, message="boom",
                              error_class=ToolErrorClass.INFRASTRUCTURE)

        dumped = outcome.model_dump(mode="json")
        assert dumped["status"] == "error"
        assert dumped["error_class"] == "infrastructure"
