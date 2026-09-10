import httpx
import openai
import pytest
from sqlalchemy.exc import (
    DataError,
    DBAPIError,
    IntegrityError,
    InternalError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
)

_PSYCOPG_SKIP_REASON = "psycopg3 arrives with the runtime extra, alongside langchain-postgres"

from elitea_sdk.tools.utils.retry import _is_deterministic_db_error, is_server_error_retriable

_INSERT_STMT = (
    "INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata) "
    "VALUES (%(id__0)s, %(collection_id__0)s, %(embedding__0)s, %(document__0)s, %(cmetadata__0)s) "
    "ON CONFLICT (id) DO UPDATE SET embedding = excluded.embedding"
)
_BOUND_VECTORS_CONTAINING_STATUS_CODE_DIGITS = {
    "embedding__0": "[0.500123, -0.429876, 0.502244, -0.503991, 0.504010]"
}

_STATUS_CODES = (429, 500, 502, 503, 504)


def assert_substring_path_would_have_retried(exception):
    assert any(str(code) in str(exception) for code in _STATUS_CODES)


def assert_substring_path_would_not_have_retried(exception):
    assert not any(str(code) in str(exception) for code in _STATUS_CODES)


def _insert_error(error_class, orig, **kwargs):
    return error_class(
        _INSERT_STMT,
        _BOUND_VECTORS_CONTAINING_STATUS_CODE_DIGITS,
        orig,
        **kwargs,
    )


def _httpx_response(status_code):
    return httpx.Response(status_code, request=httpx.Request("POST", "https://api.example.com/v1"))


class TestDeterministicDatabaseErrorsFailFast:
    @pytest.mark.parametrize("error_class,orig", [
        (DataError, Exception("unsupported Unicode escape sequence: \\u0000")),
        (DataError, Exception("expected 1536 dimensions, not 3072")),
        (IntegrityError, Exception("duplicate key value violates unique constraint")),
    ])
    def test_deterministic_error_is_not_retriable(self, error_class, orig):
        exception = _insert_error(error_class, orig)
        assert_substring_path_would_have_retried(exception)
        assert is_server_error_retriable(exception) is False


class TestTransientDatabaseErrorsAreNotDenyListed:
    """The deny-list must never widen to a class that carries transient failures.

    is_server_error_retriable still answers these from the substring fallback, so
    asserting only its return value would pass with both guards deleted. The
    classification itself is what these pin.
    """

    @pytest.mark.parametrize("error_class,orig", [
        (OperationalError, Exception("terminating connection due to administrator command")),
        (InternalError, Exception("current transaction is aborted")),
    ])
    def test_transient_error_is_not_classified_as_deterministic(self, error_class, orig):
        assert _is_deterministic_db_error(_insert_error(error_class, orig)) is False

    @pytest.mark.parametrize("error_class,orig", [
        (OperationalError, Exception("terminating connection due to administrator command")),
        (InternalError, Exception("current transaction is aborted")),
    ])
    def test_transient_error_is_retriable(self, error_class, orig):
        exception = _insert_error(error_class, orig)
        assert_substring_path_would_have_retried(exception)
        assert is_server_error_retriable(exception) is True

    @pytest.mark.parametrize("wrapper,psycopg_error_name,sqlstate,message", [
        (ProgrammingError, "DuplicatePreparedStatement", "42P05",
         'prepared statement "_pg3_0" already exists'),
        (ProgrammingError, "InvalidSqlStatementName", "26000",
         'prepared statement "_pg3_0" does not exist'),
        (NotSupportedError, "FeatureNotSupported", "0A000",
         "cached plan must not change result type"),
    ])
    def test_prepared_plan_failures_are_not_classified_as_deterministic(
        self, wrapper, psycopg_error_name, sqlstate, message
    ):
        psycopg_errors = pytest.importorskip("psycopg.errors", reason=_PSYCOPG_SKIP_REASON)
        orig = getattr(psycopg_errors, psycopg_error_name)(message)
        assert orig.sqlstate == sqlstate
        exception = _insert_error(wrapper, orig)
        assert_substring_path_would_have_retried(exception)
        assert _is_deterministic_db_error(exception) is False
        assert is_server_error_retriable(exception) is True

    def test_recycled_connection_is_retriable_without_any_status_code_digits(self):
        exception = DBAPIError(
            None,
            None,
            Exception("server closed the connection unexpectedly"),
            connection_invalidated=True,
        )
        assert_substring_path_would_not_have_retried(exception)
        assert is_server_error_retriable(exception) is True

    def test_recycled_connection_wins_over_a_deterministic_class(self):
        exception = _insert_error(
            DataError,
            Exception("connection recycled mid-statement"),
            connection_invalidated=True,
        )
        assert is_server_error_retriable(exception) is True


class TestDeterministicOperationalErrorKeepsFailingFast:
    def test_bad_password_is_not_newly_retried(self):
        exception = OperationalError(
            "connection to server failed",
            None,
            Exception('FATAL: password authentication failed for user "centry"'),
        )
        assert_substring_path_would_not_have_retried(exception)
        assert is_server_error_retriable(exception) is False


class TestProviderAndFigmaPathsAreUnchanged:
    def test_httpx_status_error_is_retriable(self):
        response = _httpx_response(503)
        exception = httpx.HTTPStatusError("boom", request=response.request, response=response)
        assert is_server_error_retriable(exception) is True

    def test_httpx_remote_protocol_error_is_retriable(self):
        assert is_server_error_retriable(httpx.RemoteProtocolError("closed prematurely")) is True

    def test_openai_status_error_is_retriable(self):
        exception = openai.APIStatusError(
            "rate limited",
            response=_httpx_response(429),
            body=None,
        )
        assert is_server_error_retriable(exception) is True

    @pytest.mark.parametrize("message", [
        "Figma API returned 500 Internal Server Error",
        "rate limit exceeded",
    ])
    def test_wrapped_string_errors_still_match_the_substring_fallback(self, message):
        assert is_server_error_retriable(Exception(message)) is True

    def test_unrelated_error_is_not_retriable(self):
        assert is_server_error_retriable(ValueError("Collection not found")) is False


class TestAcceptedResidualGap:
    """Approach B narrows the substring fallback for DB errors, it does not retire it.

    A transient OperationalError whose message carries no status-code digits and that
    did not invalidate the pooled connection still fails on attempt 1, exactly as it
    does today. Pinned so the gap is visible rather than implicit.
    """

    def test_digitless_transient_error_still_falls_through_unretried(self):
        exception = OperationalError(
            "connection to server failed",
            None,
            Exception("server closed the connection unexpectedly"),
        )
        assert_substring_path_would_not_have_retried(exception)
        assert exception.connection_invalidated is False
        assert is_server_error_retriable(exception) is False


class TestResidualDependsOnRenderedParameters:
    """The insert path escapes the digit lottery only while SQLAlchemy renders bound
    parameters into the message.

    `create_engine(..., hide_parameters=True)` replaces the whole `[parameters: ...]` block
    with a fixed notice, and a transient ProgrammingError on the insert path then reads
    exactly like the digit-free case: not deny-listed, but not retried either. Since
    embeddings and cmetadata are user content, that hardening is plausible -- pinned here so
    the coupling is explicit, and guarded at the engine in
    tests/runtime/langchain/test_6587_add_documents_retry_attempts.py.
    """

    def test_hiding_parameters_makes_a_transient_db_error_unretriable(self):
        exception = ProgrammingError(
            _INSERT_STMT,
            _BOUND_VECTORS_CONTAINING_STATUS_CODE_DIGITS,
            Exception('prepared statement "_pg3_0" already exists'),
            hide_parameters=True,
        )
        assert "hidden due to hide_parameters" in str(exception)
        assert_substring_path_would_not_have_retried(exception)
        assert _is_deterministic_db_error(exception) is False
        assert is_server_error_retriable(exception) is False
