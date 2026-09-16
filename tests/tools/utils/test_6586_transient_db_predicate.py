# Copyright (c) 2026 EPAM Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guards `is_transient_db_error`, the retry predicate for the keyed index_meta write.

The write it protects is the terminal one, which runs AFTER promote_run. Refusing to
retry a transient failure there reports a corpus that is already published and
searchable as `failed` — issue #6586's exact symptom, reached through the retry
predicate instead of through an embedding call.

The trap this pins is the same one #6587 documented: behind a transaction-pooling
PgBouncer, psycopg3's auto-prepared statements fail with codes that land on classes
which LOOK deterministic. A bare class allow-list silently excludes all three.
"""

import pytest
from sqlalchemy.exc import (
    DataError,
    IntegrityError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
)

from elitea_sdk.tools.utils.retry import is_transient_db_error

_PSYCOPG_SKIP_REASON = "psycopg3 arrives with the runtime extra, alongside langchain-postgres"
_STMT = "UPDATE langchain_pg_embedding SET cmetadata=(coalesce(cmetadata, %(p)s) || %(q)s)"
# The patch legitimately carries digits that a substring matcher would read as HTTP
# status codes; that is why this predicate must never fall back to one.
_PARAMS = {"p": "{}", "q": '{"state": "completed", "indexed": 500, "updated": 429}'}


def _db_error(error_class, orig):
    return error_class(_STMT, _PARAMS, orig)


class TestPreparedPlanFailuresAreRetried:
    """42P05 / 26000 / 0A000 recover on retry. Two of them are ProgrammingError and
    one is NotSupportedError, so neither class can be excluded wholesale."""

    @pytest.mark.parametrize("wrapper,psycopg_error_name,sqlstate,message", [
        (ProgrammingError, "DuplicatePreparedStatement", "42P05",
         'prepared statement "_pg3_0" already exists'),
        (ProgrammingError, "InvalidSqlStatementName", "26000",
         'prepared statement "_pg3_0" does not exist'),
        (NotSupportedError, "FeatureNotSupported", "0A000",
         "cached plan must not change result type"),
    ])
    def test_the_pooled_prepared_statement_codes_retry(
        self, wrapper, psycopg_error_name, sqlstate, message
    ):
        psycopg_errors = pytest.importorskip("psycopg.errors", reason=_PSYCOPG_SKIP_REASON)
        orig = getattr(psycopg_errors, psycopg_error_name)(message)
        assert orig.sqlstate == sqlstate

        assert is_transient_db_error(_db_error(wrapper, orig)) is True


class TestDeterministicFailuresAreNotRetried:
    """The whole point of not reusing `is_server_error_retriable`: a deterministic
    failure must not burn the retry chain, and the bound parameters above contain
    both "500" and "429"."""

    @pytest.mark.parametrize("error_class", [DataError, IntegrityError])
    def test_the_two_safe_deny_listed_classes_do_not_retry(self, error_class):
        assert is_transient_db_error(_db_error(error_class, Exception("boom"))) is False

    def test_digits_in_the_bound_parameters_do_not_make_it_retriable(self):
        # A ProgrammingError with no recognised SQLSTATE is not retried, even though
        # its rendered parameters carry "500" and "429".
        exception = _db_error(ProgrammingError, Exception("syntax error at or near"))
        assert "500" in str(exception)

        assert is_transient_db_error(exception) is False

    def test_an_unrelated_sqlstate_does_not_retry(self):
        psycopg_errors = pytest.importorskip("psycopg.errors", reason=_PSYCOPG_SKIP_REASON)
        orig = psycopg_errors.UndefinedTable("relation does not exist")
        assert orig.sqlstate == "42P01"

        assert is_transient_db_error(_db_error(ProgrammingError, orig)) is False


class TestConnectionFailuresAreRetried:

    def test_operational_errors_retry(self):
        assert is_transient_db_error(_db_error(OperationalError, Exception("server closed"))) is True

    def test_a_recycled_connection_retries(self):
        exception = _db_error(OperationalError, Exception("connection reset"))
        exception.connection_invalidated = True

        assert is_transient_db_error(exception) is True
