"""Opt-in integration test for the PgBouncer prepared-statement collision.

`ProgrammingError` is deliberately absent from `_DETERMINISTIC_DB_ERRORS` because it
carries 42P05 / 26000, which a transaction-pooling PgBouncer produces against psycopg3's
auto-prepared statements and which recover on retry. The unit tests construct that error;
only this test provokes it, so it is the one that pins reachability.

Skipped unless `PGBOUNCER_DSN` is set, so CI stays green without a pooler. To reconstitute
the rig against the local stack:

    docker run -d --name pgb6587 --network centry_centry -p 6432:5432 \
      -e DB_HOST=centry-postgres-1 -e DB_PORT=5432 \
      -e DB_USER=centry -e DB_PASSWORD=changeme -e DB_NAME=project_2 \
      -e POOL_MODE=transaction -e DEFAULT_POOL_SIZE=1 -e MAX_CLIENT_CONN=50 \
      -e AUTH_TYPE=scram-sha-256 edoburu/pgbouncer:latest
    docker exec pgb6587 sh -c \
      "sed -i 's/^pool_mode = transaction/pool_mode = transaction\\nmax_prepared_statements = 0/' \
       /etc/pgbouncer/pgbouncer.ini"
    docker restart pgb6587

    PGBOUNCER_DSN=postgresql+psycopg://centry:changeme@localhost:6432/project_2 \
      pytest tests/tools/utils/test_6587_pgbouncer_prepared_statement_collision.py

`max_prepared_statements = 0` is load-bearing: PgBouncer 1.25.2's default tracks the
statement names and no collision occurs, so the exposure is pooler-config dependent.

The bound parameter is a vector string because that is what the production statement carries:
`is_server_error_retriable` still answers non-deny-listed errors from the substring fallback, so
a digit-free message would fail on attempt 1 even after this fix. That residual is the deliberate
one pinned by `TestAcceptedResidualGap`; the insert path is not exposed to it because SQLAlchemy
embeds the bound embeddings in every message.

The engine runs in AUTOCOMMIT because transaction mode hands a server connection to one
client for the length of its transaction: with `default_pool_size=1`, two connections
holding SQLAlchemy's implicit transaction deadlock on `query_wait_timeout` instead of
interleaving onto the shared backend.
"""
import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from elitea_sdk.tools.utils.retry import _is_deterministic_db_error, is_server_error_retriable

PGBOUNCER_DSN = os.environ.get("PGBOUNCER_DSN")

pytestmark = pytest.mark.skipif(
    not PGBOUNCER_DSN,
    reason="set PGBOUNCER_DSN to a transaction-pooling PgBouncer with max_prepared_statements=0",
)

_PREPARED_STATEMENT = text("SELECT :embedding")
_BOUND_VECTOR = "[0.500123, -0.429876, 0.502244, -0.503991, 0.504010]"
_EXECUTIONS_PAST_PREPARE_THRESHOLD = 15


def _provoke_collision(engine):
    with engine.connect() as first, engine.connect() as second:
        for _ in range(_EXECUTIONS_PAST_PREPARE_THRESHOLD):
            for connection in (first, second):
                connection.execute(_PREPARED_STATEMENT, {"embedding": _BOUND_VECTOR}).fetchall()


def _pooled_engine():
    return create_engine(
        PGBOUNCER_DSN,
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 10},
    )


@pytest.fixture(name="engine")
def _engine():
    engine = _pooled_engine()
    yield engine
    engine.dispose()


def test_collision_surfaces_as_a_retriable_programming_error(engine):
    with pytest.raises(ProgrammingError) as raised:
        _provoke_collision(engine)

    exception = raised.value
    assert exception.orig.sqlstate == "42P05"
    assert "already exists" in str(exception.orig)
    assert _is_deterministic_db_error(exception) is False
    assert is_server_error_retriable(exception) is True


def test_a_fresh_connection_recovers(engine):
    with pytest.raises(ProgrammingError):
        _provoke_collision(engine)

    with _pooled_engine().connect() as recovered:
        assert recovered.execute(
            _PREPARED_STATEMENT, {"embedding": _BOUND_VECTOR}
        ).fetchall() == [(_BOUND_VECTOR,)]
