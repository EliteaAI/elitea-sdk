import logging

from sqlalchemy import CheckConstraint, Column, Double, Index, MetaData, String, inspect
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

# Twin contract: elitea_core mirrors this model beside EmbeddingStore in
# models/indexer.py. Bump both together. v1 columns are FROZEN — create_all never
# ALTERs an existing table, so any incompatible change ships as an admin migration
# task iterating configured connection strings, never as an edit here.
ELITEA_INDEX_RUNS_SCHEMA_VERSION = 1

INDEX_RUNS_TABLE_NAME = "elitea_index_runs"

RUN_STATUS_PENDING = "pending"
RUN_STATUS_CANCELLED = "cancelled"
RUN_STATUS_PROMOTED = "promoted"
RUN_STATUS_DISCARDED = "discarded"

RUN_STATUSES = (
    RUN_STATUS_PENDING,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_PROMOTED,
    RUN_STATUS_DISCARDED,
)

INDEX_RUNS_V1_COLUMNS = frozenset({
    "run_id",
    "collection",
    "status",
    "task_id",
    "started_on",
    "heartbeat",
    "promoted_on",
})

Base = declarative_base()


class IndexRun(Base):
    # Declared with schema=None so DML resolves through the vectorstore engine's
    # schema_translate_map into the toolkit schema; DDL goes through the
    # schema-qualified copy in ensure_index_runs_table.
    __tablename__ = INDEX_RUNS_TABLE_NAME

    run_id = Column(String, primary_key=True)
    collection = Column(String, nullable=False)
    status = Column(
        String,
        CheckConstraint(
            "status IN ({})".format(", ".join(f"'{status}'" for status in RUN_STATUSES)),
            name="ck_elitea_index_runs_status",
        ),
        nullable=False,
        default=RUN_STATUS_PENDING,
        server_default=RUN_STATUS_PENDING,
    )
    task_id = Column(String, nullable=True)
    started_on = Column(Double, nullable=False)
    heartbeat = Column(Double, nullable=False)
    promoted_on = Column(Double, nullable=True)


def live_run_where():
    # Registration's ON CONFLICT must name the exact predicate the partial unique
    # index was built from — a mismatched index_where raises "no unique or
    # exclusion constraint matching" on EVERY insert, so both sides derive it
    # from this one function.
    return IndexRun.status == RUN_STATUS_PENDING


Index(
    "uq_elitea_index_runs_live",
    IndexRun.collection,
    unique=True,
    postgresql_where=live_run_where(),
)
Index("ix_elitea_index_runs_collection", IndexRun.collection)


def _exception_sqlstate(exc: Exception):
    orig = getattr(exc, "orig", None)
    return getattr(orig, "sqlstate", None) or getattr(
        getattr(orig, "diag", None), "sqlstate", None
    )


def _is_duplicate_object_error(exc: Exception) -> bool:
    # 42P07/42710: relation/object already exists; 23505: the pg-catalog unique
    # violation two concurrent CREATEs race into (pg_type/pg_class rows).
    return _exception_sqlstate(exc) in ("42P07", "42710", "23505")


def is_undefined_table_error(exc: Exception) -> bool:
    return _exception_sqlstate(exc) == "42P01"


def is_degradable_run_lookup_error(exc: Exception) -> bool:
    """Whether a failed pending-run lookup may degrade to an unfiltered read.

    Keyed on sqlstate, never on exception class: a missing table is the only
    fail-open case, and the DBAPI classes it arrives in also carry statement
    timeouts, deadlocks, serialization failures and permission errors. Widening
    to those would answer a transient failure with an unfiltered search, which
    is exactly how an in-flight run's staged rows reach a reader.
    """
    return is_undefined_table_error(exc)


def ensure_index_runs_table(engine, schema: str) -> None:
    """Provision the per-schema runs table. Write path only — read paths never
    ensure and treat a missing table as an empty pending set."""
    ddl_metadata = MetaData()
    IndexRun.__table__.to_metadata(ddl_metadata, schema=schema)
    # create_all's check-then-CREATE races on pg catalogs when two workers
    # provision one fresh schema; the loser retries (picking up whatever the
    # winner has not created yet) and then rechecks instead of failing the run.
    for attempt in (1, 2):
        try:
            ddl_metadata.create_all(engine, checkfirst=True)
            break
        except DBAPIError as exc:
            if not _is_duplicate_object_error(exc):
                raise
            if attempt == 2 and not inspect(engine).has_table(
                INDEX_RUNS_TABLE_NAME, schema=schema
            ):
                raise
    assert_index_runs_columns(engine, schema)


def assert_index_runs_columns(engine, schema: str) -> None:
    columns = {
        column["name"]
        for column in inspect(engine).get_columns(INDEX_RUNS_TABLE_NAME, schema=schema)
    }
    missing = INDEX_RUNS_V1_COLUMNS - columns
    if missing:
        raise RuntimeError(
            f"{INDEX_RUNS_TABLE_NAME} in schema '{schema}' is missing v{ELITEA_INDEX_RUNS_SCHEMA_VERSION} "
            f"columns {sorted(missing)}; refusing to run against a drifted table"
        )
