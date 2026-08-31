import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from logging import getLogger

from ...runtime.utils.utils import IndexerKeywords

logger = getLogger(__name__)

PROMOTE_DELETE_BATCH_SIZE = 50000
# Caps how many strands one sweep reclaims. The candidate read returns only
# discarded runs whose chunks actually survived, so the cap bounds the deletes of
# a single index run without ever excluding a strand: whatever it leaves behind
# is still a candidate for the next sweep.
STRANDED_RECLAIM_RUN_LIMIT = 50


class VectorStoreAdapter(ABC):
    """Abstract base class for vector store adapters."""

    # Adapters without run staging keep the pre-run-coordination indexing
    # lifecycle: immediate dedup deletes, pre-load clean, no promote contract.
    supports_run_staging = False

    @abstractmethod
    def get_vectorstore_params(self, collection_name: str, connection_string: Optional[str] = None) -> Dict[str, Any]:
        """Get vector store specific parameters."""
        pass

    @abstractmethod
    def list_collections(self, vectorstore_wrapper) -> str:
        """List all collections in the vector store."""
        pass

    @abstractmethod
    def remove_collection(self, vectorstore_wrapper, collection_name: str):
        """Remove a collection from the vector store."""
        pass

    @abstractmethod
    def get_indexed_ids(self, vectorstore_wrapper, index_name: Optional[str] = '') -> List[str]:
        """Get all indexed document IDs from vectorstore"""
        pass

    @abstractmethod
    def clean_collection(self, vectorstore_wrapper, index_name: str = '', including_index_meta: bool = False) -> int:
        """Clean the vectorstore collection by deleting all indexed data. If including_index_meta is True, skip the index_meta records.

        Returns:
            int: Number of deleted records.
        """
        pass

    @abstractmethod
    def get_indexed_data(self, vectorstore_wrapper):
        """Get all indexed data from vectorstore for non-code content"""
        pass

    @abstractmethod
    def get_code_indexed_data(self, vectorstore_wrapper, index_name) -> Dict[str, Dict[str, Any]]:
        """Get all indexed data from vectorstore for code content"""
        pass

    @abstractmethod
    def add_to_collection(self, vectorstore_wrapper, entry_id, new_collection_value):
        """Add a new collection name to the metadata"""
        pass

    @abstractmethod
    def get_index_meta(self, vectorstore_wrapper, index_name: str) -> List[Dict[str, Any]]:
        """Get all index_meta entries from the vector store."""
        pass

    @abstractmethod
    def promote_run(self, vectorstore_wrapper, index_name: str, run_id: str,
                    superseded_ids: List[str], orphan_ids: List[str],
                    damaged_ids: List[str]) -> str:
        """Publish a run's staged rows atomically. Returns the promote outcome."""
        pass

    @abstractmethod
    def discard_run(self, vectorstore_wrapper, index_name: str, run_id: str) -> str:
        """Delete a run's staged rows and terminal its run row."""
        pass

    def get_pending_run_ids(self, vectorstore_wrapper, index_name: str,
                            include_cancelled: bool = True) -> List[str]:
        """Run ids whose rows must stay invisible to every read path."""
        return []

    def ensure_index_runs_table(self, vectorstore_wrapper) -> None:
        raise NotImplementedError("Run staging is not supported by this adapter")

    def register_index_run(self, vectorstore_wrapper, index_name: str, run_id: str,
                           task_id: Optional[str] = None,
                           meta_lock_id: Optional[str] = None):
        raise NotImplementedError("Run staging is not supported by this adapter")

    def heartbeat_index_run(self, vectorstore_wrapper, index_name: str, run_id: str,
                            meta_id: Optional[str]) -> None:
        raise NotImplementedError("Run staging is not supported by this adapter")

    def sweep_stale_index_runs(self, vectorstore_wrapper, index_name: str,
                               stale_before: float) -> List[str]:
        raise NotImplementedError("Run staging is not supported by this adapter")


class PGVectorAdapter(VectorStoreAdapter):
    """Adapter for PGVector database operations."""

    supports_run_staging = True

    def get_vectorstore_params(self, collection_name: str, connection_string: Optional[str] = None) -> Dict[str, Any]:
        try:
            from tools import this  # pylint: disable=E0401,C0415
            worker_config = this.for_module("indexer_worker").descriptor.config
        except:  # pylint: disable=W0702
            worker_config = {}
        #
        return {
            "use_jsonb": True,
            "collection_name": collection_name,
            "create_extension": worker_config.get("pgvector_create_extension", True),
            "elitea_sdk_options": {
                "target_schema": collection_name,
            },
            "connection_string": connection_string
        }

    def list_collections(self, vectorstore_wrapper) -> str:
        from sqlalchemy import func
        from sqlalchemy.orm import Session

        store = vectorstore_wrapper.vectorstore
        try:
            with Session(store.session_maker.bind) as session:
                collections = (
                    session.query(
                        func.distinct(func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'collection'))
                    )
                    .filter(store.EmbeddingStore.cmetadata.isnot(None))
                    .all()
                )
                return [collection[0] for collection in collections if collection[0] is not None]
        except Exception as e:
            logger.error(f"Failed to get unique collections from PGVector: {str(e)}")
            return []

    def remove_collection(self, vectorstore_wrapper, collection_name: str):
        from sqlalchemy import text
        from sqlalchemy.orm import Session

        schema_name = vectorstore_wrapper.vectorstore.collection_name
        with Session(vectorstore_wrapper.vectorstore.session_maker.bind) as session:
            drop_schema_query = text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")
            session.execute(drop_schema_query)
            session.commit()
            logger.info(f"Schema '{schema_name}' has been dropped.")

    def get_indexed_ids(self, vectorstore_wrapper, index_name: Optional[str] = '') -> List[str]:
        """Get all indexed document IDs from PGVector"""
        from sqlalchemy.orm import Session
        from sqlalchemy import func, or_

        store = vectorstore_wrapper.vectorstore
        try:
            with Session(store.session_maker.bind) as session:
                # Start building the query
                query = session.query(store.EmbeddingStore.id)
                # Apply filter only if index_name is provided
                if index_name:
                    query = query.filter(
                        func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'collection') == index_name,
                        or_(
                            func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'type').is_(None),
                            func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata,
                                                         'type') != IndexerKeywords.INDEX_META_TYPE.value
                        )
                    )
                ids = query.all()
                return [str(id_tuple[0]) for id_tuple in ids]
        except Exception as e:
            logger.error(f"Failed to get indexed IDs from PGVector: {str(e)}")
            return []

    def clean_collection(self, vectorstore_wrapper, index_name: str = '', including_index_meta: bool = False) -> int:
        """Clean the vectorstore collection by deleting all indexed data. If including_index_meta is True, skip the index_meta records."""
        from sqlalchemy.orm import Session
        from sqlalchemy import func, or_
        store = vectorstore_wrapper.vectorstore
        with Session(store.session_maker.bind) as session:
            if including_index_meta:
                deleted_count = session.query(store.EmbeddingStore).filter(
                    func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'collection') == index_name
                ).delete(synchronize_session=False)
            else:
                deleted_count = session.query(store.EmbeddingStore).filter(
                    func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'collection') == index_name,
                    or_(func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'type').is_(None),
                        func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'type') != IndexerKeywords.INDEX_META_TYPE.value)
                ).delete(synchronize_session=False)
            session.commit()
        if including_index_meta and index_name:
            try:
                self._delete_index_runs(vectorstore_wrapper, index_name)
            except Exception as exc:
                # Bookkeeping after an already committed destructive delete:
                # the chunks are gone, so failing the removal here would report
                # a failure for work that succeeded and skip its removal event.
                logger.warning(f"Failed to delete run rows of index '{index_name}': {exc}")
        return deleted_count

    def is_vectorstore_type(self, vectorstore) -> bool:
        """Check if the vectorstore is a PGVector store."""
        return hasattr(vectorstore, 'session_maker') and hasattr(vectorstore, 'EmbeddingStore')

    def get_indexed_data(self, vectorstore_wrapper, index_name: str)-> Dict[str, Dict[str, Any]]:
        """Get all indexed data from PGVector for non-code content per index_name."""
        from sqlalchemy.orm import Session
        from sqlalchemy import func
        from ...runtime.tools.index_runs_model import is_undefined_table_error

        result = {}
        try:
            vectorstore_wrapper._log_tool_event("Retrieving already indexed data from PGVector vectorstore",
                           tool_name="get_indexed_data")
            store = vectorstore_wrapper.vectorstore
            pending_run_ids = self.get_pending_run_ids(vectorstore_wrapper, index_name)
            with Session(store.session_maker.bind) as session:
                docs = session.query(
                    store.EmbeddingStore.id,
                    store.EmbeddingStore.cmetadata
                ).filter(
                    func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'collection') == index_name,
                    self._active_data_rows_clause(store, pending_run_ids)
                ).all()

            # Process the retrieved data
            for doc in docs:
                db_id = doc.id
                meta = doc.cmetadata or {}

                # Get document id from metadata
                doc_id = str(meta.get('id', db_id))
                dependent_docs = meta.get(IndexerKeywords.DEPENDENT_DOCS.value, [])
                if dependent_docs:
                    dependent_docs = [d.strip() for d in dependent_docs.split(';') if d.strip()]
                parent_id = meta.get(IndexerKeywords.PARENT.value, -1)

                chunk_id = meta.get('chunk_id')
                if doc_id in result and chunk_id:
                    # If document with the same id already saved, add db_id for current one as chunk
                    result[doc_id]['all_chunks'].append(db_id)
                else:
                    result[doc_id] = {
                        'metadata': meta,
                        'id': db_id,
                        'all_chunks': [db_id],
                        IndexerKeywords.DEPENDENT_DOCS.value: dependent_docs,
                        IndexerKeywords.PARENT.value: parent_id,
                        # Rows keyed by their row PK (no source id) have no stable
                        # identity: the orphan math must skip them, or they are
                        # deleted on every attested promote with no replacement.
                        'pk_fallback': 'id' not in meta,
                    }

        except Exception as e:
            # An unreadable index must never look like an empty one: the caller
            # would nominate nothing for supersede and publish this run's
            # generation on top of the surviving one, duplicating every document
            # with nothing reported. A missing table is the one honest empty —
            # no run can have indexed anything there.
            if not is_undefined_table_error(e):
                logger.error(f"Failed to get indexed data from PGVector: {str(e)}")
                raise
            logger.warning(f"No embedding table to read indexed data from: {str(e)}. Continuing with empty index.")

        return result

    def _active_data_rows_clause(self, store, pending_run_ids: Optional[List[str]] = None):
        from sqlalchemy import and_, func, or_

        clause = or_(
            func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'type').is_(None),
            func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'type') != IndexerKeywords.INDEX_META_TYPE.value
        )
        if pending_run_ids:
            run_id_text = func.jsonb_extract_path_text(
                store.EmbeddingStore.cmetadata, IndexerKeywords.RUN_ID.value
            )
            clause = and_(clause, or_(run_id_text.is_(None), run_id_text.notin_(pending_run_ids)))
        return clause

    def get_code_indexed_data(self, vectorstore_wrapper, index_name: str) -> Dict[str, Dict[str, Any]]:
        """Get all indexed code data from PGVector per collection suffix."""
        from sqlalchemy.orm import Session
        from sqlalchemy import func
        from ...runtime.tools.index_runs_model import is_undefined_table_error

        result = {}
        try:
            vectorstore_wrapper._log_tool_event(message="Retrieving already indexed code data from PGVector vectorstore",
                           tool_name="index_code_data")
            store = vectorstore_wrapper.vectorstore
            pending_run_ids = self.get_pending_run_ids(vectorstore_wrapper, index_name)
            with (Session(store.session_maker.bind) as session):
                docs = session.query(
                    store.EmbeddingStore.id,
                    store.EmbeddingStore.cmetadata
                ).filter(
                    func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'collection') == index_name,
                    self._active_data_rows_clause(store, pending_run_ids)
                ).all()

            for db_id, meta in docs:
                filename = meta.get('filename')
                commit_hash = meta.get('commit_hash')
                if not filename:
                    continue
                if filename not in result:
                    result[filename] = {
                        'metadata': meta,
                        'commit_hashes': [],
                        'ids': []
                    }
                if commit_hash is not None:
                    result[filename]['commit_hashes'].append(commit_hash)
                result[filename]['ids'].append(db_id)
        except Exception as e:
            # Same polarity as get_indexed_data: a failed read that reports an
            # empty index makes the run publish duplicates of every file.
            if not is_undefined_table_error(e):
                logger.error(f"Failed to get indexed code data from PGVector: {str(e)}")
                raise
            logger.warning(f"No embedding table to read indexed code data from: {str(e)}. Continuing with empty index.")
        return result

    def add_to_collection(self, vectorstore_wrapper, entry_id, new_collection_value):
        """Add a new collection name to the `collection` key in the `metadata` column."""
        from sqlalchemy import func
        from sqlalchemy.orm import Session

        store = vectorstore_wrapper.vectorstore
        try:
            with Session(store.session_maker.bind) as session:
                # Query the current value of the `collection` key
                current_collection_query = session.query(
                    func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'collection')
                ).filter(store.EmbeddingStore.id == entry_id).scalar()

                # If the `collection` key is NULL or doesn't contain the new value, update it
                if current_collection_query is None:
                    # If `collection` is NULL, initialize it with the new value
                    session.query(store.EmbeddingStore).filter(
                        store.EmbeddingStore.id == entry_id
                    ).update(
                        {
                            store.EmbeddingStore.cmetadata: func.jsonb_set(
                                func.coalesce(store.EmbeddingStore.cmetadata, '{}'),
                                '{collection}',  # Path to the `collection` key
                                f'"{new_collection_value}"',  # New value for the `collection` key
                                True  # Create the key if it doesn't exist
                            )
                        }
                    )
                elif new_collection_value not in current_collection_query.split(";"):
                    # If `collection` exists but doesn't contain the new value, append it
                    updated_collection_value = f"{current_collection_query};{new_collection_value}"
                    session.query(store.EmbeddingStore).filter(
                        store.EmbeddingStore.id == entry_id
                    ).update(
                        {
                            store.EmbeddingStore.cmetadata: func.jsonb_set(
                                store.EmbeddingStore.cmetadata,
                                '{collection}',  # Path to the `collection` key
                                f'"{updated_collection_value}"',  # Concatenated value as a valid JSON string
                                True  # Create the key if it doesn't exist
                            )
                        }
                    )

                session.commit()
                logger.info(f"Successfully updated collection for entry ID {entry_id}.")
        except Exception as e:
            logger.error(f"Failed to update collection for entry ID {entry_id}: {str(e)}")

    def get_index_meta(self, vectorstore_wrapper, index_name: str) -> List[Dict[str, Any]]:
        from sqlalchemy.orm import Session

        store = vectorstore_wrapper.vectorstore
        try:
            with Session(store.session_maker.bind) as session:
                meta = session.query(
                    store.EmbeddingStore.id,
                    store.EmbeddingStore.document,
                    store.EmbeddingStore.cmetadata
                ).filter(
                    self._index_meta_clause(store, index_name)
                ).all()
                result = []
                for id, document, cmetadata in meta:
                    result.append({"id": id, "content": document, "metadata": cmetadata})
                return result
        except Exception as e:
            logger.error(f"Failed to get index_meta from PGVector: {str(e)}")
            raise e

    def ensure_index_runs_table(self, vectorstore_wrapper) -> None:
        from ...runtime.tools.index_runs_model import ensure_index_runs_table

        store = vectorstore_wrapper.vectorstore
        ensure_index_runs_table(store.session_maker.bind, store.collection_name)

    def get_pending_run_ids(self, vectorstore_wrapper, index_name: str,
                            include_cancelled: bool = True) -> List[str]:
        from sqlalchemy.exc import ProgrammingError
        from sqlalchemy.orm import Session
        from ...runtime.tools.index_runs_model import (
            IndexRun, RUN_STATUS_CANCELLED, RUN_STATUS_PENDING, is_undefined_table_error,
        )

        store = vectorstore_wrapper.vectorstore
        statuses = (
            (RUN_STATUS_PENDING, RUN_STATUS_CANCELLED)
            if include_cancelled else (RUN_STATUS_PENDING,)
        )
        try:
            with Session(store.session_maker.bind) as session:
                query = session.query(IndexRun.run_id).filter(
                    IndexRun.status.in_(statuses)
                )
                if index_name:
                    query = query.filter(IndexRun.collection == index_name)
                return [row[0] for row in query.all()]
        except ProgrammingError as exc:
            # Read paths never provision the table: a schema without it means no
            # run ever staged there, so search degrades to unfiltered, not broken.
            if is_undefined_table_error(exc):
                return []
            raise

    def register_index_run(self, vectorstore_wrapper, index_name: str, run_id: str,
                           task_id: Optional[str] = None,
                           meta_lock_id: Optional[str] = None):
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.orm import Session
        from ...runtime.tools.index_runs_model import (
            IndexRun, RUN_STATUS_PENDING, live_run_where,
        )

        store = vectorstore_wrapper.vectorstore
        now = time.time()
        with Session(store.session_maker.bind) as session:
            if meta_lock_id is not None:
                # Pure lock, deliberately unused by the INSERT itself: core's
                # dispatch guard and failed-state writers read this table inside
                # a meta-row FOR UPDATE, and only queueing the registration
                # behind that same lock keeps their in-lock read authoritative.
                # Removing it silently reopens the guard TOCTOU.
                session.query(store.EmbeddingStore.id).filter(
                    store.EmbeddingStore.id == meta_lock_id
                ).with_for_update().all()
            statement = pg_insert(IndexRun).values(
                run_id=run_id,
                collection=index_name,
                status=RUN_STATUS_PENDING,
                task_id=task_id,
                started_on=now,
                heartbeat=now,
            ).on_conflict_do_nothing(
                index_elements=[IndexRun.collection],
                index_where=live_run_where(),
            ).returning(IndexRun.run_id)
            registered = session.execute(statement).first() is not None
            session.commit()
        if registered:
            return True, None
        with Session(store.session_maker.bind) as session:
            blocker = session.query(IndexRun).filter(
                IndexRun.collection == index_name,
                live_run_where(),
            ).first()
            if blocker is None:
                return False, None
            return False, {
                "run_id": blocker.run_id,
                "heartbeat": blocker.heartbeat,
                "started_on": blocker.started_on,
            }

    def heartbeat_index_run(self, vectorstore_wrapper, index_name: str, run_id: str,
                            meta_id: Optional[str]) -> None:
        import json as json_module
        from sqlalchemy import cast, exists, func, literal, update
        from sqlalchemy.dialects.postgresql import JSONB, array
        from sqlalchemy.orm import Session
        from ...runtime.tools.index_runs_model import (
            IndexRun, RUN_STATUS_CANCELLED, RUN_STATUS_PENDING, live_run_where,
        )

        store = vectorstore_wrapper.vectorstore
        now = time.time()
        # Two SEPARATE transactions: one transaction would hold the run-row lock
        # while waiting on the meta row — an AB-BA inversion against meta-first
        # promote/discard/cancel. Cancelled rows tick too, so a tombstoned
        # surviving worker stays fresh and the sweep cannot reclaim it mid-run.
        with Session(store.session_maker.bind) as session:
            session.execute(
                update(IndexRun)
                .where(
                    IndexRun.run_id == run_id,
                    IndexRun.status.in_((RUN_STATUS_PENDING, RUN_STATUS_CANCELLED)),
                )
                .values(heartbeat=now)
            )
            session.commit()
        if meta_id is None:
            return
        with Session(store.session_maker.bind) as session:
            # The EXISTS(pending own row) guard makes a tick queued behind
            # promote's meta lock a 0-row no-op after promote commits.
            session.execute(
                update(store.EmbeddingStore)
                .where(
                    store.EmbeddingStore.id == meta_id,
                    exists().where(IndexRun.run_id == run_id, live_run_where()),
                )
                .values(
                    cmetadata=func.jsonb_set(
                        func.coalesce(store.EmbeddingStore.cmetadata, cast(literal("{}"), JSONB)),
                        array(["updated_on"]),
                        cast(literal(json_module.dumps(now)), JSONB),
                    )
                )
            )
            session.commit()

    def sweep_stale_index_runs(self, vectorstore_wrapper, index_name: str,
                               stale_before: float) -> List[str]:
        from sqlalchemy.orm import Session
        from ...runtime.tools.index_runs_model import (
            IndexRun, RUN_STATUS_CANCELLED, RUN_STATUS_DISCARDED, RUN_STATUS_PENDING,
        )

        store = vectorstore_wrapper.vectorstore
        self._reclaim_discarded_run_chunks(store, index_name)
        with Session(store.session_maker.bind) as session:
            candidate_run_ids = [
                row[0]
                for row in session.query(IndexRun.run_id).filter(
                    IndexRun.collection == index_name,
                    IndexRun.status.in_((RUN_STATUS_PENDING, RUN_STATUS_CANCELLED)),
                    IndexRun.heartbeat < stale_before,
                ).all()
            ]
        reclaimed_run_ids = []
        for candidate_run_id in candidate_run_ids:
            # One transaction per run, run row FIRST. The candidate read above is
            # lock-free, and staleness is a heartbeat guess: a worker whose
            # heartbeat thread died keeps indexing and can promote inside that
            # window. Re-verifying under the row lock is what stops the DELETE
            # from erasing a corpus that just went live, and committing the flip
            # with the DELETE leaves no window where chunks outlive a terminal
            # row. Lock order run -> chunks is the tail of the universal
            # meta -> run -> chunks: the chunk DELETE excludes the index_meta
            # row, so the sweep never wants a meta row it does not already hold
            # and cannot close a cycle against promote/discard/cancel.
            with Session(store.session_maker.bind) as session:
                run_row = session.get(IndexRun, candidate_run_id, with_for_update=True)
                if (
                    run_row is None
                    or run_row.status not in (RUN_STATUS_PENDING, RUN_STATUS_CANCELLED)
                    or run_row.heartbeat >= stale_before
                ):
                    session.rollback()
                    continue
                self._delete_run_chunks(session, store, candidate_run_id)
                run_row.status = RUN_STATUS_DISCARDED
                session.commit()
                reclaimed_run_ids.append(candidate_run_id)
        return reclaimed_run_ids

    def _reclaim_discarded_run_chunks(self, store, index_name: str) -> None:
        from sqlalchemy.orm import Session
        from ...runtime.tools.index_runs_model import IndexRun, RUN_STATUS_DISCARDED

        # Chunks under a 'discarded' row are unreachable garbage: neither the read
        # filter (pending/cancelled) nor the reclaim loop below can see them, so
        # they resurface as visible phantoms once the name is reused. Only the
        # best-effort cleanup on promote's and discard's abort branches deletes
        # them today, and when it fails nothing else ever retries — this pass is
        # that retry. Runs ahead of the loop because the loop's own flips are
        # committed with their DELETEs and leave nothing behind.
        #
        # 'promoted' rows are the live corpus and stay out of the id set; ids with
        # no run row at all stay out too — rows are only ever deleted by index
        # removal or the rollback runbook, so an absent row is not abandonment.
        #
        # Candidates are picked by surviving chunks, never by recency: a run
        # stranded by the sweep stops heartbeating the moment it goes terminal, so
        # its heartbeat is older than that of every run discarded after it, and a
        # newest-first window would push it out permanently once the collection
        # has accumulated a window's worth of discarded rows. The predicate is the
        # delete's own, so every candidate is deletable and each sweep makes
        # progress; the probe per candidate row is GIN-served.
        try:
            with Session(store.session_maker.bind) as session:
                stranded_run_ids = [
                    row[0]
                    for row in session.query(IndexRun.run_id).filter(
                        IndexRun.collection == index_name,
                        IndexRun.status == RUN_STATUS_DISCARDED,
                        self._run_chunks_exist_clause(store, IndexRun.run_id),
                    ).limit(STRANDED_RECLAIM_RUN_LIMIT).all()
                ]
                if not stranded_run_ids:
                    return
                self._delete_runs_chunks(session, store, stranded_run_ids)
                session.commit()
        except Exception as exc:
            # Reclaiming a previous run's leftovers must never fail the run that
            # sweeps; the next sweep retries.
            logger.warning(
                f"Failed to reclaim stranded chunks of discarded runs in '{index_name}': {exc}"
            )

    def promote_run(self, vectorstore_wrapper, index_name: str, run_id: str,
                    superseded_ids: List[str], orphan_ids: List[str],
                    damaged_ids: List[str]) -> str:
        from sqlalchemy.orm import Session
        from ...runtime.tools.index_runs_model import (
            IndexRun, RUN_STATUS_CANCELLED, RUN_STATUS_DISCARDED,
            RUN_STATUS_PENDING, RUN_STATUS_PROMOTED,
        )

        store = vectorstore_wrapper.vectorstore
        with Session(store.session_maker.bind) as session:
            # Universal lock order for promote, discard AND cancel:
            # meta row -> run row -> chunk rows. Any other order deadlocks
            # against the other two parties.
            meta_ids = self._lock_index_meta_rows(session, store, index_name)
            run_row = session.get(IndexRun, run_id, with_for_update=True)
            if not meta_ids:
                # Every non-promoted status takes its staged rows along: a run
                # already swept to DISCARDED is out of reach of both the sweep
                # (pending/cancelled only) and the read filter, so rows this
                # worker flushed after the sweep would stay visible forever.
                if run_row is None or run_row.status != RUN_STATUS_PROMOTED:
                    self._delete_run_chunks(session, store, run_id)
                    if run_row is not None and run_row.status in (RUN_STATUS_PENDING, RUN_STATUS_CANCELLED):
                        run_row.status = RUN_STATUS_DISCARDED
                session.commit()
                return "aborted-row-deleted"
            if run_row is None or run_row.status not in (RUN_STATUS_PENDING, RUN_STATUS_CANCELLED):
                already_promoted = run_row is not None and run_row.status == RUN_STATUS_PROMOTED
                session.rollback()
                if not already_promoted:
                    self._discard_stranded_run_chunks(store, run_id)
                return "aborted-not-pending"
            if run_row.status == RUN_STATUS_CANCELLED:
                self._delete_run_chunks(session, store, run_id)
                run_row.status = RUN_STATUS_DISCARDED
                session.commit()
                return "aborted-cancelled"
            delete_ids = list(dict.fromkeys([*superseded_ids, *orphan_ids, *damaged_ids]))
            for ids_batch in self._iter_id_batches(delete_ids):
                self._delete_rows_by_pk(session, store, ids_batch)
            run_row.status = RUN_STATUS_PROMOTED
            run_row.promoted_on = time.time()
            session.commit()
            return "promoted"

    def discard_run(self, vectorstore_wrapper, index_name: str, run_id: str) -> str:
        from sqlalchemy.orm import Session
        from ...runtime.tools.index_runs_model import (
            IndexRun, RUN_STATUS_CANCELLED, RUN_STATUS_DISCARDED, RUN_STATUS_PENDING,
            RUN_STATUS_PROMOTED,
        )

        store = vectorstore_wrapper.vectorstore
        with Session(store.session_maker.bind) as session:
            self._lock_index_meta_rows(session, store, index_name)
            run_row = session.get(IndexRun, run_id, with_for_update=True)
            if run_row is None or run_row.status not in (RUN_STATUS_PENDING, RUN_STATUS_CANCELLED):
                already_promoted = run_row is not None and run_row.status == RUN_STATUS_PROMOTED
                session.rollback()
                if not already_promoted:
                    self._discard_stranded_run_chunks(store, run_id)
                return "noop"
            # The flip is committed WITH the delete, never ahead of it: 'discarded' sits
            # outside both the read filter's and the sweep's status set, so a row flipped
            # over chunks that survived turns them into permanently visible phantoms. A
            # failed cleanup therefore leaves the row pending — invisible, and reclaimable.
            self._delete_run_chunks(session, store, run_id)
            run_row.status = RUN_STATUS_DISCARDED
            session.commit()
            return "discarded"

    def _lock_index_meta_rows(self, session, store, index_name: str):
        return session.query(store.EmbeddingStore.id).filter(
            self._index_meta_clause(store, index_name)
        ).with_for_update().all()

    @staticmethod
    def _index_meta_clause(store, index_name: str):
        # Containment, not ->>: the table's only jsonb index is GIN
        # jsonb_path_ops, which serves @> alone. An extract-text predicate
        # sequential-scans the whole collection inside promote's transaction,
        # with every core meta writer queued behind it.
        return store.EmbeddingStore.cmetadata.contains({
            "type": IndexerKeywords.INDEX_META_TYPE.value,
            "collection": index_name,
        })

    def _discard_stranded_run_chunks(self, store, run_id: str) -> None:
        from sqlalchemy.orm import Session

        # Own-run rows only, in a transaction of its own so the meta lock is
        # already released. Once the run row is terminal neither the sweep
        # (pending/cancelled only) nor the read filter can reach these rows, so
        # leaving them turns them into visible phantoms the moment an index of
        # the same name is recreated.
        try:
            with Session(store.session_maker.bind) as session:
                self._delete_run_chunks(session, store, run_id)
                session.commit()
        except Exception as exc:
            # Best effort by contract: the abort branches that call this publish
            # nothing, write no terminal meta state and emit no event, and
            # raising from here would turn that silent abort into a reported
            # run failure.
            logger.warning(f"Failed to delete stranded chunks of run '{run_id}': {exc}")

    def _delete_index_runs(self, vectorstore_wrapper, index_name: str) -> None:
        from sqlalchemy.exc import ProgrammingError
        from sqlalchemy.orm import Session
        from ...runtime.tools.index_runs_model import (
            IndexRun, RUN_STATUS_CANCELLED, RUN_STATUS_PENDING, is_undefined_table_error,
        )

        store = vectorstore_wrapper.vectorstore
        try:
            with Session(store.session_maker.bind) as session:
                # Terminal rows only. A live run keeps staging chunks whose run id
                # has to stay in the pending set, and its row is what holds the
                # name against a second registration. Reclaiming a crashed run's
                # row is the stale sweep's job, which index_data runs before it
                # registers on both the fresh and the reindex branch.
                session.query(IndexRun).filter(
                    IndexRun.collection == index_name,
                    IndexRun.status.notin_((RUN_STATUS_PENDING, RUN_STATUS_CANCELLED)),
                ).delete(synchronize_session=False)
                session.commit()
        except ProgrammingError as exc:
            if not is_undefined_table_error(exc):
                raise

    def _delete_run_chunks(self, session, store, run_id: str) -> int:
        return self._delete_runs_chunks(session, store, [run_id])

    def _delete_runs_chunks(self, session, store, run_ids: List[str]) -> int:
        from sqlalchemy import or_

        # No `collection` conjunct: multi-index rows carry `collection` as an
        # appended "a;b" string that equality would miss, leaving escaped rows
        # permanently visible once the run row goes terminal. The run id is
        # globally unique; the type conjunct shields the meta row belt-and-braces.
        # Containment per id rather than one array predicate: only `@>` is served
        # by the table's jsonb_path_ops GIN, and the planner bitmap-ORs the probes.
        return session.query(store.EmbeddingStore).filter(
            or_(*[
                store.EmbeddingStore.cmetadata.contains({IndexerKeywords.RUN_ID.value: run_id})
                for run_id in run_ids
            ]),
            self._non_index_meta_clause(store),
        ).delete(synchronize_session=False)

    @staticmethod
    def _non_index_meta_clause(store):
        from sqlalchemy import func, or_

        return or_(
            func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'type').is_(None),
            func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'type') != IndexerKeywords.INDEX_META_TYPE.value
        )

    @classmethod
    def _run_chunks_exist_clause(cls, store, run_id_column):
        from sqlalchemy import String as SAString
        from sqlalchemy import cast, func, literal, select

        # Correlated on the run row, so the right-hand side of `@>` is built per
        # candidate and stays a plain jsonb value the jsonb_path_ops GIN can probe.
        # The key is cast because jsonb_build_object is VARIADIC "any": an untyped
        # bind leaves its type undeterminable to any server-side-binding driver.
        return select(store.EmbeddingStore.id).where(
            store.EmbeddingStore.cmetadata.contains(
                func.jsonb_build_object(
                    cast(literal(IndexerKeywords.RUN_ID.value), SAString), run_id_column
                )
            ),
            cls._non_index_meta_clause(store),
        ).exists()

    @staticmethod
    def _iter_id_batches(ids: List[str]):
        for start in range(0, len(ids), PROMOTE_DELETE_BATCH_SIZE):
            yield ids[start:start + PROMOTE_DELETE_BATCH_SIZE]

    @staticmethod
    def _delete_rows_by_pk(session, store, ids_batch: List[str]) -> int:
        from sqlalchemy import String as SAString
        from sqlalchemy import any_, cast, literal
        from sqlalchemy.dialects.postgresql import ARRAY

        # `= ANY(:array)` is a single bind: `.in_()` fails past 65535 parameters.
        return session.query(store.EmbeddingStore).filter(
            store.EmbeddingStore.id == any_(cast(literal(ids_batch), ARRAY(SAString)))
        ).delete(synchronize_session=False)


class ChromaAdapter(VectorStoreAdapter):
    """Adapter for Chroma database operations."""

    def get_vectorstore_params(self, collection_name: str, connection_string: Optional[str] = None) -> Dict[str, Any]:
        return {
            "collection_name": collection_name,
            "persist_directory": "./indexer_db"
        }

    def list_collections(self, vectorstore_wrapper) -> str:
        vector_client = vectorstore_wrapper.vectorstore._client
        return ','.join([collection.name for collection in vector_client.list_collections()])

    def remove_collection(self, vectorstore_wrapper, collection_name: str):
        vectorstore_wrapper.vectorstore.delete_collection()

    def get_indexed_ids(self, vectorstore_wrapper, index_name: Optional[str] = '') -> List[str]:
        """Get all indexed document IDs from Chroma, optionally filtered by index_name."""
        try:
            if index_name:
                data = vectorstore_wrapper.vectorstore.get(
                    where={"collection": index_name},
                    include=[]
                )
            else:
                data = vectorstore_wrapper.vectorstore.get(include=[])
            return data.get('ids', [])
        except Exception as e:
            logger.error(f"Failed to get indexed IDs from Chroma: {str(e)}")
            return []

    def clean_collection(self, vectorstore_wrapper, index_name: str = '', including_index_meta: bool = False) -> int:
        """Clean the vectorstore collection by deleting indexed data filtered by index_name."""

        if index_name:
            data = vectorstore_wrapper.vectorstore.get(
                where={"collection": index_name},
                include=['metadatas']
            )
            ids = data.get('ids', [])
            metadatas = data.get('metadatas', [])

            if including_index_meta:
                ids_to_delete = ids
            else:
                ids_to_delete = [
                    id_ for id_, meta in zip(ids, metadatas)
                    if meta.get('type') != IndexerKeywords.INDEX_META_TYPE.value
                ]
        else:
            data = vectorstore_wrapper.vectorstore.get(include=[])
            ids_to_delete = data.get('ids', [])

        if ids_to_delete:
            vectorstore_wrapper.vectorstore.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def get_indexed_data(self, vectorstore_wrapper):
        """Get all indexed data from Chroma for non-code content"""

        result = {}
        try:
            vectorstore_wrapper._log_data("Retrieving already indexed data from Chroma vectorstore",
                           tool_name="get_indexed_data")
            data = vectorstore_wrapper.vectorstore.get(include=['metadatas'])

            # Re-structure data to be more usable
            for meta, db_id in zip(data['metadatas'], data['ids']):
                # Get document id from metadata
                doc_id = str(meta['id'])
                dependent_docs = meta.get(IndexerKeywords.DEPENDENT_DOCS.value, [])
                if dependent_docs:
                    dependent_docs = [d.strip() for d in dependent_docs.split(';') if d.strip()]
                parent_id = meta.get(IndexerKeywords.PARENT.value, -1)

                chunk_id = meta.get('chunk_id')
                if doc_id in result and chunk_id:
                    # If document with the same id already saved, add db_id for current one as chunk
                    result[doc_id]['all_chunks'].append(db_id)
                else:
                    result[doc_id] = {
                        'metadata': meta,
                        'id': db_id,
                        'all_chunks': [db_id],
                        IndexerKeywords.DEPENDENT_DOCS.value: dependent_docs,
                        IndexerKeywords.PARENT.value: parent_id
                    }
        except Exception as e:
            logger.error(f"Failed to get indexed data from Chroma: {str(e)}. Continuing with empty index.")

        return result

    def get_code_indexed_data(self, vectorstore_wrapper, index_name) -> Dict[str, Dict[str, Any]]:
        """Get all indexed code data from Chroma."""
        result = {}
        try:
            vectorstore_wrapper._log_data("Retrieving already indexed code data from Chroma vectorstore",
                           tool_name="index_code_data")
            data = vectorstore_wrapper.vectorstore.get(include=['metadatas'])
            for meta, db_id in zip(data['metadatas'], data['ids']):
                filename = meta.get('filename')
                commit_hash = meta.get('commit_hash')
                if not filename:
                    continue
                if filename not in result:
                    result[filename] = {
                        'commit_hashes': [],
                        'ids': []
                    }
                if commit_hash is not None:
                    result[filename]['commit_hashes'].append(commit_hash)
                result[filename]['ids'].append(db_id)
        except Exception as e:
            logger.error(f"Failed to get indexed code data from Chroma: {str(e)}. Continuing with empty index.")
        return result

    def add_to_collection(self, vectorstore_wrapper, entry_id, new_collection_value):
        """Add a new collection name to the metadata - Chroma implementation"""
        # For Chroma, we would need to update the metadata through vectorstore operations
        # This is a simplified implementation - in practice, you might need more complex logic
        logger.warning("add_to_collection for Chroma is not fully implemented yet")

    def get_index_meta(self, vectorstore_wrapper, index_name: str) -> List[Dict[str, Any]]:
        logger.warning("get_index_meta for Chroma is not implemented yet")
        return []

    def promote_run(self, vectorstore_wrapper, index_name: str, run_id: str,
                    superseded_ids: List[str], orphan_ids: List[str],
                    damaged_ids: List[str]) -> str:
        logger.warning("promote_run is a no-op for Chroma: the preserve-on-failure "
                       "guarantee does not extend to Chroma")
        return "promoted"

    def discard_run(self, vectorstore_wrapper, index_name: str, run_id: str) -> str:
        logger.warning("discard_run is a no-op for Chroma: the preserve-on-failure "
                       "guarantee does not extend to Chroma")
        return "noop"


class VectorStoreAdapterFactory:
    """Factory for creating vector store adapters."""

    _adapters = {
        'PGVector': PGVectorAdapter,
        'Chroma': ChromaAdapter,
    }

    @classmethod
    def create_adapter(cls, vectorstore_type: str) -> VectorStoreAdapter:
        adapter_class = cls._adapters.get(vectorstore_type)
        if not adapter_class:
            raise ValueError(f"Unsupported vectorstore type: {vectorstore_type}")
        return adapter_class()

    @classmethod
    def register_adapter(cls, vectorstore_type: str, adapter_class: type):
        """Register a new adapter for a vector store type."""
        cls._adapters[vectorstore_type] = adapter_class
