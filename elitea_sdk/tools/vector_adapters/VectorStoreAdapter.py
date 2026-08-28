import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from logging import getLogger

from ...runtime.utils.utils import IndexerKeywords

logger = getLogger(__name__)

PROMOTE_DELETE_BATCH_SIZE = 50000


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
            return deleted_count

    def is_vectorstore_type(self, vectorstore) -> bool:
        """Check if the vectorstore is a PGVector store."""
        return hasattr(vectorstore, 'session_maker') and hasattr(vectorstore, 'EmbeddingStore')

    def get_indexed_data(self, vectorstore_wrapper, index_name: str)-> Dict[str, Dict[str, Any]]:
        """Get all indexed data from PGVector for non-code content per index_name."""
        from sqlalchemy.orm import Session
        from sqlalchemy import func

        result = {}
        try:
            vectorstore_wrapper._log_tool_event("Retrieving already indexed data from PGVector vectorstore",
                           tool_name="get_indexed_data")
            store = vectorstore_wrapper.vectorstore
            pending_run_ids = self.get_pending_run_ids(vectorstore_wrapper, index_name)
            with Session(store.session_maker.bind) as session:
                docs = session.query(
                    store.EmbeddingStore.id,
                    store.EmbeddingStore.document,
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
            logger.error(f"Failed to get indexed data from PGVector: {str(e)}. Continuing with empty index.")

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
            logger.error(f"Failed to get indexed code data from PGVector: {str(e)}. Continuing with empty index.")
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
        from sqlalchemy import func

        store = vectorstore_wrapper.vectorstore
        try:
            with Session(store.session_maker.bind) as session:
                meta = session.query(
                    store.EmbeddingStore.id,
                    store.EmbeddingStore.document,
                    store.EmbeddingStore.cmetadata
                ).filter(
                    store.EmbeddingStore.cmetadata['type'].astext == IndexerKeywords.INDEX_META_TYPE.value,
                    func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'collection') == index_name
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
        from sqlalchemy import update
        from sqlalchemy.orm import Session
        from ...runtime.tools.index_runs_model import (
            IndexRun, RUN_STATUS_CANCELLED, RUN_STATUS_DISCARDED, RUN_STATUS_PENDING,
        )

        store = vectorstore_wrapper.vectorstore
        with Session(store.session_maker.bind) as session:
            stale_run_ids = [
                row[0]
                for row in session.query(IndexRun.run_id).filter(
                    IndexRun.collection == index_name,
                    IndexRun.status.in_((RUN_STATUS_PENDING, RUN_STATUS_CANCELLED)),
                    IndexRun.heartbeat < stale_before,
                ).all()
            ]
        for stale_run_id in stale_run_ids:
            # The chunk DELETE commits lock-free BEFORE the status flip: a single
            # transaction would hold chunk-row locks while wanting the run row,
            # inverting cancel's meta -> run -> chunk order. A crash between the
            # two transactions leaves the row to be re-swept next run.
            with Session(store.session_maker.bind) as session:
                self._delete_run_chunks(session, store, stale_run_id)
                session.commit()
            with Session(store.session_maker.bind) as session:
                session.execute(
                    update(IndexRun)
                    .where(
                        IndexRun.run_id == stale_run_id,
                        IndexRun.status.in_((RUN_STATUS_PENDING, RUN_STATUS_CANCELLED)),
                    )
                    .values(status=RUN_STATUS_DISCARDED)
                )
                session.commit()
        return stale_run_ids

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
                if run_row is not None and run_row.status in (RUN_STATUS_PENDING, RUN_STATUS_CANCELLED):
                    run_row.status = RUN_STATUS_DISCARDED
                session.commit()
                return "aborted-row-deleted"
            if run_row is None or run_row.status not in (RUN_STATUS_PENDING, RUN_STATUS_CANCELLED):
                session.rollback()
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
        )

        store = vectorstore_wrapper.vectorstore
        with Session(store.session_maker.bind) as session:
            self._lock_index_meta_rows(session, store, index_name)
            run_row = session.get(IndexRun, run_id, with_for_update=True)
            if run_row is None or run_row.status not in (RUN_STATUS_PENDING, RUN_STATUS_CANCELLED):
                session.rollback()
                return "noop"
            self._delete_run_chunks(session, store, run_id)
            run_row.status = RUN_STATUS_DISCARDED
            session.commit()
            return "discarded"

    def _lock_index_meta_rows(self, session, store, index_name: str):
        from sqlalchemy import func

        return session.query(store.EmbeddingStore.id).filter(
            store.EmbeddingStore.cmetadata['type'].astext == IndexerKeywords.INDEX_META_TYPE.value,
            func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'collection') == index_name
        ).with_for_update().all()

    def _delete_run_chunks(self, session, store, run_id: str) -> int:
        from sqlalchemy import func, or_

        # No `collection` conjunct: multi-index rows carry `collection` as an
        # appended "a;b" string that equality would miss, leaving escaped rows
        # permanently visible once the run row goes terminal. The run id is
        # globally unique; the type conjunct shields the meta row belt-and-braces.
        return session.query(store.EmbeddingStore).filter(
            store.EmbeddingStore.cmetadata.contains({IndexerKeywords.RUN_ID.value: run_id}),
            or_(
                func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'type').is_(None),
                func.jsonb_extract_path_text(store.EmbeddingStore.cmetadata, 'type') != IndexerKeywords.INDEX_META_TYPE.value
            )
        ).delete(synchronize_session=False)

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
