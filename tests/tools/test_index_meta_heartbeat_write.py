"""A load-phase heartbeat must survive the row a platform dispatch actually creates.

`start_index_task` writes the index_meta row from scratch at every run, and that row
carries no `indexed_chunks` — the SDK only adds it when it counts. A heartbeat asks not
to count, so anything derived from the count has to be skipped too, or the write raises
and is swallowed by the ticker's best-effort handler, leaving the run looking abandoned
while it is still loading.

These drive the real `index_meta_update`; mocking it wholesale is what let that ship.
"""
from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit


DISPATCH_ROW = {
    "collection": "idx",
    "type": "index_meta",
    "indexed": 0,
    "updated": 0,
    "state": "in_progress",
    "index_configuration": "{}",
    "created_on": 1.0,
    "updated_on": 1.0,
    "task_id": "task-1",
    "toolkit_id": 56,
    "conversation_id": None,
    "history": "[]",
}


@pytest.fixture
def wrapper():
    instance = BaseIndexerToolkit.model_construct()
    object.__setattr__(instance, "_ensure_vectorstore_initialized", lambda: None)
    object.__setattr__(instance, "get_indexed_count", lambda _name: 42)
    object.__setattr__(instance, "_is_scheduled_run", lambda: False)
    object.__setattr__(instance, "get_indexing_stats", lambda: None)
    object.__setattr__(instance, "vectorstore", MagicMock())
    return instance


def _run(wrapper, row, **kwargs):
    written = {}

    def capture(vectorstore, documents, **kwargs):
        written['metadata'] = documents[0].metadata

    with patch.object(VectorStoreWrapperBase, 'get_index_meta', return_value={"metadata": dict(row)}), \
         patch('elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents', capture):
        wrapper.index_meta_update(
            "idx", IndexerKeywords.INDEX_META_IN_PROGRESS.value, 0, update_force=True, **kwargs,
        )
    return written.get('metadata', {})


class TestHeartbeatWrite:

    def test_writes_against_a_freshly_dispatched_row(self, wrapper):
        metadata = _run(wrapper, DISPATCH_ROW, refresh_counts=False)

        assert metadata["state"] == IndexerKeywords.INDEX_META_IN_PROGRESS.value
        assert metadata["updated_on"] > DISPATCH_ROW["updated_on"]

    def test_leaves_every_count_untouched(self, wrapper):
        # The tick knows nothing about progress; reporting zero would erase what the
        # chunk loop recorded.
        row = dict(DISPATCH_ROW, indexed=7, updated=7, indexed_chunks=90)
        metadata = _run(wrapper, row, refresh_counts=False)

        assert metadata["indexed"] == 7
        assert metadata["updated"] == 7
        assert metadata["indexed_chunks"] == 90

    def test_a_counting_write_still_refreshes_them(self, wrapper):
        metadata = _run(wrapper, DISPATCH_ROW)

        assert metadata["indexed_chunks"] == 42
        assert metadata["indexed"] == 42
