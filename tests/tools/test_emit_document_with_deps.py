"""
Tests for `BaseIndexerToolkit._emit_document_with_deps` metadata propagation.

Regression coverage for the "missing required metadata field 'id' or
'updated_on'" warning fired at base_indexer_toolkit.py against dependent
chunks. Dependents (attachment/image chunks, comment chunks, sub-page chunks)
inherit their `updated_on` from the parent so that:

  1. The metadata guard downstream does not warn per-chunk on every index run.
  2. Dedup on dependents works — same parent timestamp → skip re-embed.

Dependents that already carry an explicit `updated_on` must keep it (a loader
may have set a per-chunk timestamp that is more precise than the parent's).
"""
from unittest.mock import patch

from langchain_core.documents import Document

from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit


def _wrapper() -> BaseIndexerToolkit:
    # Bypass full Pydantic validation — the emit helper only needs
    # `_log_tool_event` and `_extract_doc_name` to be bound.
    return BaseIndexerToolkit.model_construct()


def _parent(updated_on="2026-07-10T00:00:00Z", doc_id="wi-42") -> Document:
    return Document(
        page_content="parent body",
        metadata={"id": doc_id, "name": "parent", "updated_on": updated_on},
    )


def _dep(dep_id="wi-42-att-1", **extra) -> Document:
    meta = {"id": dep_id}
    meta.update(extra)
    return Document(page_content="dep chunk", metadata=meta)


class TestEmitDocumentWithDepsUpdatedOn:
    """The parent's `updated_on` must land on dependents that lack their own."""

    def test_dependent_inherits_updated_on_from_parent(self):
        parent = _parent(updated_on="2026-07-10T12:00:00Z")
        dep = _dep()  # no updated_on

        with patch.object(BaseIndexerToolkit, "_log_tool_event"):
            emitted = list(_wrapper()._emit_document_with_deps(parent, [dep]))

        # emit order: deps first, then parent
        assert emitted[0] is dep
        assert dep.metadata["updated_on"] == "2026-07-10T12:00:00Z"

    def test_multiple_dependents_all_inherit(self):
        parent = _parent(updated_on="2026-07-10T12:00:00Z")
        deps = [_dep(dep_id="d1"), _dep(dep_id="d2"), _dep(dep_id="d3")]

        with patch.object(BaseIndexerToolkit, "_log_tool_event"):
            emitted = list(_wrapper()._emit_document_with_deps(parent, deps))

        for dep in deps:
            assert dep.metadata["updated_on"] == "2026-07-10T12:00:00Z"
        # parent yielded last
        assert emitted[-1] is parent

    def test_dependent_keeps_its_own_updated_on(self):
        # A loader may set a more precise per-chunk timestamp — preserve it.
        parent = _parent(updated_on="2026-01-01T00:00:00Z")
        dep = _dep(updated_on="2026-06-15T09:00:00Z")

        with patch.object(BaseIndexerToolkit, "_log_tool_event"):
            list(_wrapper()._emit_document_with_deps(parent, [dep]))

        assert dep.metadata["updated_on"] == "2026-06-15T09:00:00Z"

    def test_no_propagation_when_parent_missing_updated_on(self):
        # Parent without a timestamp must not inject `updated_on: None` on deps.
        parent = Document(page_content="p", metadata={"id": "wi-1"})
        dep = _dep()

        with patch.object(BaseIndexerToolkit, "_log_tool_event"):
            list(_wrapper()._emit_document_with_deps(parent, [dep]))

        assert "updated_on" not in dep.metadata

    def test_parent_metadata_unchanged_by_propagation(self):
        parent = _parent(updated_on="2026-07-10T12:00:00Z")
        original_meta = dict(parent.metadata)
        dep = _dep()

        with patch.object(BaseIndexerToolkit, "_log_tool_event"):
            list(_wrapper()._emit_document_with_deps(parent, [dep]))

        # Parent still carries its original id + timestamp untouched.
        assert parent.metadata["id"] == original_meta["id"]
        assert parent.metadata["updated_on"] == original_meta["updated_on"]


class TestEmitDocumentWithDepsRegressionSurface:
    """Existing behaviour of the emit helper must stay intact alongside the fix."""

    def test_parent_id_still_set_on_each_dep(self):
        parent = _parent(doc_id="wi-99")
        deps = [_dep(dep_id="d1"), _dep(dep_id="d2")]

        with patch.object(BaseIndexerToolkit, "_log_tool_event"):
            list(_wrapper()._emit_document_with_deps(parent, deps))

        for dep in deps:
            assert dep.metadata[IndexerKeywords.PARENT.value] == "wi-99"

    def test_dependent_docs_aggregated_on_parent(self):
        parent = _parent()
        deps = [_dep(dep_id="d1"), _dep(dep_id="d2"), _dep(dep_id="")]

        with patch.object(BaseIndexerToolkit, "_log_tool_event"):
            list(_wrapper()._emit_document_with_deps(parent, deps))

        # Only truthy dep ids are collected; empty ids are skipped.
        assert parent.metadata[IndexerKeywords.DEPENDENT_DOCS.value] == "d1;d2"

    def test_yield_order_deps_then_parent(self):
        parent = _parent()
        deps = [_dep(dep_id="d1"), _dep(dep_id="d2")]

        with patch.object(BaseIndexerToolkit, "_log_tool_event"):
            emitted = list(_wrapper()._emit_document_with_deps(parent, deps))

        assert emitted[:2] == deps
        assert emitted[-1] is parent

    def test_no_deps_yields_only_parent(self):
        parent = _parent()

        with patch.object(BaseIndexerToolkit, "_log_tool_event"):
            emitted = list(_wrapper()._emit_document_with_deps(parent, []))

        assert emitted == [parent]

    def test_existing_dependent_docs_extended_not_overwritten(self):
        # If a loader pre-populated dependent_docs on the parent, the emit helper
        # must append newly collected ids rather than clobber the existing list.
        parent = _parent()
        parent.metadata[IndexerKeywords.DEPENDENT_DOCS.value] = "old-a;old-b"
        deps = [_dep(dep_id="new-1")]

        with patch.object(BaseIndexerToolkit, "_log_tool_event"):
            list(_wrapper()._emit_document_with_deps(parent, deps))

        assert parent.metadata[IndexerKeywords.DEPENDENT_DOCS.value] == "old-a;old-b;new-1"
