"""Tests for the workers cap on ADO wiki + work_item indexers.

Both wrappers accept a `workers` knob for opt-in parallelism. Unbounded values
would risk ADO REST 429s, LLM rate limits, and pgvector pool exhaustion, so
the wrappers hard-cap at `_MAX_WORKERS = 10`:

  * The Pydantic Field for `workers` declares `le=10` so schema-level
    validation rejects >10 before it reaches runtime.
  * `_base_loader` clamps `_index_workers` to `_MAX_WORKERS` defensively
    (for callers that skip the tool schema, e.g. direct SDK use).
  * A warning is logged whenever clamping actually happens.
"""
import logging

from elitea_sdk.tools.ado.wiki.ado_wrapper import AzureDevOpsApiWrapper as WikiWrapper
from elitea_sdk.tools.ado.work_item.ado_wrapper import AzureDevOpsApiWrapper as WorkItemWrapper


def _make_wiki() -> WikiWrapper:
    w = WikiWrapper.model_construct(organization_url="https://dev.azure.com/x", project="p")
    w._init_indexing_stats = lambda: None  # skip stats bootstrap
    w._resolve_wiki_identifier = lambda _wid: "wiki-id"
    w._iter_wiki_pages = lambda _wid: iter(())
    return w


def _make_work_item() -> WorkItemWrapper:
    w = WorkItemWrapper.model_construct(organization_url="https://dev.azure.com/x", project="p")
    w._init_indexing_stats = lambda: None

    class _FakeClient:
        def query_by_wiql(self, _wiql):
            class _R:
                work_items = []
                work_item_relations = None
            return _R()

    w._client = _FakeClient()
    return w


# ------ wiki ---------------------------------------------------------------


def test_wiki_max_workers_constant():
    # Pydantic wraps underscore-prefixed class attrs as ModelPrivateAttr,
    # so we read the default through an instance (matches production access).
    assert _make_wiki()._MAX_WORKERS == 10


def test_wiki_workers_clamped_at_runtime(caplog):
    w = _make_wiki()
    with caplog.at_level(logging.WARNING, logger="elitea_sdk.tools.ado.wiki.ado_wrapper"):
        list(w._base_loader(workers=100))
    assert w._index_workers == 10
    assert any("exceeds cap" in rec.message for rec in caplog.records)


def test_wiki_workers_within_cap_not_clamped(caplog):
    w = _make_wiki()
    with caplog.at_level(logging.WARNING, logger="elitea_sdk.tools.ado.wiki.ado_wrapper"):
        list(w._base_loader(workers=5))
    assert w._index_workers == 5
    assert not any("exceeds cap" in rec.message for rec in caplog.records)


def test_wiki_workers_default_serial():
    w = _make_wiki()
    list(w._base_loader())
    assert w._index_workers == w._DEFAULT_WORKERS == 1


def test_wiki_workers_schema_declares_le_10():
    params = _make_wiki()._index_tool_params()
    _type, field = params["workers"]
    metadata = getattr(field, "metadata", [])
    # Pydantic v2 stores constraints in Field.metadata as annotated-types objects.
    assert any(getattr(m, "le", None) == 10 for m in metadata), field


# ------ work_item ----------------------------------------------------------


def test_work_item_max_workers_constant():
    assert _make_work_item()._MAX_WORKERS == 10


def test_work_item_workers_clamped_at_runtime(caplog):
    w = _make_work_item()
    with caplog.at_level(logging.WARNING, logger="elitea_sdk.tools.ado.work_item.ado_wrapper"):
        list(w._base_loader(wiql="SELECT [System.Id] FROM workitems", workers=250))
    assert w._index_workers == 10
    assert any("exceeds cap" in rec.message for rec in caplog.records)


def test_work_item_workers_within_cap_not_clamped(caplog):
    w = _make_work_item()
    with caplog.at_level(logging.WARNING, logger="elitea_sdk.tools.ado.work_item.ado_wrapper"):
        list(w._base_loader(wiql="SELECT [System.Id] FROM workitems", workers=3))
    assert w._index_workers == 3
    assert not any("exceeds cap" in rec.message for rec in caplog.records)


def test_work_item_workers_default_serial():
    w = _make_work_item()
    list(w._base_loader(wiql="SELECT [System.Id] FROM workitems"))
    assert w._index_workers == w._DEFAULT_WORKERS == 1


def test_work_item_workers_schema_declares_le_10():
    params = _make_work_item()._index_tool_params()
    _type, field = params["workers"]
    metadata = getattr(field, "metadata", [])
    assert any(getattr(m, "le", None) == 10 for m in metadata), field
