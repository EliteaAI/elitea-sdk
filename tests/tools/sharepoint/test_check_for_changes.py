"""The SharePoint source listing for the index change check, and when a run may remove rows.

Covers the listing hook, the change marker with the list-field digest, the gap signal of the
backends and the gates that let an index run remove the chunks of deleted files.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.tools import index_change_check as check
from elitea_sdk.tools.base_indexer_toolkit import _IndexRunState
from elitea_sdk.tools.sharepoint import library_metadata as lm
from elitea_sdk.tools.sharepoint.api_wrapper import SharepointApiWrapper
from elitea_sdk.tools.sharepoint.rest_wrapper import SharepointRestWrapper

SITE = "https://tenant.sharepoint.com/sites/x"
LIB_FILE_URL = f"{SITE}/Lib/a.docx"


def listed_file(name="a.docx", modified="2026-01-01", file_id="f1", link=None):
    return {"Name": name, "Path": f"/sites/x/Lib/{name}", "Created": "c", "Modified": modified,
            "Link": link or f"{SITE}/Lib/{name}", "id": file_id}


class Backend:
    """Backend double: it can list, and it has no way to read file content."""

    def __init__(self, files=(), gaps=()):
        self.files = list(files)
        self.gaps = list(gaps)
        self.calls = []

    def get_files_list(self, folder_name=None, limit_files=100, form_name=None, include_extensions=None,
                       skip_extensions=None, on_file_skipped=None, on_listing_gap=None):
        self.calls.append({"folder": folder_name, "limit": limit_files, "form": form_name,
                           "include": include_extensions, "skip": skip_extensions})
        for reason in self.gaps:
            on_listing_gap(reason)
        return self.files[:limit_files]


def toolkit_with(backend, library=None):
    toolkit = SharepointApiWrapper.model_construct(site_url=SITE)
    toolkit._backend = backend
    toolkit._log_tool_event = lambda *a, **k: None
    toolkit._library_join_context = lambda *a, **k: library
    return toolkit


def library_context(metadata, categories=None):
    index = lm.LibraryIndex()
    index.add(lm.LibraryEntry(list_name="Docs", item_id="7", path="/sites/x/Lib/a.docx", name="a.docx",
                              metadata=metadata))
    settings = lm.LibrarySettings.from_settings(library_lists=["Docs"], categories=categories)
    return settings, index, lm.JoinCounts()


def stored_row(listing_item, **extra):
    """A base row as the indexer stores it for a listed file."""
    return {"metadata": {"id": listing_item.key, "Name": listing_item.name, "Path": listing_item.path,
                         "updated_on": "2026-01-01", **extra}}


class TestListingHook:

    def test_listing_uses_the_saved_scope_and_reads_no_content(self):
        backend = Backend([listed_file()])
        toolkit = toolkit_with(backend)

        listing = toolkit._list_source_items({
            "path": "Lib/sub", "form_name": "Lib", "include_extensions": ["docx"],
            "skip_extensions": ["tmp"], "limit_files": 50,
        })

        assert backend.calls == [{"folder": "Lib/sub", "limit": 50, "form": "Lib",
                                  "include": ["docx"], "skip": ["tmp"]}]
        assert [(i.key, i.name, i.path, i.marker) for i in listing.items] == [
            ("f1", "a.docx", "/sites/x/Lib/a.docx", "2026-01-01")]
        assert listing.complete is True

    def test_the_categories_filter_of_the_saved_scope_drops_files(self):
        backend = Backend([listed_file()])
        library = library_context({"category": "hr"}, categories=["finance"])

        listing = toolkit_with(backend, library)._list_source_items({"limit_files": 50, "categories": ["finance"]})

        assert listing.items == []

    def test_onenote_rows_are_left_out_and_named_in_the_report(self):
        toolkit = toolkit_with(Backend([listed_file()]))
        listing = toolkit._list_source_items({"include_onenote": True, "limit_files": 50})

        assert toolkit._is_unchecked_row({"metadata": {"source_type": "onenote"}}) is True
        assert toolkit._is_unchecked_row({"metadata": {"Name": "a.docx"}}) is False
        report = check.build_report("idx", check.compare(listing, {}, {}), listing, True, excluded_rows=2)
        assert "OneNote pages are not checked" in report["warnings"]
        assert any("OneNote" in blocker for blocker in listing.attest_blockers)

    def test_the_limit_is_a_gap_when_files_are_left_over(self):
        toolkit = toolkit_with(Backend([listed_file(file_id="f1"), listed_file("b.docx", file_id="f2")]))
        backend_files = toolkit._backend.files
        toolkit._backend.get_files_list = lambda *a, **k: backend_files   # a backend that ignores the cap

        listing = toolkit._list_source_items({"limit_files": 1})

        assert len(listing.items) == 1
        assert listing.complete is False
        assert "limit" in listing.gaps[0]

    def test_a_backend_gap_makes_the_listing_incomplete_and_blocks_removal(self):
        toolkit = toolkit_with(Backend([listed_file()], gaps=["a document library could not be listed"]))

        listing = toolkit._list_source_items({"limit_files": 50})

        assert listing.complete is False
        assert listing.gaps == ["a document library could not be listed"]
        assert listing.attest_blockers == ["a document library could not be listed"]
        stored = {"f1": check.ListedItem("f1", "a.docx", "/sites/x/Lib/a.docx", "2026-01-01"),
                  "f2": check.ListedItem("f2", "gone.docx", "/sites/x/Lib/gone.docx", "2026-01-01")}
        comparison = check.compare(listing, stored, {})
        report = check.build_report("idx", comparison, listing, True)
        assert report["counts"]["removed"] == 0
        assert "a document library could not be listed" in report["reason"] or report["warnings"]


class TestListFieldMarker:

    def listing_marker(self, metadata):
        toolkit = toolkit_with(Backend([listed_file()]), library_context(metadata))
        return toolkit, toolkit._list_source_items({"limit_files": 50, "library_lists": ["Docs"]}).items[0].marker

    def test_a_list_field_edit_changes_the_marker_though_the_file_did_not_change(self):
        _, before = self.listing_marker({"owner": "Ann"})
        _, after = self.listing_marker({"owner": "Bob"})

        assert before != after
        assert before.startswith("2026-01-01|")

    def test_an_unchanged_row_matches_its_listing_marker(self):
        toolkit, marker = self.listing_marker({"owner": "Ann"})
        digest = marker.split("|", 1)[1]
        item = toolkit._list_source_items({"limit_files": 50, "library_lists": ["Docs"]}).items[0]

        stored = toolkit._indexed_item("f1", stored_row(item, library_digest=digest))

        assert stored.marker == item.marker

    def test_a_row_without_a_digest_is_not_reported_as_changed(self):
        toolkit, _ = self.listing_marker({"owner": "Ann"})
        item = toolkit._list_source_items({"limit_files": 50, "library_lists": ["Docs"]}).items[0]

        stored = toolkit._indexed_item("f1", stored_row(item))

        assert stored.marker == item.marker

    def test_the_loader_writes_the_same_digest_the_check_computes(self):
        toolkit = toolkit_with(Backend([listed_file()]), library_context({"owner": "Ann"}))
        docs = list(toolkit._base_loader(limit_files=50, library_lists=["Docs"]))
        item = toolkit._list_source_items({"limit_files": 50, "library_lists": ["Docs"]}).items[0]

        assert toolkit._document_marker(docs[0]) == item.marker

    def test_files_without_library_lists_keep_the_plain_marker(self):
        toolkit = toolkit_with(Backend([listed_file()]))

        item = toolkit._list_source_items({"limit_files": 50}).items[0]

        assert item.marker == "2026-01-01"


def run_state(meta_existed=True, last_scope=None, run_scope=None):
    return _IndexRunState(run_id="r1", meta_existed=meta_existed, last_run_scope=last_scope, run_scope=run_scope)


class TestOrphanRemovalGates:
    """The index run may remove rows only when it proved which files are gone."""

    def run_loader(self, files, gaps=(), run=None, **kwargs):
        toolkit = toolkit_with(Backend(files, gaps=gaps))
        scope = toolkit._index_scope(kwargs)
        toolkit._index_run = run or run_state(meta_existed=False, run_scope=scope)
        docs = list(toolkit._base_loader(limit_files=kwargs.pop("limit_files", 50), **kwargs))
        return toolkit, docs

    def test_a_new_index_with_a_clean_listing_attests(self):
        toolkit, docs = self.run_loader([listed_file()])

        assert len(docs) == 1
        assert toolkit._index_run.loader_attested is True

    def test_the_same_scope_as_the_last_run_attests(self):
        toolkit = toolkit_with(Backend([listed_file()]))
        scope = toolkit._index_scope({"path": "Lib"})
        run = run_state(last_scope=scope, run_scope=scope)

        toolkit, _ = self.run_loader([listed_file()], run=run, path="Lib")

        assert toolkit._index_run.loader_attested is True

    def test_a_narrower_scope_than_the_last_run_does_not_attest(self):
        toolkit = toolkit_with(Backend())
        wide = toolkit._index_scope({"path": "Lib"})
        narrow = toolkit._index_scope({"path": "Lib/sub"})

        toolkit, _ = self.run_loader([listed_file()], run=run_state(last_scope=wide, run_scope=narrow), path="Lib/sub")

        assert toolkit._index_run.loader_attested is False

    def test_a_row_without_a_stored_scope_does_not_attest_once(self):
        scope = toolkit_with(Backend())._index_scope({})

        toolkit, _ = self.run_loader([listed_file()], run=run_state(last_scope=None, run_scope=scope))

        assert toolkit._index_run.loader_attested is False

    def test_a_listing_gap_does_not_attest(self):
        toolkit, _ = self.run_loader([listed_file()], gaps=["a folder page could not be read"])

        assert toolkit._index_run.loader_attested is False

    def test_a_listing_at_the_limit_does_not_attest(self):
        toolkit, _ = self.run_loader([listed_file(file_id="f1"), listed_file("b.docx", file_id="f2")], limit_files=2)

        assert toolkit._index_run.loader_attested is False

    def test_onenote_in_the_run_does_not_attest(self):
        toolkit = toolkit_with(Backend([listed_file()]))
        toolkit._onenote_base_loader = lambda cfg: iter(())

        scope = toolkit._index_scope({"include_onenote": True})
        toolkit._index_run = run_state(meta_existed=False, run_scope=scope)
        list(toolkit._base_loader(limit_files=50, include_onenote=True))

        assert toolkit._index_run.loader_attested is False

    def test_a_loader_outside_an_index_run_attests_nothing(self):
        toolkit = toolkit_with(Backend([listed_file()]))

        assert list(toolkit._base_loader(limit_files=50))

    def test_scope_ignores_list_order_and_empty_values(self):
        toolkit = toolkit_with(Backend())

        assert toolkit._index_scope({"include_extensions": ["pdf", "docx"], "path": ""}) == \
            toolkit._index_scope({"include_extensions": ["docx", "pdf"], "path": None})
        assert toolkit._index_scope({"path": "a"}) != toolkit._index_scope({"path": "b"})


class TestRunScopeLifecycle:
    """index_meta_init reads the stored scope; a failed run keeps it, a good run replaces it."""

    @pytest.fixture
    def toolkit(self, monkeypatch):
        monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
        monkeypatch.setattr(SharepointApiWrapper, "_staging_active", lambda self: False)
        monkeypatch.setattr(SharepointApiWrapper, "_resolve_initiator", lambda self: None)
        monkeypatch.setattr(SharepointApiWrapper, "_log_tool_event", lambda self, *a, **k: None)
        monkeypatch.setattr("elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents",
                            lambda *a, **k: None)
        instance = SharepointApiWrapper.model_construct(site_url=SITE)
        object.__setattr__(instance, "vectorstore", object())
        object.__setattr__(instance, "_index_run", run_state(meta_existed=False))
        return instance

    def meta(self, monkeypatch, metadata):
        monkeypatch.setattr(VectorStoreWrapperBase, "get_index_meta",
                            lambda self, name: {"id": "m1", "metadata": metadata} if metadata is not None else None)

    def test_a_new_index_has_no_meta_row_and_no_stored_scope(self, toolkit, monkeypatch):
        self.meta(monkeypatch, None)

        toolkit.index_meta_init("idx", {"path": "Lib"})

        run = toolkit._index_run
        assert (run.meta_existed, run.last_run_scope) == (False, None)
        assert run.run_scope["path"] == "Lib"

    def test_an_existing_row_hands_its_stored_scope_to_the_run(self, toolkit, monkeypatch):
        self.meta(monkeypatch, {"state": "completed", "history": "[]", "last_run_scope": {"path": "Lib"}})

        toolkit.index_meta_init("idx", {"path": "Lib"})

        run = toolkit._index_run
        assert run.meta_existed is True
        assert run.last_run_scope == {"path": "Lib"}

    def test_a_row_without_the_key_is_a_legacy_row(self, toolkit, monkeypatch):
        self.meta(monkeypatch, {"state": "completed", "history": "[]"})

        toolkit.index_meta_init("idx", {"path": "Lib"})

        assert toolkit._index_run.meta_existed is True
        assert toolkit._index_run.last_run_scope is None


class TestRestGapPoints:

    def rest(self, graph_helper, client):
        wrapper = SharepointRestWrapper.__new__(SharepointRestWrapper)
        wrapper.site_url = SITE
        wrapper._client = client
        wrapper._graph_helper = lambda: graph_helper
        return wrapper

    def test_a_library_that_cannot_be_listed_is_a_gap(self):
        library = SimpleNamespace(properties={"EntityTypeName": "Lib", "Title": "Lib"})
        client = MagicMock()
        client.web.lists.filter.return_value.get.return_value.execute_query.return_value = [library]
        wrapper = self.rest(MagicMock(), client)
        gaps = []
        with patch.object(SharepointRestWrapper, "_rest_bfs_files", side_effect=RuntimeError("denied")):
            wrapper.get_files_list(limit_files=10, on_listing_gap=gaps.append)

        assert "a document library could not be listed" in gaps

    def test_a_switch_from_graph_to_rest_is_a_gap(self):
        helper = MagicMock()
        helper.get_files_list.side_effect = RuntimeError("graph down")
        client = MagicMock()
        client.web.lists.filter.return_value.get.return_value.execute_query.return_value = []
        gaps = []

        self.rest(helper, client).get_files_list("Lib/sub", 10, on_listing_gap=gaps.append)

        assert "the listing fell back from Graph to REST" in gaps

    def test_the_limit_is_a_gap(self):
        library = SimpleNamespace(properties={"EntityTypeName": "Lib", "Title": "Lib"})
        client = MagicMock()
        client.web.lists.filter.return_value.get.return_value.execute_query.return_value = [library]
        files = [SimpleNamespace(properties={
            "ServerRelativeUrl": "/sites/x/Lib/a.docx", "Name": "a.docx", "TimeCreated": "c",
            "TimeLastModified": "m", "LinkingUrl": "", "UniqueId": "u1"})]
        gaps = []
        with patch.object(SharepointRestWrapper, "_rest_bfs_files", return_value=files):
            result = self.rest(MagicMock(), client).get_files_list(limit_files=1, on_listing_gap=gaps.append)

        assert len(result) == 1
        assert gaps == ["the file limit was reached before the listing ended"]
