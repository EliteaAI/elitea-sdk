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

"""The index change check: pure comparison, report shape and the tool wiring.

Lives outside tests/tools/index/ because CI skips that directory.
"""

import json
import time
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.tools import ToolException

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.tools import index_change_check as check
from elitea_sdk.tools.base_indexer_toolkit import (
    SDK_OWNED_META_KEYS,
    TASK_DISCONNECTED_TIMEOUT_DEFAULT,
    IndexTools,
    _IndexRunState,
)
from elitea_sdk.tools.elitea_base import INDEX_TOOL_NAMES
from elitea_sdk.tools.non_code_indexer_toolkit import NonCodeIndexerToolkit


def item(key, marker="1", path=None, name=None):
    name = name or key
    return check.ListedItem(key=key, name=name, path=path or f"/lib/{name}", marker=marker)


def run_compare(listed, indexed, unindexed=None, **listing_args):
    listing = check.SourceListing(items=listed, **listing_args)
    stored = {entry.key: entry for entry in indexed}
    comparison = check.compare(listing, stored, unindexed)
    return comparison, check.build_report("idx", comparison, listing, unindexed_recorded=unindexed is not None)


class TestComparison:

    def test_no_change_is_up_to_date_with_zero_counts(self):
        _, report = run_compare([item("a"), item("b")], [item("a"), item("b")], unindexed={})

        assert report["status"] == check.STATUS_UP_TO_DATE
        assert report["counts"] == {"added": 0, "changed": 0, "removed": 0, "unchanged": 2, "not_indexed": 0}
        assert report["items"] == {"added": [], "changed": [], "removed": []}
        assert report["listing_complete"] is True
        assert report["schema_version"] == 1

    def test_new_key_is_add_only(self):
        _, report = run_compare([item("a"), item("new")], [item("a")], unindexed={})

        assert report["status"] == check.STATUS_ADD_ONLY
        assert report["counts"]["added"] == 1
        assert report["items"]["added"] == ["/lib/new"]
        assert "index_data" in report["reason"]

    def test_changed_marker_needs_clean_reindex(self):
        _, report = run_compare([item("a", marker="2")], [item("a", marker="1")], unindexed={})

        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX
        assert report["counts"]["changed"] == 1
        assert report["items"]["changed"] == ["/lib/a"]
        assert "clean_index" in report["reason"]

    def test_added_alongside_changed_still_needs_clean_reindex(self):
        _, report = run_compare([item("a", marker="2"), item("new")], [item("a")], unindexed={})

        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX

    def test_removed_key_needs_clean_reindex(self):
        _, report = run_compare([item("a")], [item("a"), item("gone")], unindexed={})

        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX
        assert report["items"]["removed"] == ["/lib/gone"]

    def test_rename_is_old_path_removed_and_new_path_added(self):
        _, report = run_compare(
            [item("a", path="/lib/new-name.docx")],
            [item("a", path="/lib/old-name.docx")],
            unindexed={},
        )

        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX
        assert report["items"]["removed"] == ["/lib/old-name.docx"]
        assert report["items"]["added"] == ["/lib/new-name.docx"]
        assert report["counts"]["changed"] == 0

    def test_incomplete_listing_never_reports_removed(self):
        comparison, report = run_compare(
            [item("a")], [item("a"), item("maybe-gone")], unindexed={},
            complete=False, gaps=["library 'Docs' was skipped after an error"],
        )

        assert comparison.removed == []
        assert report["counts"]["removed"] == 0
        assert report["listing_complete"] is False
        assert "library 'Docs' was skipped after an error" in report["warnings"]
        assert "incomplete" in report["reason"]

    def test_incomplete_listing_reports_a_path_change_as_changed(self):
        _, report = run_compare(
            [item("a", path="/lib/new.docx")], [item("a", path="/lib/old.docx")], unindexed={},
            complete=False, gaps=["gap"],
        )

        assert report["counts"]["changed"] == 1
        assert report["counts"]["removed"] == 0

    def test_a_gap_makes_the_listing_incomplete_even_when_complete_is_set(self):
        _, report = run_compare([item("a")], [item("a"), item("b")], unindexed={}, gaps=["gap"])

        assert report["listing_complete"] is False
        assert report["counts"]["removed"] == 0

    def test_listing_with_no_key_overlap_is_not_trusted(self):
        _, report = run_compare(
            [item("graph-id-1"), item("graph-id-2")], [item("rest-uid-1"), item("rest-uid-2")], unindexed={},
        )

        assert report["listing_complete"] is False
        assert report["counts"]["removed"] == 0
        assert check.KEY_MISMATCH_WARNING in report["warnings"]

    def test_empty_index_is_not_a_key_mismatch(self):
        _, report = run_compare([item("a")], [], unindexed={})

        assert report["listing_complete"] is True
        assert report["counts"]["added"] == 1

    def test_unindexed_key_with_the_same_marker_is_not_added(self):
        _, report = run_compare(
            [item("a"), item("empty-file", marker="7")], [item("a")], unindexed={"empty-file": "7"},
        )

        assert report["status"] == check.STATUS_UP_TO_DATE
        assert report["counts"]["added"] == 0
        assert report["counts"]["not_indexed"] == 1

    def test_unindexed_key_with_another_marker_is_added(self):
        _, report = run_compare(
            [item("a"), item("empty-file", marker="8")], [item("a")], unindexed={"empty-file": "7"},
        )

        assert report["status"] == check.STATUS_ADD_ONLY
        assert report["counts"]["not_indexed"] == 0

    def test_index_without_the_unindexed_record_adds_a_warning(self):
        _, report = run_compare([item("a")], [item("a")], unindexed=None)

        assert check.UNINDEXED_NOT_RECORDED_WARNING in report["warnings"]


class TestReportCaps:

    def test_path_lists_are_capped_and_counts_stay_full(self):
        total = check.MAX_LISTED_PER_GROUP + 25
        _, report = run_compare([item(f"k{n}") for n in range(total)], [], unindexed={})

        assert report["counts"]["added"] == total
        assert len(report["items"]["added"]) == check.MAX_LISTED_PER_GROUP
        assert report["items_truncated"] is True

    def test_a_short_list_is_not_truncated(self):
        _, report = run_compare([item("a")], [], unindexed={})

        assert report["items_truncated"] is False

    def test_reason_names_the_cause_when_the_next_run_cannot_remove_rows(self):
        _, report = run_compare(
            [item("a")], [item("a"), item("gone")], unindexed={},
            attest_blockers=["the file limit was reached"],
        )

        assert "will not remove the chunks of removed items" in report["reason"]
        assert "the file limit was reached" in report["reason"]

    def test_a_reason_for_changed_items_only_does_not_talk_about_removed_items(self):
        _, report = run_compare(
            [item("a", marker="2")], [item("a", marker="1")], unindexed={},
            attest_blockers=["OneNote pages are included in the index and are not checked"],
        )

        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX
        assert "Changed items found" in report["reason"]
        assert "will not remove" not in report["reason"]
        assert "OneNote" not in report["reason"]

    def test_excluded_rows_add_the_toolkit_note_and_reason(self):
        listing = check.SourceListing(
            items=[item("a")], unchecked_note="OneNote pages are not checked",
            unchecked_reason="Turn include_onenote on to keep them.",
        )
        comparison = check.compare(listing, {"a": item("a")}, {})

        report = check.build_report("idx", comparison, listing, unindexed_recorded=True, excluded_rows=3)

        assert "OneNote pages are not checked" in report["warnings"]
        assert "Turn include_onenote on to keep them." in report["reason"]


class TestReportBounds:

    def test_a_gap_repeated_for_every_folder_shows_once(self):
        _, report = run_compare([item("a")], [item("a")], unindexed={}, gaps=["a folder page could not be read"] * 500)

        assert report["warnings"].count("a folder page could not be read") == 1

    def test_distinct_warnings_are_capped_with_a_count_of_the_rest(self):
        gaps = [f"folder {n} could not be read" for n in range(check.MAX_REPORT_LINES + 7)]

        _, report = run_compare([item("a")], [item("a")], unindexed={}, gaps=gaps)

        assert len(report["warnings"]) == check.MAX_REPORT_LINES + 1
        assert "+7 more" in report["warnings"]

    def test_the_reason_lists_each_blocker_once_and_caps_the_rest(self):
        blockers = ["a folder page could not be read"] * 300 + [f"folder {n} failed" for n in range(40)]

        _, report = run_compare([item("a")], [item("a"), item("gone")], unindexed={}, attest_blockers=blockers)

        assert report["reason"].count("a folder page could not be read") == 1
        assert "+21 more" in report["reason"]
        assert len(report["reason"]) < 1000


class _Indexed(NonCodeIndexerToolkit):
    """Toolkit double with a source listing."""

    def _list_source_items(self, index_configuration):
        self.harness.seen_configuration = index_configuration
        return check.SourceListing(items=self.harness.listing_items)


class _NoListing(NonCodeIndexerToolkit):
    """Toolkit double that does not override the listing hook."""


def row(key, marker="1", **extra):
    metadata = {"id": key, "name": key, "path": f"/lib/{key}", "updated_on": marker, "collection": "idx"}
    metadata.update(extra)
    return {"metadata": metadata, "id": f"pk-{key}", "all_chunks": [f"pk-{key}"],
            "dependent_docs": [], "parent_id": -1, "pk_fallback": False}


class Harness:
    """Test data the toolkit double reads, held outside the pydantic model."""

    def __init__(self):
        self.listing_items = []
        self.stored_rows = {}
        self.stored_meta = {
            "id": "meta-1", "metadata": {"index_configuration": {"path": "/lib"}, "unindexed_items": {}},
        }
        self.writes = []
        self.seen_configuration = None
        self.toolkit = _Indexed.model_construct()
        object.__setattr__(self.toolkit, "harness", self)

    def check_for_changes(self, index_name):
        return self.toolkit.check_for_changes(index_name)


@pytest.fixture
def indexed_toolkit(monkeypatch):
    harness = Harness()
    writes = harness.writes
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_index_meta", lambda self, name: self.harness.stored_meta)
    monkeypatch.setattr(_Indexed, "_get_indexed_data", lambda self, name: self.harness.stored_rows)
    monkeypatch.setattr(_Indexed, "_available_indexes_note", lambda self: "Available indexes: idx")
    for name in ("index_meta_init", "index_meta_update", "_emit_index_event", "_clean_index"):
        monkeypatch.setattr(_Indexed, name, lambda self, *a, _name=name, **kw: writes.append(_name))
    monkeypatch.setattr(VectorStoreWrapperBase, "_clean_collection",
                        lambda self, *a, **kw: writes.append("_clean_collection"))
    return harness


class TestCheckTool:

    def test_check_for_changes_is_an_index_tool_name(self):
        assert "check_for_changes" in INDEX_TOOL_NAMES
        assert IndexTools.CHECK_FOR_CHANGES.value == "check_for_changes"

    def test_tool_is_offered_only_when_the_listing_hook_is_overridden(self, monkeypatch):
        names = {}
        for cls in (_Indexed, _NoListing):
            instance = cls.model_construct()
            monkeypatch.setattr(cls, "_index_tool_params", lambda self: {})
            names[cls] = {tool["name"] for tool in instance.get_available_tools()}

        assert "check_for_changes" in names[_Indexed]
        assert "check_for_changes" not in names[_NoListing]
        assert names[_NoListing] >= {"index_data", "search_index", "remove_index", "list_indexes"}

    def test_check_reports_the_difference_and_writes_nothing(self, indexed_toolkit):
        indexed_toolkit.listing_items = [item("a", marker="2"), item("new")]
        indexed_toolkit.stored_rows = {"a": row("a", "1"), "gone": row("gone")}

        report = indexed_toolkit.check_for_changes("idx")

        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX
        assert report["counts"] == {"added": 1, "changed": 1, "removed": 1, "unchanged": 0, "not_indexed": 0}
        assert indexed_toolkit.writes == []
        assert not hasattr(indexed_toolkit.toolkit, "_index_run")

    def test_check_hands_the_saved_configuration_to_the_hook(self, indexed_toolkit):
        indexed_toolkit.check_for_changes("idx")

        assert indexed_toolkit.seen_configuration == {"path": "/lib"}

    def test_a_configuration_stored_as_a_json_string_reaches_the_hook_as_a_dict(self, indexed_toolkit):
        indexed_toolkit.stored_meta["metadata"]["index_configuration"] = json.dumps({"path": "/lib"})

        report = indexed_toolkit.check_for_changes("idx")

        assert indexed_toolkit.seen_configuration == {"path": "/lib"}
        assert report["status"] == "up_to_date"

    @pytest.fixture
    def scoped_toolkit(self, indexed_toolkit, monkeypatch):
        monkeypatch.setattr(_Indexed, "_run_scope_keys", lambda self, cfg: {"path": cfg.get("path")})
        indexed_toolkit.listing_items = [item("a", marker="2")]
        # A removed row is what makes the scope cause matter: only then does the reason talk about removal.
        indexed_toolkit.stored_rows = {"a": row("a", "1"), "gone": row("gone")}
        return indexed_toolkit

    def test_reason_names_a_scope_that_differs_from_the_last_run(self, scoped_toolkit):
        scoped_toolkit.stored_meta["metadata"]["last_run_scope"] = {"path": "/other"}

        report = scoped_toolkit.check_for_changes("idx")

        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX
        assert "the saved scope differs from the last run" in report["reason"]

    def test_reason_names_a_missing_last_run_scope(self, scoped_toolkit):
        report = scoped_toolkit.check_for_changes("idx")

        assert "no earlier run recorded its scope" in report["reason"]

    def test_reason_reads_the_scope_from_the_history_after_the_dispatch_reset(self, scoped_toolkit):
        scoped_toolkit.stored_meta["metadata"]["history"] = json.dumps([
            {"state": "completed", "last_run_scope": {"path": "/lib"}},
            {"state": "in_progress"},
        ])

        report = scoped_toolkit.check_for_changes("idx")

        assert "scope" not in report["reason"]
        assert "will not remove" not in report["reason"]

    def test_a_toolkit_without_a_scope_names_no_scope_cause(self, indexed_toolkit):
        indexed_toolkit.listing_items = [item("a", marker="2")]
        indexed_toolkit.stored_rows = {"a": row("a", "1")}

        report = indexed_toolkit.check_for_changes("idx")

        assert "scope" not in report["reason"]

    def test_dependent_pk_fallback_and_child_rows_are_not_compared(self, indexed_toolkit):
        indexed_toolkit.listing_items = [item("a")]
        parent = row("a")
        parent["dependent_docs"] = ["a-attachment"]
        indexed_toolkit.stored_rows = {
            "a": parent,
            "a-attachment": row("a-attachment"),
            "pk-only": {**row("pk-only"), "pk_fallback": True},
            "child": {**row("child"), "parent_id": "a"},
        }

        report = indexed_toolkit.check_for_changes("idx")

        assert report["status"] == check.STATUS_UP_TO_DATE
        assert report["counts"]["removed"] == 0

    def test_recorded_unindexed_items_count_as_not_indexed(self, indexed_toolkit):
        indexed_toolkit.listing_items = [item("empty-file", marker="5")]
        indexed_toolkit.stored_meta["metadata"]["unindexed_items"] = {"empty-file": "5"}

        report = indexed_toolkit.check_for_changes("idx")

        assert report["counts"]["not_indexed"] == 1
        assert report["counts"]["added"] == 0

    def test_unindexed_items_stored_as_json_text_are_read(self, indexed_toolkit):
        indexed_toolkit.listing_items = [item("empty-file", marker="5")]
        indexed_toolkit.stored_meta["metadata"]["unindexed_items"] = '{"empty-file": "5"}'

        assert indexed_toolkit.check_for_changes("idx")["counts"]["not_indexed"] == 1

    def test_a_row_the_toolkit_marks_unchecked_is_left_out(self, indexed_toolkit, monkeypatch):
        indexed_toolkit.listing_items = [item("a")]
        indexed_toolkit.stored_rows = {"a": row("a"), "note": row("note", source_type="onenote")}
        monkeypatch.setattr(
            _Indexed, "_is_unchecked_row",
            lambda self, entry: entry["metadata"].get("source_type") == "onenote",
        )

        report = indexed_toolkit.check_for_changes("idx")

        assert report["counts"]["removed"] == 0

    @pytest.mark.parametrize("meta", [
        None,
        {"id": "meta-1", "metadata": {}},
        {"id": "meta-1", "metadata": {"index_configuration": None}},
        {"id": "meta-1", "metadata": {"index_configuration": "not a dict"}},
        {"id": "meta-1", "metadata": {"index_configuration": {}}},
        {"id": "meta-1", "metadata": {"index_configuration": "{}"}},
        {"id": "meta-1", "metadata": {"index_configuration": "[1, 2]"}},
        {"id": "meta-1", "metadata": {"index_configuration": "null"}},
    ])
    def test_missing_index_or_configuration_raises_with_the_index_list(self, indexed_toolkit, meta):
        object.__setattr__(indexed_toolkit, "stored_meta", meta)

        with pytest.raises(ToolException, match="Available indexes: idx"):
            indexed_toolkit.check_for_changes("idx")

    def test_empty_index_name_is_refused(self, indexed_toolkit):
        with pytest.raises(ToolException, match="non-empty 'index_name'"):
            indexed_toolkit.check_for_changes("  ")

    def test_a_toolkit_without_the_hook_refuses(self):
        with pytest.raises(ToolException, match="cannot check"):
            _NoListing.model_construct().check_for_changes("idx")


class TestCheckReadsAndGuards:

    def test_a_check_during_a_run_is_refused_before_the_source_is_listed(self, indexed_toolkit):
        indexed_toolkit.stored_meta["metadata"]["state"] = "in_progress"

        with pytest.raises(ToolException, match="being indexed now") as refused:
            indexed_toolkit.check_for_changes("idx")

        assert "start a new index run" in str(refused.value)
        assert indexed_toolkit.seen_configuration is None

    def stage(self, indexed_toolkit, ticked_ago, staging=True):
        adapter = SimpleNamespace(supports_run_staging=staging)
        object.__setattr__(indexed_toolkit.toolkit, "vector_adapter", adapter)
        metadata = indexed_toolkit.stored_meta["metadata"]
        metadata["state"] = "in_progress"
        metadata["updated_on"] = time.time() - ticked_ago

    def test_a_staged_run_without_a_heartbeat_for_the_timeout_no_longer_blocks_the_check(self, indexed_toolkit):
        indexed_toolkit.listing_items = [item("a")]
        indexed_toolkit.stored_rows = {"a": row("a")}
        self.stage(indexed_toolkit, ticked_ago=2 * TASK_DISCONNECTED_TIMEOUT_DEFAULT)

        report = indexed_toolkit.check_for_changes("idx")

        assert report["status"] == check.STATUS_UP_TO_DATE
        assert any("stopped before it finished" in warning for warning in report["warnings"])

    def test_a_staged_run_with_a_fresh_heartbeat_still_blocks_the_check(self, indexed_toolkit):
        self.stage(indexed_toolkit, ticked_ago=60)

        with pytest.raises(ToolException, match="being indexed now"):
            indexed_toolkit.check_for_changes("idx")

    def test_a_run_without_staging_has_no_heartbeat_so_an_old_row_does_not_prove_it_stopped(self, indexed_toolkit):
        self.stage(indexed_toolkit, ticked_ago=2 * TASK_DISCONNECTED_TIMEOUT_DEFAULT, staging=False)

        with pytest.raises(ToolException, match="being indexed now"):
            indexed_toolkit.check_for_changes("idx")

    def test_a_staged_run_with_an_unreadable_heartbeat_still_blocks_the_check(self, indexed_toolkit):
        self.stage(indexed_toolkit, ticked_ago=0)
        indexed_toolkit.stored_meta["metadata"]["updated_on"] = None

        with pytest.raises(ToolException, match="being indexed now"):
            indexed_toolkit.check_for_changes("idx")

    def test_skipped_items_come_from_the_history_after_the_dispatch_reset(self, indexed_toolkit):
        indexed_toolkit.listing_items = [item("empty-file", marker="5")]
        indexed_toolkit.stored_meta["metadata"] = {
            "index_configuration": {"path": "/lib"},
            "history": json.dumps([
                {"state": "completed", "unindexed_items": {"empty-file": "5"}},
                {"state": "failed"},
            ]),
        }

        report = indexed_toolkit.check_for_changes("idx")

        assert report["counts"]["not_indexed"] == 1
        assert report["counts"]["added"] == 0
        assert check.UNINDEXED_NOT_RECORDED_WARNING not in report["warnings"]

    def test_run_facts_stored_as_json_strings_are_read(self, indexed_toolkit, monkeypatch):
        monkeypatch.setattr(_Indexed, "_run_scope_keys", lambda self, cfg: {"path": cfg.get("path")})
        indexed_toolkit.listing_items = [item("a", marker="2"), item("empty-file", marker="5")]
        indexed_toolkit.stored_rows = {"a": row("a", "1")}
        indexed_toolkit.stored_meta["metadata"].update({
            "last_run_scope": json.dumps({"path": "/lib"}), "unindexed_items": json.dumps({"empty-file": "5"}),
        })

        report = indexed_toolkit.check_for_changes("idx")

        assert "scope" not in report["reason"]
        assert report["counts"]["not_indexed"] == 1

    def test_an_empty_listing_of_a_non_empty_index_is_incomplete_and_removes_nothing(self):
        comparison, report = run_compare([], [item("a"), item("b")], unindexed={})

        assert comparison.removed == []
        assert report["counts"]["removed"] == 0
        assert report["listing_complete"] is False
        assert check.EMPTY_LISTING_WARNING in report["warnings"]

    def test_an_empty_listing_of_an_empty_index_stays_up_to_date(self):
        _, report = run_compare([], [], unindexed={})

        assert report["listing_complete"] is True
        assert report["status"] == check.STATUS_UP_TO_DATE

    def test_the_reason_for_changed_items_does_not_demand_clean_index_for_every_change(self):
        _, report = run_compare([item("a", marker="2")], [item("a", marker="1")], unindexed={})

        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX
        assert "plain index_data run re-indexes" in report["reason"]
        assert "Removed items" not in report["reason"]

    def test_the_reason_for_removed_items_names_clean_index(self):
        _, report = run_compare([item("a")], [item("a"), item("gone")], unindexed={})

        assert "Removed items found: run index_data with clean_index" in report["reason"]
        assert "Changed items" not in report["reason"]


class TestIncompleteStatus:
    """A listing that cannot be trusted never reads as up to date or add only."""

    def test_a_listing_page_that_failed_is_incomplete_not_up_to_date(self):
        _, report = run_compare([item("a")], [item("a")], unindexed={}, complete=False, gaps=["page 2 could not be read"])

        assert report["status"] == check.STATUS_INCOMPLETE
        assert "page 2 could not be read" in report["warnings"]
        assert "incomplete" in report["reason"]

    def test_the_file_limit_reached_is_incomplete_even_with_a_new_item(self):
        _, report = run_compare([item("a"), item("new")], [item("a")], unindexed={}, complete=False,
                                gaps=["the listing stopped at the file limit"])

        assert report["status"] == check.STATUS_INCOMPLETE
        assert report["counts"]["added"] == 1
        assert "1 item(s)" in report["reason"]

    def test_ids_of_another_family_are_incomplete_not_add_only(self):
        _, report = run_compare([item("graph-1")], [item("rest-1")], unindexed={})

        assert report["status"] == check.STATUS_INCOMPLETE
        assert check.KEY_MISMATCH_WARNING in report["warnings"]

    def test_an_empty_listing_of_a_non_empty_index_is_incomplete(self):
        _, report = run_compare([], [item("a")], unindexed={})

        assert report["status"] == check.STATUS_INCOMPLETE
        assert check.EMPTY_LISTING_WARNING in report["warnings"]

    def test_a_change_the_partial_listing_did_show_still_needs_a_clean_reindex(self):
        _, report = run_compare([item("a", marker="2")], [item("a", marker="1")], unindexed={}, complete=False,
                                gaps=["gap"])

        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX
        assert "listing is incomplete" in report["reason"]

    def test_a_complete_listing_never_reads_as_incomplete(self):
        _, report = run_compare([item("a"), item("new")], [item("a")], unindexed={})

        assert report["status"] == check.STATUS_ADD_ONLY


class _RecordingAdapter:
    supports_run_staging = True

    def __init__(self):
        self.patches = []

    def update_index_meta_keys(self, wrapper, meta_id, run_id, patch):
        self.patches.append(patch)
        return 1


@pytest.fixture
def meta_toolkit(monkeypatch):
    instance = _Indexed.model_construct()
    adapter = _RecordingAdapter()
    object.__setattr__(instance, "vector_adapter", adapter)
    object.__setattr__(instance, "_index_run", _IndexRunState(run_id="run-1"))
    stored = {"id": "meta-1", "metadata": {
        "collection": "idx", "state": "in_progress", "history": '[{"state": "in_progress"}]',
        "last_run_scope": {"path": "/old"}, "unindexed_items": {"old": "1"},
    }}
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_index_meta", lambda self, name: stored)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_indexed_count", lambda self, name: 3)
    monkeypatch.setattr(_Indexed, "_is_scheduled_run", lambda self: False)
    return instance, adapter


class TestRunFacts:

    def test_a_successful_write_keeps_the_scope_in_the_row_and_both_facts_in_the_last_history_entry(
            self, meta_toolkit):
        toolkit, adapter = meta_toolkit

        toolkit.index_meta_update(
            "idx", "completed", 3, last_run_scope={"path": "/new"}, unindexed_items={"k": "5"},
        )

        patch = adapter.patches[-1]
        assert json.loads(patch["last_run_scope"]) == {"path": "/new"}
        # The row holds no copy of the large skipped-items map; the history entry does.
        assert patch["unindexed_items"] is None
        history = json.loads(patch["history"])
        assert history[-1]["last_run_scope"] == {"path": "/new"}
        assert history[-1]["unindexed_items"] == {"k": "5"}

    def test_an_older_history_entry_loses_its_skipped_items_when_a_newer_run_records_them(self, meta_toolkit):
        toolkit, adapter = meta_toolkit
        stored = VectorStoreWrapperBase.get_index_meta(toolkit, "idx")
        stored["metadata"]["history"] = json.dumps([
            {"state": "completed", "unindexed_items": {"old": "1"}},
            {"state": "in_progress"},
        ])

        toolkit.index_meta_update("idx", "completed", 3, unindexed_items={"k": "5"})

        history = json.loads(adapter.patches[-1]["history"])
        assert [entry.get("unindexed_items") for entry in history] == [None, {"k": "5"}]

    def test_a_failed_write_keeps_the_skipped_items_of_the_last_good_entry(self, meta_toolkit):
        toolkit, adapter = meta_toolkit
        stored = VectorStoreWrapperBase.get_index_meta(toolkit, "idx")
        stored["metadata"]["history"] = json.dumps([
            {"state": "completed", "unindexed_items": {"old": "1"}},
            {"state": "in_progress"},
        ])

        toolkit.index_meta_update("idx", "failed", 0, error="boom")

        history = json.loads(adapter.patches[-1]["history"])
        assert history[0]["unindexed_items"] == {"old": "1"}
        assert "unindexed_items" not in history[-1]

    def test_an_older_history_entry_loses_its_scope_when_a_newer_run_records_one(self, meta_toolkit):
        toolkit, adapter = meta_toolkit
        stored = VectorStoreWrapperBase.get_index_meta(toolkit, "idx")
        stored["metadata"]["history"] = json.dumps([
            {"state": "completed", "last_run_scope": {"path": "/old"}},
            {"state": "in_progress"},
        ])

        toolkit.index_meta_update("idx", "completed", 3, last_run_scope={"path": "/new"})

        history = json.loads(adapter.patches[-1]["history"])
        assert [entry.get("last_run_scope") for entry in history] == [None, {"path": "/new"}]

    def test_a_failed_write_leaves_the_scope_of_the_last_good_entry_alone(self, meta_toolkit):
        toolkit, adapter = meta_toolkit
        stored = VectorStoreWrapperBase.get_index_meta(toolkit, "idx")
        stored["metadata"]["history"] = json.dumps([
            {"state": "completed", "last_run_scope": {"path": "/old"}},
            {"state": "in_progress"},
        ])

        toolkit.index_meta_update("idx", "failed", 0, error="boom")

        history = json.loads(adapter.patches[-1]["history"])
        assert history[0]["last_run_scope"] == {"path": "/old"}
        assert "last_run_scope" not in history[-1]

    def test_a_write_without_run_facts_keeps_the_stored_values(self, meta_toolkit):
        toolkit, adapter = meta_toolkit

        toolkit.index_meta_update("idx", "failed", 0, error="boom")

        patch = adapter.patches[-1]
        assert patch["last_run_scope"] == {"path": "/old"}
        assert patch["unindexed_items"] == {"old": "1"}

    def test_run_facts_are_sdk_owned_keys(self):
        assert {"last_run_scope", "unindexed_items"} <= set(SDK_OWNED_META_KEYS)

    def test_empty_documents_are_noted_by_key_and_marker_for_a_listing_toolkit(self, meta_toolkit):
        toolkit, _ = meta_toolkit

        toolkit._note_unindexed_item(Document(page_content="", metadata={"id": "f1", "updated_on": "9"}))

        assert toolkit._index_run.unindexed_items == {"f1": "9"}

    @pytest.mark.parametrize("failed_set", ["pipeline_failed_keys", "explicitly_failed_keys"])
    def test_a_failed_document_is_not_noted_as_a_skipped_item(self, meta_toolkit, failed_set):
        toolkit, _ = meta_toolkit
        getattr(toolkit._index_run, failed_set).add("f1")

        toolkit._note_unindexed_item(Document(page_content="", metadata={"id": "f1", "updated_on": "9"}))
        toolkit._note_unindexed_item(Document(page_content="", metadata={"id": "f2", "updated_on": "9"}))

        assert toolkit._index_run.unindexed_items == {"f2": "9"}

    def test_a_failed_new_file_reads_as_added_not_as_not_indexed(self, meta_toolkit):
        toolkit, _ = meta_toolkit
        toolkit._index_run.explicitly_failed_keys.add("f1")
        toolkit._note_unindexed_item(Document(page_content="", metadata={"id": "f1", "updated_on": "9"}))
        listing = check.SourceListing(items=[item("f1", marker="9")])

        comparison = check.compare(listing, {}, toolkit._index_run.unindexed_items)

        assert [added.key for added in comparison.added] == ["f1"]
        assert comparison.not_indexed == 0

    def test_a_toolkit_without_a_listing_notes_nothing(self):
        toolkit = _NoListing.model_construct()
        object.__setattr__(toolkit, "_index_run", _IndexRunState(run_id="run-1"))

        toolkit._note_unindexed_item(Document(page_content="", metadata={"id": "f1", "updated_on": "9"}))

        assert toolkit._index_run.unindexed_items == {}

    def test_orphan_collection_skips_dependent_fallback_and_child_rows(self):
        toolkit = _Indexed.model_construct()
        parent = row("kept")
        parent["dependent_docs"] = ["att"]
        data = {
            "kept": parent, "att": row("att"), "gone": row("gone"),
            "fallback": {**row("fallback"), "pk_fallback": True},
            "child": {**row("child"), "parent_id": "kept"},
        }

        ids, count = toolkit._collect_orphan_candidates(data, seen_keys={"kept"})

        assert (ids, count) == (["pk-gone"], 1)
