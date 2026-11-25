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

"""`check_for_changes` on a SharePoint index of 2000 files.

The source is a faked `get_files_list`: no network and no file content. The run goes through the
listing hook, the stored-row read and the comparison.
Only the in-memory part is timed: the Graph listing and the pgvector read of the stored rows are
faked, so the time bound shows that the comparison and the report scale, not that a real
2000-file library lists fast.
Lives beside the other SharePoint tests, outside tests/tools/index/ because CI skips that directory.
"""

import json
import time

import pytest

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.tools import index_change_check as check
from elitea_sdk.tools.sharepoint.api_wrapper import SharepointApiWrapper
from tests.tools.sharepoint.helpers import (
    Backend,
    listed_file,
    stored_row,
    toolkit_with,
)

ITEM_COUNT = 2000
TIME_BOUND_SECONDS = 5.0
INDEX_NAME = "idx"
CONFIG = {"path": "Lib", "form_name": "Lib"}


def file_id(number):
    return f"f{number}"


def listed(number, modified="2026-01-01"):
    return listed_file(f"doc{number}.docx", modified=modified, file_id=file_id(number))


def stored_from(listing_files):
    """The base rows an earlier run stored for the given listed files."""
    rows = {}
    for file in listing_files:
        item = check.ListedItem(key=file["id"], name=file["Name"], path=file["Path"], marker=file["Modified"])
        rows[item.key] = stored_row(item)
    return rows


class Run:
    """One call of the tool and how long it took."""

    def __init__(self, report, seconds, backend):
        self.report = report
        self.seconds = seconds
        self.backend = backend


@pytest.fixture
def run_check(monkeypatch):
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(SharepointApiWrapper, "_available_indexes_note", lambda self: "Available indexes: idx")

    def run(files, stored_rows, unindexed=None):
        backend = Backend(files)
        toolkit = toolkit_with(backend)
        scope = toolkit._run_scope_keys(CONFIG)
        meta = {"id": "meta-1", "metadata": {
            "index_configuration": json.dumps(CONFIG),
            "unindexed_items": json.dumps(unindexed or {}),
            "history": json.dumps([{"state": "completed", "last_run_scope": scope}]),
        }}
        monkeypatch.setattr(VectorStoreWrapperBase, "get_index_meta", lambda self, name: meta)
        monkeypatch.setattr(SharepointApiWrapper, "_get_indexed_data", lambda self, name: stored_rows)
        started = time.perf_counter()
        report = toolkit.check_for_changes(INDEX_NAME)
        return Run(report, time.perf_counter() - started, backend)

    return run


class TestTwoThousandFiles:

    def test_an_unchanged_index_is_up_to_date_and_the_comparison_stays_under_the_time_bound(self, run_check):
        files = [listed(number) for number in range(ITEM_COUNT)]

        outcome = run_check(files, stored_from(files))

        report = outcome.report
        assert report["status"] == check.STATUS_UP_TO_DATE
        assert report["counts"] == {"added": 0, "changed": 0, "removed": 0, "unchanged": ITEM_COUNT,
                                    "not_indexed": 0}
        assert report["listing_complete"] is True
        assert outcome.seconds < TIME_BOUND_SECONDS

    def test_added_changed_and_removed_files_are_counted_and_the_comparison_stays_under_the_time_bound(
            self, run_check):
        indexed_files = [listed(number) for number in range(ITEM_COUNT)]
        stored = stored_from(indexed_files)
        # Source now: files 0..1997 kept (10, 20 and 30 edited), 1998 and 1999 gone, two new files.
        source = [listed(number, modified="2026-02-02" if number in (10, 20, 30) else "2026-01-01")
                  for number in range(ITEM_COUNT - 2)]
        source += [listed(number) for number in (ITEM_COUNT, ITEM_COUNT + 1)]
        assert len(source) == ITEM_COUNT

        outcome = run_check(source, stored)

        report = outcome.report
        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX
        assert report["counts"] == {"added": 2, "changed": 3, "removed": 2, "unchanged": ITEM_COUNT - 2 - 3,
                                    "not_indexed": 0}
        assert report["items"]["changed"] == [f"/sites/x/Lib/doc{number}.docx" for number in (10, 20, 30)]
        assert sorted(report["items"]["removed"]) == [
            f"/sites/x/Lib/doc{number}.docx" for number in (ITEM_COUNT - 2, ITEM_COUNT - 1)]
        assert report["items"]["added"] == [
            f"/sites/x/Lib/doc{number}.docx" for number in (ITEM_COUNT, ITEM_COUNT + 1)]
        assert report["listing_complete"] is True
        assert outcome.seconds < TIME_BOUND_SECONDS

    def test_the_source_is_listed_once_with_the_saved_scope_and_no_content_is_read(self, run_check):
        files = [listed(number) for number in range(ITEM_COUNT)]

        outcome = run_check(files, stored_from(files))

        assert [call["folder"] for call in outcome.backend.calls] == ["Lib"]
        assert outcome.backend.calls[0]["form"] == "Lib"
        assert outcome.backend.content_reads == []

    def test_every_file_removed_from_a_large_source_is_reported_as_removed(self, run_check):
        # One kept file keeps the listing and the index sharing a key, so the mismatch guard stays out of the way.
        files = [listed(number) for number in range(ITEM_COUNT)]

        outcome = run_check(files[:1], stored_from(files))

        report = outcome.report
        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX
        assert report["counts"]["removed"] == ITEM_COUNT - 1
        assert report["counts"]["unchanged"] == 1
        assert outcome.seconds < TIME_BOUND_SECONDS
