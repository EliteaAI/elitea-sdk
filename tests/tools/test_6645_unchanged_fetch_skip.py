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

"""Guards the fetch-free unchanged skip in the code loader (#6645).

A file whose provider-side identity is unchanged is never downloaded, and the run
must still report, count and promote exactly as it did when every file was read.

Lives outside tests/tools/index/ because CI skips that directory.
"""

import hashlib
import json

import pytest
from langchain_core.documents import Document

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit
from elitea_sdk.tools.code_indexer_toolkit import CodeIndexerToolkit

def git_blob_sha(content):
    body = content.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()


PY_BODY = "def alpha():\n    return 1\n"
PY_BODY_HASH = hashlib.sha256(PY_BODY.encode("utf-8")).hexdigest()
SHA_PY_BODY = git_blob_sha(PY_BODY)
CHANGED_BODY = "def alpha():\n    return 2\n"


class IdentityToolkit(CodeIndexerToolkit):
    def _get_files(self, path="", branch=None):
        return list(self.tree)

    def _get_files_with_identity(self, path="", branch=None):
        return dict(self.tree)

    def _read_file(self, file_path, branch, **kwargs):
        self.reads.append(file_path)
        return self.contents.get(file_path, PY_BODY)


class AttestingIdentityToolkit(IdentityToolkit):
    loader_attests_completion = True


class FakeStagingAdapter:
    supports_run_staging = True

    def __init__(self):
        self.calls = []
        self.promote_args = []
        self.stamped = {}
        self.stamp_calls = []

    def ensure_index_runs_table(self, wrapper):
        pass

    def register_index_run(self, wrapper, index_name, run_id, task_id=None, meta_lock_id=None):
        return (True, None)

    def sweep_stale_index_runs(self, wrapper, index_name, stale_before):
        return []

    def heartbeat_index_run(self, wrapper, index_name, run_id, meta_id, chunks_written=None):
        pass

    def update_index_meta_keys(self, wrapper, meta_id, run_id, patch):
        stored = wrapper._stored_meta
        if stored is None:
            return 0
        merged = {**stored.get("metadata", {}), **patch}
        object.__setattr__(wrapper, "_stored_meta", {**stored, "metadata": merged})
        return 1

    def stamp_code_identity(self, wrapper, identity_by_row_id):
        self.stamp_calls.append(dict(identity_by_row_id))
        self.stamped.update(identity_by_row_id)
        return len(identity_by_row_id)

    def promote_run(self, wrapper, index_name, run_id, superseded_ids, orphan_ids, damaged_ids):
        self.calls.append("promote")
        self.promote_args.append({"orphan_ids": list(orphan_ids)})
        return "promoted"

    def discard_run(self, wrapper, index_name, run_id):
        self.calls.append("discard")
        return "discarded"

    def get_pending_run_ids(self, wrapper, index_name, include_cancelled=True):
        return []

    def get_index_meta(self, wrapper, index_name):
        return [wrapper._stored_meta] if wrapper._stored_meta else []


def entry_for(filename, rows):
    """rows: list of (row_id, commit_hash, blob_sha), stored as the adapter stores them."""
    return {
        "metadata": {"collection": "x", "filename": filename, "collection_name": "x"},
        "commit_hashes": [commit_hash for _, commit_hash, _ in rows],
        "blob_shas": [blob_sha for _, _, blob_sha in rows],
        "ids": [row_id for row_id, _, _ in rows],
    }


def indexed_entry(filename, blob_sha, commit_hash="stored-hash"):
    return entry_for(filename, [(f"row-{filename}", commit_hash, blob_sha)])


def build_toolkit(monkeypatch, tree, indexed, toolkit_cls=IdentityToolkit, contents=None):
    instance = toolkit_cls.model_construct()
    object.__setattr__(instance, "_stored_meta", None)
    object.__setattr__(instance, "toolkit_id", None)
    object.__setattr__(instance, "max_docs_per_add", 100)
    object.__setattr__(instance, "llm", None)
    object.__setattr__(instance, "tree", dict(tree))
    object.__setattr__(instance, "contents", dict(contents or {}))
    object.__setattr__(instance, "reads", [])
    object.__setattr__(instance, "indexed", dict(indexed))
    object.__setattr__(instance, "indexed_fetches", [])
    object.__setattr__(instance, "vector_adapter", FakeStagingAdapter())
    flushed = []
    object.__setattr__(instance, "flushed", flushed)

    def fake_add_documents(vectorstore=None, documents=None, ids=None):
        flushed.append([d.metadata.get("filename") for d in documents])
        object.__setattr__(
            instance, "_stored_meta",
            {"id": "meta-1", "content": "index_meta_x", "metadata": dict(documents[0].metadata)},
        )
        return [f"row-{n}" for n in range(len(documents))]

    def fake_get_indexed_data(self, index_name):
        self.indexed_fetches.append(index_name)
        return dict(self.indexed)

    monkeypatch.setattr(
        "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents", fake_add_documents
    )
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_index_meta", lambda self, name: self._stored_meta)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_indexed_count", lambda self, name: 7)
    monkeypatch.setattr(BaseIndexerToolkit, "_is_scheduled_run", lambda self: False)
    monkeypatch.setattr(BaseIndexerToolkit, "_emit_index_event",
                        lambda self, name, error=None, state=None: None)
    monkeypatch.setattr(BaseIndexerToolkit, "_clean_index", lambda self, name: None)
    monkeypatch.setattr(BaseIndexerToolkit, "_log_tool_event", lambda self, *a, **kw: None)
    monkeypatch.setattr(CodeIndexerToolkit, "_get_indexed_data", fake_get_indexed_data)
    return instance


def seed_completed_run(toolkit):
    metadata = {
        "collection": "x",
        "state": IndexerKeywords.INDEX_META_COMPLETED.value,
        "indexed": 3,
        "total": 3,
        "report": json.dumps({"status": "ok", "totals": {"indexed": 3}}),
        "skipped": None,
        "error": None,
        "history": json.dumps([{"state": "created"}, {"state": "completed"}]),
    }
    object.__setattr__(toolkit, "_stored_meta", {"id": "meta-1", "content": "c", "metadata": metadata})


def run_index(toolkit, **kwargs):
    return toolkit.index_data(index_name="x", **kwargs)


def totals_of(result):
    return result["report"]["totals"]


@pytest.fixture
def all_unchanged(monkeypatch):
    tree = {"a.py": "sha-a", "b.py": "sha-b", "c.py": "sha-c"}
    indexed = {name: indexed_entry(name, sha) for name, sha in tree.items()}
    toolkit = build_toolkit(monkeypatch, tree, indexed)
    seed_completed_run(toolkit)
    return toolkit


class TestAnUnchangedRepositoryIsNeverDownloaded:
    def test_no_file_is_read(self, all_unchanged):
        run_index(all_unchanged)
        assert all_unchanged.reads == []

    def test_the_run_reads_as_up_to_date(self, all_unchanged):
        result = run_index(all_unchanged)
        totals = totals_of(result)
        assert result["status"] == "ok"
        assert totals["indexed"] == 0
        assert totals["unchanged"] == 3
        assert result["message"].splitlines()[0] == "Up to date — 3 files unchanged."

    def test_the_loader_is_not_mistaken_for_an_empty_source(self, all_unchanged):
        result = run_index(all_unchanged)
        assert "no content" not in result["message"]
        assert all_unchanged._stored_meta["metadata"]["state"] == \
            IndexerKeywords.INDEX_META_COMPLETED.value

    def test_nothing_is_recorded_as_failed(self, all_unchanged):
        run_index(all_unchanged)
        stats = all_unchanged.get_indexing_stats()
        assert stats.documents_skipped_error == set()
        assert stats.items_withdrawn == 0

    def test_the_stats_invariant_holds(self, all_unchanged):
        run_index(all_unchanged)
        stats = all_unchanged.get_indexing_stats()
        assert stats.total_fetched == stats.items_processed + stats.total_skipped


class TestSkippedFilesStayVisibleToTheOrphanMath:
    def test_an_attesting_toolkit_deletes_nothing(self, monkeypatch):
        tree = {"a.py": "sha-a", "b.py": "sha-b"}
        indexed = {name: indexed_entry(name, sha) for name, sha in tree.items()}
        toolkit = build_toolkit(monkeypatch, tree, indexed, toolkit_cls=AttestingIdentityToolkit)
        seed_completed_run(toolkit)

        run_index(toolkit)

        assert toolkit._index_run.orphan_candidate_ids == []
        assert toolkit.vector_adapter.promote_args[-1]["orphan_ids"] == []

    def test_no_retained_orphan_warning_is_raised(self, all_unchanged):
        result = run_index(all_unchanged)
        assert result["report"].get("warnings", []) == []


class TestAPartiallyChangedRepository:
    @pytest.fixture
    def one_changed(self, monkeypatch):
        tree = {"a.py": "sha-a", "b.py": "sha-b", "c.py": "sha-c-new"}
        indexed = {
            "a.py": indexed_entry("a.py", "sha-a"),
            "b.py": indexed_entry("b.py", "sha-b"),
            "c.py": indexed_entry("c.py", "sha-c-old"),
        }
        toolkit = build_toolkit(monkeypatch, tree, indexed, contents={"c.py": CHANGED_BODY})
        seed_completed_run(toolkit)
        return toolkit

    def test_only_the_changed_file_is_downloaded(self, one_changed):
        run_index(one_changed)
        assert one_changed.reads == ["c.py"]

    def test_counts_split_between_indexed_and_unchanged(self, one_changed):
        totals = totals_of(run_index(one_changed))
        assert totals["indexed"] == 1
        assert totals["unchanged"] == 2

    def test_processed_count_covers_skipped_and_downloaded_alike(self, one_changed):
        run_index(one_changed)
        stats = one_changed.get_indexing_stats()
        assert stats.items_processed - len(stats.documents_already_indexed) == 1


class TestTheSkipIsArmedOnlyWhenDedupWouldSkipToo:
    def test_a_clean_index_run_downloads_everything(self, all_unchanged):
        totals = totals_of(run_index(all_unchanged, clean_index=True))
        assert sorted(all_unchanged.reads) == ["a.py", "b.py", "c.py"]
        assert totals["unchanged"] == 0

    def test_a_first_run_downloads_everything(self, monkeypatch):
        tree = {"a.py": "sha-a", "b.py": "sha-b"}
        toolkit = build_toolkit(monkeypatch, tree, indexed={})
        run_index(toolkit)
        assert sorted(toolkit.reads) == ["a.py", "b.py"]

    def test_rows_without_a_stored_identity_are_downloaded(self, monkeypatch):
        tree = {"a.py": "sha-a"}
        legacy = {"a.py": entry_for("a.py", [("row-a", "stored-hash", None)])}
        toolkit = build_toolkit(monkeypatch, tree, legacy)
        seed_completed_run(toolkit)
        run_index(toolkit)
        assert toolkit.reads == ["a.py"]

    def test_a_provider_without_identities_downloads_everything(self, monkeypatch):
        class NoIdentityToolkit(IdentityToolkit):
            def _get_files_with_identity(self, path="", branch=None):
                return None

        tree = {"a.py": "sha-a", "b.py": "sha-b"}
        indexed = {name: indexed_entry(name, sha) for name, sha in tree.items()}
        toolkit = build_toolkit(monkeypatch, tree, indexed, toolkit_cls=NoIdentityToolkit)
        seed_completed_run(toolkit)
        run_index(toolkit)
        assert sorted(toolkit.reads) == ["a.py", "b.py"]


class TestTheIdentityIsTheProvidersNotAContentHash:
    def test_a_matching_identity_skips_even_when_the_content_hash_differs(self, monkeypatch):
        tree = {"a.py": "sha-a"}
        indexed = {"a.py": indexed_entry("a.py", "sha-a", commit_hash="a-hash-nothing-will-match")}
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        seed_completed_run(toolkit)

        totals = totals_of(run_index(toolkit))

        assert toolkit.reads == []
        assert totals["unchanged"] == 1


class TestTheStoredCorpusIsReadOnce:
    def test_one_fetch_serves_both_the_loader_and_dedup(self, all_unchanged):
        run_index(all_unchanged)
        assert all_unchanged.indexed_fetches == ["x"]

    def test_the_identity_map_never_reaches_the_persisted_configuration(self, all_unchanged):
        run_index(all_unchanged)
        stored_config = all_unchanged._stored_meta["metadata"].get("index_configuration", {})
        assert "unchanged_identities" not in stored_config
        assert "indexed_data" not in stored_config


class TestAPreUpgradeIndexGainsItsIdentity:
    """Without a back-fill the skip can never engage on an index that already exists."""

    @pytest.fixture
    def legacy_index(self, monkeypatch):
        tree = {"a.py": SHA_PY_BODY, "b.py": SHA_PY_BODY}
        indexed = {
            name: entry_for(name, [(f"row-{name}-1", PY_BODY_HASH, None),
                                   (f"row-{name}-2", PY_BODY_HASH, None)])
            for name in tree
        }
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        seed_completed_run(toolkit)
        return toolkit

    def test_the_first_run_still_costs_what_it_costs_today(self, legacy_index):
        run_index(legacy_index)
        assert sorted(legacy_index.reads) == ["a.py", "b.py"]

    def test_the_first_run_stamps_every_existing_row(self, legacy_index):
        run_index(legacy_index)
        assert legacy_index.vector_adapter.stamped == {
            "row-a.py-1": SHA_PY_BODY, "row-a.py-2": SHA_PY_BODY,
            "row-b.py-1": SHA_PY_BODY, "row-b.py-2": SHA_PY_BODY,
        }

    def test_nothing_is_re_embedded_to_achieve_it(self, legacy_index):
        result = run_index(legacy_index)
        assert totals_of(result)["indexed"] == 0
        assert totals_of(result)["unchanged"] == 2

    def test_a_row_that_already_has_an_identity_is_not_restamped(self, monkeypatch):
        tree = {"a.py": "sha-a"}
        indexed = {"a.py": indexed_entry("a.py", "sha-a", commit_hash=PY_BODY_HASH)}
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        seed_completed_run(toolkit)

        run_index(toolkit)

        assert toolkit.vector_adapter.stamped == {}

    def test_a_failing_stamp_does_not_fail_the_run(self, legacy_index):
        def explode(wrapper, identity_by_row_id):
            raise RuntimeError("no write access")

        legacy_index.vector_adapter.stamp_code_identity = explode

        result = run_index(legacy_index)

        assert result["status"] == "ok"

    def test_the_next_run_downloads_nothing_once_the_stamps_landed(self, legacy_index):
        run_index(legacy_index)
        stamped = legacy_index.vector_adapter.stamped
        for entry in legacy_index.indexed.values():
            entry["blob_shas"] = [stamped.get(row_id, identity) for row_id, identity
                                  in zip(entry["ids"], entry["blob_shas"])]
        object.__setattr__(legacy_index, "reads", [])

        result = run_index(legacy_index)

        assert legacy_index.reads == []
        assert totals_of(result)["unchanged"] == 2


class TestTheStampNeverTouchesARowHoldingOtherContent:
    """A file with rows from two content versions must not be cemented as unchanged."""

    @pytest.fixture
    def mixed_rows(self, monkeypatch):
        tree = {"a.py": SHA_PY_BODY}
        indexed = {"a.py": entry_for("a.py", [("row-current", PY_BODY_HASH, None),
                                              ("row-stale", "hash-of-an-older-body", None)])}
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        seed_completed_run(toolkit)
        return toolkit

    def test_only_the_rows_holding_this_content_are_stamped(self, mixed_rows):
        run_index(mixed_rows)
        assert mixed_rows.vector_adapter.stamped == {"row-current": SHA_PY_BODY}

    def test_a_partially_stamped_file_is_still_downloaded_next_run(self, mixed_rows):
        run_index(mixed_rows)
        entry = mixed_rows.indexed["a.py"]
        stamped = mixed_rows.vector_adapter.stamped
        entry["blob_shas"] = [stamped.get(row_id, identity) for row_id, identity
                              in zip(entry["ids"], entry["blob_shas"])]
        object.__setattr__(mixed_rows, "reads", [])

        run_index(mixed_rows)

        assert mixed_rows.reads == ["a.py"]

    def test_rows_disagreeing_on_an_identity_never_arm_the_skip(self, monkeypatch):
        tree = {"a.py": "sha-a"}
        indexed = {"a.py": entry_for("a.py", [("row-1", PY_BODY_HASH, "sha-a"),
                                              ("row-2", PY_BODY_HASH, "sha-something-else")])}
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        seed_completed_run(toolkit)

        run_index(toolkit)

        assert toolkit.reads == ["a.py"]


class TestAHalfStampedFileCanStillBeRepaired:
    """A batch boundary does not follow file boundaries, so a failed batch can leave a
    healthy file half stamped. It must top up on the next run, not lock itself out."""

    @pytest.fixture
    def half_stamped(self, monkeypatch):
        tree = {"a.py": SHA_PY_BODY}
        indexed = {"a.py": entry_for("a.py", [("row-1", PY_BODY_HASH, SHA_PY_BODY),
                                              ("row-2", PY_BODY_HASH, None)])}
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        seed_completed_run(toolkit)
        return toolkit

    def test_only_the_rows_still_missing_it_are_written(self, half_stamped):
        run_index(half_stamped)
        assert half_stamped.vector_adapter.stamped == {"row-2": SHA_PY_BODY}

    def test_the_skip_is_not_armed_until_every_row_agrees(self, half_stamped):
        run_index(half_stamped)
        assert half_stamped.reads == ["a.py"]

    def test_a_fully_stamped_file_is_not_restamped(self, monkeypatch):
        tree = {"a.py": SHA_PY_BODY}
        indexed = {"a.py": entry_for("a.py", [("row-1", PY_BODY_HASH, SHA_PY_BODY),
                                              ("row-2", PY_BODY_HASH, SHA_PY_BODY)])}
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        seed_completed_run(toolkit)

        run_index(toolkit)

        assert toolkit.vector_adapter.stamped == {}


class _CountingRows(list):
    def __init__(self, rows):
        super().__init__(rows)
        self.walks = 0

    def __iter__(self):
        self.walks += 1
        return super().__iter__()


class TestTheBackfillDoesNotGrowWithChunkCount:
    """Dedup sees one document per CHUNK, so a per-chunk walk of the file's rows is
    quadratic in its chunk count."""

    def _walks_for(self, monkeypatch, chunk_count):
        body = "".join(f"def f{n}():\n    return {n}\n\n" for n in range(chunk_count))
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        entry = entry_for("a.py", [("row-1", body_hash, None)])
        rows = _CountingRows(entry["blob_shas"])
        entry["blob_shas"] = rows
        toolkit = build_toolkit(monkeypatch, {"a.py": git_blob_sha(body)}, {"a.py": entry},
                                contents={"a.py": body})
        seed_completed_run(toolkit)

        run_index(toolkit)

        assert toolkit.get_indexing_stats().documents_already_indexed == {"a.py"}
        assert toolkit.vector_adapter.stamped == {"row-1": git_blob_sha(body)}
        return rows.walks

    def test_the_row_walks_do_not_grow_with_the_chunk_count(self, monkeypatch):
        few = self._walks_for(monkeypatch, 3)
        many = self._walks_for(monkeypatch, 40)

        assert few == many


class TestTheBackfillIsFlushedAsItFills:
    """A first run over a large index would otherwise hold every pending stamp in memory
    until dedup ended, and lose all of it to one failure."""

    def _run_with_batch_size(self, monkeypatch, batch_size, file_count):
        monkeypatch.setattr(
            "elitea_sdk.tools.base_indexer_toolkit.IDENTITY_STAMP_BATCH_SIZE", batch_size)
        names = [f"f{n}.py" for n in range(file_count)]
        tree = {name: SHA_PY_BODY for name in names}
        indexed = {name: entry_for(name, [(f"row-{name}", PY_BODY_HASH, None)])
                   for name in names}
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        seed_completed_run(toolkit)

        run_index(toolkit)

        assert toolkit.vector_adapter.stamped == {f"row-{name}": SHA_PY_BODY for name in names}
        return toolkit.vector_adapter.stamp_calls

    def test_one_write_per_full_buffer(self, monkeypatch):
        writes = self._run_with_batch_size(monkeypatch, batch_size=2, file_count=6)

        assert [len(write) for write in writes] == [2, 2, 2]

    def test_a_larger_buffer_means_fewer_writes(self, monkeypatch):
        writes = self._run_with_batch_size(monkeypatch, batch_size=3, file_count=6)

        assert [len(write) for write in writes] == [3, 3]

    def test_a_partial_buffer_still_lands_at_the_end_of_the_pass(self, monkeypatch):
        writes = self._run_with_batch_size(monkeypatch, batch_size=4, file_count=6)

        assert [len(write) for write in writes] == [4, 2]


class TestDedupWithoutARunDoesNotExplode:

    def test_reduce_duplicates_tolerates_a_missing_run(self, monkeypatch):
        tree = {"a.py": "sha-a"}
        indexed = {"a.py": indexed_entry("a.py", "sha-a", commit_hash=PY_BODY_HASH)}
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        document = Document(page_content=PY_BODY, metadata={
            "filename": "a.py", "commit_hash": PY_BODY_HASH, "blob_sha": "sha-a"})

        surviving = list(toolkit._reduce_duplicates(iter([document]), "x"))

        assert surviving == []


class TestAStoredIdentityThatIsConsistentButWrong:
    """Normalisation in the loader (json.dumps, errors="ignore") can change the blob
    while leaving the content hash equal, so the stored identity goes stale without the
    file ever being re-indexed. Both sides must agree that it needs repair."""

    @pytest.fixture
    def stale_identity(self, monkeypatch):
        tree = {"a.py": SHA_PY_BODY}
        indexed = {"a.py": entry_for("a.py", [("row-1", PY_BODY_HASH, "sha-OLD"),
                                              ("row-2", PY_BODY_HASH, "sha-OLD")])}
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        seed_completed_run(toolkit)
        return toolkit

    def test_the_stale_identity_is_replaced(self, stale_identity):
        run_index(stale_identity)
        assert stale_identity.vector_adapter.stamped == {"row-1": SHA_PY_BODY,
                                                         "row-2": SHA_PY_BODY}

    def test_the_file_stops_being_downloaded_on_the_next_run(self, stale_identity):
        run_index(stale_identity)
        stamped = stale_identity.vector_adapter.stamped
        entry = stale_identity.indexed["a.py"]
        entry["blob_shas"] = [stamped.get(row_id, identity) for row_id, identity
                              in zip(entry["ids"], entry["blob_shas"])]
        object.__setattr__(stale_identity, "reads", [])

        run_index(stale_identity)

        assert stale_identity.reads == []

    def test_a_current_identity_is_never_rewritten(self, monkeypatch):
        tree = {"a.py": SHA_PY_BODY}
        indexed = {"a.py": entry_for("a.py", [("row-1", PY_BODY_HASH, SHA_PY_BODY)])}
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        seed_completed_run(toolkit)

        run_index(toolkit)

        assert toolkit.vector_adapter.stamped == {}
        assert toolkit.reads == []


class TestAFileLeftHoldingTwoGenerationsIsNeverArmed:
    """A run that could not confirm every replacement chunk keeps the superseded rows
    (_assemble_promote_sets skips the nomination while pending_chunk_counts != 0). Arming
    the skip there would cement the stale rows in search, so the file keeps being read
    however well stamped its current generation is."""

    def _toolkit_with(self, monkeypatch, current_identity):
        tree = {"a.py": SHA_PY_BODY}
        indexed = {"a.py": entry_for("a.py", [
            ("row-current", PY_BODY_HASH, current_identity),
            ("row-superseded", "hash-of-the-generation-that-was-kept", None),
        ])}
        toolkit = build_toolkit(monkeypatch, tree, indexed)
        seed_completed_run(toolkit)
        return toolkit

    def test_a_fully_stamped_current_generation_is_still_read(self, monkeypatch):
        toolkit = self._toolkit_with(monkeypatch, current_identity=SHA_PY_BODY)

        run_index(toolkit)

        assert toolkit.reads == ["a.py"]

    def test_nothing_is_rewritten_once_the_current_generation_carries_it(self, monkeypatch):
        toolkit = self._toolkit_with(monkeypatch, current_identity=SHA_PY_BODY)

        run_index(toolkit)

        assert toolkit.vector_adapter.stamped == {}

    def test_the_superseded_rows_are_never_given_the_current_identity(self, monkeypatch):
        toolkit = self._toolkit_with(monkeypatch, current_identity=None)

        run_index(toolkit)

        assert toolkit.vector_adapter.stamped == {"row-current": SHA_PY_BODY}

    def test_the_file_is_read_again_on_the_next_run(self, monkeypatch):
        toolkit = self._toolkit_with(monkeypatch, current_identity=None)
        run_index(toolkit)
        entry = toolkit.indexed["a.py"]
        stamped = toolkit.vector_adapter.stamped
        entry["blob_shas"] = [stamped.get(row_id, identity) for row_id, identity
                              in zip(entry["ids"], entry["blob_shas"])]
        toolkit.vector_adapter.stamped.clear()

        run_index(toolkit)

        assert toolkit.vector_adapter.stamped == {}
        assert toolkit.reads == ["a.py", "a.py"]


class TestTheArmingPredicateItself:
    """Asserted directly, because the end-to-end version depends on which identity an
    arming mutant happens to choose - and that made it a detector that only fired on
    some hash seeds."""

    def _identities_for(self, monkeypatch, blob_shas):
        toolkit = build_toolkit(monkeypatch, {}, {})
        entry = entry_for("a.py", [(f"row-{n}", PY_BODY_HASH, identity)
                                   for n, identity in enumerate(blob_shas)])
        return toolkit._collect_unchanged_identities({"a.py": entry})

    def test_rows_that_agree_arm_with_that_identity(self, monkeypatch):
        assert self._identities_for(monkeypatch, ["sha-a", "sha-a"]) == {"a.py": "sha-a"}

    def test_rows_that_disagree_arm_nothing(self, monkeypatch):
        assert self._identities_for(monkeypatch, ["sha-a", "sha-b"]) == {}

    def test_a_missing_identity_among_them_arms_nothing(self, monkeypatch):
        assert self._identities_for(monkeypatch, ["sha-a", None]) == {}

    def test_a_leading_missing_identity_arms_nothing(self, monkeypatch):
        assert self._identities_for(monkeypatch, [None, "sha-a"]) == {}

    def test_no_rows_arm_nothing(self, monkeypatch):
        assert self._identities_for(monkeypatch, []) == {}
