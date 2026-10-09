"""#5261 phase 2 — a resumed run skips fetching files it already holds in full.

Chunk adoption removes the embedding cost but not the download: the run still pulls every
file from the provider, and on a fetch-bound corpus that is most of the wall clock. #6645
already skips unchanged files before `_read_file`, but it compares against the promoted
generation, which cannot see an interrupted run's own rows. This path can.

Skipping a file whose chunk set is only partial would silently drop the missing chunks, so
the gate is a positive count — `chunk_total`, stamped at chunk time and not recoverable
afterwards — never the identity alone.
"""

from uuid import uuid4

import pytest

from elitea_sdk.tools.base_indexer_toolkit import _IndexRunState
from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import PGVectorAdapter
from tests.tools.code_loader_rig import SHA_PY_BODY, IdentityToolkit, build_toolkit
from tests.tools.code_loader_rig import FakeStagingAdapter as CodeLoaderStagingAdapter
from tests.tools.test_6586_meta_row_writes import (  # noqa: F401  (fixtures)
    FakeStagingAdapter,
    StagingToolkit,
    sessions,
    toolkit,
)


class CompleteFilesAdapter(FakeStagingAdapter):
    def __init__(self):
        super().__init__()
        self.complete_files = {}
        self.read_failure = None

    def read_run_staged_digests(self, wrapper, run_id, digest_of, cap):
        return {}, set(), False

    def read_run_complete_files(self, wrapper, run_id):
        if self.read_failure is not None:
            raise self.read_failure
        return self.complete_files


@pytest.fixture
def resuming(toolkit):  # noqa: F811
    adapter = CompleteFilesAdapter()
    toolkit.vector_adapter = adapter
    run = _IndexRunState(run_id=uuid4().hex[:12])
    toolkit._index_run = run
    return toolkit, run, adapter


class TestASkippedFilesRowsSurvivePromote:
    """The trap this phase creates: a file skipped before the fetch never reaches
    _consume_pipeline_output, so nothing claims its rows, and the promote fence would
    supersede exactly the chunks the skip was relying on."""

    def test_claiming_a_resumed_file_keeps_its_rows_out_of_the_supersede_set(self, resuming):
        """Mutation: drop the adopt_resumed_file call from the loader skip."""
        toolkit, run, _ = resuming
        run.adopted_row_pks = {"pk-1", "pk-2"}
        run.resumable_files = {"a.py": ("sha-a", ["pk-1", "pk-2"])}
        run.indexed_data = {}

        toolkit.adopt_resumed_file("a.py", "x")
        superseded, _, _ = toolkit._assemble_promote_sets()

        assert run.reused_row_pks == {"pk-1", "pk-2"}
        assert superseded == []

    def test_without_the_claim_the_rows_would_be_deleted(self, resuming):
        toolkit, run, _ = resuming
        run.adopted_row_pks = {"pk-1", "pk-2"}
        run.resumable_files = {"a.py": ("sha-a", ["pk-1", "pk-2"])}

        superseded, _, _ = toolkit._assemble_promote_sets()

        assert sorted(superseded) == ["pk-1", "pk-2"]

    def test_claiming_an_unknown_file_changes_nothing(self, resuming):
        toolkit, run, _ = resuming
        run.indexed_data = {}
        toolkit.adopt_resumed_file("missing.py", "x")
        assert run.reused_row_pks == set()
        assert run.resumed_files == set()


class TestTheSkipGate:

    def test_a_file_held_in_full_reports_its_identity(self, resuming):
        toolkit, run, _ = resuming
        run.resumable_files = {"a.py": ("sha-a", ["pk-1"])}
        assert toolkit.resumable_file_identity("a.py") == "sha-a"

    def test_a_file_the_run_does_not_hold_reports_nothing(self, resuming):
        toolkit, _, _ = resuming
        assert toolkit.resumable_file_identity("a.py") is None

    def test_no_run_reports_nothing(self, toolkit):  # noqa: F811
        toolkit.vector_adapter = CompleteFilesAdapter()
        assert toolkit.resumable_file_identity("a.py") is None


class TestTheMapIsOnlyLoadedForAnAdoptedRun:

    def test_a_run_that_adopted_nothing_loads_no_map(self, resuming):
        toolkit, run, adapter = resuming
        adapter.complete_files = {"a.py": ("sha-a", ["pk-1"])}

        toolkit._load_adopted_chunk_digests()

        assert run.resumable_files == {}

    def test_an_adopted_run_loads_the_map(self, resuming):
        toolkit, run, adapter = resuming
        run.adopted_from_run_id = "old-run"
        adapter.complete_files = {"a.py": ("sha-a", ["pk-1"])}

        toolkit._load_adopted_chunk_digests()

        assert run.resumable_files == {"a.py": ("sha-a", ["pk-1"])}

    def test_a_failed_read_costs_the_fetch_but_not_the_run(self, resuming):
        """The chunks stay reusable; only the download they would have saved is lost."""
        toolkit, run, adapter = resuming
        run.adopted_from_run_id = "old-run"
        adapter.read_failure = RuntimeError("read blew up")

        toolkit._load_adopted_chunk_digests()

        assert run.resumable_files == {}


class TestChunkTotalIsStampedAtChunkTime:
    """The gate is a count, not an identity: a half-flushed file's rows all carry the
    same blob_sha, so without the total a short set is indistinguishable from a whole one."""

    def parse(self, content, name="a.py"):
        from elitea_sdk.tools.chunkers.code.codeparser import parse_code_files_for_db

        return list(parse_code_files_for_db(
            [{"file_name": name, "file_content": content, "commit_hash": "h1", "blob_sha": "s1"}]
        ))

    def test_every_chunk_of_a_file_carries_that_file_s_total(self):
        """Mutation: stop stamping chunk_total in either codeparser branch."""
        documents = self.parse("def f():\n    return 1\n\ndef g():\n    return 2\n")

        assert len(documents) > 1
        assert {d.metadata.get("chunk_total") for d in documents} == {len(documents)}

    def test_the_total_matches_the_number_of_rows_the_file_will_write(self):
        documents = self.parse("def f():\n    return 1\n")
        stored = [d for d in documents if d.page_content]

        assert documents[0].metadata.get("chunk_total") == len(stored)

    def test_an_unparseable_language_still_gets_a_total(self):
        documents = self.parse("plain text body\n", name="notes.txt")

        assert documents
        assert {d.metadata.get("chunk_total") for d in documents} == {len(documents)}

    def test_chunk_id_is_left_alone(self):
        """chunk_id is part of a row's identity; renumbering it would break dedup."""
        documents = self.parse("def f():\n    return 1\n\ndef g():\n    return 2\n")

        assert [d.metadata["chunk_id"] for d in documents] == list(range(1, len(documents) + 1))

    def test_an_empty_chunk_is_not_counted_toward_the_total(self):
        """Empty content is dropped before it reaches the store, so counting it would
        leave the file one row short of its own total forever — never resumable.
        Mutation: count len(file_documents) instead of the non-empty ones."""
        from langchain_core.documents import Document

        from elitea_sdk.tools.chunkers.code.codeparser import with_chunk_total

        documents = with_chunk_total([
            Document(page_content="body", metadata={}),
            Document(page_content="", metadata={}),
            Document(page_content="more", metadata={}),
        ])

        assert {d.metadata["chunk_total"] for d in documents} == {2}


class TestTheLoaderSkipClaimsTheRowsItRelliesOn:
    """In-CI mirror of the live-SQL path. Asserting on adopt_resumed_file directly passes
    even when the loader never calls it, so these drive the real loader instead."""

    def build(self, monkeypatch, held):
        toolkit = build_toolkit(monkeypatch, {"a.py": SHA_PY_BODY, "b.py": "sha-b"}, {},
                                toolkit_cls=IdentityToolkit)
        run = _IndexRunState(run_id="new-run")
        run.resumable_files = held
        object.__setattr__(toolkit, "_index_run", run)
        return toolkit, run

    def drive(self, toolkit):
        preskipped = set()
        list(toolkit.loader(index_name="x", preskipped_keys=preskipped))
        return preskipped

    def test_a_file_this_run_already_holds_is_never_downloaded(self, monkeypatch):
        toolkit, _ = self.build(monkeypatch, {"a.py": (SHA_PY_BODY, ["pk-1", "pk-2"])})

        preskipped = self.drive(toolkit)

        assert "a.py" not in toolkit.reads
        assert "a.py" in preskipped

    def test_the_skipped_file_s_rows_are_claimed_for_reuse(self, monkeypatch):
        """The trap: a loader-skipped file never reaches _consume_pipeline_output, so
        nothing else claims its rows and the promote fence deletes exactly the chunks the
        skip relied on. Mutation: drop the adopt_resumed_file call from the loader."""
        toolkit, run = self.build(monkeypatch, {"a.py": (SHA_PY_BODY, ["pk-1", "pk-2"])})

        self.drive(toolkit)

        assert run.reused_row_pks == {"pk-1", "pk-2"}
        assert run.resumed_files == {"a.py"}

    def test_a_file_whose_identity_moved_on_is_downloaded_again(self, monkeypatch):
        toolkit, run = self.build(monkeypatch, {"a.py": ("sha-stale", ["pk-1"])})

        self.drive(toolkit)

        assert "a.py" in toolkit.reads
        assert run.reused_row_pks == set()

    def test_a_run_holding_nothing_downloads_every_file(self, monkeypatch):
        toolkit, run = self.build(monkeypatch, {})

        self.drive(toolkit)

        assert "a.py" in toolkit.reads
        assert run.reused_row_pks == set()


class TestTheCompletenessGate:
    """In-CI mirror of TestCompleteFilesGateTheFetchSkip, which lives under tests/tools/index/
    and is excluded from PR CI. Covers the grouping rules without a database."""

    def gate(self, rows):
        adapter, staged = PGVectorAdapter(), {}
        for row_pk, metadata in rows:
            adapter._collect_staged_file_row(staged, row_pk, metadata)
        return {name: file for name, file in staged.items() if adapter._file_is_whole(file)}

    def file_rows(self, chunks, total, identity="sha-a", filename="a.py"):
        return [(f"pk-{index}", {"filename": filename, "blob_sha": identity,
                                 "chunk_total": total})
                for index in range(chunks)]

    def test_a_whole_file_passes(self):
        assert list(self.gate(self.file_rows(chunks=3, total=3))) == ["a.py"]

    def test_a_half_flushed_file_is_rejected(self):
        """The whole reason the gate is a count and not an identity: every row of a
        half-written file carries the same blob_sha as a complete one.
        Mutation: return True instead of comparing the row count to the total."""
        assert self.gate(self.file_rows(chunks=2, total=5)) == {}

    def test_a_file_with_more_rows_than_its_total_is_rejected(self):
        assert self.gate(self.file_rows(chunks=4, total=3)) == {}

    def test_a_file_with_no_total_is_rejected(self):
        rows = [("pk-0", {"filename": "a.py", "blob_sha": "sha-a"})]
        assert self.gate(rows) == {}

    def test_a_file_with_disagreeing_identities_is_rejected(self):
        rows = (self.file_rows(chunks=1, total=2)
                + [("pk-1", {"filename": "a.py", "blob_sha": "sha-b", "chunk_total": 2})])
        assert self.gate(rows) == {}

    def test_a_file_with_no_identity_is_rejected(self):
        rows = [("pk-0", {"filename": "a.py", "chunk_total": 1})]
        assert self.gate(rows) == {}

    def test_a_row_with_no_filename_is_ignored(self):
        assert self.gate([("pk-0", {"blob_sha": "sha-a", "chunk_total": 1})]) == {}

    def test_whole_and_partial_files_are_separated(self):
        rows = (self.file_rows(chunks=2, total=2, filename="whole.py")
                + self.file_rows(chunks=1, total=4, filename="partial.py", identity="sha-b"))
        assert list(self.gate(rows)) == ["whole.py"]


class TestAResumedFileRetiresTheGenerationItReplaces:
    """A resume-skipped file reaches neither _reduce_duplicates nor the pipeline. The
    first is the only place that nominates the previous generation's rows for removal,
    so without an explicit nomination promote publishes the old rows beside the new.

    An interrupted run only writes rows for changed or new files, so on any index that
    already holds a generation every resumed file is a changed file — which makes this
    the common case for the feature, not an edge case.
    """

    OLD_GEN = ["old-a-1", "old-a-2"]
    ADOPTED = ["runA-a-1"]

    def build(self, monkeypatch, resume_map):
        from tests.tools.code_loader_rig import (
            CHANGED_BODY, PY_BODY_HASH, SHA_PY_BODY, entry_for, git_blob_sha,
            run_index, seed_completed_run,
        )

        class Adapter(CodeLoaderStagingAdapter):

            def __init__(self):
                super().__init__()
                self.superseded = None

            def claim_adoptable_run(self, wrapper, index_name, stale_before, max_chunks=None):
                return "run-A"

            def adopt_run_chunks(self, wrapper, index_name, source_run_id, target_run_id):
                return 1

            def read_run_staged_digests(self, wrapper, run_id, digest_of, cap):
                return {"placeholder": []}, set(TestAResumedFileRetiresTheGenerationItReplaces.ADOPTED), False

            def read_run_complete_files(self, wrapper, run_id):
                return resume_map

            def promote_run(self, wrapper, index_name, run_id, superseded_ids, orphan_ids,
                            damaged_ids):
                self.superseded = list(superseded_ids)
                return super().promote_run(wrapper, index_name, run_id, superseded_ids,
                                           orphan_ids, damaged_ids)

        class Claiming(IdentityToolkit):
            def _claim_adopted_row(self, document):
                run = self._index_run
                held = TestAResumedFileRetiresTheGenerationItReplaces.ADOPTED[0]
                if document.metadata.get("filename") == "a.py" and held not in run.reused_row_pks:
                    run.reused_row_pks.add(held)
                    return held
                return None

        changed_sha = git_blob_sha(CHANGED_BODY)
        indexed = {"a.py": entry_for("a.py", [(row, PY_BODY_HASH, SHA_PY_BODY)
                                              for row in self.OLD_GEN])}
        toolkit = build_toolkit(monkeypatch, {"a.py": changed_sha}, indexed,
                               toolkit_cls=Claiming, contents={"a.py": CHANGED_BODY})
        object.__setattr__(toolkit, "vector_adapter", Adapter())
        seed_completed_run(toolkit)
        result = run_index(toolkit)
        return toolkit, toolkit.vector_adapter, result

    def resumed(self, monkeypatch):
        from tests.tools.code_loader_rig import CHANGED_BODY, git_blob_sha

        return self.build(monkeypatch, {"a.py": (git_blob_sha(CHANGED_BODY), list(self.ADOPTED))})

    def test_the_previous_generations_rows_are_superseded(self, monkeypatch):
        """Mutation: drop the _stage_resumed_file_removal call from adopt_resumed_file.
        Without it the old and new rows are both published and the index holds every
        chunk of that file twice."""
        _, adapter, _ = self.resumed(monkeypatch)

        assert set(self.OLD_GEN) <= set(adapter.superseded)

    def test_the_reused_rows_are_not_superseded_with_them(self, monkeypatch):
        _, adapter, _ = self.resumed(monkeypatch)

        assert set(self.ADOPTED).isdisjoint(adapter.superseded)

    def test_the_file_is_still_never_downloaded(self, monkeypatch):
        toolkit, _, _ = self.resumed(monkeypatch)

        assert "a.py" not in toolkit.reads

    def test_a_changed_file_is_not_reported_as_unchanged(self, monkeypatch):
        """Mutation: call _track_document_unchanged for resumed keys too."""
        from tests.tools.code_loader_rig import totals_of

        _, _, result = self.resumed(monkeypatch)
        totals = totals_of(result)

        assert totals["unchanged"] == 0
        assert totals["indexed"] == 1

    def test_the_reused_chunks_are_counted_as_published(self, monkeypatch):
        """Mutation: stop adding resumed_chunk_count into result["count"]."""
        toolkit, _, _ = self.resumed(monkeypatch)

        assert toolkit._stored_meta["metadata"]["updated"] == len(self.ADOPTED)

    def test_without_a_resume_the_old_path_is_unchanged(self, monkeypatch):
        """The pre-change behaviour this must not regress: the file is fetched and
        _reduce_duplicates nominates the old generation itself."""
        toolkit, adapter, _ = self.build(monkeypatch, {})

        assert "a.py" in toolkit.reads
        assert set(self.OLD_GEN) <= set(adapter.superseded)

    def test_rows_belonging_to_another_index_are_never_nominated(self, monkeypatch):
        """Multi-index rows are shared, so a stored entry under a different collection
        is not this index's previous generation. Nominating it would delete another
        index's content. Same guard _reduce_duplicates applies.
        Mutation: drop the collection comparison from _stage_resumed_file_removal."""
        from tests.tools.code_loader_rig import (
            CHANGED_BODY, PY_BODY_HASH, SHA_PY_BODY, entry_for, git_blob_sha,
        )

        toolkit, adapter, _ = self.resumed(monkeypatch)
        foreign = entry_for("a.py", [(row, PY_BODY_HASH, SHA_PY_BODY) for row in self.OLD_GEN])
        foreign["metadata"]["collection"] = "a-different-index"
        run = toolkit._index_run
        run.staged_removal_ids.clear()
        run.indexed_data = {"a.py": foreign}

        toolkit._stage_resumed_file_removal("a.py", "x")

        assert run.staged_removal_ids == {}
