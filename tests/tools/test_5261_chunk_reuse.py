"""#5261 — reusing an adopted row instead of embedding its chunk again.

A hit asserts one thing only: a row already exists whose stored payload equals the payload
this run was about to write. That is why the digest covers the *stored* form of the
metadata, not the in-flight form, and why a wrong prediction can only ever cost an
embedding it could have saved.

The counterpart rule lives in _assemble_promote_sets: an adopted row this run did not
reuse must be superseded, or a resumed generation is not the same corpus a clean run
would have produced.
"""

from hashlib import sha256
from uuid import uuid4

import pytest
from langchain_core.documents import Document

from elitea_sdk.tools.base_indexer_toolkit import (
    IDLESS_STAGING_KEY,
    _IndexRunState,
    candidate_chunk_digest,
    chunk_digest,
    stored_metadata_form,
)
from tests.tools.test_6586_meta_row_writes import (  # noqa: F401  (fixtures)
    FakeStagingAdapter,
    StagingToolkit,
    sessions,
    toolkit,
)


def server_digest_of(text):
    return sha256(text.encode("utf-8")).hexdigest()


class TestTheDigestPredictsTheStoredForm:

    def test_a_list_is_compared_as_the_joined_string_add_documents_stores(self):
        candidate = Document(page_content="body", metadata={"id": "1", "tags": ["a", "b"]})
        assert candidate_chunk_digest(candidate) == chunk_digest(
            server_digest_of("body"), {"id": "1", "tags": "a; b"}
        )

    def test_a_dict_is_compared_as_the_json_add_documents_stores(self):
        candidate = Document(page_content="body", metadata={"id": "1", "extra": {"k": 1}})
        assert candidate_chunk_digest(candidate) == chunk_digest(
            server_digest_of("body"), {"id": "1", "extra": '{"k": 1}'}
        )

    def test_the_run_id_is_excluded_so_a_re_stamped_row_still_matches(self):
        """The stamp is what adoption rewrites, so including it would zero every hit."""
        candidate = Document(page_content="body",
                             metadata={"id": "1", "_elitea_run_id": "new-run"})
        assert candidate_chunk_digest(candidate) == chunk_digest(
            server_digest_of("body"), {"id": "1", "_elitea_run_id": "old-run"}
        )

    def test_key_order_does_not_change_the_digest(self):
        first = Document(page_content="body", metadata={"a": 1, "b": 2})
        second = Document(page_content="body", metadata={"b": 2, "a": 1})
        assert candidate_chunk_digest(first) == candidate_chunk_digest(second)

    def test_different_content_gives_a_different_digest(self):
        assert candidate_chunk_digest(Document(page_content="one", metadata={"id": "1"})) \
            != candidate_chunk_digest(Document(page_content="two", metadata={"id": "1"}))

    def test_changed_metadata_gives_a_different_digest(self):
        """A file edit bumps commit_hash on every chunk, so an unchanged chunk of a
        changed file still misses — which is what stops mixed hashes reaching promote."""
        assert candidate_chunk_digest(
            Document(page_content="body", metadata={"id": "1", "commit_hash": "old"})
        ) != candidate_chunk_digest(
            Document(page_content="body", metadata={"id": "1", "commit_hash": "new"})
        )

    def test_the_stored_form_drops_only_the_run_id(self):
        assert stored_metadata_form({"id": "1", "_elitea_run_id": "r"}) == {"id": "1"}


class DropRecordingAdapter(FakeStagingAdapter):
    def __init__(self):
        super().__init__()
        self.dropped_runs = []

    def drop_run_chunks(self, wrapper, run_id):
        self.dropped_runs.append(run_id)


@pytest.fixture
def run_state(toolkit):  # noqa: F811
    toolkit.vector_adapter = DropRecordingAdapter()
    run = _IndexRunState(run_id=uuid4().hex[:12])
    toolkit._index_run = run
    return toolkit, run


class TestClaimingAnAdoptedRow:

    def test_a_matching_chunk_claims_a_row(self, run_state):
        toolkit, run = run_state
        document = Document(page_content="body", metadata={"id": "1"})
        run.adoptable_chunks = {candidate_chunk_digest(document): ["pk-1"]}

        assert toolkit._claim_adopted_row(document) == "pk-1"
        assert run.reused_row_pks == {"pk-1"}

    def test_identical_chunks_consume_distinct_rows(self, run_state):
        """Mutation: peek instead of pop. The unclaimed twin then survives promote."""
        toolkit, run = run_state
        document = Document(page_content="body", metadata={"id": "1"})
        run.adoptable_chunks = {candidate_chunk_digest(document): ["pk-1", "pk-2"]}

        claimed = {toolkit._claim_adopted_row(document), toolkit._claim_adopted_row(document)}

        assert claimed == {"pk-1", "pk-2"}
        assert toolkit._claim_adopted_row(document) is None

    def test_a_chunk_with_no_match_claims_nothing(self, run_state):
        toolkit, run = run_state
        run.adoptable_chunks = {
            candidate_chunk_digest(Document(page_content="other", metadata={"id": "9"})): ["pk-1"]
        }
        assert toolkit._claim_adopted_row(Document(page_content="body", metadata={"id": "1"})) is None
        assert run.reused_row_pks == set()

    def test_nothing_is_claimed_when_no_run_was_adopted(self, run_state):
        toolkit, _ = run_state
        assert toolkit._claim_adopted_row(Document(page_content="body", metadata={"id": "1"})) is None


class TestUnreusedAdoptedRowsAreSuperseded:

    def test_an_adopted_row_the_run_did_not_reuse_is_superseded(self, run_state):
        """Mutation: drop the adopted_row_pks - reused_row_pks extend."""
        toolkit, run = run_state
        run.adopted_row_pks = {"pk-1", "pk-2"}
        run.reused_row_pks = {"pk-1"}

        superseded, _, _ = toolkit._assemble_promote_sets()

        assert "pk-2" in superseded
        assert "pk-1" not in superseded

    def test_a_fully_reused_adoption_supersedes_nothing_extra(self, run_state):
        toolkit, run = run_state
        run.adopted_row_pks = {"pk-1"}
        run.reused_row_pks = {"pk-1"}

        superseded, _, _ = toolkit._assemble_promote_sets()

        assert superseded == []

    def test_a_run_that_adopted_nothing_is_unaffected(self, run_state):
        toolkit, run = run_state
        run.staged_removal_ids = {"doc-1": {"old-1"}}

        superseded, _, _ = toolkit._assemble_promote_sets()

        assert superseded == ["old-1"]


class TestTheReuseMapIsOnlyBuiltWhenItIsSafe:

    def test_no_adoption_means_no_read(self, run_state):
        toolkit, run = run_state
        toolkit._load_adopted_chunk_digests()
        assert run.adoptable_chunks == {}

    def test_a_truncated_read_drops_the_adopted_rows(self, run_state):
        """The rows already carry this run's id, so leaving them unreferenced publishes
        them at promote beside the copies this run embeds — every chunk twice.
        Mutation: return instead of calling _release_adopted_rows."""
        toolkit, run = run_state
        run.adopted_from_run_id = "old-run"
        toolkit.vector_adapter.read_run_staged_digests = (
            lambda *a, **kw: ({b"d": ["pk-1"]}, {"pk-1"}, True)
        )

        toolkit._load_adopted_chunk_digests()

        assert toolkit.vector_adapter.dropped_runs == [run.run_id]
        assert run.adoptable_chunks == {}
        assert run.adopted_row_pks == set()
        assert run.adopted_from_run_id is None

    def test_an_unreadable_index_drops_the_adopted_rows(self, run_state):
        toolkit, run = run_state
        run.adopted_from_run_id = "old-run"

        def explode(*_a, **_kw):
            raise RuntimeError("read blew up")

        toolkit.vector_adapter.read_run_staged_digests = explode

        toolkit._load_adopted_chunk_digests()

        assert toolkit.vector_adapter.dropped_runs == [run.run_id]
        assert run.adoptable_chunks == {}
        assert run.adopted_from_run_id is None

    def test_a_failed_drop_abandons_the_run_rather_than_risking_duplicates(self, run_state):
        """There is no safe way to continue: the rows are live under this run id and
        nothing downstream can reach them. That drop_run_chunks actually raises is pinned
        on the real adapter in tests/tools/vector_adapters/test_5261_adoption.py — a stub
        here would only assert that a stub raises."""
        toolkit, run = run_state
        run.adopted_from_run_id = "old-run"
        toolkit.vector_adapter.read_run_staged_digests = (
            lambda *a, **kw: ({}, set(), True)
        )

        def explode(*_a, **_kw):
            raise RuntimeError("drop blew up")

        toolkit.vector_adapter.drop_run_chunks = explode

        with pytest.raises(RuntimeError):
            toolkit._load_adopted_chunk_digests()
        assert run.adopted_from_run_id == "old-run"

    def test_a_dropped_adoption_supersedes_nothing_at_promote(self, run_state):
        """The counterpart of the drop: with the rows gone, the promote set must stay
        empty rather than naming ids that no longer exist."""
        toolkit, run = run_state
        run.adopted_from_run_id = "old-run"
        toolkit.vector_adapter.read_run_staged_digests = (
            lambda *a, **kw: ({b"d": ["pk-1"]}, {"pk-1"}, True)
        )

        toolkit._load_adopted_chunk_digests()
        superseded, _, _ = toolkit._assemble_promote_sets()

        assert superseded == []

    def test_a_successful_read_arms_reuse(self, run_state):
        toolkit, run = run_state
        run.adopted_from_run_id = "old-run"
        toolkit.vector_adapter.read_run_staged_digests = (
            lambda *a, **kw: ({b"d": ["pk-1"]}, {"pk-1"}, False)
        )

        toolkit._load_adopted_chunk_digests()

        assert run.adoptable_chunks == {b"d": ["pk-1"]}
        assert run.adopted_row_pks == {"pk-1"}
