from uuid import uuid4

import httpx
import openai
import pytest
from tenacity import wait_none

from elitea_sdk.tools import base_indexer_toolkit
from elitea_sdk.tools.base_indexer_toolkit import (
    AdoptionProbeUnavailable,
    _IndexRunState,
    probe_failure_can_clear,
    vectors_agree,
)
from tests.tools.code_loader_rig import SameSpaceEmbeddings
from tests.tools.test_6586_meta_row_writes import (  # noqa: F401  (fixtures)
    FakeStagingAdapter,
    StagingToolkit,
    sessions,
    toolkit,
)


class OtherModelEmbeddings:
    def __init__(self, vector=(0.0, 1.0)):
        self.vector = list(vector)
        self.texts = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return [list(self.vector) for _ in texts]


TOLERANCE = StagingToolkit.adoption_probe_max_relative_distance


class TestTheVectorComparison:

    def test_an_identical_vector_agrees(self):
        assert vectors_agree([0.6, 0.8], [0.6, 0.8], TOLERANCE)

    def test_same_model_jitter_agrees(self):
        assert vectors_agree([0.6, 0.8], [0.61, 0.79], TOLERANCE)

    def test_a_different_dimension_disagrees(self):
        assert not vectors_agree([0.6, 0.8], [0.6, 0.8, 0.0], TOLERANCE)

    def test_a_different_direction_disagrees(self):
        assert not vectors_agree([1.0, 0.0], [0.0, 1.0], TOLERANCE)

    def test_a_different_normalisation_disagrees(self):
        assert not vectors_agree([0.6, 0.8], [1.2, 1.6], TOLERANCE)

    def test_zero_vectors_prove_nothing(self):
        assert not vectors_agree([0.0, 0.0], [0.0, 0.0], TOLERANCE)


class ProbeAdapter(FakeStagingAdapter):
    def __init__(self):
        super().__init__()
        self.dropped_runs = []
        self.digest_reads = 0
        self.sample_requests = []

    def read_run_embedding_samples(self, wrapper, run_id, limit):
        self.sample_requests.append((run_id, limit))
        return super().read_run_embedding_samples(wrapper, run_id, limit)

    def read_run_staged_digests(self, wrapper, run_id, digest_of, cap):
        self.digest_reads += 1
        return {b"d": ["pk-1"]}, {"pk-1"}, False

    def read_run_complete_files(self, wrapper, run_id):
        return {"a.py": ("sha", ["pk-1"])}

    def drop_run_chunks(self, wrapper, run_id):
        self.dropped_runs.append(run_id)


@pytest.fixture
def adopted(toolkit):  # noqa: F811
    adapter = ProbeAdapter()
    toolkit.vector_adapter = adapter
    run = _IndexRunState(run_id=uuid4().hex[:12])
    run.adopted_from_run_id = "old-run"
    toolkit._index_run = run
    return toolkit, run, adapter


def assert_released(run, adapter):
    assert adapter.dropped_runs == [run.run_id]
    assert adapter.digest_reads == 0
    assert run.adopted_from_run_id is None
    assert run.adoptable_chunks == {}
    assert run.adopted_row_pks == set()
    assert run.resumable_files == {}


class TestAdoptedRowsFromAnotherModelAreReleased:

    def test_rows_embedded_by_another_model_are_dropped(self, adopted):
        toolkit, run, adapter = adopted
        toolkit.embeddings = OtherModelEmbeddings()

        toolkit._load_adopted_chunk_digests()

        assert_released(run, adapter)

    def test_a_changed_vector_dimension_is_dropped(self, adopted):
        toolkit, run, adapter = adopted
        toolkit.embeddings = OtherModelEmbeddings(vector=(1.0, 0.0, 0.0))

        toolkit._load_adopted_chunk_digests()

        assert_released(run, adapter)

    def test_rows_the_probe_cannot_read_are_dropped(self, adopted):
        toolkit, run, adapter = adopted
        adapter.embedding_samples = []

        toolkit._load_adopted_chunk_digests()

        assert_released(run, adapter)

    def test_rows_whose_vectors_cannot_be_read_are_dropped(self, adopted):
        toolkit, run, adapter = adopted

        def explode(*_a, **_kw):
            raise RuntimeError("sample read blew up")

        adapter.read_run_embedding_samples = explode

        toolkit._load_adopted_chunk_digests()

        assert_released(run, adapter)

    def test_a_short_embedding_response_drops_the_rows(self, adopted):
        toolkit, run, adapter = adopted

        class Short:
            def embed_documents(self, texts):
                return []

        toolkit.embeddings = Short()

        toolkit._load_adopted_chunk_digests()

        assert_released(run, adapter)

    def test_one_mismatching_sample_is_enough_to_drop(self, adopted):
        toolkit, run, adapter = adopted
        adapter.embedding_samples = [("same", [1.0, 0.0]), ("moved", [0.0, 1.0])]
        toolkit.embeddings = SameSpaceEmbeddings()

        toolkit._load_adopted_chunk_digests()

        assert_released(run, adapter)


def gateway_error(status):
    request = httpx.Request("POST", "http://gateway/llm/v1/embeddings")
    return httpx.HTTPStatusError(f"{status}", request=request,
                                 response=httpx.Response(status, request=request))


def provider_rejection(error_class, status):
    request = httpx.Request("POST", "http://gateway/llm/v1/embeddings")
    return error_class("rejected", response=httpx.Response(status, request=request), body=None)


class FlakyEmbeddings:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return SameSpaceEmbeddings().embed_documents(texts)


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(base_indexer_toolkit, "embed_probe_texts",
                        base_indexer_toolkit.embed_probe_texts.retry_with(wait=wait_none()))


class TestAnUnreachableModelIsNotAMismatch:

    def test_a_gateway_blip_is_retried_like_a_flush(self, adopted, no_backoff):
        toolkit, run, adapter = adopted
        flaky = FlakyEmbeddings([gateway_error(503), gateway_error(429)])
        toolkit.embeddings = flaky

        toolkit._load_adopted_chunk_digests()

        assert flaky.calls == 3
        assert adapter.dropped_runs == []
        assert run.adoptable_chunks == {b"d": ["pk-1"]}

    def test_the_probe_gives_up_where_add_documents_does(self, adopted, no_backoff):
        toolkit, run, adapter = adopted
        flaky = FlakyEmbeddings([gateway_error(503)] * 5)
        toolkit.embeddings = flaky

        with pytest.raises(AdoptionProbeUnavailable):
            toolkit._load_adopted_chunk_digests()

        assert flaky.calls == 5

    @pytest.mark.parametrize("failure", [
        provider_rejection(openai.InternalServerError, 503),
        provider_rejection(openai.APIStatusError, 408),
        openai.APIConnectionError(request=httpx.Request("POST", "http://gateway")),
        openai.APITimeoutError(request=httpx.Request("POST", "http://gateway")),
    ], ids=["provider-503", "proxy-timeout-408", "connection", "client-timeout"])
    def test_an_unreachable_model_keeps_the_rows(self, adopted, no_backoff, failure):
        toolkit, run, adapter = adopted
        toolkit.embeddings = FlakyEmbeddings([failure] * 5)

        with pytest.raises(AdoptionProbeUnavailable) as raised:
            toolkit._load_adopted_chunk_digests()

        assert adapter.dropped_runs == []
        assert run.adopted_from_run_id == "old-run"
        assert raised.value.__cause__ is failure


class TestEachTransientSignalIsNeeded:

    def test_a_status_only_named_in_text_is_caught_by_the_retry_predicate(self):
        assert probe_failure_can_clear(RuntimeError("upstream returned 502"))

    def test_a_proxy_timeout_is_caught_by_its_status_whatever_its_message(self):
        assert probe_failure_can_clear(provider_rejection(openai.APIStatusError, 408))

    def test_a_timeout_known_only_by_its_text_is_caught_by_the_classifier(self):
        assert probe_failure_can_clear(RuntimeError("connection reset by peer"))

    def test_a_quota_known_only_by_name_still_can_clear(self):
        class RateLimitExceededException(Exception):
            pass

        assert probe_failure_can_clear(RateLimitExceededException("budget exhausted"))

    @pytest.mark.parametrize("error_class, status", [
        (openai.BadRequestError, 400),
        (openai.AuthenticationError, 401),
        (openai.PermissionDeniedError, 403),
        (openai.NotFoundError, 404),
        (openai.UnprocessableEntityError, 422),
    ])
    def test_a_permanent_rejection_cannot_clear(self, error_class, status):
        assert not probe_failure_can_clear(provider_rejection(error_class, status))


class TestAPermanentRejectionIsNotParked:

    @pytest.mark.parametrize("error_class, status", [
        (openai.BadRequestError, 400),
        (openai.AuthenticationError, 401),
        (openai.NotFoundError, 404),
    ])
    def test_a_rejection_releases_the_rows_without_retrying(self, adopted, no_backoff,
                                                            error_class, status):
        toolkit, run, adapter = adopted
        flaky = FlakyEmbeddings([provider_rejection(error_class, status)])
        toolkit.embeddings = flaky

        toolkit._load_adopted_chunk_digests()

        assert flaky.calls == 1
        assert_released(run, adapter)

    def test_a_spent_quota_is_still_parked(self, adopted, no_backoff):
        class RateLimitExceededException(Exception):
            pass

        toolkit, run, adapter = adopted
        flaky = FlakyEmbeddings([RateLimitExceededException("budget exhausted")])
        toolkit.embeddings = flaky

        with pytest.raises(AdoptionProbeUnavailable):
            toolkit._load_adopted_chunk_digests()

        assert flaky.calls == 1
        assert adapter.dropped_runs == []


class TestAdoptedRowsFromThisModelAreStillReused:

    def test_matching_vectors_keep_the_reuse_map(self, adopted):
        toolkit, run, adapter = adopted

        toolkit._load_adopted_chunk_digests()

        assert adapter.dropped_runs == []
        assert run.adopted_from_run_id == "old-run"
        assert run.adoptable_chunks == {b"d": ["pk-1"]}
        assert run.resumable_files == {"a.py": ("sha", ["pk-1"])}

    def test_the_probe_re_embeds_the_stored_texts_of_this_run(self, adopted):
        toolkit, run, adapter = adopted
        adapter.embedding_samples = [("chunk one", [1.0, 0.0]), ("chunk two", [1.0, 0.0])]
        recorder = OtherModelEmbeddings(vector=(1.0, 0.0))
        toolkit.embeddings = recorder

        toolkit._load_adopted_chunk_digests()

        assert recorder.texts == ["chunk one", "chunk two"]
        assert adapter.sample_requests == [(run.run_id, toolkit.adoption_probe_rows)]

    def test_no_adoption_means_no_probe(self, adopted):
        toolkit, run, adapter = adopted
        run.adopted_from_run_id = None
        recorder = OtherModelEmbeddings()
        toolkit.embeddings = recorder

        toolkit._load_adopted_chunk_digests()

        assert recorder.texts == []
        assert adapter.sample_requests == []


class TestAWholeRunAfterTheModelChanged:

    ADOPTED = ["runA-a-1"]

    def run(self, monkeypatch, embeddings):
        from tests.tools.code_loader_rig import (
            CHANGED_BODY, IdentityToolkit, build_toolkit, git_blob_sha, run_index,
            seed_completed_run,
        )
        from tests.tools.code_loader_rig import FakeStagingAdapter as RigAdapter

        adopted = self.ADOPTED
        changed_sha = git_blob_sha(CHANGED_BODY)

        class Adapter(RigAdapter):
            def __init__(self):
                super().__init__()
                self.dropped_runs = []
                self.superseded = None
                self.order = []
                self.discards = []

            def claim_adoptable_run(self, wrapper, index_name, stale_before, max_chunks=None):
                return "run-A"

            def adopt_run_chunks(self, wrapper, index_name, source_run_id, target_run_id):
                return len(adopted)

            def read_run_staged_digests(self, wrapper, run_id, digest_of, cap):
                return {}, set(adopted), False

            def read_run_complete_files(self, wrapper, run_id):
                return {"a.py": (changed_sha, list(adopted))}

            def drop_run_chunks(self, wrapper, run_id):
                self.dropped_runs.append(run_id)

            def heartbeat_index_run(self, wrapper, index_name, run_id, meta_id,
                                    chunks_written=None):
                self.order.append("heartbeat")

            def read_run_embedding_samples(self, wrapper, run_id, limit):
                self.order.append("probe")
                return super().read_run_embedding_samples(wrapper, run_id, limit)

            def discard_run(self, wrapper, index_name, run_id, retain_chunks=False):
                self.discards.append(retain_chunks)
                return super().discard_run(wrapper, index_name, run_id, retain_chunks)

            def promote_run(self, wrapper, index_name, run_id, superseded_ids, orphan_ids,
                            damaged_ids):
                self.superseded = list(superseded_ids)
                return super().promote_run(wrapper, index_name, run_id, superseded_ids,
                                           orphan_ids, damaged_ids)

        toolkit = build_toolkit(monkeypatch, {"a.py": changed_sha}, {},
                               toolkit_cls=IdentityToolkit, contents={"a.py": CHANGED_BODY})
        self.last_adapter = Adapter()
        object.__setattr__(toolkit, "vector_adapter", self.last_adapter)
        object.__setattr__(toolkit, "embeddings", embeddings)
        seed_completed_run(toolkit)
        run_index(toolkit)
        return toolkit, toolkit.vector_adapter

    def test_after_a_model_change_the_file_is_fetched_and_its_old_rows_dropped(self, monkeypatch):
        toolkit, adapter = self.run(monkeypatch, OtherModelEmbeddings())

        assert "a.py" in toolkit.reads
        assert adapter.dropped_runs == [toolkit._index_run.run_id]
        assert toolkit._index_run.reused_row_pks == set()

    def test_an_unreachable_model_parks_the_rows_for_the_next_run(self, monkeypatch, no_backoff):
        with pytest.raises(AdoptionProbeUnavailable):
            self.run(monkeypatch, FlakyEmbeddings([gateway_error(503)] * 5))
        adapter = self.last_adapter
        assert adapter.discards == [True]
        assert adapter.dropped_runs == []

    def test_a_rejecting_model_releases_the_rows_and_the_run_carries_on(self, monkeypatch,
                                                                       no_backoff):
        toolkit, adapter = self.run(
            monkeypatch, FlakyEmbeddings([provider_rejection(openai.BadRequestError, 400)]))
        assert adapter.dropped_runs == [toolkit._index_run.run_id]
        assert True not in adapter.discards
        assert "a.py" in toolkit.reads

    def test_the_heartbeat_runs_before_the_probe(self, monkeypatch):
        _, adapter = self.run(monkeypatch, SameSpaceEmbeddings())
        assert adapter.order.index("heartbeat") < adapter.order.index("probe")

    def test_with_the_same_model_the_file_still_resumes(self, monkeypatch):
        toolkit, adapter = self.run(monkeypatch, SameSpaceEmbeddings())

        assert "a.py" not in toolkit.reads
        assert adapter.dropped_runs == []
        assert toolkit._index_run.reused_row_pks == set(self.ADOPTED)
