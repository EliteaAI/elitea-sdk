"""#5261 — the order index_meta_init runs adoption in.

The order is the safety property, not an implementation detail:

  claim -> sweep(except claimed) -> register -> re-stamp

Claim must precede the sweep or the sweep reclaims the very rows being adopted.
The re-stamp must follow registration: a row carrying a run id that has no run row
yet matches no pending run, and `_active_data_rows_clause` therefore lets it through
to search. Parking the claimed run as 'cancelled' is what keeps it hidden in between
while releasing the slot `live_run_where()` guards.
"""

from uuid import uuid4

import pytest

from elitea_sdk.tools.base_indexer_toolkit import _IndexRunState
from tests.tools.test_6586_meta_row_writes import (  # noqa: F401  (fixtures)
    FakeStagingAdapter,
    StagingToolkit,
    seed_meta,
    sessions,
    toolkit,
)


class AdoptionRecordingAdapter(FakeStagingAdapter):
    def __init__(self):
        super().__init__()
        self.claimable = "old-run"
        self.swept_except = []
        self.adoptions = []
        self.adopted_count = 7

    def claim_adoptable_run(self, wrapper, index_name, stale_before, max_chunks=None):
        self.calls.append("claim")
        return self.claimable

    def sweep_stale_index_runs(self, wrapper, index_name, stale_before, except_run_id=None):
        self.calls.append("sweep")
        self.swept_except.append(except_run_id)
        return []

    def adopt_run_chunks(self, wrapper, index_name, source_run_id, target_run_id):
        self.calls.append("adopt")
        self.adoptions.append((source_run_id, target_run_id))
        return self.adopted_count


@pytest.fixture
def adopting(toolkit):  # noqa: F811
    adapter = AdoptionRecordingAdapter()
    toolkit.vector_adapter = adapter
    toolkit._index_run = _IndexRunState(run_id=uuid4().hex[:12])
    return toolkit, adapter


def init_fresh(toolkit):
    toolkit._stored_meta = None
    toolkit.index_meta_init("docs", {})


def init_reindex(toolkit):
    seed_meta(toolkit)
    toolkit.index_meta_init("docs", {})


class TestTheOrderIsClaimSweepRegisterAdopt:

    @pytest.mark.parametrize("start", [init_fresh, init_reindex])
    def test_every_branch_runs_the_four_steps_in_order(self, adopting, start):
        """Mutation: move the _adopt_claimed_run call above _register_index_run."""
        toolkit, adapter = adopting
        start(toolkit)
        staged = [call for call in adapter.calls if call in ("claim", "sweep", "register", "adopt")]
        assert staged == ["claim", "sweep", "register", "adopt"]

    @pytest.mark.parametrize("start", [init_fresh, init_reindex])
    def test_the_sweep_spares_the_claimed_run(self, adopting, start):
        """Mutation: drop except_run_id from the sweep call."""
        toolkit, adapter = adopting
        start(toolkit)
        assert adapter.swept_except == ["old-run"]

    @pytest.mark.parametrize("start", [init_fresh, init_reindex])
    def test_the_claimed_rows_move_onto_this_run(self, adopting, start):
        toolkit, adapter = adopting
        start(toolkit)
        assert adapter.adoptions == [("old-run", toolkit._index_run.run_id)]
        assert toolkit._index_run.adopted_from_run_id == "old-run"
        assert toolkit._index_run.adopted_chunk_count == 7


class TestAdoptionIsSkippedWhenItMustNotRun:

    def test_nothing_to_claim_means_nothing_to_adopt(self, adopting):
        toolkit, adapter = adopting
        adapter.claimable = None
        init_reindex(toolkit)
        assert "adopt" not in adapter.calls
        assert adapter.swept_except == [None]
        assert toolkit._index_run.adopted_from_run_id is None

    def test_a_clean_index_rebuild_never_adopts(self, adopting):
        """clean_index is the user asking for a full rebuild. Mutation: drop the
        clean_index term from _adoption_is_available."""
        toolkit, adapter = adopting
        seed_meta(toolkit)
        toolkit._index_run.clean_index = True
        toolkit.index_meta_init("docs", {})
        assert "claim" not in adapter.calls
        assert "adopt" not in adapter.calls

    def test_the_kill_switch_disables_adoption(self, adopting, monkeypatch):
        toolkit, adapter = adopting
        monkeypatch.setattr(type(toolkit), "adoption_enabled", False)
        init_reindex(toolkit)
        assert "claim" not in adapter.calls
        assert "adopt" not in adapter.calls

    def test_an_adapter_without_adoption_support_is_tolerated(self, toolkit):  # noqa: F811
        toolkit.vector_adapter = FakeStagingAdapter()
        toolkit._index_run = _IndexRunState(run_id=uuid4().hex[:12])
        seed_meta(toolkit)
        toolkit.index_meta_init("docs", {})
        assert toolkit._index_run.adopted_from_run_id is None


class TestAdoptionFailuresNeverFailTheRun:

    def test_a_failed_claim_leaves_the_run_indexing_from_scratch(self, adopting):
        toolkit, adapter = adopting

        def explode(*_args, **_kwargs):
            raise RuntimeError("claim blew up")

        adapter.claim_adoptable_run = explode
        init_reindex(toolkit)
        assert toolkit._index_run.adopted_from_run_id is None
        assert "register" in adapter.calls

    def test_a_failed_restamp_leaves_the_parked_rows_for_a_later_run(self, adopting):
        toolkit, adapter = adopting

        def explode(*_args, **_kwargs):
            raise RuntimeError("restamp blew up")

        adapter.adopt_run_chunks = explode
        init_reindex(toolkit)
        assert toolkit._index_run.adopted_from_run_id is None
        assert "register" in adapter.calls

    def test_an_adoption_that_moves_no_rows_is_not_recorded(self, adopting):
        toolkit, adapter = adopting
        adapter.adopted_count = 0
        init_reindex(toolkit)
        assert toolkit._index_run.adopted_from_run_id is None
