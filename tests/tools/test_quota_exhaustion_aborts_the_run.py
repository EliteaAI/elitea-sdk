"""A spent source quota must abort the run, not be skipped one file at a time.

Measured before this guard: an anonymous GitHub credential ran out of quota 47 files into
a 305-file repository, every remaining file was filed under read_error, and the run was
PROMOTED as `partly_indexed` with 382 of 1575 rows — a quarter-full index published as a
partial success, carrying no error_class at all.

The predicate is narrower than ToolErrorClass.INFRASTRUCTURE on purpose: a timeout or a
5xx affects one file and the run should carry on.
"""

import pytest
from github.GithubException import GithubException, RateLimitExceededException
from langchain_core.tools import ToolException

from elitea_sdk.runtime.tool_outcome import quota_exhausted
from elitea_sdk.tools.github.github_client import BoundedGithubRetry, GitHubClient

RATE_LIMIT = RateLimitExceededException(403, {"message": "API rate limit exceeded"},
                                       {"x-ratelimit-remaining": "0"})
THROTTLED = GithubException(429, {"message": "Too Many Requests"}, {})
QUOTA_403 = GithubException(403, {"message": "quota gone"},
                            {"x-ratelimit-remaining": "0", "x-ratelimit-limit": "60"})
NO_ACCESS = GithubException(403, {"message": "Resource not accessible"},
                            {"x-ratelimit-remaining": "4999"})
SERVER_DOWN = GithubException(503, {"message": "unavailable"}, {})


class TestTheQuotaPredicateIsNarrow:

    @pytest.mark.parametrize("exc", [RATE_LIMIT, THROTTLED, QUOTA_403])
    def test_a_spent_quota_is_recognised(self, exc):
        assert quota_exhausted(exc) is True

    @pytest.mark.parametrize("exc", [NO_ACCESS, SERVER_DOWN, TimeoutError("slow"),
                                     ValueError("nonsense")])
    def test_everything_else_is_not(self, exc):
        """Mutation: widen the predicate to ToolErrorClass.INFRASTRUCTURE — a 503 or a
        timeout would then abort a whole run over one flaky file."""
        assert quota_exhausted(exc) is False

    def test_it_is_found_through_a_wrapper(self):
        """Toolkits wrap provider errors; the chain is what carries the type."""
        try:
            try:
                raise RATE_LIMIT
            except Exception as inner:
                raise ToolException("could not read file") from inner
        except ToolException as wrapped:
            assert quota_exhausted(wrapped) is True

    def test_it_does_not_depend_on_wording(self):
        reworded = RateLimitExceededException(403, {"message": "secondary quota spent"},
                                             {"x-ratelimit-remaining": "0"})
        assert quota_exhausted(reworded) is True


class TestReadFileStopsRetryingASpentQuota:

    def client_raising(self, error, calls):
        class Repo:
            def get_contents(self, path, ref=None):
                calls.append(path)
                raise error
        client = GitHubClient.model_construct()
        object.__setattr__(client, "_github_repo_instance", Repo())
        object.__setattr__(client, "active_branch", "main")
        return client

    def test_a_spent_quota_is_not_retried(self):
        """Mutation: drop the quota_exhausted early raise — three attempts inside 1.5s
        cannot outlast an hourly reset, and the retries only burn wall clock."""
        calls = []
        with pytest.raises(Exception):
            self.client_raising(RATE_LIMIT, calls)._read_file("a.py", "main")
        assert len(calls) == 1

    def test_an_ordinary_failure_still_retries(self):
        calls = []
        with pytest.raises(ToolException):
            self.client_raising(GithubException(500, {"message": "boom"}, {}), calls)._read_file("a.py", "main")
        assert len(calls) == 3

    def test_the_cause_survives_the_retry_wrapper(self):
        """Mutation: drop `from last_exception` — classification loses the provider type."""
        calls = []
        with pytest.raises(ToolException) as raised:
            self.client_raising(GithubException(500, {"message": "boom"}, {}), calls)._read_file("a.py", "main")
        assert raised.value.__cause__ is not None

    def test_a_refusal_is_no_longer_called_a_missing_file(self):
        """Mutation: restore the "File not found" wording for every cause."""
        calls = []
        with pytest.raises(ToolException) as raised:
            self.client_raising(NO_ACCESS, calls)._read_file("a.py", "main")
        assert "not found" not in str(raised.value).lower()


class TestALongRetryAfterIsNotSleptThrough:
    """PyGithub honours Retry-After instead of the backoff, leaving get_backoff_time at 0.
    A bound reading only the backoff sleeps through every secondary rate limit.
    Mutation: inspect get_backoff_time alone."""

    class Resp:
        def __init__(self, status, headers):
            self.status, self.headers, self.reason = status, headers, "x"
        def get_redirect_location(self): return None
        @property
        def data(self): return b'{"message":"rate limit"}'

    @pytest.mark.parametrize("status", [403, 429])
    def test_an_hour_long_retry_after_raises(self, status):
        with pytest.raises(GithubException):
            BoundedGithubRetry(total=10).increment(
                method="GET", url="/x", response=self.Resp(status, {"Retry-After": "3600"}))

    def test_a_short_retry_after_still_retries(self):
        """A brief secondary pause is a legitimate retry, not an outage."""
        out = BoundedGithubRetry(total=10).increment(
            method="GET", url="/x", response=self.Resp(403, {"Retry-After": "5"}))
        assert out is not None

    def test_the_larger_of_the_two_waits_decides(self):
        retry = BoundedGithubRetry(total=10)
        class R:
            respect_retry_after_header = True
            def get_backoff_time(self): return 2.0
            def get_retry_after(self, response): return 900.0
        assert retry._intended_wait(R(), None) == 900.0

    def test_an_unreadable_wait_is_treated_as_no_wait(self):
        retry = BoundedGithubRetry(total=10)
        class R:
            respect_retry_after_header = True
            def get_backoff_time(self): raise RuntimeError("no idea")
            def get_retry_after(self, response): raise RuntimeError("no idea")
        assert retry._intended_wait(R(), None) == 0.0


class TestTheLoaderAbortsRatherThanFilingEveryFileAsAReadError:
    """The fix that prevents the measured outcome: 47 of 305 files indexed, 219 filed as
    read errors, and the run PROMOTED as partly_indexed with 382 of 1575 rows."""

    def loader_hitting(self, monkeypatch, error):
        from tests.tools.code_loader_rig import SHA_PY_BODY, IdentityToolkit, build_toolkit

        class Failing(IdentityToolkit):
            def _read_file(self, file_path, branch, **kwargs):
                raise error

        toolkit = build_toolkit(monkeypatch, {"a.py": SHA_PY_BODY, "b.py": "sha-b"}, {},
                               toolkit_cls=Failing)
        return toolkit

    def drive(self, toolkit):
        return list(toolkit.loader(index_name="x", preskipped_keys=set()))

    def test_a_spent_quota_aborts_the_loader(self, monkeypatch):
        """Mutation: let the per-file handler swallow a spent quota."""
        toolkit = self.loader_hitting(monkeypatch, RATE_LIMIT)
        with pytest.raises(Exception) as raised:
            self.drive(toolkit)
        assert quota_exhausted(raised.value) or isinstance(raised.value, RateLimitExceededException)

    def test_no_file_is_filed_as_a_read_error_on_a_spent_quota(self):
        """read_error is what made the run look partly successful."""
        import pytest as _pytest
        from tests.tools.code_loader_rig import SHA_PY_BODY, IdentityToolkit, build_toolkit

        class Failing(IdentityToolkit):
            def _read_file(self, file_path, branch, **kwargs):
                raise RATE_LIMIT

        with _pytest.MonkeyPatch.context() as mp:
            toolkit = build_toolkit(mp, {"a.py": SHA_PY_BODY, "b.py": "sha-b"}, {},
                                    toolkit_cls=Failing)
            with _pytest.raises(Exception):
                list(toolkit.loader(index_name="x", preskipped_keys=set()))
            assert toolkit.get_indexing_stats().files_skipped_read_error == set()

    def test_an_ordinary_read_failure_still_skips_just_that_file(self, monkeypatch):
        """The narrowness matters: one unreadable file must not abort a whole run."""
        toolkit = self.loader_hitting(monkeypatch, GithubException(500, {"message": "boom"}, {}))
        self.drive(toolkit)
        assert toolkit.get_indexing_stats().files_skipped_read_error == {"a.py", "b.py"}


class TestAQuotaAbortKeepsItsProgress:
    """A corpus larger than one quota window must still converge.

    Discarding the run's rows on a quota abort means the next run spends the same quota on
    the same first files and throws them away again — so a 305-file repo on a 60/hour
    credential could never finish, which is worse than the sleep-until-reset baseline it
    replaced. 'cancelled' is the only terminal status a later run can adopt from.
    """

    def toolkit_aborting_with(self, monkeypatch, error):
        from tests.tools.test_6586_meta_row_writes import StagingToolkit  # noqa: F401
        from tests.tools.code_loader_rig import SHA_PY_BODY, IdentityToolkit, build_toolkit

        class Failing(IdentityToolkit):
            def _read_file(self, file_path, branch, **kwargs):
                raise error

        toolkit = build_toolkit(monkeypatch, {"a.py": SHA_PY_BODY}, {}, toolkit_cls=Failing)
        calls = []
        adapter = toolkit.vector_adapter
        original = adapter.discard_run

        def recording(wrapper, index_name, run_id, retain_chunks=False):
            calls.append(retain_chunks)
            return "retained" if retain_chunks else "discarded"

        object.__setattr__(adapter, "discard_run", recording)
        return toolkit, calls

    def test_a_spent_quota_parks_the_rows_for_the_next_run(self, monkeypatch):
        """Mutation: always discard, ignoring quota_exhausted."""
        toolkit, calls = self.toolkit_aborting_with(monkeypatch, RATE_LIMIT)
        with pytest.raises(Exception):
            toolkit.index_data(index_name="x")
        assert calls and calls[-1] is True

    def test_an_ordinary_abort_still_discards(self, monkeypatch):
        """Retaining rows for every failure would leave garbage generations behind.

        Raised from the listing, not a file read: an ordinary per-file error is skipped
        rather than aborting, so it never reaches the discard at all.
        Mutation: retain unconditionally."""
        from tests.tools.code_loader_rig import SHA_PY_BODY, IdentityToolkit, build_toolkit

        class ListingBroken(IdentityToolkit):
            def _get_files_with_identity(self, path="", branch=None):
                raise RuntimeError("listing is unrelated to any quota")

        toolkit = build_toolkit(monkeypatch, {"a.py": SHA_PY_BODY}, {},
                                toolkit_cls=ListingBroken)
        calls = []
        object.__setattr__(toolkit.vector_adapter, "discard_run",
                           lambda wrapper, index_name, run_id, retain_chunks=False:
                           (calls.append(retain_chunks), "discarded")[1])
        with pytest.raises(Exception):
            toolkit.index_data(index_name="x")
        assert calls and calls[-1] is False

    def test_retained_rows_are_parked_as_cancelled_not_discarded(self):
        """Only pending/cancelled are adoptable, and discarded chunks get reclaimed."""
        import inspect

        from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import PGVectorAdapter

        source = inspect.getsource(PGVectorAdapter.discard_run)
        retained = source.split("if retain_chunks:")[1].split("return")[0]
        assert "RUN_STATUS_CANCELLED" in retained
        assert "RUN_STATUS_DISCARDED" not in retained


class TestOnlyARateLimitIsCapped:
    """A 5xx carrying Retry-After is a server asking us to wait, not a quota. Capping it
    skipped a legitimate wait AND reported it as a rate limit.
    Mutation: cap every status that carries Retry-After."""

    @staticmethod
    def Resp(status, retry_after, body):
        """A real urllib3 response: get_content reads the stream, so a hand-rolled stub
        that only exposes .data cannot exercise the body extraction at all."""
        import io

        import urllib3
        return urllib3.HTTPResponse(
            body=io.BytesIO(body), status=status, preload_content=False,
            headers={"Retry-After": retry_after, "Content-Type": "application/json",
                     "x-ratelimit-remaining": "0", "x-ratelimit-limit": "60"})

    def increment(self, status, retry_after, body):
        return BoundedGithubRetry(total=10, raise_on_status=False).increment(
            method="GET", url="/x", response=self.Resp(status, retry_after, body))

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_a_server_error_is_left_to_wait(self, status):
        assert self.increment(status, "120", b'{"message":"unavailable"}') is not None

    def test_a_rate_limited_403_still_raises(self):
        with pytest.raises(GithubException):
            self.increment(403, "3600", b'{"message":"API rate limit exceeded"}')

    def test_the_raised_error_carries_the_quota_headers(self):
        """The body cannot be used: it is readable once, and super() needs it to recognise
        the rate limit at all. The headers survive and carry the facts.
        Mutation: drop response.headers from the raise."""
        with pytest.raises(GithubException) as raised:
            self.increment(429, "3600", b'{"message":"Too Many Requests"}')
        headers = {str(k).lower(): v for k, v in (raised.value.headers or {}).items()}
        assert headers.get("x-ratelimit-remaining") == "0"
        assert headers.get("x-ratelimit-limit") == "60"

    def test_a_rate_limited_403_is_typed_as_a_rate_limit(self):
        """The real body is what lets createException pick RateLimitExceededException,
        which is what quota_exhausted matches on."""
        from github.GithubException import RateLimitExceededException as RLE
        with pytest.raises(RLE):
            self.increment(403, "3600", b'{"message":"API rate limit exceeded"}')

    def test_an_unparseable_body_still_raises_a_typed_rate_limit(self):
        """Independent of the body by design."""
        from github.GithubException import RateLimitExceededException as RLE
        with pytest.raises(RLE):
            self.increment(403, "3600", b'<html>nope</html>')


class TestRetriesRunningOutStayTyped:
    """raise_on_status=True let urllib3 raise MaxRetryError once retries ran out, which
    requests turns into RetryError — untyped, so quota_exhausted and classify_tool_error
    both miss it and the file is filed as a read error.
    Mutation: drop raise_on_status=False."""

    def test_the_clients_do_not_raise_on_status(self):
        import ast
        import inspect

        from elitea_sdk.tools.github import github_client

        tree = ast.parse(inspect.getsource(github_client))
        found = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Both the plain constructor and the reads-only classmethod build a policy, and
            # each one that reaches a client must carry raise_on_status.
            direct = getattr(node.func, "id", None) == "BoundedGithubRetry"
            via_factory = (getattr(node.func, "attr", None) == "for_reads_only"
                           and getattr(getattr(node.func, "value", None), "id", None)
                           == "BoundedGithubRetry")
            if direct or via_factory:
                assert "raise_on_status" in {kw.arg for kw in node.keywords}
                found += 1
        assert found >= 3


class TestARefusalIsNotRelabelledAsAQuota:
    """A 403 carrying Retry-After — a GHE proxy or firewall does this — was being raised as
    RateLimitExceededException with a fabricated "API rate limit exceeded", classified
    infrastructure, and parked as retriable. No amount of waiting fixes a refusal.

    The body is readable here precisely because PyGithub reads it only on the branch where
    Retry-After is ABSENT, so the real message decides the type.
    Mutation: always raise RateLimitExceededException from the headers.
    """

    def resp(self, headers, body, status=403):
        import io

        import urllib3
        return urllib3.HTTPResponse(
            body=io.BytesIO(body), status=status, preload_content=False,
            headers={"Content-Type": "application/json", **headers})

    def raised_by(self, headers, body, status=403):
        with pytest.raises(GithubException) as raised:
            BoundedGithubRetry(total=10, raise_on_status=False).increment(
                method="GET", url="/x", response=self.resp(headers, body, status))
        return raised.value

    REFUSAL = b'{"message":"Resource not accessible by integration"}'
    SPENT = b'{"message":"API rate limit exceeded"}'

    def test_a_refusal_keeps_its_own_message(self):
        error = self.raised_by({"Retry-After": "120", "x-ratelimit-remaining": "4999"},
                               self.REFUSAL)
        assert "not accessible" in str(error.data)
        assert "rate limit" not in str(error.data).lower()

    def test_a_refusal_is_not_typed_as_a_rate_limit(self):
        from github.GithubException import RateLimitExceededException as RLE

        assert not isinstance(
            self.raised_by({"Retry-After": "120", "x-ratelimit-remaining": "4999"},
                           self.REFUSAL), RLE)

    def test_a_refusal_is_not_parked_for_resume(self):
        """Parking it would retain rows for a run that can never make progress."""
        error = self.raised_by({"Retry-After": "120", "x-ratelimit-remaining": "4999"},
                               self.REFUSAL)
        assert quota_exhausted(error) is False

    def test_a_refusal_classifies_as_policy(self):
        from elitea_sdk.runtime.tool_outcome import classify_tool_error

        error = self.raised_by({"Retry-After": "120", "x-ratelimit-remaining": "4999"},
                               self.REFUSAL)
        assert classify_tool_error(error).value == "policy"

    def test_a_real_quota_is_still_parked(self):
        error = self.raised_by({"Retry-After": "3600", "x-ratelimit-remaining": "0"},
                               self.SPENT)
        assert quota_exhausted(error) is True


class TestExhaustedRetriesOnARateLimitStayTyped:
    """A secondary limit sends no Retry-After, so PyGithub sets its documented 60s backoff —
    not MORE than the 60s bound, so all ten retries run. With raise_on_status off, urlopen
    then hands back a response whose body is already read, and the failure surfaced as an
    untyped 403 that classified as POLICY: a credential problem waiting cannot fix.

    The 10x60s wait itself predates this work. What is new is that the wait is now paid once,
    because the run aborts and parks, instead of once per file.
    Mutation: drop the MaxRetryError branch from increment.
    """

    SECONDARY = b'{"message":"You have exceeded a secondary rate limit."}'

    def exhaust(self, status=403, body=None, total=10):
        import io

        import urllib3
        retry = BoundedGithubRetry(total=total, raise_on_status=False)
        for _ in range(total + 3):
            response = urllib3.HTTPResponse(
                body=io.BytesIO(body or self.SECONDARY), status=status,
                preload_content=False,
                headers={"Content-Type": "application/json",
                         "x-ratelimit-remaining": "4999"})
            retry = retry.increment(method="GET", url="/x", response=response)
        raise AssertionError("retries never ran out")

    def test_it_surfaces_as_a_rate_limit_not_a_refusal(self):
        from github.GithubException import RateLimitExceededException as RLE

        with pytest.raises(RLE):
            self.exhaust()

    def test_it_classifies_as_retriable_infrastructure(self):
        from elitea_sdk.runtime.tool_outcome import classify_tool_error, retriable_for

        with pytest.raises(GithubException) as raised:
            self.exhaust()
        error_class = classify_tool_error(raised.value)
        assert error_class.value == "infrastructure"
        assert retriable_for(error_class) is True

    def test_it_parks_the_run_so_the_wait_is_paid_once(self):
        with pytest.raises(GithubException) as raised:
            self.exhaust()
        assert quota_exhausted(raised.value) is True

    def test_a_server_error_exhausting_its_retries_is_not_relabelled(self):
        """Only a rate-limited status gets this treatment; a 503 stays urllib3's problem."""
        from urllib3.exceptions import MaxRetryError

        with pytest.raises(MaxRetryError):
            self.exhaust(status=503, body=b'{"message":"down"}', total=2)


class TestOnlyTheAppClientGivesUpWriteRetries:
    """GithubIntegration defaults to retry=None, so an App client retried nothing. Handing it
    the shared policy newly made writes retryable — and PyGithub's allowed_methods includes
    POST, so a 502 on create_issue or a PR comment could produce duplicates. Token clients
    already retried POST under PyGithub's own default, so they are left alone.
    Mutation: give the App client the shared BoundedGithubRetry."""

    def integration_retry_call(self):
        import ast
        import inspect

        from elitea_sdk.tools.github import github_client

        tree = ast.parse(inspect.getsource(github_client))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "GithubIntegration"):
                for keyword in node.keywords:
                    if keyword.arg == "retry":
                        return keyword.value
        raise AssertionError("no GithubIntegration construction with a retry policy")

    def test_the_app_client_uses_the_reads_only_policy(self):
        node = self.integration_retry_call()
        assert getattr(node.func, "attr", None) == "for_reads_only"

    def test_the_reads_only_policy_refuses_post(self):
        assert BoundedGithubRetry.for_reads_only(
            total=10, raise_on_status=False)._is_method_retryable("POST") is False

    def test_it_still_retries_reads(self):
        """Giving up write retries must not cost the rate-limit bound on reads."""
        assert BoundedGithubRetry.for_reads_only(
            total=10, raise_on_status=False)._is_method_retryable("GET") is True

    def test_token_clients_are_left_as_they_were(self):
        """They retried POST before this work; changing that is not in scope."""
        assert BoundedGithubRetry(
            total=10, raise_on_status=False)._is_method_retryable("POST") is True


class TestAdoptionOutlivesTheDigestCeiling:
    """A generation above adoption_max_chunks used to be dropped, so a corpus needing
    several quota windows never converged: each run spent the quota on the same first files
    and threw them away, while reporting itself retriable.

    The ceiling exists for the digest map, which holds a digest AND a pk list per row. The
    supersede fence needs only the ids — about 16MB at 200k rows, 81MB at a million — so
    adoption is kept and only chunk-content reuse is given up.
    """

    def toolkit_adopting(self, monkeypatch, truncated, row_pks=None, digests=None):
        from tests.tools.test_6586_meta_row_writes import StagingToolkit  # noqa: F401
        from tests.tools.code_loader_rig import SHA_PY_BODY, IdentityToolkit, build_toolkit

        toolkit = build_toolkit(monkeypatch, {"a.py": SHA_PY_BODY}, {},
                               toolkit_cls=IdentityToolkit)
        adapter = toolkit.vector_adapter
        object.__setattr__(adapter, "read_run_staged_digests",
                           lambda w, r, d, cap: (digests or {}, set(row_pks or []), truncated))
        object.__setattr__(adapter, "read_run_row_pks",
                           lambda w, r: set(row_pks or []))
        object.__setattr__(adapter, "read_run_complete_files",
                           lambda w, r: {"a.py": (SHA_PY_BODY, list(row_pks or []))})
        # the release path drops the rows rather than risk publishing them twice
        object.__setattr__(adapter, "drop_run_chunks", lambda w, r: None)
        from elitea_sdk.tools.base_indexer_toolkit import _IndexRunState

        run = getattr(toolkit, "_index_run", None)
        if run is None:
            run = _IndexRunState(run_id="r")
            object.__setattr__(toolkit, "_index_run", run)
        run.adopted_from_run_id = "old-run"
        return toolkit, run

    def test_an_oversized_generation_keeps_its_rows(self, monkeypatch):
        """Mutation: release the adopted rows when the digest read truncates."""
        toolkit, run = self.toolkit_adopting(monkeypatch, truncated=True,
                                             row_pks=["pk-1", "pk-2", "pk-3"])
        toolkit._load_adopted_chunk_digests()
        assert run.adopted_row_pks == {"pk-1", "pk-2", "pk-3"}
        assert run.adopted_from_run_id == "old-run"

    def test_a_partial_digest_map_is_still_used(self, monkeypatch):
        """What the capped read did manage to index is still reusable; the fence covers the
        rest. Mutation: discard the partial map, which only loses reuse."""
        toolkit, run = self.toolkit_adopting(monkeypatch, truncated=True,
                                             row_pks=["pk-1", "pk-2"],
                                             digests={"d": ["pk-1"]})
        toolkit._load_adopted_chunk_digests()
        assert run.adoptable_chunks == {"d": ["pk-1"]}
        assert run.adopted_row_pks == {"pk-1", "pk-2"}

    def test_an_oversized_generation_can_still_resume_by_file(self, monkeypatch):
        """File-level resumption is what makes the kept rows useful."""
        toolkit, run = self.toolkit_adopting(monkeypatch, truncated=True, row_pks=["pk-1"])
        toolkit._load_adopted_chunk_digests()
        assert "a.py" in run.resumable_files

    def test_the_fence_still_covers_every_adopted_row(self, monkeypatch):
        """Without the full id set, unreused adopted rows would be published beside the
        re-indexed copies — every chunk twice."""
        toolkit, run = self.toolkit_adopting(monkeypatch, truncated=True,
                                             row_pks=["pk-1", "pk-2"])
        toolkit._load_adopted_chunk_digests()
        superseded, _, _ = toolkit._assemble_promote_sets()
        assert sorted(superseded) == ["pk-1", "pk-2"]

    def test_a_generation_inside_the_ceiling_still_reuses_chunks(self, monkeypatch):
        toolkit, run = self.toolkit_adopting(monkeypatch, truncated=False,
                                             row_pks=["pk-1"], digests={"d": ["pk-1"]})
        toolkit._load_adopted_chunk_digests()
        assert run.adoptable_chunks == {"d": ["pk-1"]}

    def test_an_unreadable_id_read_still_fails_safe(self, monkeypatch):
        """Publishing rows the fence cannot cover is the duplication bug; dropping is safe."""
        toolkit, run = self.toolkit_adopting(monkeypatch, truncated=True, row_pks=["pk-1"])
        def explode(wrapper, run_id):
            raise RuntimeError("read blew up")
        object.__setattr__(toolkit.vector_adapter, "read_run_row_pks", explode)
        toolkit._load_adopted_chunk_digests()
        assert run.adopted_row_pks == set()
        assert run.adopted_from_run_id is None


class TestTheClaimNoLongerRefusesALargeGeneration:

    def test_the_claim_ceiling_is_the_restamp_one_not_the_digest_one(self):
        """Two different costs, two different ceilings: the digest map is memory, the claim
        is how long the index meta row stays locked while every row is re-stamped.
        Mutation: pass adoption_max_chunks, or None, to the claim."""
        import ast
        import inspect

        from elitea_sdk.tools import base_indexer_toolkit

        tree = ast.parse(inspect.getsource(base_indexer_toolkit))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "claim_adoptable_run"):
                last = node.args[-1]
                assert getattr(last, "attr", None) == "adoption_restamp_max_chunks", \
                    "the claim must be bounded by the restamp ceiling"
                return
        raise AssertionError("no claim_adoptable_run call found")

    def test_the_restamp_ceiling_is_above_the_digest_one(self):
        """Otherwise the id-only path it exists to enable is unreachable."""
        from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit

        assert (BaseIndexerToolkit.adoption_restamp_max_chunks
                > BaseIndexerToolkit.adoption_max_chunks)

    def test_the_restamp_ceiling_keeps_the_lock_to_seconds(self):
        """Measured ~10.2s at 500k rows and ~26s at 1M; Stop and promote block throughout."""
        from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit

        assert BaseIndexerToolkit.adoption_restamp_max_chunks <= 1_000_000


class TestAnAdapterThatCannotEnumerateItsRowsFailsLoudly:
    """The id read backs the supersede fence. An adapter that reported a truncated digest
    read but returned an empty id set would leave the fence covering nothing, and every
    adopted row would be published beside its re-indexed copy — the duplication this work
    started by fixing. Raising makes the caller release the rows instead.
    Mutation: return an empty set from the base implementation.
    """

    def test_the_base_adapter_refuses_rather_than_returning_nothing(self):
        from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import VectorStoreAdapter

        # Called unbound: the class is abstract, and the method does not touch self.
        with pytest.raises(NotImplementedError):
            VectorStoreAdapter.read_run_row_pks(object(), None, "run-1")

    def test_the_adapter_that_does_support_it_still_works(self):
        """Guards the guard: if PGVector had inherited the raise, adoption would be dead."""
        from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import PGVectorAdapter

        assert (PGVectorAdapter.read_run_row_pks
                is not PGVectorAdapter.__mro__[1].read_run_row_pks)


class TestACorpusTooLargeToResumeStopsInsteadOfLooping:
    """Above adoption_restamp_max_chunks the next run's claim declines, and the sweep then
    deletes the parked rows once the heartbeat ages out. Parking such a generation loses it a
    window later while every attempt reports retriable: true — a silent forever-loop on
    roughly 50k-file repos, which GitHub's 100k-entry tree listing puts well in range.

    This is close to a fix I applied and then reverted. The difference is that it now applies
    only ABOVE the lock bound, so nothing under 500k is stranded by it.
    """

    def toolkit_with_generation(self, monkeypatch, chunks_written, adopted=(), reused=()):
        from tests.tools.code_loader_rig import SHA_PY_BODY, IdentityToolkit, build_toolkit

        class Failing(IdentityToolkit):
            def _read_file(self, file_path, branch, **kwargs):
                raise RATE_LIMIT

        toolkit = build_toolkit(monkeypatch, {"a.py": SHA_PY_BODY}, {}, toolkit_cls=Failing)
        retained = []
        object.__setattr__(toolkit.vector_adapter, "discard_run",
                           lambda wrapper, index_name, run_id, retain_chunks=False:
                           (retained.append(retain_chunks), "x")[1])
        original = toolkit._generation_fits_the_restamp_bound

        def sized():
            run = toolkit._index_run
            if run is not None:
                run.chunks_written = chunks_written
                run.adopted_row_pks = set(adopted)
                run.reused_row_pks = set(reused)
            return original()

        object.__setattr__(toolkit, "_generation_fits_the_restamp_bound", sized)
        return toolkit, retained

    def outcome(self, monkeypatch, chunks_written, adopted=(), reused=()):
        import json
        toolkit, retained = self.toolkit_with_generation(
            monkeypatch, chunks_written, adopted, reused)
        with pytest.raises(Exception):
            toolkit.index_data(index_name="x")
        stored = toolkit._stored_meta["metadata"]
        report = stored.get("report")
        return retained, stored, json.loads(report) if isinstance(report, str) else (report or {})

    def test_a_resumable_generation_is_parked_and_stays_retriable(self, monkeypatch):
        retained, _, report = self.outcome(monkeypatch, 1_575)
        assert retained and retained[-1] is True
        assert report.get("retriable") is True

    def test_an_unresumable_generation_is_discarded_not_parked(self, monkeypatch):
        """Mutation: park regardless of size, deferring the loss by one window."""
        retained, _, _ = self.outcome(monkeypatch, 600_000)
        assert retained and retained[-1] is False

    def test_an_unresumable_generation_is_not_reported_retriable(self, monkeypatch):
        """Mutation: drop retriable= from build_error_report, leaving the classifier's
        'infrastructure' to advertise an endless retry."""
        _, _, report = self.outcome(monkeypatch, 600_000)
        assert report.get("retriable") is False

    def test_the_error_says_what_to_do_about_it(self, monkeypatch):
        _, stored, _ = self.outcome(monkeypatch, 600_000)
        error = stored.get("error") or ""
        assert "adoption_restamp_max_chunks" in error
        assert "quota" in error and "scope" in error

    def test_adopted_rows_not_reused_count_toward_the_bound(self, monkeypatch):
        """They stay in the generation, so counting only this run's writes reads low."""
        adopted = [f"pk-{i}" for i in range(80)]
        retained, _, _ = self.outcome(monkeypatch, 499_950, adopted=adopted)
        assert retained and retained[-1] is False

    def test_reused_adopted_rows_are_not_counted_twice(self, monkeypatch):
        adopted = [f"pk-{i}" for i in range(80)]
        retained, _, _ = self.outcome(monkeypatch, 1_000, adopted=adopted, reused=adopted)
        assert retained and retained[-1] is True

    def test_a_generation_between_the_two_ceilings_is_still_parked(self, monkeypatch):
        """The digest ceiling (200k) is about memory and must not gate parking: a generation
        above it is still adoptable through the id-only read. Only the restamp ceiling
        (500k, the meta lock) decides whether a later run can take it.
        Mutation: bound the check against adoption_max_chunks."""
        from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit

        between = (BaseIndexerToolkit.adoption_max_chunks
                   + BaseIndexerToolkit.adoption_restamp_max_chunks) // 2
        assert BaseIndexerToolkit.adoption_max_chunks < between \
            < BaseIndexerToolkit.adoption_restamp_max_chunks

        retained, _, report = self.outcome(monkeypatch, between)
        assert retained and retained[-1] is True
        assert report.get("retriable") is True


class TestParkingRequiresThatAnyoneWouldClaimTheRows:
    """The park decision checked staging, quota and the restamp bound, but not whether a
    later run would adopt at all. _adoption_is_available() refuses a clean_index run and an
    index with adoption switched off, so those parked rows are never claimed and the sweep
    deletes them — the same silent loop, reachable through a saved config or a schedule
    passing clean_index=true.
    """

    def outcome(self, monkeypatch, clean_index=False, adoption_enabled=True,
                chunks_written=1_575):
        import json

        from tests.tools.code_loader_rig import SHA_PY_BODY, IdentityToolkit, build_toolkit

        class Failing(IdentityToolkit):
            def _read_file(self, file_path, branch, **kwargs):
                raise RATE_LIMIT

        toolkit = build_toolkit(monkeypatch, {"a.py": SHA_PY_BODY}, {}, toolkit_cls=Failing)
        object.__setattr__(toolkit, "adoption_enabled", adoption_enabled)
        retained = []
        object.__setattr__(toolkit.vector_adapter, "discard_run",
                           lambda wrapper, index_name, run_id, retain_chunks=False:
                           (retained.append(retain_chunks), "x")[1])
        original = toolkit._unresumable_quota_reason

        def primed():
            run = toolkit._index_run
            if run is not None:
                run.clean_index = clean_index
                run.chunks_written = chunks_written
            return original()

        object.__setattr__(toolkit, "_unresumable_quota_reason", primed)
        raised = None
        try:
            toolkit.index_data(index_name="x")
        except Exception as exc:
            raised = exc
        stored = toolkit._stored_meta["metadata"]
        report = stored.get("report")
        return retained, stored, (json.loads(report) if isinstance(report, str)
                                  else (report or {})), raised

    def test_a_clean_index_run_is_not_parked(self, monkeypatch):
        """Mutation: drop the adoption-availability check from the park condition."""
        retained, _, _, _ = self.outcome(monkeypatch, clean_index=True)
        assert retained and retained[-1] is False

    def test_a_clean_index_run_is_not_reported_retriable(self, monkeypatch):
        _, _, report, _ = self.outcome(monkeypatch, clean_index=True)
        assert report.get("retriable") is False

    def test_the_remedy_names_clean_index(self, monkeypatch):
        """A generic "raise the quota" would send the reader at the wrong thing."""
        _, stored, _, _ = self.outcome(monkeypatch, clean_index=True)
        assert "Clean Index" in (stored.get("error") or "")

    def test_adoption_switched_off_is_also_unresumable(self, monkeypatch):
        retained, stored, report, _ = self.outcome(monkeypatch, adoption_enabled=False)
        assert retained and retained[-1] is False
        assert report.get("retriable") is False
        assert "switched off" in (stored.get("error") or "")

    def test_an_ordinary_resumable_run_is_still_parked(self, monkeypatch):
        retained, _, report, _ = self.outcome(monkeypatch)
        assert retained and retained[-1] is True
        assert report.get("retriable") is True

    def test_the_over_bound_reason_still_names_the_bound(self, monkeypatch):
        _, stored, _, _ = self.outcome(monkeypatch, chunks_written=600_000)
        assert "adoption_restamp_max_chunks" in (stored.get("error") or "")


class TestTheRemedyTravelsWithTheException:
    """An agent calling index_data sees the exception, not the meta row. Re-raising the bare
    quota error gave it the provider's message with no remedy, so a retry restarts from zero
    and burns another window.
    Mutation: re-raise the original error on the unresumable path.
    """

    def raised_for(self, monkeypatch, clean_index):
        return TestParkingRequiresThatAnyoneWouldClaimTheRows().outcome(
            monkeypatch, clean_index=clean_index)[3]

    def test_an_unresumable_failure_carries_the_remedy(self, monkeypatch):
        raised = self.raised_for(monkeypatch, clean_index=True)
        assert "Clean Index" in str(raised)

    def test_the_provider_error_stays_reachable_for_the_classifier(self, monkeypatch):
        """Replacing the exception must not cost the cause chain the classifier reads."""
        raised = self.raised_for(monkeypatch, clean_index=True)
        assert quota_exhausted(raised) is True

    def test_a_resumable_failure_is_re_raised_unchanged(self, monkeypatch):
        """Only the new path changes shape; the proven one keeps its exception."""
        from github.GithubException import RateLimitExceededException as RLE

        raised = self.raised_for(monkeypatch, clean_index=False)
        assert isinstance(raised, RLE)
