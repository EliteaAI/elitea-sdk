"""#6981: Tavily toolkit + credential, exercised against a fake Tavily HTTP server.

The fake server is reached through the deployment-level TAVILY_API_BASE_URL env var, the
real tavily-python client and the real metered_call run end to end; only the platform
(check/record endpoints) is a stub object.
"""
import json
import logging
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from langchain_core.tools import ToolException

from elitea_sdk.configurations.tavily import TavilyConfiguration
from elitea_sdk.tools import tavily as tavily_toolkit
from elitea_sdk.tools.tavily import api_wrapper as wrapper_module
from elitea_sdk.tools.tavily.api_wrapper import TRUNCATED_NOTE, TavilyApiWrapper
from elitea_sdk.tools.utils import metering
from elitea_sdk.tools.utils.metering import ToolSpendDenied

KEY = "tvly-secret-key-123"


class FakeTavily:
    """Queue of (status, body) per path; records every request it receives."""

    def __init__(self):
        self.replies = {}
        self.requests = []
        self.lock = threading.Lock()

    def reply(self, path, *responses):
        self.replies[path] = list(responses)

    def next(self, path):
        with self.lock:
            queue = self.replies.get(path) or [(200, {})]
            return queue.pop(0) if len(queue) > 1 else queue[0]


@pytest.fixture
def fake(monkeypatch):
    state = FakeTavily()

    class Handler(BaseHTTPRequestHandler):
        def _serve(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            with state.lock:
                state.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
            status, payload = state.next(self.path)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _serve

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("TAVILY_API_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    yield state
    server.shutdown()


class Platform:
    """Stub of the EliteAClient tool-spend surface."""

    def __init__(self, allowed=True, run_id="run-1"):
        self.project_id = 3
        self.headers = {"X-Elitea-Run-Id": run_id} if run_id else {}
        self.allowed = allowed
        self.checks, self.records = [], []
        self.lock = threading.Lock()

    def check_tool_spend(self, payload, timeout=None):
        with self.lock:
            self.checks.append(payload)
        if not self.allowed:
            return {"allowed": False, "code": "tool_spend_limit_reached", "message": "Web search limit reached."}
        return {"allowed": True, "exempt": False, "permit": f"P{len(self.checks)}"}

    def record_tool_spend(self, payload, timeout=None):
        with self.lock:
            self.records.append(payload)
        return {"recorded": True}


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    wrapper_module._run_calls.clear()  # pylint: disable=W0212
    metering._exempt_until.clear()  # pylint: disable=W0212
    monkeypatch.setattr(wrapper_module, "_sleep", lambda _s: None)
    monkeypatch.setattr(metering, "_sleep", lambda _s: None)


def tools(platform=None, selected=None, **settings):
    tool = {"type": "tavily", "toolkit_name": "web", "settings": {
        "selected_tools": selected or [],
        "tavily_configuration": {"api_key": KEY, "configuration_uuid": "u-1",
                                 "configuration_project_id": 1, "configuration_type": "tavily"},
        "elitea": platform, **settings}}
    return {t.name: t for t in tavily_toolkit.get_tools(tool)}


def search_ok(results=None, credits=1):
    body = {"results": results if results is not None else [
        {"title": "Example", "url": "https://example.com/a", "content": "hello"}]}
    if credits is not None:
        body["usage"] = {"credits": credits}
    return 200, body


# ---- schema and credential ----------------------------------------------------------------

class TestSchema:
    def test_tool_spend_metadata_matches_ts0_contract(self):
        meta = tavily_toolkit.TavilyToolkit.toolkit_config_schema().model_json_schema()["metadata"]
        spend = meta["tool_spend"]
        assert spend["feature"] == "web_search" and spend["backend"] == "tavily"
        assert spend["configuration_type"] == "tavily" and spend["unit"] == "credit"
        assert spend["max_units_per_call"] == 250
        assert set(spend["operations"]) == {"search", "extract", "crawl", "map", "research"}
        assert spend["chat_tools"] == ["web_search", "web_extract"]

    def test_lists_four_tools_and_honors_selected_tools(self):
        assert set(tools()) == {"web_search", "web_extract", "web_crawl", "web_map"}
        assert set(tools(selected=["web_search"])) == {"web_search"}

    def test_credential_holds_only_the_key(self):
        assert set(TavilyConfiguration.model_fields) == {"api_key"}

    def test_no_base_url_field_anywhere(self):
        toolkit = json.dumps(tavily_toolkit.TavilyToolkit.toolkit_config_schema().model_json_schema())
        credential = json.dumps(TavilyConfiguration.model_json_schema())
        for schema in (toolkit, credential):
            assert "base_url" not in schema and "api_base_url" not in schema

    def test_credential_ids_reach_the_wrapper(self):
        wrapper = tools()["web_search"].api_wrapper
        assert (wrapper.configuration_uuid, wrapper.configuration_project_id,
                wrapper.configuration_type) == ("u-1", 1, "tavily")


class TestCheckConnection:
    def test_uses_free_usage_endpoint(self, fake):
        fake.reply("/usage", (200, {"key": {"usage": 3}}))
        assert TavilyConfiguration.check_connection({"api_key": KEY}) is None
        assert [r["path"] for r in fake.requests] == ["/usage"]
        assert fake.requests[0]["headers"]["Authorization"] == f"Bearer {KEY}"

    def test_bad_key_message_has_no_key(self, fake):
        fake.reply("/usage", (401, {"detail": {"error": f"bad {KEY}"}}))
        message = TavilyConfiguration.check_connection({"api_key": KEY})
        assert message == "Invalid Tavily API key" and KEY not in message

    def test_empty_key(self):
        assert TavilyConfiguration.check_connection({"api_key": " "}) == "Tavily API key is required"


# ---- metering ---------------------------------------------------------------------------------

class TestMetering:
    def test_every_http_request_runs_inside_metered_call(self, fake, monkeypatch):
        """Patch the session layer: any request outside an open metered block fails the test."""
        inside = threading.local()
        real_metered = wrapper_module.metered_call

        @contextmanager
        def tracking(*args, **kwargs):
            with real_metered(*args, **kwargs) as call:
                inside.open = True
                try:
                    yield call
                finally:
                    inside.open = False

        real_request = requests.Session.request

        def guarded(self, method, url, *args, **kwargs):
            assert getattr(inside, "open", False), f"{method} {url} outside metered_call"
            return real_request(self, method, url, *args, **kwargs)

        monkeypatch.setattr(wrapper_module, "metered_call", tracking)
        monkeypatch.setattr(requests.Session, "request", guarded)
        fake.reply("/search", search_ok())
        fake.reply("/extract", (200, {"results": [{"url": "https://example.com", "raw_content": "x"}],
                                      "usage": {"credits": 1}}))
        fake.reply("/crawl", (200, {"results": [], "usage": {"credits": 1}}))
        fake.reply("/map", (200, {"results": [], "usage": {"credits": 1}}))
        t = tools(Platform(), per_run_call_cap=10)
        t["web_search"].invoke({"query": "q"})
        t["web_extract"].invoke({"urls": ["https://example.com"]})
        t["web_crawl"].invoke({"url": "https://example.com"})
        t["web_map"].invoke({"url": "https://example.com"})
        assert len(fake.requests) == 4

    def test_one_check_and_one_record_with_reported_credits(self, fake):
        fake.reply("/search", search_ok(credits=1))
        platform = Platform()
        tools(platform)["web_search"].invoke({"query": "q"})
        assert [c["op"] for c in platform.checks] == ["search"]
        assert platform.checks[0]["configuration_uuid"] == "u-1"
        assert platform.checks[0]["configuration_project_id"] == 1
        record = platform.records[0]
        assert record["units"] == 1 and record["ok"] is True
        assert record["meta"] == {"depth": "basic", "results_count": 1, "credits_source": "reported"}

    def test_missing_usage_charges_estimate_and_logs_loudly(self, fake, caplog):
        fake.reply("/search", search_ok(credits=None))
        platform = Platform()
        with caplog.at_level(logging.ERROR):
            tools(platform, search_depth_allowed="basic_advanced")["web_search"].invoke(
                {"query": "q", "search_depth": "advanced"})
        assert platform.records[0]["units"] == 2
        assert platform.records[0]["meta"]["credits_source"] == "estimated"
        assert "TAVILY USAGE MISSING" in caplog.text

    def test_zero_reported_on_paid_call_is_not_trusted(self, fake):
        fake.reply("/search", search_ok(credits=0))
        platform = Platform()
        tools(platform)["web_search"].invoke({"query": "q"})
        assert platform.records[0]["units"] == 1

    def test_crawl_estimate_uses_requested_limit(self, fake):
        fake.reply("/crawl", (200, {"results": []}))
        platform = Platform()
        tools(platform)["web_crawl"].invoke({"url": "https://example.com", "limit": 20})
        # map ceil(20/10)=2 + extract ceil(20/5)=4
        assert platform.records[0]["units"] == 6

    def test_retry_is_its_own_metered_block(self, fake):
        fake.reply("/search", (502, {}), search_ok())
        platform = Platform()
        tools(platform)["web_search"].invoke({"query": "q"})
        assert len(platform.checks) == 2
        assert [r["ok"] for r in platform.records] == [False, True]
        assert platform.records[0]["permit"] != platform.records[1]["permit"]

    def test_at_most_one_retry(self, fake):
        fake.reply("/search", (503, {}))
        platform = Platform()
        with pytest.raises(ToolException, match="temporarily unavailable"):
            tools(platform)["web_search"].api_wrapper.web_search("q")
        assert len(fake.requests) == 2

    def test_denial_propagates_without_calling_tavily(self, fake):
        platform = Platform(allowed=False)
        with pytest.raises(ToolSpendDenied) as denied:
            tools(platform)["web_search"].api_wrapper.web_search("q")
        assert "budget_exceeded" not in str(denied.value)
        assert fake.requests == []

    def test_pinned_client_has_no_retry_adapter(self):
        wrapper = tools(Platform())["web_search"].api_wrapper
        session = wrapper._tavily().session  # pylint: disable=W0212
        for adapter in session.adapters.values():
            assert adapter.max_retries.total == 0

    def test_sends_project_attribution_header(self, fake):
        fake.reply("/search", search_ok())
        tools(Platform())["web_search"].invoke({"query": "q"})
        assert fake.requests[0]["headers"]["X-Project-ID"] == "3"


# ---- bounds, policy, output ------------------------------------------------------------------

class TestBounds:
    def test_search_fixed_params_and_depth_coerced(self, fake):
        fake.reply("/search", search_ok())
        tools(Platform(), max_results=50)["web_search"].invoke({"query": "q", "search_depth": "advanced"})
        body = fake.requests[0]["body"]
        assert body["search_depth"] == "basic" and body["max_results"] == 10
        assert body["include_answer"] is False and body["include_raw_content"] is False
        assert body["auto_parameters"] is False and body["include_usage"] is True

    def test_extract_rejects_before_any_call(self, fake):
        api = tools(Platform(), blocked_domains=["evil.com"])["web_extract"].api_wrapper
        for urls, match in ((["https://a.com"] * 6, "At most 5"), (["ftp://a.com"], "http/https"),
                            (["https://x.evil.com/p"], "blocked")):
            with pytest.raises(ToolException, match=match):
                api.web_extract(urls)
        assert fake.requests == []

    def test_allowed_domains_only(self, fake):
        api = tools(Platform(), allowed_domains=["docs.python.org"])["web_extract"].api_wrapper
        with pytest.raises(ToolException, match="allowed domains"):
            api.web_extract(["https://example.com"])

    def test_crawl_and_map_clamped(self, fake):
        fake.reply("/crawl", (200, {"results": [], "usage": {"credits": 1}}))
        fake.reply("/map", (200, {"results": [], "usage": {"credits": 1}}))
        api = tools(Platform())["web_crawl"].api_wrapper
        api.web_crawl("https://example.com", max_depth=9, limit=999)
        api.web_map("https://example.com", max_depth=9, limit=999)
        crawl, site_map = fake.requests[0]["body"], fake.requests[1]["body"]
        assert (crawl["max_depth"], crawl["limit"], crawl["allow_external"], crawl["timeout"]) == (2, 20, False, 60)
        assert (site_map["max_depth"], site_map["limit"], site_map["allow_external"]) == (2, 50, False)

    def test_page_truncated_with_standard_marker_and_wrapped(self, fake):
        fake.reply("/extract", (200, {"results": [{"url": "https://example.com", "raw_content": "x" * 1500}],
                                      "usage": {"credits": 1}}))
        out = tools(Platform(), max_page_chars=1000)["web_extract"].invoke({"urls": ["https://example.com"]})
        assert TRUNCATED_NOTE.format(500) in out
        assert out.startswith('<external_web_content tool="web_extract">') and "untrusted" in out

    def test_snippet_capped(self, fake):
        fake.reply("/search", search_ok([{"title": "t", "url": "https://e.com", "content": "y" * 900}]))
        out = tools(Platform())["web_search"].invoke({"query": "q"})
        assert "y" * 500 + TRUNCATED_NOTE.format(400) in out


class TestRunCap:
    def test_blocks_call_after_cap(self, fake):
        fake.reply("/search", search_ok())
        api = tools(Platform(), per_run_call_cap=2)["web_search"].api_wrapper
        api.web_search("a")
        api.web_search("b")
        with pytest.raises(ToolException, match="call limit for this run"):
            api.web_search("c")
        assert len(fake.requests) == 2

    def test_cap_survives_a_rebuilt_toolkit_in_same_run(self, fake):
        """A resume rebuilds the toolkit; the counter is keyed by run id, not by instance."""
        fake.reply("/search", search_ok())
        tools(Platform(), per_run_call_cap=1)["web_search"].api_wrapper.web_search("a")
        with pytest.raises(ToolException, match="call limit"):
            tools(Platform(), per_run_call_cap=1)["web_search"].api_wrapper.web_search("b")
        tools(Platform(run_id="run-2"), per_run_call_cap=1)["web_search"].api_wrapper.web_search("c")

    def test_parallel_calls_respect_cap(self, fake):
        fake.reply("/search", search_ok())
        api = tools(Platform(), per_run_call_cap=3)["web_search"].api_wrapper
        outcomes = []

        def call():
            try:
                api.web_search("q")
                outcomes.append("ok")
            except ToolException:
                outcomes.append("capped")

        threads = [threading.Thread(target=call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert outcomes.count("ok") == 3 and outcomes.count("capped") == 7


class TestErrorsAndSecrets:
    @pytest.mark.parametrize("status,match", [
        (401, "rejected the API key"), (432, "plan or credit limit"), (433, "plan or credit limit"),
        (429, "rate limit"), (400, "rejected the request"),
    ])
    def test_clear_messages_without_key(self, fake, caplog, status, match):
        fake.reply("/search", (status, {"detail": {"error": f"oops {KEY}"}}))
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ToolException, match=match) as error:
                tools(Platform())["web_search"].api_wrapper.web_search("q")
        assert KEY not in str(error.value) and KEY not in caplog.text

    def test_no_results(self, fake):
        fake.reply("/search", search_ok(results=[]))
        assert "No results found" in tools(Platform())["web_search"].invoke({"query": "q"})

    def test_key_never_in_output(self, fake):
        fake.reply("/search", search_ok())
        out = tools(Platform())["web_search"].invoke({"query": "q"})
        assert KEY not in out
