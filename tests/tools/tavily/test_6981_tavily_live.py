"""#6981 live tests against the real Tavily API (~15 credits per run).

Skipped unless TAVILY_LIVE_API_KEY is set. Harmless queries only. The platform is a stub, so
these prove the real API shape: reported usage credits, extract/crawl/map payloads, error
mapping, and that what we record never undercounts what Tavily's GET /usage shows.
"""
import os
import re
import threading
import time
from urllib.parse import urlparse

import pytest
import requests
from langchain_core.tools import ToolException

from elitea_sdk.configurations.tavily import TavilyConfiguration, tavily_base_url
from elitea_sdk.tools import tavily as tavily_toolkit
from elitea_sdk.tools.tavily import api_wrapper as wrapper_module

KEY = os.getenv("TAVILY_LIVE_API_KEY")
pytestmark = pytest.mark.skipif(not KEY, reason="TAVILY_LIVE_API_KEY not set")


class Platform:
    def __init__(self):
        self.project_id = 3
        self.headers = {"X-Elitea-Run-Id": f"live-{time.time()}"}
        self.records = []
        self.lock = threading.Lock()

    def check_tool_spend(self, payload, timeout=None):
        return {"allowed": True, "exempt": True, "permit": "live"}

    def record_tool_spend(self, payload, timeout=None):
        with self.lock:
            self.records.append(payload)
        return {"recorded": True}


def hosts(text):
    """Hostnames of every URL in a tool output (exact match, not substring)."""
    return {(urlparse(u).hostname or "").lower() for u in re.findall(r"https?://[^\s|)\]]+", text)}


def key_usage(key=KEY):
    response = requests.get(f"{tavily_base_url()}/usage", headers={"Authorization": f"Bearer {key}"}, timeout=15)
    response.raise_for_status()
    return float((response.json().get("key") or {}).get("usage") or 0)


def toolkit(platform, key=KEY, **settings):
    tool = {"type": "tavily", "toolkit_name": "live", "settings": {
        "tavily_configuration": {"api_key": key, "configuration_uuid": "live", "configuration_project_id": 3},
        "elitea": platform, "per_run_call_cap": 50, **settings}}
    return {t.name: t for t in tavily_toolkit.get_tools(tool)}


@pytest.fixture(scope="module")
def ledger():
    """Shared stub platform; at the end compares our recorded units with Tavily's own counter."""
    platform = Platform()
    before = key_usage()
    yield platform
    time.sleep(5)  # Tavily's usage counter is updated asynchronously
    spent = key_usage() - before
    recorded = sum(r["units"] for r in platform.records)
    print(f"\nTavily usage delta={spent} recorded={recorded} rows={len(platform.records)}")
    if spent == 0:
        print("Tavily /usage counter not updated yet (it lags); undercount check skipped")
        return
    assert recorded >= spent, f"undercount: recorded {recorded} < Tavily {spent}"


def test_check_connection_is_free():
    before = key_usage()
    assert TavilyConfiguration.check_connection({"api_key": KEY}) is None
    time.sleep(2)
    assert key_usage() == before


def test_check_connection_bad_key():
    assert TavilyConfiguration.check_connection({"api_key": "tvly-invalid-000"}) == "Invalid Tavily API key"


def test_search_basic(ledger):
    out = toolkit(ledger)["web_search"].invoke({"query": "Python programming language official website"})
    assert hosts(out) & {"python.org", "www.python.org"} and "untrusted" in out
    last = ledger.records[-1]
    assert last["ok"] and last["meta"]["credits_source"] == "reported" and last["units"] == 1


def test_search_advanced(ledger):
    toolkit(ledger, search_depth_allowed="basic_advanced")["web_search"].invoke(
        {"query": "Wikipedia main page", "search_depth": "advanced"})
    last = ledger.records[-1]
    assert last["meta"]["depth"] == "advanced" and last["units"] == 2


def test_extract_one_url(ledger):
    out = toolkit(ledger)["web_extract"].invoke({"urls": ["https://example.com"]})
    assert "documentation examples" in out
    assert ledger.records[-1]["units"] >= 1


def test_extract_five_urls_with_a_bad_one(ledger):
    urls = ["https://example.com", "https://www.python.org", "https://www.wikipedia.org",
            "https://www.iana.org", "https://nonexistent-domain-elitea-test.invalid"]
    out = toolkit(ledger, max_page_chars=2000)["web_extract"].invoke({"urls": urls})
    assert "Could not fetch" in out and "...[truncated:" in out
    last = ledger.records[-1]
    assert last["meta"]["urls_count"] == 5 and last["meta"]["failed_urls"] >= 1


def test_map_small(ledger):
    out = toolkit(ledger)["web_map"].invoke({"url": "https://docs.python.org/3/", "limit": 5})
    assert "docs.python.org" in hosts(out)
    assert ledger.records[-1]["op"] == "map"


def test_crawl_small(ledger):
    out = toolkit(ledger, max_page_chars=3000)["web_crawl"].invoke({"url": "https://example.com", "limit": 2})
    assert "external_web_content" in out
    assert ledger.records[-1]["op"] == "crawl"


def test_bad_key_message():
    with pytest.raises(ToolException, match="rejected the API key") as error:
        toolkit(Platform(), key="tvly-invalid-000")["web_search"].api_wrapper.web_search("hello")
    assert "tvly-invalid-000" not in str(error.value)


def test_forced_timeout(ledger, monkeypatch):
    monkeypatch.setattr(wrapper_module, "SEARCH_TIMEOUT", 0.001)
    monkeypatch.setattr(wrapper_module, "_sleep", lambda _s: None)
    before = len(ledger.records)
    with pytest.raises(ToolException, match="did not respond in time"):
        toolkit(ledger)["web_search"].api_wrapper.web_search("weather")
    attempts = ledger.records[before:]
    assert len(attempts) == 2 and not any(r["ok"] for r in attempts)
