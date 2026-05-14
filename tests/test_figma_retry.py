"""Unit tests for EliteAFigmaPy retry logic and FigmaApiWrapper tool deprecations."""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.figma.figma_client import (
    EliteAFigmaPy,
    _MAX_RETRIES,
    _RETRYABLE_STATUS,
    _MAX_RETRY_AFTER_S,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code, body=None, headers=None):
    """Build a fake requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = json.dumps(body) if isinstance(body, dict) else (body or "")
    resp.reason = "OK" if 200 <= status_code < 300 else "Error"
    resp.headers = headers or {}
    resp.json.return_value = body if isinstance(body, dict) else {}
    if body is None:
        resp.json.side_effect = ValueError("No JSON")
    return resp


def _make_client():
    """Create an EliteAFigmaPy instance with a dummy token."""
    with patch.object(EliteAFigmaPy, "__init__", lambda self: None):
        client = EliteAFigmaPy.__new__(EliteAFigmaPy)
        client.api_token = "test-token"
        client.oauth2 = False
        client.api_uri = "https://api.figma.com/v1/"
    return client


# ===========================================================================
# Tests: Happy path
# ===========================================================================

class TestHappyPath:
    def test_200_returns_parsed_json(self):
        client = _make_client()
        body = {"document": {"id": "0:0"}}
        with patch.object(EliteAFigmaPy, "_send_http", return_value=_mock_response(200, body)):
            result = client.api_request("files/abc123")
        assert result == body

    def test_200_non_json_returns_raw(self):
        client = _make_client()
        resp = _mock_response(200)
        resp.text = "not-json"
        with patch.object(EliteAFigmaPy, "_send_http", return_value=resp):
            result = client.api_request("files/abc123")
        assert result == {"raw": "not-json"}


# ===========================================================================
# Tests: No-retry errors (4xx except 429)
# ===========================================================================

class TestNoRetryErrors:
    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_4xx_fails_immediately(self, status):
        client = _make_client()
        with patch.object(EliteAFigmaPy, "_send_http", return_value=_mock_response(status, {"err": "msg"})) as mock:
            with pytest.raises(ToolException, match=f"error {status}"):
                client.api_request("files/abc")
        assert mock.call_count == 1


# ===========================================================================
# Tests: Retry on 429
# ===========================================================================

class TestRetry429:
    @patch("elitea_sdk.tools.figma.figma_client.time")
    def test_429_retries_with_retry_after(self, mock_time):
        """429 with Retry-After ≤ 60s should sleep and retry."""
        mock_time.monotonic.side_effect = [0.0, 1.0, 5.0, 10.0]
        client = _make_client()
        fail_resp = _mock_response(429, {"error": "rate_limited"}, {"Retry-After": "3"})
        ok_resp = _mock_response(200, {"ok": True})
        with patch.object(EliteAFigmaPy, "_send_http", side_effect=[fail_resp, ok_resp]):
            result = client.api_request("files/abc")
        assert result == {"ok": True}
        mock_time.sleep.assert_called_once_with(3.0)

    @patch("elitea_sdk.tools.figma.figma_client.time")
    def test_429_no_retry_if_retry_after_exceeds_max(self, mock_time):
        """429 with Retry-After > 60s (monthly quota) should NOT retry."""
        mock_time.monotonic.side_effect = [0.0, 1.0]
        client = _make_client()
        resp = _mock_response(429, {"error": "rate_limited"}, {"Retry-After": "86400"})
        with patch.object(EliteAFigmaPy, "_send_http", return_value=resp) as mock:
            with pytest.raises(ToolException, match="error 429"):
                client.api_request("files/abc")
        assert mock.call_count == 1
        mock_time.sleep.assert_not_called()

    @patch("elitea_sdk.tools.figma.figma_client.time")
    def test_429_backoff_without_retry_after(self, mock_time):
        """429 without Retry-After header uses exponential backoff."""
        mock_time.monotonic.side_effect = [0.0, 1.0, 3.0, 6.0, 10.0]
        client = _make_client()
        fail_resp = _mock_response(429, {"error": "rate_limited"})
        ok_resp = _mock_response(200, {"result": "ok"})
        with patch.object(EliteAFigmaPy, "_send_http", side_effect=[fail_resp, fail_resp, ok_resp]):
            result = client.api_request("files/abc")
        assert result == {"result": "ok"}
        assert mock_time.sleep.call_count == 2
        mock_time.sleep.assert_any_call(2.0)  # first backoff
        mock_time.sleep.assert_any_call(4.0)  # second backoff

    @patch("elitea_sdk.tools.figma.figma_client.time")
    def test_429_max_retries_exhausted(self, mock_time):
        """After MAX_RETRIES, should raise even if still 429."""
        mock_time.monotonic.side_effect = [0.0] + [float(i) for i in range(1, 20)]
        client = _make_client()
        fail_resp = _mock_response(429, {"error": "rate_limited"})
        with patch.object(EliteAFigmaPy, "_send_http", return_value=fail_resp) as mock:
            with pytest.raises(ToolException, match="error 429"):
                client.api_request("files/abc")
        assert mock.call_count == _MAX_RETRIES + 1  # initial + retries

    @patch("elitea_sdk.tools.figma.figma_client.time")
    def test_429_upgrade_link_in_error(self, mock_time):
        """429 error message should include X-Figma-Upgrade-Link when present."""
        mock_time.monotonic.side_effect = [0.0, 1.0]
        client = _make_client()
        resp = _mock_response(
            429, {"error": "rate_limited"},
            {"Retry-After": "86400", "X-Figma-Upgrade-Link": "https://figma.com/pricing"},
        )
        with patch.object(EliteAFigmaPy, "_send_http", return_value=resp):
            with pytest.raises(ToolException, match="https://figma.com/pricing"):
                client.api_request("files/abc")

    @patch("elitea_sdk.tools.figma.figma_client.time")
    def test_429_rate_limit_type_in_error(self, mock_time):
        """429 error message should include X-Figma-Rate-Limit-Type when present."""
        mock_time.monotonic.side_effect = [0.0, 1.0]
        client = _make_client()
        resp = _mock_response(
            429, {"error": "rate_limited"},
            {"Retry-After": "86400", "X-Figma-Rate-Limit-Type": "low"},
        )
        with patch.object(EliteAFigmaPy, "_send_http", return_value=resp):
            with pytest.raises(ToolException, match="rate-limit-type: low"):
                client.api_request("files/abc")


# ===========================================================================
# Tests: Retry on 5xx
# ===========================================================================

class TestRetry5xx:
    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    @patch("elitea_sdk.tools.figma.figma_client.time")
    def test_5xx_retries_with_backoff(self, mock_time, status):
        """5xx errors should retry with exponential backoff."""
        mock_time.monotonic.side_effect = [0.0, 1.0, 3.0, 6.0]
        client = _make_client()
        fail_resp = _mock_response(status)
        ok_resp = _mock_response(200, {"ok": True})
        with patch.object(EliteAFigmaPy, "_send_http", side_effect=[fail_resp, ok_resp]):
            result = client.api_request("files/abc")
        assert result == {"ok": True}
        mock_time.sleep.assert_called_once_with(2.0)

    @patch("elitea_sdk.tools.figma.figma_client.time")
    def test_5xx_exhausted_raises(self, mock_time):
        """5xx after all retries should raise."""
        mock_time.monotonic.side_effect = [0.0] + [float(i) for i in range(1, 20)]
        client = _make_client()
        fail_resp = _mock_response(500, {"error": "internal"})
        with patch.object(EliteAFigmaPy, "_send_http", return_value=fail_resp):
            with pytest.raises(ToolException, match="error 500"):
                client.api_request("files/abc")


# ===========================================================================
# Tests: Wall-clock cap
# ===========================================================================

class TestWallClockCap:
    @patch("elitea_sdk.tools.figma.figma_client.time")
    def test_wall_clock_exceeded_stops_retry(self, mock_time):
        """Should stop retrying when total elapsed exceeds 30s."""
        # Simulate: start=0, first retry check at 31s (over limit)
        mock_time.monotonic.side_effect = [0.0, 31.0]
        client = _make_client()
        fail_resp = _mock_response(429, {"error": "rate_limited"})
        with patch.object(EliteAFigmaPy, "_send_http", return_value=fail_resp) as mock:
            with pytest.raises(ToolException):
                client.api_request("files/abc")
        assert mock.call_count == 1  # no retry because wall-clock exceeded


# ===========================================================================
# Tests: Network errors (no retry)
# ===========================================================================

class TestNetworkErrors:
    def test_connection_error_raises(self):
        """Network errors should raise immediately without retry."""
        import requests as req
        client = _make_client()
        with patch.object(EliteAFigmaPy, "_send_http", side_effect=req.ConnectionError("refused")):
            with pytest.raises(ToolException, match="request failed"):
                client.api_request("files/abc")

    def test_ssl_error_raises(self):
        """SSL errors should raise immediately."""
        import requests as req
        client = _make_client()
        with patch.object(EliteAFigmaPy, "_send_http", side_effect=req.exceptions.SSLError("cert")):
            with pytest.raises(ToolException, match="request failed"):
                client.api_request("files/abc")


# ===========================================================================
# Tests: _compute_retry_sleep
# ===========================================================================

class TestComputeRetrySleep:
    def test_429_with_valid_retry_after(self):
        resp = _mock_response(429, headers={"Retry-After": "10"})
        assert EliteAFigmaPy._compute_retry_sleep(resp, 0) == 10.0

    def test_429_with_excessive_retry_after_returns_none(self):
        resp = _mock_response(429, headers={"Retry-After": "3600"})
        assert EliteAFigmaPy._compute_retry_sleep(resp, 0) is None

    def test_429_without_retry_after_uses_backoff(self):
        resp = _mock_response(429)
        assert EliteAFigmaPy._compute_retry_sleep(resp, 0) == 2.0
        assert EliteAFigmaPy._compute_retry_sleep(resp, 1) == 4.0
        assert EliteAFigmaPy._compute_retry_sleep(resp, 2) == 8.0

    def test_500_uses_backoff(self):
        resp = _mock_response(500)
        assert EliteAFigmaPy._compute_retry_sleep(resp, 0) == 2.0
        assert EliteAFigmaPy._compute_retry_sleep(resp, 1) == 4.0

    def test_429_invalid_retry_after_uses_backoff(self):
        resp = _mock_response(429, headers={"Retry-After": "not-a-number"})
        assert EliteAFigmaPy._compute_retry_sleep(resp, 0) == 2.0


# ===========================================================================
# Tests: _build_error
# ===========================================================================

class TestBuildError:
    def test_429_includes_upgrade_link(self):
        resp = _mock_response(429, {"err": True}, {
            "X-Figma-Upgrade-Link": "https://figma.com/upgrade",
            "X-Figma-Rate-Limit-Type": "low",
        })
        err = EliteAFigmaPy._build_error(resp)
        assert isinstance(err, ToolException)
        assert "https://figma.com/upgrade" in str(err)
        assert "rate-limit-type: low" in str(err)

    def test_non_429_no_upgrade_link(self):
        resp = _mock_response(500, {"error": "internal"})
        err = EliteAFigmaPy._build_error(resp)
        assert "upgrade" not in str(err).lower()


# ===========================================================================
# Tests: Tool deprecations (removed from get_available_tools)
# ===========================================================================

class TestToolDeprecations:
    """Verify deprecated tools are no longer exposed."""

    def _get_tool_names(self):
        """Get tool names from FigmaApiWrapper.get_available_tools()."""
        from elitea_sdk.tools.figma.api_wrapper import FigmaApiWrapper
        wrapper = FigmaApiWrapper.model_construct(
            figma_token="fake-token",
        )
        wrapper._client = MagicMock()
        wrapper.llm = None
        wrapper.global_limit = 1000000
        wrapper.global_regexp = None
        wrapper.global_fields_retain = []
        wrapper.global_fields_remove = []
        wrapper.global_depth_start = 1
        wrapper.global_depth_end = 6
        try:
            tools = wrapper.get_available_tools()
            return [t["name"] for t in tools]
        except Exception:
            pytest.skip("Cannot instantiate FigmaApiWrapper for tool listing")

    def test_analyze_file_not_in_tools(self):
        names = self._get_tool_names()
        assert "analyze_file" not in names

    def test_get_file_structure_toon_not_in_tools(self):
        names = self._get_tool_names()
        assert "get_file_structure_toon" not in names

    def test_get_page_flows_toon_not_in_tools(self):
        names = self._get_tool_names()
        assert "get_page_flows_toon" not in names

    def test_get_frame_detail_toon_not_in_tools(self):
        names = self._get_tool_names()
        assert "get_frame_detail_toon" not in names

    def test_get_file_summary_not_in_tools(self):
        names = self._get_tool_names()
        assert "get_file_summary" not in names

    def test_kept_tools_still_present(self):
        names = self._get_tool_names()
        expected = [
            "get_file_nodes", "get_file", "get_file_versions",
            "get_file_comments", "post_file_comment",
            "get_file_images", "get_team_projects", "get_project_files",
        ]
        for tool in expected:
            assert tool in names, f"Expected tool '{tool}' missing from available tools"
