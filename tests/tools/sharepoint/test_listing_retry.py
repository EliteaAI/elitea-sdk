"""A throttled SharePoint listing is retried, and a page that cannot be read is reported as a gap.

Without the retry a single 429 in the middle of a listing ends the walk, and a
change check would read the missing files as removed.
"""

import io
import json
from unittest.mock import patch

import pytest
import requests
from langchain_core.tools import ToolException

from elitea_sdk.tools.sharepoint import library_metadata
from elitea_sdk.tools.sharepoint.authorization_helper import (
    SharepointAuthorizationHelper,
)
from elitea_sdk.tools.sharepoint.graph_wrapper import SharepointGraphWrapper

GRAPH_MODULE = "elitea_sdk.tools.sharepoint.graph_wrapper"
AUTH_MODULE = "elitea_sdk.tools.sharepoint.authorization_helper"
SITE = "https://tenant.sharepoint.com/sites/x"
DRIVE = {"id": "drive-1", "webUrl": f"{SITE}/Shared%20Documents"}
FILE = {"id": "f1", "name": "a.txt", "file": {}, "webUrl": "https://x/a.txt",
        "parentReference": {"path": "/drives/drive-1/root:"}}


def _response(status: int, body: dict | None = None, headers: dict | None = None) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(body if body is not None else {}).encode()
    resp.raw = io.BytesIO()  # a real response has a raw stream that close() releases
    resp.url = "https://graph.microsoft.com/v1.0/drives/drive-1/root/children"
    resp.headers.update(headers or {})
    return resp


@pytest.fixture
def sleeps():
    waited = []
    with patch.object(library_metadata.time, "sleep", side_effect=waited.append):
        yield waited


def _graph() -> SharepointGraphWrapper:
    wrapper = SharepointGraphWrapper(site_url=SITE, token="t", scopes=["Sites.Read.All"])
    wrapper._list_drives = lambda: [DRIVE]
    return wrapper


class TestGraphListing:

    @pytest.mark.parametrize("status", [429, 503, 504])
    def test_a_throttled_listing_page_is_retried(self, status, sleeps):
        pages = [_response(status), _response(200, {"value": [FILE]})]
        with patch(f"{GRAPH_MODULE}.requests.get", side_effect=pages) as get:
            result = _graph().get_files_list()

        assert [item["Name"] for item in result] == ["a.txt"]
        assert get.call_count == 2
        assert len(sleeps) == 1

    def test_retry_after_sets_the_wait(self, sleeps):
        pages = [_response(429, headers={"Retry-After": "7"}), _response(200, {"value": []})]
        with patch(f"{GRAPH_MODULE}.requests.get", side_effect=pages):
            _graph().get_files_list()

        assert sleeps == [7.0]

    def test_a_listing_that_stays_throttled_raises_after_the_attempts(self, sleeps):
        with (
            patch(f"{GRAPH_MODULE}.requests.get", return_value=_response(429)) as get,
            pytest.raises(ToolException),
        ):
            _graph().get_files_list()

        assert get.call_count == library_metadata.MAX_RETRIES + 1

    def test_other_reads_do_not_retry(self, sleeps):
        with (
            patch(f"{GRAPH_MODULE}.requests.get", return_value=_response(429)) as get,
            pytest.raises(requests.HTTPError),
        ):
            _graph()._get("https://graph.microsoft.com/v1.0/sites/x")

        assert get.call_count == 1
        assert sleeps == []

    def test_a_dropped_connection_in_a_listing_is_retried(self, sleeps):
        calls = [requests.ConnectionError("reset"), _response(200, {"value": [FILE]})]
        with patch(f"{GRAPH_MODULE}.requests.get", side_effect=calls):
            result = _graph().get_files_list()

        assert len(result) == 1

    def test_the_listing_reports_a_gap_when_the_limit_cuts_it_short(self):
        gaps = []
        with patch(f"{GRAPH_MODULE}.requests.get", return_value=_response(200, {"value": [FILE]})):
            _graph().get_files_list(limit_files=1, on_listing_gap=gaps.append)

        assert len(gaps) == 1

    def test_a_complete_listing_reports_no_gap(self):
        gaps = []
        with patch(f"{GRAPH_MODULE}.requests.get", return_value=_response(200, {"value": [FILE]})):
            _graph().get_files_list(limit_files=10, on_listing_gap=gaps.append)

        assert gaps == []

    def test_a_form_name_that_matches_no_library_reports_a_gap(self):
        gaps = []
        with patch(f"{GRAPH_MODULE}.requests.get", return_value=_response(200, {"value": [FILE]})) as get:
            result = _graph().get_files_list(form_name="Renamed Library", on_listing_gap=gaps.append)

        assert result == []
        assert get.call_count == 0
        assert len(gaps) == 1

    def test_a_form_name_that_matches_a_library_reports_no_gap(self):
        gaps = []
        with patch(f"{GRAPH_MODULE}.requests.get", return_value=_response(200, {"value": [FILE]})):
            result = _graph().get_files_list(form_name="shared documents", on_listing_gap=gaps.append)

        assert len(result) == 1
        assert gaps == []


class TestAuthorizationHelperPaging:

    def _helper(self):
        helper = SharepointAuthorizationHelper("t", "c", "s", None, None)
        helper.generate_token_and_site_id = lambda site_url: ("token", "site-1")
        return helper

    def _drives(self):
        return _response(200, {"value": [{"id": "drive-1", "webUrl": f"{SITE}/Shared%20Documents"}]})

    def _item(self):
        return {"id": "f1", "name": "a.txt", "webUrl": "https://x/a.txt"}

    def test_a_throttled_page_is_retried_and_no_gap_is_reported(self, sleeps):
        pages = [self._drives(), _response(429), _response(200, {"value": [self._item()]})]
        gaps = []
        with patch(f"{AUTH_MODULE}.requests.get", side_effect=pages):
            result = self._helper().get_files_list(SITE, on_listing_gap=gaps.append)

        assert [item["Name"] for item in result] == ["a.txt"]
        assert gaps == []

    def test_a_page_that_stays_unreadable_reports_a_gap(self, sleeps):
        pages = [self._drives()] + [_response(500)]
        gaps = []
        with patch(f"{AUTH_MODULE}.requests.get", side_effect=pages):
            result = self._helper().get_files_list(SITE, on_listing_gap=gaps.append)

        assert result == []
        assert len(gaps) == 1

    def test_a_page_without_items_reports_a_gap(self):
        pages = [self._drives(), _response(200, {"error": "x"})]
        gaps = []
        with patch(f"{AUTH_MODULE}.requests.get", side_effect=pages):
            self._helper().get_files_list(SITE, on_listing_gap=gaps.append)

        assert len(gaps) == 1

    def test_the_limit_reports_a_gap(self):
        pages = [self._drives(), _response(200, {"value": [self._item()]})]
        gaps = []
        with patch(f"{AUTH_MODULE}.requests.get", side_effect=pages):
            self._helper().get_files_list(SITE, limit_files=1, on_listing_gap=gaps.append)

        assert len(gaps) == 1

    def test_the_drives_and_page_requests_carry_a_timeout(self):
        pages = [self._drives(), _response(200, {"value": [self._item()]})]
        with patch(f"{AUTH_MODULE}.requests.get", side_effect=pages) as get:
            self._helper().get_files_list(SITE)

        assert get.call_count == 2
        assert all(call.kwargs.get("timeout") == 60 for call in get.call_args_list)
