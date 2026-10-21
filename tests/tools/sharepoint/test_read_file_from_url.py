"""Tests for reading a SharePoint file straight from its browser or sharing URL (#6882)."""
import base64
from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.tools import ToolException

from elitea_sdk.runtime.langchain.document_loaders.EliteAExcelLoader import (
    ExcelReadEstimate,
    ExcelReadLimitExceeded,
)
from elitea_sdk.tools.sharepoint import SharepointToolkit
from elitea_sdk.tools.sharepoint.api_wrapper import ReadFromUrl, SharepointApiWrapper
from elitea_sdk.tools.sharepoint.browser_url import (
    NOT_A_FILE_URL_MESSAGE,
    SharepointFileUrl,
    parse_sharepoint_file_url,
)
from elitea_sdk.tools.sharepoint.graph_wrapper import SharepointGraphWrapper
from elitea_sdk.tools.sharepoint.rest_wrapper import SharepointRestWrapper

GRAPH = "https://graph.microsoft.com/v1.0"
HOST = "contoso.sharepoint.com"
GUID = "1a2b3c4d-0000-1111-2222-333344445555"
SITE_ID = "contoso.sharepoint.com,site-guid,web-guid"


def _share_id(url: str) -> str:
    encoded = base64.b64encode(url.encode("utf-8")).decode("utf-8")
    return "u!" + encoded.rstrip("=").replace("/", "_").replace("+", "-")


def _decode_share_id(share_id: str) -> str:
    payload = share_id[2:].replace("_", "/").replace("-", "+")
    return base64.b64decode(payload + "=" * (-len(payload) % 4)).decode("utf-8")


def _response(status: int, body=None):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.json.return_value = body or {}
    resp.url = "graph"
    return resp


def _download_response(content: bytes):
    resp = MagicMock()
    resp.iter_content = lambda chunk_size=8192: iter([content])
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _file_item(name="report.txt", size=12):
    return {
        "name": name,
        "size": size,
        "file": {"mimeType": "text/plain"},
        "@microsoft.graph.downloadUrl": "https://download.example/report",
    }


def _wrapper():
    return SharepointGraphWrapper(
        site_url=f"https://{HOST}/sites/Team", token="token", scopes=["Files.Read"])


class _Graph:
    """Answers Graph GETs from a {url: response} table and records every call."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, headers=None, params=None, **kwargs):
        self.calls.append(url)
        return self.routes.get(url, _response(404))


@pytest.fixture
def graph():
    fake = _Graph({})
    with patch("elitea_sdk.tools.sharepoint.graph_wrapper.requests") as requests_mock, \
            patch("elitea_sdk.tools.utils.http_utils.requests") as download_mock, \
            patch("elitea_sdk.tools.sharepoint.graph_wrapper.parse_file_content") as parse_mock:
        requests_mock.get.side_effect = fake.get
        download_mock.get.return_value = _download_response(b"file content")
        parse_mock.return_value = "parsed content"
        fake.parse = parse_mock
        yield fake


class TestParseSharepointFileUrl:

    @pytest.mark.parametrize("url, expected_path", [
        (f"https://{HOST}/sites/Team/Shared%20Documents/Specs/report.pdf",
         "/sites/Team/Shared Documents/Specs/report.pdf"),
        (f"https://{HOST}/sites/Team/Shared%20Documents/Forms/AllItems.aspx"
         "?id=%2Fsites%2FTeam%2FShared%20Documents%2Freport.pdf&parent=%2Fsites%2FTeam",
         "/sites/Team/Shared Documents/report.pdf"),
        ("https://contoso-my.sharepoint.com/personal/jane_contoso_com/_layouts/15/onedrive.aspx"
         "?id=%2Fpersonal%2Fjane_contoso_com%2FDocuments%2Fnotes.txt&parent=%2Fpersonal",
         "/personal/jane_contoso_com/Documents/notes.txt"),
        (f"https://{HOST}/Shared%20Documents/root-file.docx",
         "/Shared Documents/root-file.docx"),
    ])
    def test_path_urls_resolve_to_the_server_relative_file_path(self, url, expected_path):
        assert parse_sharepoint_file_url(url).server_relative_paths == (expected_path,)

    @pytest.mark.parametrize("url, site_path", [
        (f"https://{HOST}/sites/Team/_layouts/15/Doc.aspx?sourcedoc=%7B{GUID}%7D"
         "&file=Plan.docx&action=default&mobileredirect=true", "/sites/Team"),
        (f"https://{HOST}/:w:/r/sites/Team/Sub/_layouts/15/Doc.aspx?sourcedoc=%7B{GUID}%7D",
         "/sites/Team/Sub"),
        (f"https://{HOST}/_layouts/15/xlviewer.aspx?SourceDoc={GUID}", ""),
    ])
    def test_office_urls_resolve_to_the_document_id_and_site(self, url, site_path):
        assert parse_sharepoint_file_url(url) == SharepointFileUrl(
            host=HOST, site_path=site_path, unique_id=GUID)

    @pytest.mark.parametrize("url", [
        f"https://{HOST}/:x:/s/Team/EbC123ABC?e=xyz",
        "https://contoso-my.sharepoint.com/:w:/p/jane/EaBcD?e=1",
        f"https://{HOST}/:f:/g/Team/EfOlDeR",
    ])
    def test_sharing_links_are_flagged(self, url):
        assert parse_sharepoint_file_url(url).is_sharing_link

    def test_form_encoded_id_reads_plus_as_a_space_first(self):
        target = parse_sharepoint_file_url(
            f"https://{HOST}/sites/Team/Shared+Documents/Forms/AllItems.aspx"
            "?id=/sites/Team/Shared+Documents/a.pdf")

        assert target.server_relative_paths == (
            "/sites/Team/Shared Documents/a.pdf", "/sites/Team/Shared+Documents/a.pdf")

    def test_percent_encoded_plus_stays_a_plus(self):
        target = parse_sharepoint_file_url(
            f"https://{HOST}/sites/Team/Shared%20Documents/Forms/AllItems.aspx"
            "?id=%2Fsites%2FTeam%2FShared%20Documents%2FC%2B%2B%20notes.pdf")

        assert target.server_relative_paths == ("/sites/Team/Shared Documents/C++ notes.pdf",)

    def test_existing_access_link_is_parsed_as_a_browser_url(self):
        target = parse_sharepoint_file_url(
            f"https://{HOST}/:b:/r/sites/Team/Shared%20Documents/a.pdf?csf=1&web=1")
        assert not target.is_sharing_link
        assert target.server_relative_paths == ("/sites/Team/Shared Documents/a.pdf",)

    @pytest.mark.parametrize("url", [
        f"https://{HOST}/sites/Team/Shared%20Documents/Forms/AllItems.aspx",
        f"https://{HOST}/sites/Team/SitePages/Home.aspx",
        f"https://{HOST}/sites/Team/_layouts/15/viewlsts.aspx",
        f"https://{HOST}/",
    ])
    def test_urls_without_a_file_are_rejected(self, url):
        with pytest.raises(ToolException, match="does not identify a single file"):
            parse_sharepoint_file_url(url)

    @pytest.mark.parametrize("url", [None, "", "not a url", f"http://{HOST}/sites/Team/a.pdf"])
    def test_non_https_urls_are_rejected(self, url):
        with pytest.raises(ToolException, match="Must be a full HTTPS URL"):
            parse_sharepoint_file_url(url)

    @pytest.mark.parametrize("url", [
        f"https://{HOST}/sites/Team/_layouts/15/WopiFrame.aspx"
        "?sourcedoc=/sites/Team/Shared%20Documents/a.docx&action=default",
        f"https://{HOST}/sites/Team/_layouts/15/Doc.aspx"
        "?sourcedoc=%2Fsites%2FTeam%2FShared%20Documents%2Fa.docx",
    ])
    def test_sourcedoc_holding_a_path_resolves_to_that_path(self, url):
        assert parse_sharepoint_file_url(url).server_relative_paths == (
            "/sites/Team/Shared Documents/a.docx",)

    def test_malformed_sourcedoc_is_rejected(self):
        with pytest.raises(ToolException, match="not a document id"):
            parse_sharepoint_file_url(f"https://{HOST}/sites/Team/_layouts/15/Doc.aspx?sourcedoc=abc")

    def test_uppercase_id_query_key_is_recognized(self):
        target = parse_sharepoint_file_url(
            f"https://{HOST}/sites/Team/Shared%20Documents/Forms/AllItems.aspx"
            "?ID=%2Fsites%2FTeam%2FShared%20Documents%2Freport.pdf")

        assert target.server_relative_paths == ("/sites/Team/Shared Documents/report.pdf",)


class TestGraphReadFileFromUrl:

    def test_direct_file_url_is_read_with_one_lookup_and_no_listing(self, graph):
        file_url = f"https://{HOST}/sites/Team/Shared%20Documents/Specs/report.txt"
        item_url = f"{GRAPH}/shares/{_share_id(file_url)}/driveItem"
        graph.routes[item_url] = _response(200, _file_item())

        result = _wrapper().read_file_from_url(
            f"https://{HOST}/sites/Team/Shared%20Documents/Specs/report.txt")

        assert result == "parsed content"
        assert graph.calls == [item_url]

    def test_library_viewer_url_resolves_the_file_it_previews(self, graph):
        file_url = f"https://{HOST}/sites/Team/Shared%20Documents/report.txt"
        graph.routes[f"{GRAPH}/shares/{_share_id(file_url)}/driveItem"] = _response(200, _file_item())

        result = _wrapper().read_file_from_url(
            f"https://{HOST}/sites/Team/Shared%20Documents/Forms/AllItems.aspx"
            "?id=%2Fsites%2FTeam%2FShared%20Documents%2Freport.txt")

        assert result == "parsed content"
        assert _decode_share_id(graph.calls[0].split("/shares/")[1].split("/")[0]) == file_url

    def test_literal_plus_reading_is_tried_when_the_space_reading_misses(self, graph):
        literal_url = f"https://{HOST}/sites/Team/Shared%20Documents/C%2B%2B%20notes.txt"
        item_url = f"{GRAPH}/shares/{_share_id(literal_url)}/driveItem"
        graph.routes[item_url] = _response(200, _file_item("C++ notes.txt"))

        result = _wrapper().read_file_from_url(
            f"https://{HOST}/sites/Team/Shared%20Documents/Forms/AllItems.aspx"
            "?id=/sites/Team/Shared%20Documents/C++%20notes.txt")

        assert result == "parsed content"
        assert [_decode_share_id(call.split("/shares/")[1].split("/")[0]) for call in graph.calls] == [
            f"https://{HOST}/sites/Team/Shared%20Documents/C%20%20%20notes.txt", literal_url]

    def test_403_on_space_reading_falls_through_to_literal_reading(self, graph):
        space_url = f"https://{HOST}/sites/Team/Shared%20Documents/C%20%20%20notes.txt"
        literal_url = f"https://{HOST}/sites/Team/Shared%20Documents/C%2B%2B%20notes.txt"
        graph.routes[f"{GRAPH}/shares/{_share_id(space_url)}/driveItem"] = _response(403)
        item_url = f"{GRAPH}/shares/{_share_id(literal_url)}/driveItem"
        graph.routes[item_url] = _response(200, _file_item("C++ notes.txt"))

        result = _wrapper().read_file_from_url(
            f"https://{HOST}/sites/Team/Shared%20Documents/Forms/AllItems.aspx"
            "?id=/sites/Team/Shared%20Documents/C++%20notes.txt")

        assert result == "parsed content"
        assert [_decode_share_id(call.split("/shares/")[1].split("/")[0]) for call in graph.calls] == [
            space_url, literal_url]

    @pytest.mark.parametrize("status", [429, 500])
    def test_space_reading_5xx_raises_without_trying_literal_candidate(self, graph, status):
        space_url = f"https://{HOST}/sites/Team/Shared%20Documents/C%20%20%20notes.txt"
        item_url = f"{GRAPH}/shares/{_share_id(space_url)}/driveItem"
        resp = _response(status)
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status}")
        graph.routes[item_url] = resp

        with pytest.raises(requests.exceptions.HTTPError):
            _wrapper().read_file_from_url(
                f"https://{HOST}/sites/Team/Shared%20Documents/Forms/AllItems.aspx"
                "?id=/sites/Team/Shared%20Documents/C++%20notes.txt")

        assert graph.calls == [item_url]

    def test_id_without_plus_that_misses_yields_exactly_one_graph_call(self, graph):
        with pytest.raises(ToolException, match="File not found"):
            _wrapper().read_file_from_url(
                f"https://{HOST}/sites/Team/Shared%20Documents/Forms/AllItems.aspx"
                "?id=%2Fsites%2FTeam%2FShared%20Documents%2Fmissing.pdf")

        assert len(graph.calls) == 1

    def test_office_url_reads_the_document_from_the_default_library(self, graph):
        graph.routes[f"{GRAPH}/sites/{HOST}:/sites/Team"] = _response(200, {"id": SITE_ID})
        item_url = f"{GRAPH}/sites/{SITE_ID}/drive/items/{GUID}"
        graph.routes[item_url] = _response(200, _file_item("Plan.docx"))

        result = _wrapper().read_file_from_url(
            f"https://{HOST}/sites/Team/_layouts/15/Doc.aspx?sourcedoc=%7B{GUID}%7D&file=Plan.docx")

        assert result == "parsed content"
        assert graph.calls == [f"{GRAPH}/sites/{HOST}:/sites/Team", item_url]

    def test_office_url_finds_a_document_in_a_non_default_library(self, graph):
        graph.routes[f"{GRAPH}/sites/{HOST}:/sites/Team"] = _response(200, {"id": SITE_ID})
        graph.routes[f"{GRAPH}/sites/{SITE_ID}/drives"] = _response(
            200, {"value": [{"id": "default"}, {"id": "archive"}]})
        graph.routes[f"{GRAPH}/drives/archive/items/{GUID}"] = _response(200, _file_item("Old.docx"))

        result = _wrapper().read_file_from_url(
            f"https://{HOST}/sites/Team/_layouts/15/Doc.aspx?sourcedoc={GUID}")

        assert result == "parsed content"
        assert graph.calls[-1] == f"{GRAPH}/drives/archive/items/{GUID}"

    def test_root_site_office_url_resolves_the_root_site(self, graph):
        graph.routes[f"{GRAPH}/sites/{HOST}"] = _response(200, {"id": SITE_ID})
        graph.routes[f"{GRAPH}/sites/{SITE_ID}/drive/items/{GUID}"] = _response(200, _file_item())

        assert _wrapper().read_file_from_url(
            f"https://{HOST}/_layouts/15/Doc.aspx?sourcedoc={GUID}") == "parsed content"

    def test_content_endpoint_is_used_when_graph_returns_no_download_url(self, graph):
        file_url = f"https://{HOST}/sites/Team/Shared%20Documents/report.txt"
        item_url = f"{GRAPH}/shares/{_share_id(file_url)}/driveItem"
        item = _file_item()
        del item["@microsoft.graph.downloadUrl"]
        graph.routes[item_url] = _response(200, item)

        with patch("elitea_sdk.tools.utils.http_utils.requests") as download_mock:
            download_mock.get.return_value = _download_response(b"bytes")
            _wrapper().read_file_from_url(file_url)

        assert download_mock.get.call_args[0][0] == f"{item_url}/content"
        assert download_mock.get.call_args[1]["headers"]["Authorization"] == "Bearer token"

    def test_sharing_link_is_redeemed_through_the_shares_api(self, graph):
        sharing_url = f"https://{HOST}/:t:/s/Team/EbC123ABC?e=xyz"
        item_url = f"{GRAPH}/shares/{_share_id(sharing_url)}/driveItem"
        graph.routes[item_url] = _response(200, _file_item())

        assert _wrapper().read_file_from_url(sharing_url) == "parsed content"
        assert graph.calls == [item_url]

    def test_folder_url_is_rejected_before_download(self, graph):
        file_url = f"https://{HOST}/sites/Team/Shared%20Documents/Specs"
        graph.routes[f"{GRAPH}/shares/{_share_id(file_url)}/driveItem"] = _response(
            200, {"name": "Specs", "folder": {"childCount": 3}})

        with pytest.raises(ToolException) as exc_info:
            _wrapper().read_file_from_url(file_url)

        assert str(exc_info.value) == NOT_A_FILE_URL_MESSAGE
        graph.parse.assert_not_called()

    @pytest.mark.parametrize("status", [400, 404])
    def test_unresolvable_url_reports_file_not_found(self, graph, status):
        file_url = f"https://{HOST}/sites/Team/Shared%20Documents/missing.txt"
        graph.routes[f"{GRAPH}/shares/{_share_id(file_url)}/driveItem"] = _response(status)

        with pytest.raises(ToolException, match="File not found"):
            _wrapper().read_file_from_url(file_url)

    def test_unknown_office_document_reports_file_not_found(self, graph):
        graph.routes[f"{GRAPH}/sites/{HOST}:/sites/Team"] = _response(200, {"id": SITE_ID})
        graph.routes[f"{GRAPH}/sites/{SITE_ID}/drives"] = _response(200, {"value": [{"id": "d1"}]})

        with pytest.raises(ToolException, match="File not found"):
            _wrapper().read_file_from_url(
                f"https://{HOST}/sites/Team/_layouts/15/Doc.aspx?sourcedoc={GUID}")

    def test_path_lookup_403_is_reported_as_not_found_or_denied(self, graph):
        file_url = f"https://{HOST}/sites/Team/Shared%20Documents/does-not-exist.txt"
        graph.routes[f"{GRAPH}/shares/{_share_id(file_url)}/driveItem"] = _response(403)

        with pytest.raises(ToolException, match="File not found or access denied"):
            _wrapper().read_file_from_url(file_url)

    def test_inaccessible_site_reports_access_denied(self, graph):
        graph.routes[f"{GRAPH}/sites/{HOST}:/sites/Private"] = _response(403)

        with pytest.raises(ToolException, match="^Access denied"):
            _wrapper().read_file_from_url(
                f"https://{HOST}/sites/Private/_layouts/15/Doc.aspx?sourcedoc={GUID}")

    def test_unsupported_file_type_is_rejected_before_download(self, graph):
        file_url = f"https://{HOST}/sites/Team/Shared%20Documents/bundle.zip"
        graph.routes[f"{GRAPH}/shares/{_share_id(file_url)}/driveItem"] = _response(
            200, _file_item("bundle.zip"))

        with pytest.raises(ToolException, match="not supported"):
            _wrapper().read_file_from_url(file_url)
        graph.parse.assert_not_called()

    def test_oversized_content_is_bounded_with_a_line_range(self, graph):
        file_url = f"https://{HOST}/sites/Team/Shared%20Documents/big.txt"
        graph.routes[f"{GRAPH}/shares/{_share_id(file_url)}/driveItem"] = _response(
            200, _file_item("big.txt"))
        graph.parse.return_value = "\n".join(f"line {i}" for i in range(1, 50001))

        assert _wrapper().read_file_from_url(file_url)["__result_status__"] == "content_too_large"
        assert _wrapper().read_file_from_url(file_url, start_line=2, end_line=3) == "line 2\nline 3\n"

    @pytest.mark.parametrize("url", [
        f"https://{HOST}/sites/Team/Shared%20Documents/huge.xlsx",
        f"https://{HOST}/:x:/s/Team/EbC123ABC?e=xyz",
    ])
    def test_oversized_workbook_guidance_names_this_tool(self, graph, url):
        graph.routes[f"{GRAPH}/shares/{_share_id(url)}/driveItem"] = _response(200, _file_item("huge.xlsx"))
        graph.parse.side_effect = ExcelReadLimitExceeded("too big", estimate=ExcelReadEstimate(
            sheets=[{"name": "Data", "max_row": 500000, "max_column": 20}],
            total_rows_workbook=500000, target_sheet="Data",
            target_sheet_total_rows=500000, requested_start_row=1,
            requested_end_row=500000, requested_rows=500000, sampled_rows=10,
            sampled_chars=5000, estimated_output_chars=25_000_000, embedded_images=0,
            file_size_bytes=18_000_000, is_unbounded_read=True, violations=["too big"],
        ))

        guidance = str(_wrapper().read_file_from_url(url))

        assert "read_file_from_url" in guidance
        assert "read_file_from_sharing_link" not in guidance


class TestToolkitExposure:

    def test_rest_backend_explains_that_url_reads_need_delegated_access(self):
        wrapper = SharepointRestWrapper(client=MagicMock(), site_url=f"https://{HOST}/sites/Team")

        with pytest.raises(ToolException, match="Graph API delegated access"):
            wrapper.read_file_from_url(f"https://{HOST}/sites/Team/Shared%20Documents/a.pdf")

    def test_tool_is_registered_with_its_schema(self):
        tools = {t["name"]: t for t in SharepointApiWrapper.model_construct().get_available_tools()}

        assert tools["read_file_from_url"]["args_schema"] is ReadFromUrl
        assert set(ReadFromUrl.model_fields) == {"url", "is_capture_image", "start_line", "end_line"}

    def test_model_receives_the_whole_description(self):
        with patch.object(SharepointApiWrapper, "_backend", create=True):
            toolkit = SharepointToolkit.get_toolkit(
                selected_tools=["read_file_from_url"], toolkit_name="team_docs",
                site_url=f"https://{HOST}/sites/EPAMAlita_test_sam",
                token="token", scopes=["Files.Read"], llm=None)

        description = toolkit.get_tools()[0].description
        assert description.endswith("Toolkit: team_docs")
        assert "do not call\n        get_files_list" in description
