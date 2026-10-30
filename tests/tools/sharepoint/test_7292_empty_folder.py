"""An empty SharePoint folder is an empty list, not a failure (#7292).

Graph answers 200 with ``{"value": []}`` for an empty library or folder. That must
reach the caller (and the indexing loader) as zero files; only a real Graph error
(non-2xx, transport failure) may raise.
"""

import json
from unittest.mock import patch

import pytest
import requests
from langchain_core.tools import ToolException

from elitea_sdk.tools.sharepoint.api_wrapper import SharepointApiWrapper
from elitea_sdk.tools.sharepoint.graph_wrapper import SharepointGraphWrapper

GRAPH_MODULE = "elitea_sdk.tools.sharepoint.graph_wrapper"
SITE = "https://tenant.sharepoint.com/sites/x"
DRIVE = {"id": "drive-1", "webUrl": f"{SITE}/Shared%20Documents"}


def _response(status: int, body: dict) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(body).encode()
    resp.url = "https://graph.microsoft.com/v1.0/drives/drive-1/root/children"
    return resp


def _graph() -> SharepointGraphWrapper:
    wrapper = SharepointGraphWrapper(site_url=SITE, token="t", scopes=["Sites.Read.All"])
    wrapper._list_drives = lambda: [DRIVE]
    return wrapper


def test_graph_empty_200_returns_empty_list():
    with patch(f"{GRAPH_MODULE}.requests.get", return_value=_response(200, {"value": []})):
        assert _graph().get_files_list() == []


def test_graph_empty_200_in_form_scoped_folder_returns_empty_list():
    with patch(f"{GRAPH_MODULE}.requests.get", return_value=_response(200, {"value": []})):
        assert _graph().get_files_list(folder_name="Empty", form_name="Shared Documents") == []


def test_graph_folder_with_only_empty_subfolder_returns_empty_list():
    pages = [
        _response(200, {"value": [{"id": "sub-1", "name": "sub", "folder": {"childCount": 0}}]}),
        _response(200, {"value": []}),
    ]
    with patch(f"{GRAPH_MODULE}.requests.get", side_effect=pages):
        assert _graph().get_files_list() == []


def test_graph_files_are_still_returned():
    item = {"id": "f1", "name": "a.txt", "file": {}, "webUrl": "https://x/a.txt",
            "parentReference": {"path": "/drives/drive-1/root:"}}
    with patch(f"{GRAPH_MODULE}.requests.get", return_value=_response(200, {"value": [item]})):
        result = _graph().get_files_list()

    assert [f["Name"] for f in result] == ["a.txt"]


def test_graph_http_error_still_raises_with_message():
    forbidden = _response(403, {"error": {"code": "accessDenied", "message": "Access denied"}})
    with patch(f"{GRAPH_MODULE}.requests.get", return_value=forbidden):
        with pytest.raises(ToolException, match="Can not get files.*403"):
            _graph().get_files_list()


def test_graph_transport_error_still_raises_with_message():
    with patch(f"{GRAPH_MODULE}.requests.get", side_effect=requests.ConnectionError("boom")):
        with pytest.raises(ToolException, match="Can not get files.*boom"):
            _graph().get_files_list()


class _Backend:
    def __init__(self, files=None, error=None):
        self._files = files
        self._error = error
        self.elitea = None
        self.llm = None

    def get_files_list(self, *args, **kwargs):
        if self._error:
            raise self._error
        return self._files


def _toolkit(backend) -> SharepointApiWrapper:
    wrapper = SharepointApiWrapper.model_construct(site_url=SITE)
    wrapper._backend = backend
    wrapper._log_tool_event = lambda *a, **k: None
    return wrapper


def test_indexing_loader_yields_nothing_for_an_empty_library():
    toolkit = _toolkit(_Backend(files=[]))

    assert list(toolkit._base_loader(path=None, limit_files=100)) == []


def test_indexing_loader_yields_a_document_per_file():
    files = [{"Name": "a.txt", "Path": "/p/a.txt", "Modified": "2026-01-01", "id": "f1"}]
    toolkit = _toolkit(_Backend(files=files))

    docs = list(toolkit._base_loader(path=None, limit_files=100))

    assert [d.metadata["Name"] for d in docs] == ["a.txt"]
    assert docs[0].metadata["updated_on"] == "2026-01-01"


def test_indexing_loader_real_error_still_raises_with_message():
    toolkit = _toolkit(_Backend(error=ToolException("Can not get files: 403 Forbidden")))

    with pytest.raises(ToolException, match="Unable to extract files.*403 Forbidden"):
        list(toolkit._base_loader(path=None, limit_files=100))
