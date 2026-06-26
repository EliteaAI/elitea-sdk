"""Regression tests for ``load_file_content_in_bytes`` document-library path
resolution (issue #5323).

These tests mock ONLY the HTTP boundary (``_get`` for the ``/drives`` listing and
``_get_raw`` for content/probe GETs) and let the REAL ``_resolve_drive_and_folder``
run, so the suite would fail again if the strip-by-name bug were reintroduced.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.sharepoint.graph_wrapper import SharepointGraphWrapper

SITE_URL = "https://test.sharepoint.com/sites/Test"
DRIVE_DEFAULT = "b!DEFAULT_DRIVE_ID"
DRIVE_ELITEA = "b!ELITEA_DRIVE_ID"

# Default library: human-facing "Shared Documents" but internal name "Documents".
# Non-default library: "EliteA_test".
DRIVES = [
    {
        "id": DRIVE_DEFAULT,
        "name": "Documents",
        "webUrl": "https://test.sharepoint.com/sites/Test/Shared Documents",
    },
    {
        "id": DRIVE_ELITEA,
        "name": "EliteA_test",
        "webUrl": "https://test.sharepoint.com/sites/Test/EliteA_test",
    },
]


def _make_content_response(content=b"file-bytes", status_code=200):
    resp = MagicMock()
    resp.ok = 200 <= status_code < 400
    resp.status_code = status_code
    resp.content = content
    return resp


def _make_probe_response(status_code=200, kind="folder"):
    """Build a probe response. ``kind`` controls the JSON body:
    'folder' -> {id, folder} (a folder), 'file' -> {id} only (a file, no
    'folder' key), 'none' -> {} (neither; does not match)."""
    resp = MagicMock()
    resp.status_code = status_code
    if kind == "folder":
        body = {"id": "item-id", "folder": {}}
    elif kind == "file":
        body = {"id": "item-id"}
    else:
        body = {}
    resp.json = MagicMock(return_value=body)
    return resp


def _build_wrapper(content_status=200, probe_status=200, probe_kind="folder",
                   drives=None):
    """Construct a wrapper with HTTP boundary mocked but resolver REAL.

    Returns (wrapper, get_mock, get_raw_mock). ``get_raw_mock`` records every URL
    it is called with so tests can assert probe vs content traffic. ``drives``
    overrides the document-library listing (default: module-level ``DRIVES``).
    """
    drives = DRIVES if drives is None else drives
    wrapper = SharepointGraphWrapper(
        site_url=SITE_URL,
        token="test-token",
        scopes=["Files.Read"],
    )
    # Pre-resolve the site id so _list_drives only needs the /drives GET.
    wrapper._SharepointGraphWrapper__site_id = "site-id"

    def fake_get(url, params=None):
        if url.endswith("/drives"):
            return {"value": drives}
        if url.endswith("/drive"):
            return {"id": DRIVE_DEFAULT}
        raise AssertionError(f"Unexpected _get URL: {url}")

    get_mock = MagicMock(side_effect=fake_get)

    def fake_get_raw(url, **kwargs):
        if url.endswith(":/content"):
            return _make_content_response(status_code=content_status)
        # Probe request from _resolve_drive_and_folder fallback leg.
        return _make_probe_response(status_code=probe_status, kind=probe_kind)

    get_raw_mock = MagicMock(side_effect=fake_get_raw)

    wrapper._get = get_mock
    wrapper._get_raw = get_raw_mock
    return wrapper, get_mock, get_raw_mock


def _content_urls(get_raw_mock):
    return [c.args[0] for c in get_raw_mock.call_args_list if c.args[0].endswith(":/content")]


def _probe_urls(get_raw_mock):
    return [c.args[0] for c in get_raw_mock.call_args_list if not c.args[0].endswith(":/content")]


class TestLoadFileContentResolution:
    def test_full_delegated_default_library_path(self):
        """(1)+(6) Full '/sites/Test/Shared Documents/...' resolves to the default
        drive, strips 'Shared Documents', and single-encodes the space (%20)."""
        wrapper, _, get_raw = _build_wrapper()
        wrapper.load_file_content_in_bytes(
            "/sites/Test/Shared Documents/folder/with image.docx")
        urls = _content_urls(get_raw)
        assert len(urls) == 1
        assert urls[0] == (
            f"https://graph.microsoft.com/v1.0/drives/{DRIVE_DEFAULT}"
            "/root:/folder/with%20image.docx:/content")
        assert "%2520" not in urls[0]

    def test_default_library_without_site_prefix(self):
        """(2) '/Shared Documents/...' (no site prefix) resolves to default drive."""
        wrapper, _, get_raw = _build_wrapper()
        wrapper.load_file_content_in_bytes("/Shared Documents/folder/file.docx")
        urls = _content_urls(get_raw)
        assert urls[0] == (
            f"https://graph.microsoft.com/v1.0/drives/{DRIVE_DEFAULT}"
            "/root:/folder/file.docx:/content")

    def test_non_default_library_path(self):
        """(3) Non-default library path routes to its own drive."""
        wrapper, _, get_raw = _build_wrapper()
        wrapper.load_file_content_in_bytes("/sites/Test/EliteA_test/sub/file.docx")
        urls = _content_urls(get_raw)
        assert urls[0] == (
            f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ELITEA}"
            "/root:/sub/file.docx:/content")

    def test_bare_path_uses_probe_fallback_and_warns(self, caplog):
        """(4) Bare path (no library prefix) exercises the probe-fallback leg and
        logs a warning when more than one drive matches."""
        wrapper, _, get_raw = _build_wrapper()
        with caplog.at_level(logging.WARNING):
            wrapper.load_file_content_in_bytes("sample/with image.docx")
        # Both drives probe-match (stubbed 200) -> ambiguous -> default preferred.
        urls = _content_urls(get_raw)
        assert urls[0] == (
            f"https://graph.microsoft.com/v1.0/drives/{DRIVE_DEFAULT}"
            "/root:/sample/with%20image.docx:/content")
        assert _probe_urls(get_raw)  # probes actually happened
        assert any("ambiguous bare path" in r.message for r in caplog.records)

    def test_ambiguous_prefers_default_drive_regardless_of_order(self, caplog):
        """(4b) When an ambiguous bare path matches multiple libraries, the DEFAULT
        drive is chosen even if it is not first in the drive enumeration order."""
        # Default library deliberately listed LAST.
        reordered = [DRIVES[1], DRIVES[0]]
        wrapper, _, get_raw = _build_wrapper(drives=reordered)
        with caplog.at_level(logging.WARNING):
            wrapper.load_file_content_in_bytes("sample/file.docx")
        urls = _content_urls(get_raw)
        # Without default-preference this would resolve to DRIVE_ELITEA (first).
        assert urls[0] == (
            f"https://graph.microsoft.com/v1.0/drives/{DRIVE_DEFAULT}"
            "/root:/sample/file.docx:/content")
        assert any("ambiguous bare path" in r.message for r in caplog.records)

    def test_probe_accepts_file_via_id_without_folder_key(self):
        """(4c) The probe leg matches a FILE (response has 'id' but no 'folder'
        key) — pins the comment's claim that files are accepted via 'id'."""
        # Single library so the match is unambiguous and isolates the 'id' path.
        wrapper, _, get_raw = _build_wrapper(probe_kind="file", drives=[DRIVES[0]])
        wrapper.load_file_content_in_bytes("sample/file.docx")
        urls = _content_urls(get_raw)
        assert urls[0] == (
            f"https://graph.microsoft.com/v1.0/drives/{DRIVE_DEFAULT}"
            "/root:/sample/file.docx:/content")

    def test_all_probes_fail_falls_back_to_default_drive(self):
        """(4d) When no library prefix matches and every probe fails, the resolver
        falls back to the default drive with the cleaned path as-is."""
        wrapper, _, get_raw = _build_wrapper(probe_status=404)
        wrapper.load_file_content_in_bytes("sample/file.docx")
        urls = _content_urls(get_raw)
        assert urls[0] == (
            f"https://graph.microsoft.com/v1.0/drives/{DRIVE_DEFAULT}"
            "/root:/sample/file.docx:/content")

    def test_branch1_graph_format_path_skips_resolver(self):
        """(5)+(7) Branch-1 '/drives/{id}/root:/...' path does NOT call the
        resolver and preserves the embedded drive_id verbatim; space single-encoded."""
        wrapper, _, get_raw = _build_wrapper()
        with patch.object(
            wrapper, "_resolve_drive_and_folder",
            wraps=wrapper._resolve_drive_and_folder,
        ) as resolver_spy:
            wrapper.load_file_content_in_bytes(
                f"/drives/{DRIVE_DEFAULT}/root:/folder/with image.txt")
        resolver_spy.assert_not_called()
        urls = _content_urls(get_raw)
        assert urls[0] == (
            f"https://graph.microsoft.com/v1.0/drives/{DRIVE_DEFAULT}"
            "/root:/folder/with%20image.txt:/content")
        assert "%2520" not in urls[0]

    def test_matched_path_makes_zero_probe_gets(self):
        """(8) A library-prefixed match issues only the single content GET — no
        per-read probe and no per-read GET /drives/{id}/root."""
        wrapper, _, get_raw = _build_wrapper()
        wrapper.load_file_content_in_bytes(
            "/sites/Test/Shared Documents/folder/file.docx")
        assert _probe_urls(get_raw) == []
        assert len(_content_urls(get_raw)) == 1

    def test_drives_listing_cached_across_reads(self):
        """(9) GET /drives fires at most once across two sequential reads."""
        wrapper, get_mock, _ = _build_wrapper()
        wrapper.load_file_content_in_bytes(
            "/sites/Test/Shared Documents/a/file1.docx")
        wrapper.load_file_content_in_bytes(
            "/sites/Test/EliteA_test/b/file2.docx")
        drives_calls = [c for c in get_mock.call_args_list
                        if c.args[0].endswith("/drives")]
        assert len(drives_calls) == 1

    def test_content_404_raises_sanitized_exception(self):
        """(10) A 404 from the content GET raises a ToolException that contains the
        caller path but NOT the resolved drive_id or the full Graph URL."""
        wrapper, _, _ = _build_wrapper(content_status=404)
        path = "/sites/Test/Shared Documents/missing/file.docx"
        with pytest.raises(ToolException) as exc_info:
            wrapper.load_file_content_in_bytes(path)
        msg = str(exc_info.value)
        assert path in msg
        assert "404" in msg
        assert DRIVE_DEFAULT not in msg
        assert "graph.microsoft.com" not in msg

    def test_read_file_sanitizes_unexpected_error(self):
        """(11) read_file's catch-all raises a ToolException that echoes the caller
        path but NOT raw exception text (which may carry internal URLs/ids)."""
        wrapper, _, _ = _build_wrapper()
        leaky = "https://graph.microsoft.com/v1.0/drives/b!SECRET/root:/x leaked"
        wrapper.load_file_content_in_bytes = MagicMock(
            side_effect=ValueError(leaky))
        path = "/sites/Test/Shared Documents/x.docx"
        with pytest.raises(ToolException) as exc_info:
            wrapper.read_file(path)
        msg = str(exc_info.value)
        assert path in msg
        assert "SECRET" not in msg
        assert "graph.microsoft.com" not in msg

    def test_library_root_without_filename_raises_clear_error(self):
        """(12) A path that resolves to a library/folder root (no filename) raises
        a clear ToolException and issues NO content GET (no '.../root:/:/content')."""
        wrapper, _, get_raw = _build_wrapper()
        with pytest.raises(ToolException) as exc_info:
            wrapper.load_file_content_in_bytes("/sites/Test/Shared Documents")
        assert "does not point to a file" in str(exc_info.value)
        assert _content_urls(get_raw) == []
