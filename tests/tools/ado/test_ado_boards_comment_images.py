"""Unit tests for ADO Boards comment image processing and the attachment image tool."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bs4 import BeautifulSoup
from langchain_core.tools import ToolException
from pydantic import SecretStr

from elitea_sdk.runtime.langchain.document_loaders.EliteAImageLoader import MAX_IMAGE_READ_BYTES
from elitea_sdk.runtime.langchain.document_loaders.image_cache import ImageDescriptionCache
from elitea_sdk.tools.ado.work_item import ado_wrapper as wrapper_module
from elitea_sdk.tools.ado.work_item.ado_wrapper import (
    ADOGetComments,
    ADOGetImageByUrl,
    AzureDevOpsApiWrapper,
)

ORG_URL = "https://dev.azure.com/MyOrg"
_TINY_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00'
    b'\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe'
    b'\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82'
)
_HUGE_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="100000" height="100000">'
    b'<rect width="100000" height="100000" fill="red"/></svg>'
)
_MB = 1024 * 1024


def _org_img(guid="guid1", name="shot.png", host="dev.azure.com/MyOrg"):
    return f"https://{host}/Proj/_apis/wit/attachments/{guid}?fileName={name}"


ALIAS_IMG = _org_img(host="myorg.visualstudio.com")


class _CountingStream:
    """Chunk generator that records how many chunks were pulled before abandonment."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    def close(self):
        self.closed = True


def _stream(content, chunk_size=None):
    if chunk_size is None:
        return _CountingStream([content])
    return _CountingStream([content[i:i + chunk_size] for i in range(0, len(content), chunk_size)])


def _router(streams):
    def _get(id, download):
        if id not in streams:
            raise KeyError(f"unexpected attachment id {id}")
        value = streams[id]
        if isinstance(value, Exception):
            raise value
        return value

    return _get


def _comment(text, rendered_text=None):
    payload = {"id": 1, "text": text}
    if rendered_text is not None:
        payload["rendered_text"] = rendered_text
    return SimpleNamespace(as_dict=lambda: dict(payload))


def _page(comments, token=None):
    return SimpleNamespace(comments=comments, continuation_token=token)


def _make_wrapper(limit=5):
    wrapper = AzureDevOpsApiWrapper.model_construct(
        organization_url=ORG_URL, project="Proj", token=SecretStr("x"), limit=limit)
    wrapper.llm = MagicMock()
    object.__setattr__(wrapper, '_client', MagicMock())
    object.__setattr__(wrapper, '_image_cache', ImageDescriptionCache())
    return wrapper


def _with_comments(wrapper, *pages):
    wrapper._client.get_comments.side_effect = list(pages)
    return wrapper


def _work_item(fields, relations=None, item_id=7):
    item = SimpleNamespace(id=item_id, fields=fields)
    if relations is not None:
        item.relations = relations
    return item


def _fake_parser(return_value="described"):
    return MagicMock(return_value=return_value)


def _descriptions_in(html):
    return [img.get('image-description') for img in BeautifulSoup(html, 'html.parser').find_all('img')]


class TestExtractAttachmentRef:

    def test_filename_first_param(self):
        wrapper = _make_wrapper()
        assert wrapper._extract_attachment_ref(_org_img()) == ("guid1", "shot.png")

    def test_filename_after_api_version(self):
        wrapper = _make_wrapper()
        url = f"{ORG_URL}/Proj/_apis/wit/attachments/guid1?api-version=7.1&fileName=a.png"
        assert wrapper._extract_attachment_ref(url) == ("guid1", "a.png")

    def test_no_filename_returns_none_name(self):
        wrapper = _make_wrapper()
        url = f"{ORG_URL}/Proj/_apis/wit/attachments/guid1"
        assert wrapper._extract_attachment_ref(url) == ("guid1", None)

    def test_urlencoded_filename_decoded(self):
        wrapper = _make_wrapper()
        url = f"{ORG_URL}/Proj/_apis/wit/attachments/guid1?fileName=my%20screen.png"
        assert wrapper._extract_attachment_ref(url) == ("guid1", "my screen.png")

    def test_non_attachment_url_returns_none(self):
        wrapper = _make_wrapper()
        assert wrapper._extract_attachment_ref("https://evil.example.com/x.png") is None

    def test_relative_attachment_path(self):
        wrapper = _make_wrapper()
        assert wrapper._extract_attachment_ref(
            "/Proj/_apis/wit/attachments/guid1?fileName=a.png") == ("guid1", "a.png")


class TestCappedDownload:

    def test_under_cap_returns_bytes(self):
        wrapper = _make_wrapper()
        stream = _stream(b"x" * 2048, chunk_size=512)
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": stream})
        assert wrapper._get_attachment_content_capped("guid1", limit=MAX_IMAGE_READ_BYTES) == b"x" * 2048
        assert stream.consumed == 4
        assert stream.closed is True

    def test_over_cap_aborts_stream(self):
        wrapper = _make_wrapper()
        stream = _stream(b"x" * (10 * _MB), chunk_size=_MB)
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": stream})
        assert wrapper._get_attachment_content_capped("guid1", limit=MAX_IMAGE_READ_BYTES) is None
        assert stream.consumed == 6
        assert stream.closed is True

    def test_limit_parameterized(self):
        wrapper = _make_wrapper()
        stream = _stream(b"x" * (10 * _MB), chunk_size=_MB)
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": stream})
        content = wrapper._get_attachment_content_capped(
            "guid1", limit=wrapper_module._ATTACHMENT_STREAM_CEILING_BYTES)
        assert content == b"x" * (10 * _MB)
        assert stream.consumed == 10


class TestGetCommentsBaseline:

    def test_paging_baseline_unchanged(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _with_comments(
            _make_wrapper(limit=2),
            _page([_comment("first"), _comment("second")], token="tok"),
            _page([_comment("third")]),
        )
        result = wrapper.get_comments(work_item_id=1, limit_total=3)
        assert [comment["text"] for comment in result] == ["first", "second", "third"]
        second_call = wrapper._client.get_comments.call_args_list[1].kwargs
        assert second_call["continuation_token"] == "tok"
        assert second_call["top"] == 3
        parser.assert_not_called()

    def test_flag_absent_no_image_processing(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        text = f'<div><img src="{_org_img()}"></div>'
        wrapper = _with_comments(_make_wrapper(), _page([_comment(text)]))
        result = wrapper.get_comments(work_item_id=1)
        assert result == [{"id": 1, "text": text}]
        parser.assert_not_called()
        wrapper._client.get_attachment_content.assert_not_called()

    def test_limit_total_slice_processed_only(self, monkeypatch):
        parser = _fake_parser("a screenshot")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _with_comments(
            _make_wrapper(limit=5),
            _page([_comment(f'<img src="{_org_img("guid1")}">'),
                   _comment(f'<img src="{_org_img("guid2")}">')]),
        )
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        result = wrapper.get_comments(work_item_id=1, limit_total=1, process_images=True)
        assert len(result) == 1
        assert parser.call_count == 1


class TestGetCommentsGate:

    def test_flag_on_llm_none_returns_raw_with_warning(self, monkeypatch, caplog):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        text = f'<img src="{_org_img()}">'
        wrapper = _with_comments(_make_wrapper(), _page([_comment(text)]))
        wrapper.llm = None
        with caplog.at_level("WARNING"):
            result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert result == [{"id": 1, "text": text}]
        assert "no LLM is configured" in caplog.text
        parser.assert_not_called()
        wrapper._client.get_attachment_content.assert_not_called()

    def test_flag_on_no_markup_no_scan(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _with_comments(_make_wrapper(), _page([_comment("plain comment text")]))
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert result == [{"id": 1, "text": "plain comment text"}]
        parser.assert_not_called()
        wrapper._client.get_attachment_content.assert_not_called()

    def test_expand_forced_to_rendered_text(self):
        wrapper = _with_comments(_make_wrapper(), _page([_comment("plain")]))
        wrapper.get_comments(work_item_id=1, process_images=True)
        assert wrapper._client.get_comments.call_args.kwargs["expand"] == "renderedText"

    def test_explicit_expand_respected(self):
        wrapper = _with_comments(_make_wrapper(), _page([_comment("plain")]))
        wrapper.get_comments(work_item_id=1, process_images=True, expand="all")
        assert wrapper._client.get_comments.call_args.kwargs["expand"] == "all"

    def test_expand_untouched_when_flag_off(self):
        wrapper = _with_comments(_make_wrapper(), _page([_comment("plain")]))
        wrapper.get_comments(work_item_id=1, expand="none")
        assert wrapper._client.get_comments.call_args.kwargs["expand"] == "none"


class TestGetCommentsImagePass:

    def test_html_image_described(self, monkeypatch):
        parser = _fake_parser("a login screenshot")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _with_comments(_make_wrapper(), _page([_comment(f'<img src="{_org_img()}">')]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert parser.call_count == 1
        assert _descriptions_in(result[0]["text"]) == ["a login screenshot"]

    def test_alias_host_image_described(self, monkeypatch):
        parser = _fake_parser("alias described")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _with_comments(_make_wrapper(), _page([_comment(f'<img src="{ALIAS_IMG}">')]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert _descriptions_in(result[0]["text"]) == ["alias described"]
        assert wrapper._client.get_attachment_content.call_args.kwargs["id"] == "guid1"

    def test_markdown_image_described(self, monkeypatch):
        parser = _fake_parser("a diagram")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        url = _org_img()
        wrapper = _with_comments(_make_wrapper(), _page([_comment(f"see ![shot]({url}) please")]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert f"![shot]({url})" in result[0]["text"]
        assert "[image-description: a diagram]" in result[0]["text"]

    def test_markdown_repeated_reference_linear_descriptions(self, monkeypatch):
        parser = _fake_parser("a diagram")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        url = _org_img()
        text = f"before ![shot]({url}) middle ![shot]({url}) end"
        wrapper = _with_comments(_make_wrapper(), _page([_comment(text)]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert result[0]["text"].count("[image-description: a diagram]") == 2

    def test_markdown_title_syntax_described(self, monkeypatch):
        parser = _fake_parser("a diagram")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        url = _org_img()
        wrapper = _with_comments(_make_wrapper(), _page([_comment(f'![shot]({url} "hover title")')]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert f'![shot]({url} "hover title")' in result[0]["text"]
        assert "[image-description: a diagram]" in result[0]["text"]
        assert "[image unavailable" not in result[0]["text"]

    def test_external_url_placeholder(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _with_comments(
            _make_wrapper(), _page([_comment('<img src="https://evil.example.com/x.png">')]))
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert isinstance(result, list)
        wrapper._client.get_attachment_content.assert_not_called()
        parser.assert_not_called()
        assert "[image unavailable" in result[0]["text"]

    def test_filename_param_order_end_to_end(self, monkeypatch):
        parser = _fake_parser("ordered")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        url = f"{ORG_URL}/Proj/_apis/wit/attachments/guid1?api-version=7.1&fileName=shot.png"
        wrapper = _with_comments(_make_wrapper(), _page([_comment(f'<img src="{url}">')]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert _descriptions_in(result[0]["text"]) == ["ordered"]

    def test_unsupported_extension_no_download(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _with_comments(
            _make_wrapper(), _page([_comment(f'<img src="{_org_img(name="scan.tiff")}">')]))
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert "unsupported image format" in result[0]["text"]
        assert "ToolException" not in result[0]["text"]
        wrapper._client.get_attachment_content.assert_not_called()
        parser.assert_not_called()

    def test_non_image_attachment_not_spliced(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _with_comments(
            _make_wrapper(), _page([_comment(f'<img src="{_org_img(name="trace.log")}">')]))
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert "[image unavailable" in result[0]["text"]
        wrapper._client.get_attachment_content.assert_not_called()
        parser.assert_not_called()

    def test_oversize_image_streaming_abort(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        stream = _stream(b"x" * (10 * _MB), chunk_size=_MB)
        wrapper = _with_comments(_make_wrapper(), _page([_comment(f'<img src="{_org_img()}">')]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": stream})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert "exceeds the 5 MB" in result[0]["text"]
        parser.assert_not_called()
        assert stream.consumed == 6

    def test_pixel_budget_exceeded_placeholder(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        monkeypatch.setattr(wrapper_module, '_MAX_IMAGE_PIXELS_FOR_LLM', 0)
        wrapper = _with_comments(_make_wrapper(), _page([_comment(f'<img src="{_org_img()}">')]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert isinstance(result, list)
        assert "dimensions" in result[0]["text"] and "exceed" in result[0]["text"]
        parser.assert_not_called()

    def test_unreadable_dimensions_placeholder(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _with_comments(_make_wrapper(), _page([_comment(f'<img src="{_org_img()}">')]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(b"not-an-image")})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert "could not read image dimensions" in result[0]["text"]
        parser.assert_not_called()

    def test_svg_canvas_bomb_gated(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _with_comments(
            _make_wrapper(), _page([_comment(f'<img src="{_org_img(name="diagram.svg")}">')]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_HUGE_SVG)})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert "[image unavailable" in result[0]["text"]
        parser.assert_not_called()

    def test_partial_failure_placeholders(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        text = "".join(f'<img src="{_org_img(f"guid{i}")}">' for i in (1, 2, 3))
        wrapper = _with_comments(_make_wrapper(), _page([_comment(text)]))
        wrapper._client.get_attachment_content.side_effect = _router({
            "guid1": _stream(_TINY_PNG),
            "guid2": RuntimeError("attachment gone"),
            "guid3": _stream(_TINY_PNG),
        })
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert isinstance(result, list)
        descriptions = _descriptions_in(result[0]["text"])
        assert descriptions.count("ok") == 2
        assert sum(1 for d in descriptions if d and d.startswith("[image unavailable")) == 1

    def test_data_uri_no_network(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        text = f'<img src="data:image/png;base64,AAAA"><img src="{_org_img()}">'
        wrapper = _with_comments(_make_wrapper(), _page([_comment(text)]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        images = BeautifulSoup(result[0]["text"], 'html.parser').find_all('img')
        assert images[0].get('image-description') is None
        assert wrapper._client.get_attachment_content.call_count == 1

    def test_image_cap_enforced(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        text = "".join(f'<img src="{_org_img(f"guid{i}")}">' for i in range(25))
        wrapper = _with_comments(_make_wrapper(), _page([_comment(text)]))
        wrapper._client.get_attachment_content.side_effect = _router(
            {f"guid{i}": _stream(_TINY_PNG) for i in range(25)})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert parser.call_count == 20
        descriptions = _descriptions_in(result[0]["text"])
        assert descriptions.count("ok") == 20
        assert sum(1 for d in descriptions if d and "limit of 20" in d) == 5

    def test_url_dedup_single_fetch(self, monkeypatch):
        parser = _fake_parser("shared")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        url = _org_img()
        wrapper = _with_comments(
            _make_wrapper(), _page([_comment(f'<img src="{url}">', rendered_text=f'<img src="{url}">')]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        result = wrapper.get_comments(work_item_id=1, process_images=True)
        assert wrapper._client.get_attachment_content.call_count == 1
        assert _descriptions_in(result[0]["text"]) == ["shared"]
        assert _descriptions_in(result[0]["rendered_text"]) == ["shared"]

    def test_comments_pool_engages_on_main_thread(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        real_executor = wrapper_module.ThreadPoolExecutor
        instantiations = []

        def counting_executor(*args, **kwargs):
            instantiations.append(kwargs)
            return real_executor(*args, **kwargs)

        monkeypatch.setattr(wrapper_module, 'ThreadPoolExecutor', counting_executor)
        text = "".join(f'<img src="{_org_img(f"guid{i}")}">' for i in range(3))
        wrapper = _with_comments(_make_wrapper(), _page([_comment(text)]))
        wrapper._client.get_attachment_content.side_effect = _router(
            {f"guid{i}": _stream(_TINY_PNG) for i in range(3)})
        wrapper.get_comments(work_item_id=1, process_images=True)
        assert instantiations

    def test_comments_pool_engages_off_main_thread(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        real_executor = wrapper_module.ThreadPoolExecutor
        instantiations = []

        def counting_executor(*args, **kwargs):
            instantiations.append(kwargs)
            return real_executor(*args, **kwargs)

        monkeypatch.setattr(wrapper_module, 'ThreadPoolExecutor', counting_executor)
        text = "".join(f'<img src="{_org_img(f"guid{i}")}">' for i in range(3))
        wrapper = _with_comments(_make_wrapper(), _page([_comment(text)]))
        wrapper._client.get_attachment_content.side_effect = _router(
            {f"guid{i}": _stream(_TINY_PNG) for i in range(3)})
        worker = threading.Thread(
            target=lambda: wrapper.get_comments(work_item_id=1, process_images=True),
            name="tool-dispatch-0")
        worker.start()
        worker.join()
        assert instantiations

    def test_comments_serial_inside_fetch_pool(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        instantiations = []
        monkeypatch.setattr(wrapper_module, 'ThreadPoolExecutor',
                            lambda *a, **k: instantiations.append(k) or (_ for _ in ()).throw(AssertionError))
        text = "".join(f'<img src="{_org_img(f"guid{i}")}">' for i in range(3))
        wrapper = _with_comments(_make_wrapper(), _page([_comment(text)]))
        wrapper._client.get_attachment_content.side_effect = _router(
            {f"guid{i}": _stream(_TINY_PNG) for i in range(3)})
        results = []
        worker = threading.Thread(
            target=lambda: results.append(wrapper.get_comments(work_item_id=1, process_images=True)),
            name="ado-wi-fetch_0")
        worker.start()
        worker.join()
        assert not instantiations
        assert results and isinstance(results[0], list)

    def test_cache_and_default_prompt_passthrough(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _with_comments(_make_wrapper(), _page([_comment(f'<img src="{_org_img()}">')]))
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        wrapper.get_comments(work_item_id=1, process_images=True)
        kwargs = parser.call_args.kwargs
        assert kwargs["prompt"] is None
        assert kwargs["image_cache"] is wrapper._image_cache
        assert kwargs["llm"] is wrapper.llm


class TestGetImageByUrl:

    def test_happy_path(self, monkeypatch):
        parser = _fake_parser("a red pixel")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _make_wrapper()
        wrapper._client.get_attachment_content.side_effect = _router(
            {"guid1": _stream(_TINY_PNG, chunk_size=16)})
        assert wrapper.get_image_by_url(_org_img()) == "a red pixel"

    def test_alias_host_accepted(self, monkeypatch):
        parser = _fake_parser("a red pixel")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _make_wrapper()
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        assert wrapper.get_image_by_url(ALIAS_IMG) == "a red pixel"
        assert wrapper._client.get_attachment_content.call_args.kwargs["id"] == "guid1"

    def test_malformed_url(self):
        wrapper = _make_wrapper()
        with pytest.raises(ToolException):
            wrapper.get_image_by_url("not a url")

    def test_non_attachment_path_rejected(self):
        wrapper = _make_wrapper()
        with pytest.raises(ToolException):
            wrapper.get_image_by_url("https://dev.azure.com/MyOrg/Proj/_apis/wit/workitems/1")
        wrapper._client.get_attachment_content.assert_not_called()

    def test_data_uri_rejected(self):
        wrapper = _make_wrapper()
        with pytest.raises(ToolException):
            wrapper.get_image_by_url("data:image/png;base64,AAAA")
        wrapper._client.get_attachment_content.assert_not_called()

    def test_no_llm(self):
        wrapper = _make_wrapper()
        wrapper.llm = None
        with pytest.raises(ToolException, match="LLM"):
            wrapper.get_image_by_url(_org_img())

    def test_unsupported_extension_before_download(self):
        wrapper = _make_wrapper()
        with pytest.raises(ToolException):
            wrapper.get_image_by_url(_org_img(name="doc.pdf"))
        wrapper._client.get_attachment_content.assert_not_called()

    def test_oversize_streaming_abort(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        stream = _stream(b"x" * (10 * _MB), chunk_size=_MB)
        wrapper = _make_wrapper()
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": stream})
        with pytest.raises(ToolException, match="exceeds the 5 MB"):
            wrapper.get_image_by_url(_org_img())
        parser.assert_not_called()
        assert stream.consumed == 6

    def test_pixel_budget_rejected(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        monkeypatch.setattr(wrapper_module, '_MAX_IMAGE_PIXELS_FOR_LLM', 0)
        wrapper = _make_wrapper()
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        with pytest.raises(ToolException, match="dimensions"):
            wrapper.get_image_by_url(_org_img())
        parser.assert_not_called()

    def test_svg_canvas_bomb_rejected(self, monkeypatch):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _make_wrapper()
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_HUGE_SVG)})
        with pytest.raises(ToolException):
            wrapper.get_image_by_url(_org_img(name="diagram.svg"))
        parser.assert_not_called()

    def test_download_error_returns_tool_exception(self, monkeypatch):
        monkeypatch.setattr(wrapper_module, 'parse_file_content', _fake_parser())
        wrapper = _make_wrapper()
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": RuntimeError("boom")})
        with pytest.raises(ToolException, match="boom"):
            wrapper.get_image_by_url(_org_img())

    def test_client_not_initialized(self, monkeypatch):
        monkeypatch.setattr(wrapper_module, 'parse_file_content', _fake_parser())
        wrapper = _make_wrapper()
        object.__setattr__(wrapper, '_client', None)
        with pytest.raises(ToolException, match="client not initialized"):
            wrapper.get_image_by_url(_org_img())

    def test_missing_filename_needs_param(self, monkeypatch):
        parser = _fake_parser("described")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        url = f"{ORG_URL}/Proj/_apis/wit/attachments/guid1"
        wrapper = _make_wrapper()
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        with pytest.raises(ToolException):
            wrapper.get_image_by_url(url)
        assert wrapper.get_image_by_url(url, file_name="a.png") == "described"


class TestGetWorkItemImagePass:

    def _wrapper_with_fields(self, fields, relations=None):
        wrapper = _make_wrapper()
        wrapper._client.get_work_item.return_value = _work_item(fields, relations)
        return wrapper

    def test_one_bad_image_among_three(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        html = (f'<img src="{_org_img("guid1")}">'
                f'<img src="https://evil.example.com/x.png">'
                f'<img src="data:image/png;base64,AAAA">'
                f'<img src="{_org_img("guid2")}">')
        wrapper = self._wrapper_with_fields({"System.Description": html})
        wrapper._client.get_attachment_content.side_effect = _router({
            "guid1": _stream(_TINY_PNG), "guid2": _stream(_TINY_PNG)})
        result = wrapper.get_work_item(id=7, parse_attachments=True, process_images=True)
        assert isinstance(result, dict)
        descriptions = _descriptions_in(result["System.Description"])
        assert descriptions.count("ok") == 2
        assert descriptions[1].startswith("[image unavailable")
        assert descriptions[2] is None

    def test_work_item_pass_is_serial(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)

        def forbidden_executor(*args, **kwargs):
            raise AssertionError("the work item field pass must not use a thread pool")

        monkeypatch.setattr(wrapper_module, 'ThreadPoolExecutor', forbidden_executor)
        html = "".join(f'<img src="{_org_img(f"guid{i}")}">' for i in range(4))
        wrapper = self._wrapper_with_fields({"System.Description": html})
        wrapper._client.get_attachment_content.side_effect = _router(
            {f"guid{i}": _stream(_TINY_PNG) for i in range(4)})
        result = wrapper.get_work_item(id=7, parse_attachments=True, process_images=True)
        assert _descriptions_in(result["System.Description"]) == ["ok"] * 4

    def test_gates_off_for_work_item(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        huge = b"x" * (8 * _MB)
        huge_stream = _stream(huge, chunk_size=_MB)
        odd_stream = _stream(b"drawio-bytes")
        html = (f'<img src="{_org_img("guid1", "huge.png")}">'
                f'<img src="{_org_img("guid2", "diagram.drawio")}">')
        wrapper = self._wrapper_with_fields({"System.Description": html})
        wrapper._client.get_attachment_content.side_effect = _router(
            {"guid1": huge_stream, "guid2": odd_stream})
        result = wrapper.get_work_item(id=7, parse_attachments=True, process_images=True)
        assert _descriptions_in(result["System.Description"]) == ["ok", "ok"]
        assert huge_stream.consumed == 8
        payloads = [call.kwargs["file_content"] for call in parser.call_args_list]
        assert huge in payloads
        assert b"drawio-bytes" in payloads

    def test_gates_off_no_pixel_gate(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = self._wrapper_with_fields(
            {"System.Description": f'<img src="{_org_img("guid1", "big.svg")}">'})
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_HUGE_SVG)})
        result = wrapper.get_work_item(id=7, parse_attachments=True, process_images=True)
        assert _descriptions_in(result["System.Description"]) == ["ok"]
        assert parser.call_args.kwargs["file_content"] == _HUGE_SVG

    def test_work_item_stream_ceiling_backstop(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        monkeypatch.setattr(wrapper_module, '_ATTACHMENT_STREAM_CEILING_BYTES', 4 * _MB)
        stream = _stream(b"x" * (6 * _MB), chunk_size=_MB)
        html = f'<img src="{_org_img("guid1", "huge.png")}"><img src="{_org_img("guid2")}">'
        wrapper = self._wrapper_with_fields({"System.Description": html})
        wrapper._client.get_attachment_content.side_effect = _router(
            {"guid1": stream, "guid2": _stream(_TINY_PNG)})
        result = wrapper.get_work_item(id=7, parse_attachments=True, process_images=True)
        assert isinstance(result, dict)
        descriptions = _descriptions_in(result["System.Description"])
        assert descriptions[0].startswith("[image unavailable")
        assert descriptions[1] == "ok"
        assert parser.call_count == 1
        assert stream.consumed == 5

    def test_work_item_stream_budget_backstop(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        monkeypatch.setattr(wrapper_module, '_WORK_ITEM_STREAM_BUDGET_BYTES', 2 * _MB)
        html = "".join(f'<img src="{_org_img(f"guid{i}")}">' for i in range(3))
        wrapper = self._wrapper_with_fields({"System.Description": html})
        streams = {f"guid{i}": _stream(b"x" * (3 * _MB // 2), chunk_size=_MB) for i in range(3)}
        wrapper._client.get_attachment_content.side_effect = _router(streams)
        result = wrapper.get_work_item(id=7, parse_attachments=True, process_images=True)
        assert isinstance(result, dict)
        descriptions = _descriptions_in(result["System.Description"])
        assert descriptions[:2] == ["ok", "ok"]
        assert descriptions[2] == "[image skipped: per-call download budget reached]"
        assert parser.call_count == 2
        assert streams["guid2"].consumed == 0

    def test_work_item_no_low_cap(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        html = "".join(f'<img src="{_org_img(f"guid{i}")}">' for i in range(25))
        wrapper = self._wrapper_with_fields({"System.Description": html})
        wrapper._client.get_attachment_content.side_effect = _router(
            {f"guid{i}": _stream(_TINY_PNG) for i in range(25)})
        result = wrapper.get_work_item(id=7, parse_attachments=True, process_images=True)
        assert parser.call_count == 25
        assert _descriptions_in(result["System.Description"]) == ["ok"] * 25
        assert "[image skipped" not in result["System.Description"]

    def test_work_item_image_ceiling_backstop(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        monkeypatch.setattr(wrapper_module, '_WORK_ITEM_IMAGE_CEILING', 3)
        html = "".join(f'<img src="{_org_img(f"guid{i}")}">' for i in range(5))
        wrapper = self._wrapper_with_fields({"System.Description": html})
        wrapper._client.get_attachment_content.side_effect = _router(
            {f"guid{i}": _stream(_TINY_PNG) for i in range(5)})
        result = wrapper.get_work_item(id=7, parse_attachments=True, process_images=True)
        assert isinstance(result, dict)
        assert parser.call_count == 3
        descriptions = _descriptions_in(result["System.Description"])
        assert descriptions[:3] == ["ok"] * 3
        assert descriptions[3:] == ["[image skipped: per-call image limit of 3 reached]"] * 2

    def test_work_item_dedup(self, monkeypatch):
        parser = _fake_parser("ok")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        url = _org_img()
        wrapper = self._wrapper_with_fields({
            "System.Description": f'<img src="{url}">',
            "Microsoft.VSTS.TCM.ReproSteps": f'<img src="{url}">',
        })
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        result = wrapper.get_work_item(id=7, parse_attachments=True, process_images=True)
        assert wrapper._client.get_attachment_content.call_count == 1
        assert _descriptions_in(result["System.Description"]) == ["ok"]
        assert _descriptions_in(result["Microsoft.VSTS.TCM.ReproSteps"]) == ["ok"]

    def test_relations_loop_untouched(self, monkeypatch):
        parser = _fake_parser("pdf text")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        relation = SimpleNamespace(as_dict=lambda: {
            "rel": "AttachedFile",
            "url": _org_img("guid9", "spec.pdf"),
            "attributes": {"name": "spec.pdf"},
        })
        wrapper = self._wrapper_with_fields({"System.Title": "plain"}, relations=[relation])
        wrapper._client.get_attachment_content.side_effect = _router({"guid9": _stream(b"%PDF-")})
        result = wrapper.get_work_item(
            id=7, expand="all", parse_attachments=True, process_images=True)
        assert result["relations"][0]["content"] == "pdf text"
        assert parser.call_args.kwargs["file_name"] == "spec.pdf"


def _indexer_wrapper(fields):
    wrapper = _make_wrapper()
    object.__setattr__(wrapper, '_index_process_images', True)
    wrapper._client.get_work_item.return_value = SimpleNamespace(id=7, fields=fields, relations=None)
    return wrapper


class TestIndexerImagePass:

    def test_description_embedded_in_payload(self, monkeypatch):
        parser = _fake_parser("a red pixel")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _indexer_wrapper({"System.Description": f'<img src="{_org_img()}">'})
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        doc = wrapper._fetch_work_item_document(7)
        assert "a red pixel" in doc.page_content

    def test_tool_exception_not_spliced(self, monkeypatch):
        parser = MagicMock(return_value=ToolException("Not supported type of files entered."))
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _indexer_wrapper({"System.Description": f'<img src="{_org_img(name="scan.tiff")}">'})
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        doc = wrapper._fetch_work_item_document(7)
        assert "ToolException" not in doc.page_content
        assert "Not supported type" not in doc.page_content

    def test_failed_image_silent_in_payload_but_logged(self, monkeypatch, caplog):
        parser = _fake_parser()
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        wrapper = _indexer_wrapper({"System.Description": '<img src="https://evil.example.com/x.png">'})
        with caplog.at_level("WARNING", logger=wrapper_module.logger.name):
            doc = wrapper._fetch_work_item_document(7)
        assert "image unavailable" not in doc.page_content
        assert "image skipped" not in doc.page_content
        wrapper._client.get_attachment_content.assert_not_called()
        parser.assert_not_called()
        failure_logs = [r.getMessage() for r in caplog.records
                        if "image description failed" in r.getMessage()]
        assert len(failure_logs) == 1
        assert "https://evil.example.com/x.png" in failure_logs[0]
        assert "work item 7" in failure_logs[0]

    def test_field_scan_failure_skips_field_not_run(self, monkeypatch, caplog):
        parser = _fake_parser("a red pixel")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        real_soup = wrapper_module.BeautifulSoup

        class _PoisonSoup:
            def find_all(self, *args, **kwargs):
                raise RuntimeError("poisoned field")

        monkeypatch.setattr(
            wrapper_module, 'BeautifulSoup',
            lambda value, features: _PoisonSoup() if "poison" in value else real_soup(value, features))
        wrapper = _indexer_wrapper({
            "Custom.Poison": '<img src="poison">',
            "System.Description": f'<img src="{_org_img()}">',
        })
        object.__setattr__(wrapper, '_index_sanitize', False)
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        with caplog.at_level("WARNING", logger=wrapper_module.logger.name):
            doc = wrapper._fetch_work_item_document(7)
        assert "a red pixel" in doc.page_content
        assert any("process_images pass failed" in r.getMessage() and "Custom.Poison" in r.getMessage()
                   for r in caplog.records)

    def test_all_note_kinds_are_image_notes(self, monkeypatch):
        monkeypatch.setattr(
            wrapper_module, 'parse_file_content',
            MagicMock(side_effect=ToolException("Not supported type of files entered.")))
        wrapper = _make_wrapper()
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        notes = {
            "external": wrapper._describe_attachment_safe("https://evil.example.com/x.png"),
            "no_name": wrapper._describe_attachment_safe(f"{ORG_URL}/Proj/_apis/wit/attachments/guid1"),
            "parser": wrapper._describe_attachment_safe(_org_img()),
            "limit": wrapper._describe_image_urls([_org_img(), _org_img("guid2")], None, max_images=1)[
                _org_img("guid2")],
        }
        for kind, note in notes.items():
            assert isinstance(note, wrapper_module._ImageNote), kind

    def test_repeated_url_single_fetch(self, monkeypatch):
        parser = _fake_parser("a red pixel")
        monkeypatch.setattr(wrapper_module, 'parse_file_content', parser)
        url = _org_img()
        wrapper = _indexer_wrapper({
            "System.Description": f'<img src="{url}">',
            "Custom.ReproSteps": f'<img src="{url}">',
        })
        wrapper._client.get_attachment_content.side_effect = _router({"guid1": _stream(_TINY_PNG)})
        doc = wrapper._fetch_work_item_document(7)
        assert wrapper._client.get_attachment_content.call_count == 1
        assert doc.page_content.count("a red pixel") == 2


class TestRegistration:

    def test_get_image_by_url_registered(self):
        tools = _make_wrapper().get_available_tools()
        entry = next(tool for tool in tools if tool["name"] == "get_image_by_url")
        assert entry["args_schema"] is ADOGetImageByUrl

    def test_get_comments_schema_defaults(self):
        assert ADOGetComments.model_fields["process_images"].default is False
        assert ADOGetComments.model_fields["image_description_prompt"].default is None

    def test_get_image_by_url_docstring_budget(self):
        assert len(AzureDevOpsApiWrapper.get_image_by_url.__doc__) < 800
