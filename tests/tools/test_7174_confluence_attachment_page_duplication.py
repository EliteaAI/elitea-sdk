"""
Issue #7174: with include_attachments, Confluence attachment text was also appended to the page body.

The page loader extracted every attachment itself and joined the text onto the page, which was then stored
as one chunk per page, outside chunking_config and the include/skip extension filters. Attachments must reach
the index only as their own dependent documents, page bodies must be chunked, and a failed attachment
download must be reported instead of indexed as text.
"""
from unittest.mock import MagicMock

import pytest
from langchain_community.document_loaders.confluence import ContentFormat
from langchain_core.documents import Document

from elitea_sdk.runtime.utils.utils import IndexerKeywords

DEPENDENT_DOCS = IndexerKeywords.DEPENDENT_DOCS.value
from elitea_sdk.tools.base_indexer_toolkit import snapshot_loader_metadata
from elitea_sdk.tools.confluence.api_wrapper import BLANK_PAGE_KEY, PAGE_SOURCE_KEY, ConfluenceAPIWrapper

PAGE_BODY = 'Release checklist for the payments service'
ATTACHMENT_TEXT = 'ONLY-IN-THE-PDF quarterly figures'


def make_wrapper(client):
    instance = ConfluenceAPIWrapper.model_construct(
        space=None, client=client, base_url='https://example.atlassian.net', llm=None, limit=5, max_pages=10,
        api_version='1', number_of_retries=1, min_retry_seconds=0, max_retry_seconds=0,
        content_format=ContentFormat.VIEW, labels=[],
    )
    instance._errors = None
    instance._init_indexing_stats()
    return instance


def confluence_client():
    client = MagicMock()
    client.url = 'https://example.atlassian.net/wiki'
    client.get_page_by_id.return_value = {
        'id': '101', 'title': 'Payments', 'status': 'current', 'type': 'page',
        'body': {'view': {'value': f'<p>{PAGE_BODY}</p>'}},
        'version': {'when': '2026-10-01T00:00:00.000Z'},
        'space': {'key': 'PAY'},
        '_links': {'webui': '/spaces/PAY/pages/101'},
    }
    client.get_attachments_from_content.return_value = {'results': [{
        'id': 'att1', 'title': 'figures.pdf',
        'metadata': {'mediaType': 'application/pdf'},
        '_links': {'download': '/download/attachments/101/figures.pdf'},
    }]}
    client.request.return_value = MagicMock(status_code=200, content=ATTACHMENT_TEXT.encode())
    return client


def page_document(wrapper, **index_kwargs):
    documents = list(wrapper._base_loader(page_ids=['101'], include_attachments=True,
                                          include_restricted_content=True, **index_kwargs))
    assert len(documents) == 1
    return documents[0]


class TestPageBodyExcludesAttachments:
    @pytest.fixture(autouse=True)
    def ocr_returns_attachment_text(self, monkeypatch):
        monkeypatch.setattr('elitea_sdk.tools.confluence.loader.EliteAConfluenceLoader.process_pdf',
                            lambda self, link, ocr_languages=None: ATTACHMENT_TEXT)

    def test_attachment_text_is_not_appended_to_the_page(self):
        document = page_document(make_wrapper(confluence_client()))

        assert PAGE_BODY in document.page_content
        assert ATTACHMENT_TEXT not in document.page_content

    def test_skipped_extension_is_not_searchable_through_the_page(self):
        wrapper = make_wrapper(confluence_client())
        document = page_document(wrapper, skip_extensions=['*.pdf'])

        attachments = list(wrapper._process_document(document))

        assert ATTACHMENT_TEXT not in document.page_content
        assert attachments == []

    def test_attachment_is_still_indexed_as_its_own_document(self):
        wrapper = make_wrapper(confluence_client())
        document = page_document(wrapper)

        [attachment] = wrapper._process_document(document)

        assert attachment.metadata['id'] == 'att1'
        assert attachment.metadata[IndexerKeywords.CONTENT_FILE_NAME.value] == 'figures.pdf'
        assert attachment.metadata[IndexerKeywords.CONTENT_IN_BYTES.value] == ATTACHMENT_TEXT.encode()


class TestPageBodyIsChunked:
    @staticmethod
    def long_page():
        sections = [f'## Section {n}\n\n' + ' '.join(f'word{n}_{i}' for i in range(150)) for n in range(8)]
        return Document(page_content='\n\n'.join(sections), metadata={'id': '101', 'updated_on': 'v1'})

    def chunks(self, wrapper, document, chunking_tool, max_tokens=256):
        wrapper._chunking_tool = chunking_tool
        wrapper.embeddings = None
        return list(wrapper._clean_metadata(wrapper._apply_loaders_chunkers(
            wrapper._extend_data(iter([document])), chunking_tool, {'.txt': {'max_tokens': max_tokens}})))

    def test_index_defaults_to_the_text_chunker(self):
        field = make_wrapper(MagicMock())._index_tool_params()['chunking_tool'][1]

        assert field.default == 'text'

    def test_page_chunks_keep_the_page_url_as_source(self):
        page = self.long_page()
        page.metadata['source'] = 'https://example.atlassian.net/wiki/spaces/PAY/pages/101'

        chunks = self.chunks(make_wrapper(MagicMock()), page, 'text')

        assert {chunk.metadata['source'] for chunk in chunks} == {page.metadata['source']}
        assert all(PAGE_SOURCE_KEY not in chunk.metadata for chunk in chunks)

    def test_text_chunking_tool_splits_a_long_page(self):
        chunks = self.chunks(make_wrapper(MagicMock()), self.long_page(), 'text')

        assert len(chunks) > 1
        assert all(chunk.metadata['id'] == '101' for chunk in chunks)
        assert 'word7_149' in chunks[-1].page_content

    def test_page_chunks_follow_the_text_max_tokens(self):
        small = self.chunks(make_wrapper(MagicMock()), self.long_page(), 'text', max_tokens=64)
        large = self.chunks(make_wrapper(MagicMock()), self.long_page(), 'text', max_tokens=2000)

        assert len(small) > len(large)

    def test_default_chunking_keeps_indentation_and_blank_lines(self):
        body = "def f(x):\n    if x:\n        return 1\n\nPara one\n\nPara two\n\tbuild:\n\t\tgo build\n"
        page = Document(page_content=body, metadata={'id': '101', 'updated_on': 'v1'})
        wrapper = make_wrapper(MagicMock())
        default_chunking_tool = wrapper._index_tool_params()['chunking_tool'][1].default

        [chunk] = self.chunks(wrapper, page, default_chunking_tool)

        assert chunk.page_content == body

    def test_without_a_chunking_tool_the_page_stays_whole(self):
        page = self.long_page()

        [chunk] = self.chunks(make_wrapper(MagicMock()), page, '')

        assert chunk.page_content == page.page_content

    def test_page_without_body_is_not_reported_as_empty(self):
        wrapper = make_wrapper(MagicMock())
        page = Document(page_content='', metadata={'id': '101', 'title': 'Attachments only', 'updated_on': 'v1'})

        self.chunks(wrapper, page, 'text')

        assert wrapper.get_indexing_stats().files_skipped_empty == set()


class TestFailedAttachmentDownload:
    def page_with_attachment(self, wrapper):
        return Document(page_content=PAGE_BODY, metadata={'id': '101', '_attachments_data': [{
            'id': 'att1', 'title': 'figures.pdf',
            'metadata': {'mediaType': 'application/pdf'},
            '_links': {'download': '/download/attachments/101/figures.pdf'},
        }]})

    def process(self, client):
        wrapper = make_wrapper(client)
        wrapper._index_include_attachments = True
        wrapper._include_extensions = []
        wrapper._skip_extensions = []
        return wrapper, list(wrapper._process_document(self.page_with_attachment(wrapper)))

    def test_http_error_is_reported_not_indexed(self):
        client = confluence_client()
        client.request.return_value = MagicMock(status_code=403, content=b'')

        wrapper, documents = self.process(client)

        assert documents == []
        assert wrapper.get_indexing_stats().dependent_items_skipped == {'figures.pdf'}

    def test_request_exception_is_reported_not_indexed(self):
        client = confluence_client()
        client.request.side_effect = ConnectionError('connection reset')

        wrapper, documents = self.process(client)

        assert documents == []
        assert wrapper.get_indexing_stats().dependent_items_skipped == {'figures.pdf'}

    def test_one_failed_download_does_not_drop_the_others(self):
        client = confluence_client()
        client.request.side_effect = [MagicMock(status_code=500, content=b''),
                                      MagicMock(status_code=200, content=b'second')]
        wrapper = make_wrapper(client)
        wrapper._index_include_attachments = True
        wrapper._include_extensions = []
        wrapper._skip_extensions = []
        page = self.page_with_attachment(wrapper)
        page.metadata['_attachments_data'].append({
            'id': 'att2', 'title': 'notes.txt', 'metadata': {'mediaType': 'text/plain'},
            '_links': {'download': '/download/attachments/101/notes.txt'},
        })

        [document] = wrapper._process_document(page)

        assert document.metadata['id'] == 'att2'
        assert wrapper.get_indexing_stats().dependent_items_skipped == {'figures.pdf'}


class TestFailedDownloadDoesNotForceReindex:
    def diverged_on_next_run(self, failure):
        client = confluence_client()
        if isinstance(failure, Exception):
            client.request.side_effect = failure
        else:
            client.request.return_value = MagicMock(status_code=failure, content=b'')
        wrapper = make_wrapper(client)
        wrapper._index_include_attachments = True
        wrapper._include_extensions = []
        wrapper._skip_extensions = []
        listing = client.get_attachments_from_content.return_value['results']
        page = Document(page_content=PAGE_BODY, metadata={'id': '101', '_attachments_data': list(listing)})
        *_, stored_page = wrapper._emit_document_with_deps(page, list(wrapper._process_document(page)))
        stored = {DEPENDENT_DOCS: (stored_page.metadata.get(DEPENDENT_DOCS) or '').split(';')}
        next_run = Document(page_content=PAGE_BODY, metadata={'id': '101', '_attachments_data': list(listing)})
        return wrapper._dependents_diverged(next_run, stored)

    @pytest.mark.parametrize('failure', [400, 401, 403, 404, 410, 413, 429, 500, 502, 503, 504,
                                         ConnectionError('reset')])
    def test_failed_download_leaves_an_unchanged_page_alone(self, failure):
        assert self.diverged_on_next_run(failure) is False


class TestPageWithoutBody:
    @staticmethod
    def holder_client(title='report.pdf'):
        client = confluence_client()
        client.get_page_by_id.return_value['body']['view']['value'] = ''
        client.get_attachments_from_content.return_value['results'][0]['title'] = title
        return client

    @staticmethod
    def indexed_page_chunks(wrapper, document):
        wrapper._chunking_tool = 'text'
        wrapper.embeddings = None
        chunks = wrapper._clean_metadata(wrapper._apply_loaders_chunkers(
            wrapper._collect_dependencies(wrapper._extend_data(iter([document]))), 'text', {}))
        return [chunk for chunk in chunks if chunk.metadata.get('id') == '101']

    def test_page_with_an_indexable_attachment_keeps_a_row(self):
        wrapper = make_wrapper(self.holder_client())

        [page_chunk] = self.indexed_page_chunks(wrapper, page_document(wrapper))

        assert page_chunk.page_content == 'Payments'

    def test_page_keeps_a_row_when_the_attachment_prefetch_fails(self):
        client = self.holder_client()
        listing = client.get_attachments_from_content.return_value
        client.get_attachments_from_content.side_effect = [ConnectionError('429 Too Many Requests'), listing]
        wrapper = make_wrapper(client)

        [page_chunk] = self.indexed_page_chunks(wrapper, page_document(wrapper))

        assert page_chunk.page_content == 'Payments'

    def test_page_with_a_whitespace_only_body_keeps_a_row(self):
        client = self.holder_client()
        client.get_page_by_id.return_value['body']['view']['value'] = '<p class="auto-cursor-target"><br /></p>'
        wrapper = make_wrapper(client)
        document = page_document(wrapper, keep_markdown_format=False, keep_newlines=True)

        [page_chunk] = self.indexed_page_chunks(wrapper, document)

        assert page_chunk.page_content == 'Payments'

    @pytest.mark.parametrize('invisible', ['&#8203;', '&#65279;', '&#8288;', '&#173;', '&#8204;'])
    def test_page_with_only_invisible_characters_keeps_a_row(self, invisible):
        client = self.holder_client()
        client.get_page_by_id.return_value['body']['view']['value'] = f'<p>{invisible}</p>'
        wrapper = make_wrapper(client)

        [page_chunk] = self.indexed_page_chunks(wrapper, page_document(wrapper))

        assert page_chunk.page_content == 'Payments'
        assert page_chunk.metadata[DEPENDENT_DOCS] == 'att1'

    @pytest.mark.parametrize('failure', [MagicMock(status_code=404, content=b''),
                                         MagicMock(status_code=503, content=b''), ConnectionError('reset')])
    def test_page_whose_downloads_all_fail_keeps_a_row_with_its_attachments(self, failure):
        client = self.holder_client()
        client.request.side_effect = [failure]
        wrapper = make_wrapper(client)

        [page_chunk] = self.indexed_page_chunks(wrapper, page_document(wrapper))

        assert page_chunk.page_content == 'Payments'
        assert page_chunk.metadata[DEPENDENT_DOCS] == 'att1'
        assert wrapper.get_indexing_stats().dependent_items_skipped == {'report.pdf'}

    def test_whitespace_page_whose_attachments_cannot_be_listed_is_failed(self):
        client = self.holder_client()
        client.get_page_by_id.return_value['body']['view']['value'] = '<p class="auto-cursor-target"><br /></p>'
        client.get_attachments_from_content.side_effect = ConnectionError('429 Too Many Requests')
        wrapper = make_wrapper(client)
        document = page_document(wrapper, keep_markdown_format=False, keep_newlines=True)

        assert self.indexed_page_chunks(wrapper, document) == []
        assert wrapper.get_indexing_stats().runtime_skipped_error == {'Payments'}

    def test_pipeline_retry_starts_from_the_blank_body(self):
        client = self.holder_client()
        listing = client.get_attachments_from_content.return_value
        client.get_attachments_from_content.side_effect = [
            ConnectionError('prefetch reset'), listing, ConnectionError('retry reset')]
        wrapper = make_wrapper(client)
        document = page_document(wrapper)
        snapshot = snapshot_loader_metadata(document)
        list(wrapper._collect_dependencies(wrapper._extend_data(iter([document]))))
        assert document.page_content == 'Payments'

        document.metadata = snapshot
        page_chunks = self.indexed_page_chunks(wrapper, document)

        assert page_chunks == []
        assert wrapper.get_indexing_stats().runtime_skipped_error == {'Payments'}

    def test_blank_page_marker_is_not_stored(self):
        wrapper = make_wrapper(self.holder_client())

        [page_chunk] = self.indexed_page_chunks(wrapper, page_document(wrapper))

        assert BLANK_PAGE_KEY not in page_chunk.metadata

    def test_page_whose_attachments_are_all_filtered_stays_empty(self):
        wrapper = make_wrapper(self.holder_client())

        assert self.indexed_page_chunks(wrapper, page_document(wrapper, skip_extensions=['*.pdf'])) == []
