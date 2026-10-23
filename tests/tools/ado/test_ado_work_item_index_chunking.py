"""#7175: ADO work item indexing — markdown chunking of the work item body,
include_attachments toggle with extension filters, and attachment download
failures reported as skipped attachments instead of failing the work item."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.ado.work_item.ado_wrapper import AzureDevOpsApiWrapper
from elitea_sdk.tools.chunkers.utils import tiktoken_length

ORG_URL = "https://dev.azure.com/MyOrg"
BYTES = IndexerKeywords.CONTENT_IN_BYTES.value
FILE_NAME = IndexerKeywords.CONTENT_FILE_NAME.value


def _relation(att_id, name):
    return SimpleNamespace(rel='AttachedFile', url=f"{ORG_URL}/_apis/wit/attachments/{att_id}",
                           attributes={'name': name})


def _wrapper(fields, relations=None):
    wrapper = AzureDevOpsApiWrapper.model_construct(
        organization_url=ORG_URL, project="Proj", token=SecretStr("x"), limit=5)
    wrapper.llm = None
    wrapper.embeddings = None
    object.__setattr__(wrapper, '_client', MagicMock())
    wrapper._client.get_work_item.return_value = SimpleNamespace(id=7, fields=fields, relations=relations)
    wrapper._client.query_by_wiql.return_value = SimpleNamespace(
        work_items=[SimpleNamespace(id=7)], work_item_relations=None)
    object.__setattr__(wrapper, '_track_skipped_attachment', MagicMock())
    return wrapper


def _load(wrapper, **kwargs):
    return list(wrapper._base_loader(wiql="SELECT [System.Id] FROM workitems", **kwargs))


def _long_fields():
    paragraph = "The service returns HTTP 500 when the payload contains nested arrays. " * 8
    return {
        "System.Title": "Checkout fails",
        "System.State": "Active",
        "System.Description": "\n".join([paragraph] * 12),
        "Microsoft.VSTS.TCM.ReproSteps": "\n".join(f"{i}. step {paragraph}" for i in range(10)),
    }


class TestBodyChunking:

    def test_markdown_rendering(self):
        doc, = _load(_wrapper(_long_fields()), chunking_tool='markdown')
        md = doc.metadata[BYTES].decode('utf-8')
        assert md.startswith("# 7: Checkout fails\n")
        assert "- **System.State:** Active" in md
        assert "## System.Description\n" in md
        assert "## Microsoft.VSTS.TCM.ReproSteps\n" in md
        assert FILE_NAME not in doc.metadata

    def test_empty_chunking_tool_keeps_json(self):
        doc, = _load(_wrapper(_long_fields()), chunking_tool='')
        assert json.loads(doc.page_content)["System.Title"] == "Checkout fails"
        assert BYTES not in doc.metadata

    @pytest.mark.parametrize("max_tokens", [128, 512])
    def test_long_work_item_split_at_max_tokens(self, max_tokens):
        wrapper = _wrapper(_long_fields())
        doc, = _load(wrapper, chunking_tool='markdown')
        chunks = list(wrapper._apply_loaders_chunkers(
            iter([doc]), 'markdown', {'.md': {'max_tokens': max_tokens}}))
        assert len(chunks) > 1
        assert max(tiktoken_length(c.page_content) for c in chunks) <= max_tokens
        assert [c.metadata['chunk_id'] for c in chunks] == list(range(1, len(chunks) + 1))
        assert all(c.metadata['id'] == '7' for c in chunks)

    def test_tool_schema_exposes_chunking_tool(self):
        params = _wrapper({})._index_tool_params()
        _type, field = params["chunking_tool"]
        assert field.default == 'markdown'
        assert params["include_attachments"][1].default is False


class TestAttachments:

    def _doc(self, wrapper, **kwargs):
        doc, = _load(wrapper, **kwargs)
        return doc

    def test_not_downloaded_by_default(self):
        wrapper = _wrapper({"System.Title": "t"}, [_relation("a1", "spec.pdf")])
        deps = list(wrapper._process_document(self._doc(wrapper)))
        assert deps == []
        wrapper._client.get_attachment_content.assert_not_called()

    def test_extension_filters(self):
        wrapper = _wrapper({"System.Title": "t"}, [
            _relation("a1", "spec.PDF"), _relation("a2", "dump.zip"), _relation("a3", "notes.txt")])
        wrapper._client.get_attachment_content.side_effect = lambda id, download: iter([b"data"])
        doc = self._doc(wrapper, include_attachments=True,
                        include_extensions=["*.pdf", "*.zip"], skip_extensions=["*.zip"])
        deps = list(wrapper._process_document(doc))
        assert [d.metadata['id'] for d in deps] == ["a1"]
        assert deps[0].metadata[FILE_NAME] == "spec.PDF"
        assert deps[0].metadata[BYTES] == b"data"
        skipped = {c.args[0]: c.kwargs['reason'] for c in wrapper._track_skipped_attachment.call_args_list}
        assert skipped == {"dump.zip": "filtered", "notes.txt": "filtered"}

    def test_failed_download_reported_not_raised(self):
        wrapper = _wrapper({"System.Title": "t"}, [_relation("bad", "a.pdf"), _relation("ok", "b.pdf")])

        def fetch(id, download):
            if id == "bad":
                raise RuntimeError("404")
            return iter([b"ok"])

        wrapper._client.get_attachment_content.side_effect = fetch
        deps = list(wrapper._process_document(self._doc(wrapper, include_attachments=True)))
        assert [d.metadata['id'] for d in deps] == ["ok"]
        wrapper._track_skipped_attachment.assert_called_once_with("a.pdf", reason="error")

    def test_oversized_and_empty_skipped(self, monkeypatch):
        from elitea_sdk.tools.ado.work_item import ado_wrapper as module
        monkeypatch.setattr(module, '_ATTACHMENT_STREAM_CEILING_BYTES', 4)
        wrapper = _wrapper({"System.Title": "t"}, [_relation("big", "big.pdf"), _relation("nil", "nil.pdf")])
        wrapper._client.get_attachment_content.side_effect = (
            lambda id, download: iter([b"12345"] if id == "big" else []))
        deps = list(wrapper._process_document(self._doc(wrapper, include_attachments=True)))
        assert deps == []
        skipped = {c.args[0]: c.kwargs['reason'] for c in wrapper._track_skipped_attachment.call_args_list}
        assert skipped == {"big.pdf": "too_large", "nil.pdf": "empty"}

    def test_dependents_diverged(self):
        wrapper = _wrapper({"System.Title": "t"}, [_relation("a1", "spec.pdf"), _relation("a2", "x.zip")])
        doc = self._doc(wrapper)
        assert wrapper._dependents_diverged(doc, {IndexerKeywords.DEPENDENT_DOCS.value: []}) is False
        doc = self._doc(wrapper, include_attachments=True, skip_extensions=["*.zip"])
        dep_key = IndexerKeywords.DEPENDENT_DOCS.value
        assert wrapper._dependents_diverged(doc, {dep_key: []}) is True
        assert wrapper._dependents_diverged(doc, {dep_key: ["a1"]}) is False
