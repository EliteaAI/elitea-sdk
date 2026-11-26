"""delete_page_by_path must send the wiki's branch: ADO rejects code wiki deletes by path without it (#7766)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.ado.wiki.ado_wrapper import AzureDevOpsApiWrapper, DeletePageByPathInput


def _wrapper(wiki):
    w = AzureDevOpsApiWrapper.model_construct(organization_url="https://dev.azure.com/x", project="p",
                                              default_wiki_identifier=None)
    object.__setattr__(w, "_client", MagicMock())
    w._client.get_wiki.return_value = wiki
    return w


def _sent_version(w):
    vd = w._client.delete_page.call_args.kwargs["version_descriptor"]
    return vd.version, vd.version_type


CODE_WIKI = SimpleNamespace(type="codeWiki", versions=[SimpleNamespace(version="main")], mapped_path="/")
PROJECT_WIKI = SimpleNamespace(type="projectWiki", versions=[SimpleNamespace(version="wikiMaster")])


def test_code_wiki_uses_published_branch():
    w = _wrapper(CODE_WIKI)
    w.delete_page_by_path(wiki_identified="Code_wiki", page_name="/Page")
    w._client.get_wiki.assert_called_once_with(project="p", wiki_identifier="Code_wiki")
    assert w._client.delete_page.call_args.kwargs["path"] == "/Page"
    assert _sent_version(w) == ("main", "branch")


def test_project_wiki_uses_wikimaster():
    w = _wrapper(PROJECT_WIKI)
    w.delete_page_by_path(wiki_identified="p.wiki", page_name="/Page")
    assert _sent_version(w) == ("wikiMaster", "branch")


def test_code_wiki_without_versions_falls_back_to_wikimaster():
    w = _wrapper(SimpleNamespace(type="codeWiki", versions=[]))
    w.delete_page_by_path(wiki_identified="Code_wiki", page_name="/Page")
    assert _sent_version(w) == ("wikiMaster", "branch")


def test_explicit_version_overrides_published_branch():
    w = _wrapper(CODE_WIKI)
    w.delete_page_by_path(wiki_identified="Code_wiki", page_name="/Page",
                          version_identifier="feature/x", version_type="branch")
    w._client.get_wiki.assert_not_called()
    assert _sent_version(w) == ("feature/x", "branch")


def test_error_names_the_branch_used():
    w = _wrapper(CODE_WIKI)
    w._client.delete_page.side_effect = Exception("boom")
    with pytest.raises(ToolException, match="on branch 'main'.*boom"):
        w.delete_page_by_path(wiki_identified="Code_wiki", page_name="/Page")


def test_schema_version_fields_are_optional():
    args = DeletePageByPathInput(page_name="/Page")
    assert args.version_identifier is None and args.version_type == "branch"
