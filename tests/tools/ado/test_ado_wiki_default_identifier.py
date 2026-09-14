"""Default Wiki Identifier should be applied consistently across tool usage and indexing.

Regular tools (get_wiki, get_wiki_page, ...) already resolve `default_wiki_identifier`
via `_resolve_wiki_identifier` and advertise it in their tool descriptions. Indexing
(`_base_loader`, called by both manual and scheduled `index_data` runs) resolves the
same way, but the `index_data` tool's `wiki_identifier` parameter did not previously
mention the configured default, leaving agents no signal that it's safe to omit it.
"""
from elitea_sdk.tools.ado.wiki.ado_wrapper import AzureDevOpsApiWrapper as WikiWrapper


def _make_wiki(default_wiki_identifier=None) -> WikiWrapper:
    w = WikiWrapper.model_construct(
        organization_url="https://dev.azure.com/x",
        project="p",
        default_wiki_identifier=default_wiki_identifier,
    )
    w._init_indexing_stats = lambda: None
    return w


def test_resolve_wiki_identifier_falls_back_to_default():
    w = _make_wiki(default_wiki_identifier="MyDefaultWiki")
    assert w._resolve_wiki_identifier(None) == "MyDefaultWiki"


def test_resolve_wiki_identifier_explicit_param_overrides_default():
    w = _make_wiki(default_wiki_identifier="MyDefaultWiki")
    assert w._resolve_wiki_identifier("OtherWiki") == "OtherWiki"


def test_resolve_wiki_identifier_raises_without_default():
    w = _make_wiki(default_wiki_identifier=None)
    try:
        w._resolve_wiki_identifier(None)
        assert False, "expected ToolException"
    except Exception as e:
        assert "default_wiki_identifier" in str(e)


def test_base_loader_uses_default_when_identifier_omitted():
    w = _make_wiki(default_wiki_identifier="MyDefaultWiki")
    w._iter_wiki_pages = lambda _wid: iter(())
    list(w._base_loader())
    assert w._index_wiki_identifier == "MyDefaultWiki"


def test_index_data_param_description_advertises_configured_default():
    w = _make_wiki(default_wiki_identifier="MyDefaultWiki")
    params = w._index_tool_params()
    _type, field = params["wiki_identifier"]
    assert "MyDefaultWiki" in field.description


def test_index_data_param_description_silent_without_default():
    w = _make_wiki(default_wiki_identifier=None)
    params = w._index_tool_params()
    _type, field = params["wiki_identifier"]
    assert "Default wiki:" not in field.description
