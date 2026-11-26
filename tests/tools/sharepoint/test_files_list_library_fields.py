"""get_files_list adds the library list columns of each file when library_lists is set.

The wrapper is the real SharepointApiWrapper; only the backend listing and the Graph
list endpoints are faked. Every tenant, site and field here is invented by the test.
"""
import logging

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.sharepoint import library_metadata as lm
from elitea_sdk.tools.sharepoint.api_wrapper import GetFiles
from tests.tools.sharepoint.test_library_mode import (
    LIB,
    LIB_ID,
    P1,
    P2,
    FakeGraph,
    _wrapper,
    graph_file,
    item,
    rest_file,
)

MAPPING = ["_Category=category"]
JOIN_KEYS = {"web_url", "file_name", "metadata_status", "metadata_match", "library_list",
             "library_item_id", "library_digest"}


def _lists(*rows):
    return {LIB_ID: list(rows)}


def _files_list(w, **kwargs):
    return w.get_files_list(**kwargs)


def test_files_list_adds_the_mapped_column_as_category(monkeypatch):
    graph = FakeGraph(_lists(item(1, P1, _Category="External Origin")))
    w = _wrapper(monkeypatch, [graph_file(P1)], graph)

    rows = _files_list(w, library_lists=[LIB], metadata_fields=MAPPING)

    assert rows[0]["category"] == "External Origin"
    assert rows[0]["metadata_status"] == lm.STATUS_OK
    assert rows[0]["metadata_match"] == "path"
    assert rows[0]["library_list"] == LIB
    assert rows[0]["Name"] == "Travel Rules.pdf"


def test_files_list_copies_only_the_mapped_column(monkeypatch):
    graph = FakeGraph(_lists(item(1, P1, _Category="Embedded", SubjectDoc="Leave", Owner0="x")))
    original = graph_file(P1)
    w = _wrapper(monkeypatch, [original], graph)

    rows = _files_list(w, library_lists=[LIB], metadata_fields=MAPPING)

    assert set(rows[0]) == set(original) | {"category"} | JOIN_KEYS
    assert "subject_doc" not in rows[0] and "SubjectDoc" not in rows[0]


def test_files_list_does_not_change_the_rows_of_the_backend(monkeypatch):
    graph = FakeGraph(_lists(item(1, P1, _Category="Embedded")))
    backend_rows = [graph_file(P1)]
    w = _wrapper(monkeypatch, backend_rows, graph)

    _files_list(w, library_lists=[LIB], metadata_fields=MAPPING)

    assert backend_rows == [graph_file(P1)]


@pytest.mark.parametrize("library_lists", [None, [], ""])
def test_files_list_without_library_lists_returns_rows_unchanged(monkeypatch, library_lists):
    graph = FakeGraph(_lists(item(1, P1, _Category="Embedded")))
    files = [graph_file(P1), rest_file(P2)]
    w = _wrapper(monkeypatch, files, graph)

    rows = _files_list(w, library_lists=library_lists, metadata_fields=MAPPING)

    assert rows == [graph_file(P1), rest_file(P2)]
    assert graph.calls == []


def test_files_list_default_call_keeps_the_old_signature(monkeypatch):
    graph = FakeGraph(_lists(item(1, P1, _Category="Embedded")))
    w = _wrapper(monkeypatch, [graph_file(P1)], graph)

    assert w.get_files_list("Handbook Library", 10) == [graph_file(P1)]
    assert graph.calls == []


@pytest.mark.parametrize("case", ["no item", "ambiguous name"])
def test_files_list_file_without_list_item_has_no_category(monkeypatch, case):
    if case == "no item":
        graph = FakeGraph(_lists(item(1, P2, _Category="Other")))
        file = graph_file(P1)
        expected = lm.STATUS_NO_METADATA
    else:
        graph = FakeGraph(_lists(
            item(1, "/sites/handbook/Handbook Library/a/Same.pdf", _Category="X"),
            item(2, "/sites/handbook/Handbook Library/b/Same.pdf", _Category="Y")))
        file = graph_file("/sites/handbook/Handbook Library/c/Same.pdf")
        expected = lm.STATUS_AMBIGUOUS
    w = _wrapper(monkeypatch, [file], graph)

    rows = _files_list(w, library_lists=[LIB], metadata_fields=MAPPING)

    assert rows[0]["metadata_status"] == expected
    assert "category" not in rows[0]


@pytest.mark.parametrize("params, message", [
    ({"library_lists": ["Missing Library"], "metadata_fields": MAPPING},
     "Library list 'Missing Library' was not found on the site."),
    ({"library_lists": [LIB], "metadata_fields": ["=category"]}, "no source field name"),
    ({"library_lists": [LIB], "metadata_fields": ["_Category="]}, "maps to an empty name"),
    ({"library_lists": [f"List {n}" for n in range(11)], "metadata_fields": MAPPING}, "At most 10"),
    ({"library_lists": [LIB], "metadata_fields": [7]}, "must be text"),
    ({"library_lists": [LIB, 5], "metadata_fields": MAPPING}, "must be text"),
])
def test_files_list_bad_library_settings_raise_a_tool_error(monkeypatch, params, message):
    graph = FakeGraph(_lists(item(1, P1, _Category="Embedded")))
    w = _wrapper(monkeypatch, [graph_file(P1)], graph)

    with pytest.raises(ToolException, match=message):
        _files_list(w, **params)


def test_files_list_column_that_matches_no_field_adds_no_category(monkeypatch, caplog):
    graph = FakeGraph(_lists(item(1, P1, _Category="Embedded")))
    w = _wrapper(monkeypatch, [graph_file(P1)], graph)

    with caplog.at_level(logging.WARNING):
        rows = _files_list(w, library_lists=[LIB], metadata_fields=["NoSuchColumn=category"])

    assert rows[0]["metadata_status"] == lm.STATUS_OK
    assert "category" not in rows[0]
    assert "match no field" in caplog.text
    assert "NoSuchColumn" not in caplog.text


def test_files_list_runs_the_extension_filter_before_the_join(monkeypatch):
    graph = FakeGraph(_lists(
        item(1, P1, _Category="Embedded"),
        item(2, P2, _Category="Work Product")))
    w = _wrapper(monkeypatch, [], graph)
    seen = []

    def backend_listing(*args, **kwargs):
        seen.append(args)
        return [graph_file(P1)]

    monkeypatch.setattr(w._backend, "get_files_list", backend_listing)

    rows = _files_list(w, include_extensions=["pdf"], library_lists=[LIB], metadata_fields=MAPPING)

    assert seen[0][3] == ["pdf"]
    assert [r["Name"] for r in rows] == ["Travel Rules.pdf"]
    assert rows[0]["category"] == "Embedded"


@pytest.mark.parametrize("value, expected", [
    ("Qualität Übersicht 質量", "Qualität Übersicht 質量"),
    (["People", "Remote"], ["People", "Remote"]),
    (7, 7),
])
def test_files_list_keeps_the_type_of_the_column_value(monkeypatch, value, expected):
    graph = FakeGraph(_lists(item(1, P1, _Category=value)))
    w = _wrapper(monkeypatch, [graph_file(P1)], graph)

    rows = _files_list(w, library_lists=[LIB], metadata_fields=MAPPING)

    assert rows[0]["category"] == expected


def test_files_list_with_no_files_still_reads_nothing_into_rows(monkeypatch):
    graph = FakeGraph(_lists(item(1, P1, _Category="Embedded")))
    w = _wrapper(monkeypatch, [], graph)

    assert _files_list(w, library_lists=[LIB], metadata_fields=MAPPING) == []


def test_get_files_schema_offers_library_fields():
    fields = GetFiles.model_fields

    assert fields["library_lists"].default is None
    assert fields["metadata_fields"].default == []
    parsed = GetFiles(library_lists=[LIB], metadata_fields=MAPPING)
    assert parsed.library_lists == [LIB] and parsed.metadata_fields == MAPPING
    assert GetFiles().library_lists is None


def test_get_files_tool_is_listed_with_the_new_fields(monkeypatch):
    w = _wrapper(monkeypatch, [], FakeGraph({}))

    tool = next(t for t in w.get_available_tools() if t["name"] == "get_files_list")

    assert {"library_lists", "metadata_fields"} <= set(tool["args_schema"].model_fields)
    assert "library_lists" in tool["description"]
