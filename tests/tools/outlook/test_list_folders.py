"""Tests for Outlook list_folders (recursion, filtering, sorting, truncation) and folder path resolution (#6831)."""

from unittest.mock import MagicMock

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.outlook import graph_wrapper
from elitea_sdk.tools.outlook.graph_wrapper import OutlookGraphWrapper

ROOT = "https://graph.microsoft.com/v1.0/me/mailFolders"


def _folder(fid, name, parent="msgroot", children=0, total=1, unread=0):
    return {"id": fid, "displayName": name, "parentFolderId": parent, "childFolderCount": children,
            "totalItemCount": total, "unreadItemCount": unread}


TREE = {
    ROOT: [_folder("root-inbox", "Inbox", children=2, total=50, unread=5),
           _folder("root-sent", "Sent Items", total=10),
           _folder("root-arch", "Archive", children=1, total=900)],
    f"{ROOT}/root-inbox/childFolders": [_folder("inbox-proj", "Projects", "root-inbox", 1, total=30, unread=12),
                                        _folder("inbox-misc", "Misc", "root-inbox", total=2)],
    f"{ROOT}/inbox-proj/childFolders": [_folder("proj-2026", "2026", "inbox-proj", total=7, unread=1)],
    f"{ROOT}/root-arch/childFolders": [_folder("arch-2026", "2026", "root-arch", total=400)],
}
TREE[f"{ROOT}/inbox/childFolders"] = TREE[f"{ROOT}/root-inbox/childFolders"]
NODES = {f["id"]: f for folders in TREE.values() for f in folders}
NODES["inbox"] = NODES["root-inbox"]
NODES["msgfolderroot"] = {"id": "msgroot", "displayName": "Top of Information Store", "parentFolderId": None}


def _fake_get(url, params=None, prefer=None):
    if url in TREE:
        return {"value": TREE[url]}
    node_id = url.rsplit("/", 1)[-1]
    if url.startswith(f"{ROOT}/") and node_id in NODES:
        return NODES[node_id]
    return {"value": []}


@pytest.fixture
def wrapper():
    w = OutlookGraphWrapper(token="stub", scopes=[])
    w._get = MagicMock(side_effect=_fake_get)
    return w


def _paths(result):
    return [f["path"] for f in result["folders"]]


def _called(wrapper):
    return [c.args[0] for c in wrapper._get.call_args_list]


# --- listing -------------------------------------------------------------------------

def test_defaults_return_wrapper_with_all_levels_sorted_by_path(wrapper):
    result = wrapper.list_folders()
    assert _paths(result) == [
        "Archive", "Archive/2026", "Inbox", "Inbox/Misc", "Inbox/Projects", "Inbox/Projects/2026", "Sent Items",
    ]
    assert result["total"] == 7 and result["truncated"] is False and "hint" not in result
    nested = next(f for f in result["folders"] if f["path"] == "Inbox/Projects/2026")
    assert nested["id"] == "proj-2026" and nested["parentFolderId"] == "inbox-proj"


def test_leaf_folders_cost_no_child_request(wrapper):
    wrapper.list_folders()
    assert f"{ROOT}/root-sent/childFolders" not in _called(wrapper)
    assert len(_called(wrapper)) == 4


def test_default_limit_is_20_and_truncation_is_reported(wrapper, monkeypatch):
    many = [_folder(f"f{i}", f"Folder {i:03d}") for i in range(35)]
    wrapper._get = MagicMock(return_value={"value": many})
    result = wrapper.list_folders()
    assert len(result["folders"]) == 20
    assert result["total"] == 35 and result["truncated"] is True
    assert "Showing 20 of 35" in result["hint"]


def test_limit_can_be_raised(wrapper):
    wrapper._get = MagicMock(return_value={"value": [_folder(f"f{i}", f"F{i}") for i in range(35)]})
    result = wrapper.list_folders(limit=50)
    assert len(result["folders"]) == 35 and result["truncated"] is False


def test_scan_cap_marks_truncated_even_under_limit(wrapper, monkeypatch):
    monkeypatch.setattr(graph_wrapper, "_MAX_SCAN", 3)
    result = wrapper.list_folders(limit=100)
    assert result["truncated"] is True and "scanned" in result["hint"]


# --- depth / parent ------------------------------------------------------------------

def test_depth_one_lists_top_level_only_and_skips_child_requests(wrapper):
    result = wrapper.list_folders(depth=1)
    assert _paths(result) == ["Archive", "Inbox", "Sent Items"]
    assert _called(wrapper) == [ROOT]


def test_parent_by_path_lists_only_that_subtree_with_full_paths(wrapper):
    result = wrapper.list_folders(parent="Inbox/Projects")
    assert _paths(result) == ["Inbox/Projects/2026"]


def test_parent_by_well_known_name_derives_path_prefix_from_graph(wrapper):
    result = wrapper.list_folders(parent="inbox")
    assert _paths(result) == ["Inbox/Misc", "Inbox/Projects", "Inbox/Projects/2026"]


def test_parent_with_depth(wrapper):
    result = wrapper.list_folders(parent="Inbox", depth=1)
    assert _paths(result) == ["Inbox/Misc", "Inbox/Projects"]


# --- filter / sort -------------------------------------------------------------------

def test_name_contains_matches_path_case_insensitively(wrapper):
    assert _paths(wrapper.list_folders(name_contains="PROJ")) == ["Inbox/Projects", "Inbox/Projects/2026"]
    assert _paths(wrapper.list_folders(name_contains="2026")) == ["Archive/2026", "Inbox/Projects/2026"]


def test_sort_by_unread_and_total_largest_first(wrapper):
    assert _paths(wrapper.list_folders(sort_by="unread"))[:3] == ["Inbox/Projects", "Inbox", "Inbox/Projects/2026"]
    assert _paths(wrapper.list_folders(sort_by="total"))[:2] == ["Archive", "Archive/2026"]


def test_sort_applies_before_limit(wrapper):
    result = wrapper.list_folders(sort_by="unread", limit=1)
    assert _paths(result) == ["Inbox/Projects"] and result["truncated"] is True


def test_invalid_sort_by_rejected(wrapper):
    with pytest.raises(ToolException, match="Invalid sort_by"):
        wrapper.list_folders(sort_by="size")


# --- path resolution -----------------------------------------------------------------

def test_path_resolves_segment_by_segment_without_full_walk(wrapper):
    assert wrapper._resolve_folder("Inbox/Projects/2026") == "proj-2026"
    assert _called(wrapper) == [ROOT, f"{ROOT}/root-inbox/childFolders", f"{ROOT}/inbox-proj/childFolders"]


def test_resolution_is_case_insensitive_and_cached(wrapper):
    assert wrapper._resolve_folder("inbox/misc") == "inbox-misc"
    calls = len(_called(wrapper))
    assert wrapper._resolve_folder("Inbox/Misc/") == "inbox-misc"
    assert len(_called(wrapper)) == calls


def test_unique_nested_name_falls_back_to_walk(wrapper):
    assert wrapper._resolve_folder("Projects") == "inbox-proj"


def test_passthrough_needs_no_lookup(wrapper):
    long_id = "A" * 120
    assert wrapper._resolve_folder("inbox") == "inbox"
    assert wrapper._resolve_folder("all") == "all"
    assert wrapper._resolve_folder(long_id) == long_id
    wrapper._get.assert_not_called()


def test_ambiguous_and_missing(wrapper):
    with pytest.raises(ToolException, match="ambiguous"):
        wrapper._resolve_folder("2026")
    with pytest.raises(ToolException, match="not found"):
        wrapper._resolve_folder("Nope")
    with pytest.raises(ToolException, match="not found"):
        wrapper._resolve_folder("Inbox/Nope/2026")


def test_list_messages_in_subfolder_uses_resolved_id(wrapper):
    wrapper.list_messages(folder="Inbox/Projects/2026")
    assert wrapper._get.call_args.args[0] == f"{ROOT}/proj-2026/messages"
