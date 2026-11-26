"""#7648: list_pull_request_files filters, pages and limits before downloading any file content."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from azure.devops.v7_0.git.git_client import GitClient

from elitea_sdk.tools.ado.repos.repos_wrapper import ReposApiWrapper

SOURCE, TARGET = "src-commit", "tgt-commit"


def _change(path, change_type="edit", original_path=None, folder=False):
    props = {"item": {"path": path, **({"isFolder": True} if folder else {})}, "changeType": change_type}
    if original_path:
        props["originalPath"] = original_path
    return SimpleNamespace(additional_properties=props)


def _wrapper(changes, contents=None, page_size=None):
    """Wrapper over a mocked GitClient; `contents` maps (commit, path) to text or an Exception."""
    client = MagicMock(spec=GitClient)
    repository = MagicMock(id="alpha-id", default_branch="refs/heads/main")
    repository.name = "alpha"
    client.get_repository.return_value = repository
    client.get_pull_request_iterations.return_value = [
        SimpleNamespace(id=3, source_ref_commit=SimpleNamespace(commit_id=SOURCE),
                        target_ref_commit=SimpleNamespace(commit_id=TARGET))
    ]

    def iteration_changes(repository_id, project, pull_request_id, iteration_id, top=None, skip=None):
        size = page_size or top or 100
        skip = skip or 0
        chunk = changes[skip:skip + min(size, top or 100)]
        nxt = skip + len(chunk) if skip + len(chunk) < len(changes) else 0
        return SimpleNamespace(change_entries=chunk, next_skip=nxt, next_top=size if nxt else 0)

    client.get_pull_request_iteration_changes.side_effect = iteration_changes

    def item_text(repository_id, project, path, version_descriptor):
        value = (contents or {}).get((version_descriptor.version, path), "line\n")
        if isinstance(value, Exception):
            raise value
        return [value.encode()]

    client.get_item_text.side_effect = item_text
    with patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient", return_value=client):
        wrapper = ReposApiWrapper(
            llm=MagicMock(), organization_url="https://dev.azure.com/org", project="Digital",
            token="fake-token", repository_id="alpha", base_branch="main", active_branch="main",
        )
    return wrapper, client


def _run(wrapper, **kwargs):
    return json.loads(wrapper.list_pull_request_diffs("12", **kwargs))


def test_paths_only_downloads_no_content_and_skips_folders():  # AC-1
    changes = [_change(f"/src/f{i}.cs") for i in range(300)] + [_change("/src", folder=True)]
    wrapper, client = _wrapper(changes)

    result = _run(wrapper, include_diff=False, max_files=1000)

    assert result["total_files"] == 300 and result["returned_files"] == 300
    assert result["files"][0] == {"path": "/src/f0.cs", "change_type": "edit"}
    assert client.get_item_text.call_count == 0


def test_include_and_exclude_patterns():  # AC-2, AC-3
    changes = [_change(p) for p in ["/src/Api/A.CS", "/src/web/b.cs", "/docs/c.md", "/src/api/app.min.js", "/package.lock"]]
    wrapper, _ = _wrapper(changes)

    def paths(**kw):
        return [f["path"] for f in _run(wrapper, include_diff=False, **kw)["files"]]

    assert paths(include_patterns=["*.cs"]) == ["/src/Api/A.CS", "/src/web/b.cs"]
    assert paths(include_patterns=["src/api/**"]) == ["/src/Api/A.CS", "/src/api/app.min.js"]
    assert paths(include_patterns=["/src/api/**"], exclude_patterns=["*.min.js"]) == ["/src/Api/A.CS"]
    assert paths(exclude_patterns=["*.lock", "*.min.js"]) == ["/src/Api/A.CS", "/src/web/b.cs", "/docs/c.md"]


def test_change_types_filter_and_rename_original_path():  # AC-4
    changes = [_change("/a", "add"), _change("/b", "edit"), _change("/c", "delete"),
               _change("/d", "rename", original_path="/old_d"), _change("/e", "edit, rename", original_path="/old_e")]
    wrapper, _ = _wrapper(changes)

    assert [f["path"] for f in _run(wrapper, include_diff=False, change_types=["add", "edit"])["files"]] == ["/a", "/b", "/e"]
    renamed = _run(wrapper, include_diff=False, change_types=["rename"])["files"]
    assert renamed[0] == {"path": "/d", "change_type": "rename", "original_path": "/old_d"}


def test_max_files_and_offset_page_through_matches():  # AC-5
    wrapper, _ = _wrapper([_change(f"/f{i:02}.cs") for i in range(40)] + [_change("/x.md")])

    first = _run(wrapper, include_diff=False, include_patterns=["*.cs"], max_files=10)
    second = _run(wrapper, include_diff=False, include_patterns=["*.cs"], max_files=10, offset=10)
    last = _run(wrapper, include_diff=False, include_patterns=["*.cs"], max_files=10, offset=30)

    assert (first["matched_files"], first["returned_files"], first["has_more"]) == (40, 10, True)
    assert [f["path"] for f in second["files"]] == [f"/f{i:02}.cs" for i in range(10, 20)]
    assert last["has_more"] is False


def test_next_offset_counts_files_that_failed_to_read():  # review: paging must not repeat or stall
    changes = [_change(f"/f{i}.cs") for i in range(6)]
    broken = {(TARGET, f"/f{i}.cs"): Exception("item not found") for i in (0, 1, 2)}
    wrapper, _ = _wrapper(changes, broken)

    first = _run(wrapper, max_files=3)
    second = _run(wrapper, max_files=3, offset=first["next_offset"])

    assert (first["returned_files"], len(first["errors"]), first["has_more"], first["next_offset"]) == (0, 3, True, 3)
    assert [f["path"] for f in second["files"]] == ["/f3.cs", "/f4.cs", "/f5.cs"]
    assert (second["has_more"], second["next_offset"]) == (False, None)


def test_brackets_in_patterns_match_literally():  # review: Next.js route folders
    changes = [_change("/src/pages/[id]/index.tsx"), _change("/src/pages/i/index.tsx")]
    wrapper, _ = _wrapper(changes)

    files = _run(wrapper, include_diff=False, include_patterns=["src/pages/[id]/*"])["files"]

    assert [f["path"] for f in files] == ["/src/pages/[id]/index.tsx"]


def test_edit_rename_is_diffed_against_its_original_path():  # review: "edit, rename" got no diff
    changes = [_change("/new.py", "edit, rename", original_path="/old.py")]
    wrapper, _ = _wrapper(changes, {(TARGET, "/old.py"): "x = 1\n", (SOURCE, "/new.py"): "x = 2\n"})

    file = _run(wrapper, change_types=["edit"])["files"][0]

    assert "-x = 1\n+x = 2\n" in file["diff"]
    assert file["original_path"] == "/old.py"


def test_changes_past_the_first_page_are_read():  # AC-6
    wrapper, client = _wrapper([_change(f"/f{i}.cs") for i in range(250)], page_size=100)

    result = _run(wrapper, include_diff=False, offset=200)

    assert result["total_files"] == 250
    assert result["files"][-1]["path"] == "/f249.cs"
    assert client.get_pull_request_iteration_changes.call_count == 3


def test_long_diff_is_truncated():  # AC-7
    base = "".join(f"old {i}\n" for i in range(1000))
    new = "".join(f"new {i}\n" for i in range(1000))
    wrapper, _ = _wrapper([_change("/big.txt")], {(TARGET, "/big.txt"): base, (SOURCE, "/big.txt"): new})

    file = _run(wrapper, max_diff_lines=100)["files"][0]

    assert file["diff_truncated"] is True
    assert len(file["diff"].splitlines()) == 101
    assert file["diff"].endswith("... diff truncated (1,903 more lines)")
    assert _run(wrapper, max_diff_lines=0)["files"][0]["diff_truncated"] is False


def test_one_failed_file_does_not_drop_the_others():  # AC-8
    changes = [_change("/ok1.cs"), _change("/gone.cs"), _change("/ok2.cs")]
    wrapper, _ = _wrapper(changes, {(TARGET, "/gone.cs"): Exception("item not found")})

    result = _run(wrapper)

    assert [f["path"] for f in result["files"]] == ["/ok1.cs", "/ok2.cs"]
    assert result["errors"][0]["path"] == "/gone.cs"
    assert "item not found" in result["errors"][0]["error"]


def test_default_call_on_small_pr_keeps_paths_and_diffs():  # AC-10
    changes = [_change("/a.py"), _change("/new.py", "add")]
    wrapper, _ = _wrapper(changes, {(TARGET, "/a.py"): "x = 1\n", (SOURCE, "/a.py"): "x = 2\n"})

    files = _run(wrapper)["files"]

    assert "-x = 1\n+x = 2\n" in files[0]["diff"]
    assert files[1]["diff"] == "Change Type: add"


def test_binary_files_are_not_diffed():  # AC-11
    changes = [_change("/logo.png"), _change("/blob.dat")]
    wrapper, client = _wrapper(changes, {(TARGET, "/blob.dat"): "a\x00b", (SOURCE, "/blob.dat"): "a\x00c"})

    files = _run(wrapper)["files"]

    assert [f["diff"] for f in files] == ["Binary file, diff not shown"] * 2
    assert all(call.kwargs["path"] != "/logo.png" for call in client.get_item_text.call_args_list)


def test_tool_schema_exposes_the_new_parameters():  # AC-12
    wrapper, _ = _wrapper([])
    tool = next(t for t in wrapper.get_available_tools() if t["name"] == "list_pull_request_files")
    props = tool["args_schema"].model_json_schema()["properties"]

    assert {"include_diff", "include_patterns", "exclude_patterns", "change_types",
            "max_files", "offset", "max_diff_lines"} <= set(props)
    assert "include_diff: false" in tool["description"]
