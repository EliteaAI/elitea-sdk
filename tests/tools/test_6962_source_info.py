# Copyright (c) 2026 EPAM Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guards the `source_info` object on index reports and search results (#6962)."""

import importlib
import json
import re
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from elitea_sdk.runtime.tools.artifact import ArtifactWrapper
from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.tools.ado.repos.repos_wrapper import ReposApiWrapper
from elitea_sdk.tools.ado.test_plan.test_plan_wrapper import TestPlanApiWrapper
from elitea_sdk.tools.ado.wiki.ado_wrapper import AzureDevOpsApiWrapper as AdoWikiWrapper
from elitea_sdk.tools.ado.work_item.ado_wrapper import AzureDevOpsApiWrapper as AdoBoardsWrapper
from elitea_sdk.tools.base_indexer_toolkit import (
    BaseIndexerToolkit,
    IndexingStats,
    IndexingStatus,
    render_report_text,
)
from elitea_sdk.tools.bitbucket.api_wrapper import BitbucketAPIWrapper
from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper
from elitea_sdk.tools.figma.api_wrapper import FigmaApiWrapper
from elitea_sdk.tools.github.api_wrapper import EliteAGitHubAPIWrapper
from elitea_sdk.tools.gitlab.api_wrapper import GitLabAPIWrapper
from elitea_sdk.tools.jira.api_wrapper import JiraApiWrapper
from elitea_sdk.tools.qtest.api_wrapper import QtestApiWrapper
from elitea_sdk.tools.sharepoint.api_wrapper import SharepointApiWrapper
from elitea_sdk.tools.testrail.api_wrapper import TestrailAPIWrapper
from elitea_sdk.tools.utils.source_info import build_source_info, reuse_url, single_line, web_base
from elitea_sdk.tools.xray.api_wrapper import XrayApiWrapper
from elitea_sdk.tools.zephyr_enterprise.api_wrapper import ZephyrApiWrapper
from elitea_sdk.tools.zephyr_essential.api_wrapper import ZephyrEssentialApiWrapper
from elitea_sdk.tools.zephyr_scale.api_wrapper import ZephyrScaleApiWrapper
from tests.tools.code_loader_rig import (
    PY_BODY_HASH,
    SHA_PY_BODY,
    FakeStagingAdapter,
    IdentityToolkit,
    build_toolkit,
    run_index,
    seed_completed_run,
)

ALLOWED_KEYS = {"type", "location", "url", "id"}

# The older UI reformats any tool-message line this matches (base_indexer_toolkit.render_report_text).
LEGACY_UI_LINE = re.compile(r"^(.+?)\s+(\d+\s+\w+)\.?\s*$")


def wrapper(cls, **fields):
    instance = cls.model_construct()
    for name, value in fields.items():
        object.__setattr__(instance, name, value)
    return instance


def source_info_of(instance, metadata):
    return instance._with_source_info([{"metadata": metadata}])[0]["metadata"]["source_info"]


class TestTheTypeIsTheToolkitName:
    @pytest.mark.parametrize("cls, module", [
        (EliteAGitHubAPIWrapper, "elitea_sdk.tools.github"),
        (GitLabAPIWrapper, "elitea_sdk.tools.gitlab"),
        (BitbucketAPIWrapper, "elitea_sdk.tools.bitbucket"),
        (ReposApiWrapper, "elitea_sdk.tools.ado.repos"),
        (AdoBoardsWrapper, "elitea_sdk.tools.ado.work_item"),
        (AdoWikiWrapper, "elitea_sdk.tools.ado.wiki"),
        (TestPlanApiWrapper, "elitea_sdk.tools.ado.test_plan"),
        (ConfluenceAPIWrapper, "elitea_sdk.tools.confluence"),
        (JiraApiWrapper, "elitea_sdk.tools.jira"),
        (SharepointApiWrapper, "elitea_sdk.tools.sharepoint"),
        (TestrailAPIWrapper, "elitea_sdk.tools.testrail"),
        (ZephyrScaleApiWrapper, "elitea_sdk.tools.zephyr_scale"),
        (ZephyrEssentialApiWrapper, "elitea_sdk.tools.zephyr_essential"),
        (ZephyrApiWrapper, "elitea_sdk.tools.zephyr_enterprise"),
        (QtestApiWrapper, "elitea_sdk.tools.qtest"),
        (XrayApiWrapper, "elitea_sdk.tools.xray"),
        (FigmaApiWrapper, "elitea_sdk.tools.figma"),
    ])
    def test_each_indexer_names_its_toolkit(self, cls, module):
        assert cls.index_source_type == importlib.import_module(module).name

    def test_the_artifact_indexer_names_its_toolkit(self):
        assert ArtifactWrapper.index_source_type == "artifact"

    def test_content_cannot_change_the_type(self):
        info = source_info_of(wrapper(JiraApiWrapper, base_url="https://jira.corp"),
                              {"id": "1", "issue_key": "AB-1", "type": "github"})
        assert info["type"] == "jira"


class TestTheShape:
    def test_unknown_fields_are_left_out(self):
        assert build_source_info("jira", location="", url=None, item_id=None) == {"type": "jira"}

    def test_values_are_strings(self):
        assert build_source_info("jira", item_id=42) == {"type": "jira", "id": "42"}

    def test_location_is_one_line_and_capped(self):
        assert single_line("a\nb\rc\x00d\x7fe") == "a b c d e"
        assert len(single_line("x" * 600)) == 512

    def test_only_the_four_keys_appear(self):
        info = source_info_of(wrapper(JiraApiWrapper, base_url="https://jira.corp"),
                              {"id": "1", "issue_key": "AB-1", "author": "a@b.c", "creator": "x"})
        assert set(info) <= ALLOWED_KEYS
        assert all(isinstance(value, str) for value in info.values())


class TestUrlRules:
    @pytest.mark.parametrize("configured", ["https://svc:TOKEN@jira.corp", "https://jira.corp/?token=x#frag"])
    def test_credentials_query_and_fragment_never_reach_the_base(self, configured):
        assert web_base(configured) == "https://jira.corp"

    def test_http_stays_http(self):
        assert web_base("http://jira.intranet:8080/") == "http://jira.intranet:8080"

    def test_https_is_never_downgraded(self):
        assert reuse_url("http://jira.corp/browse/A-1", "https://jira.corp") is None

    @pytest.mark.parametrize("configured", ["ftp://host/x", "javascript:alert(1)", "file:///etc/passwd", ""])
    def test_other_schemes_produce_no_link(self, configured):
        assert web_base(configured) is None

    def test_a_reused_link_on_another_host_is_dropped(self):
        assert reuse_url("https://evil.example/pages/1", "https://wiki.corp") is None

    def test_a_reused_link_loses_userinfo_and_signing_parameters(self):
        url = reuse_url("https://svc:TOKEN@tenant.sharepoint.com/a.docx?X-Amz-Signature=s&sig=t&downloadUrl=u",
                        "https://tenant.sharepoint.com/sites/x")
        assert url == "https://tenant.sharepoint.com/a.docx"


class TestCodeLinks:
    def test_github_links_go_to_the_web_host(self):
        github = wrapper(EliteAGitHubAPIWrapper, github_base_url="https://api.github.com")
        assert github._code_web_url("org/repo", "main", "src/app.py") == \
            "https://github.com/org/repo/blob/main/src/app.py"

    def test_a_configured_github_toolkit_builds_links(self):
        values = EliteAGitHubAPIWrapper.validate_environment(
            {"access_token": "token", "base_url": "https://ghe.corp/api/v3/", "github_repository": "org/repo"})
        github = wrapper(EliteAGitHubAPIWrapper, github_base_url=values["github_base_url"])
        assert github._code_web_url("org/repo", "main", "a.py") == "https://ghe.corp/org/repo/blob/main/a.py"

    def test_github_enterprise_drops_the_api_prefix(self):
        github = wrapper(EliteAGitHubAPIWrapper, github_base_url="https://user:tok@ghe.corp/api/v3")
        assert github._code_web_url("org/repo", "main") == "https://ghe.corp/org/repo/tree/main"

    def test_gitlab(self):
        gitlab = wrapper(GitLabAPIWrapper, url="https://gitlab.corp/")
        assert gitlab._code_web_url("grp/sub/proj", "feature/x", "docs/read me.md") == \
            "https://gitlab.corp/grp/sub/proj/-/blob/feature%2Fx/docs/read%20me.md"

    def test_gitlab_records_the_project_path_when_configured_by_id(self):
        gitlab = wrapper(GitLabAPIWrapper, repository="12345",
                         _repo_instance=SimpleNamespace(path_with_namespace="grp/proj"))
        assert gitlab._index_target_metadata({}) == {"repository": "grp/proj"}

    def test_bitbucket_cloud_given_the_api_host(self):
        bitbucket = wrapper(BitbucketAPIWrapper, url="https://api.bitbucket.org", cloud=True)
        assert bitbucket._code_web_url("ws/repo", "feature/x", "a b#c.py") == \
            "https://bitbucket.org/ws/repo/src/feature%2Fx/a%20b%23c.py"

    def test_bitbucket_server_keeps_its_context_path(self):
        bitbucket = wrapper(BitbucketAPIWrapper, url="https://git.corp/bitbucket", cloud=False)
        url = bitbucket._code_web_url("PROJ/repo", "feature/x", "a b#c.py")
        assert url.startswith("https://git.corp/bitbucket/projects/PROJ/repos/repo/browse/a%20b%23c.py?")
        assert parse_qs(urlsplit(url).query) == {"at": ["refs/heads/feature/x"]}

    def test_bitbucket_server_personal_repository(self):
        bitbucket = wrapper(BitbucketAPIWrapper, url="https://git.corp", cloud=False)
        assert bitbucket._code_web_url("~jdoe/repo", "main", "a.py").startswith(
            "https://git.corp/users/jdoe/repos/repo/browse/a.py?")

    def test_ado_repos_query_cannot_be_forged_by_a_file_name(self):
        ado = wrapper(ReposApiWrapper, organization_url="https://dev.azure.com/org")
        url = ado._code_web_url("Proj/repo", "feature/x", "a#b?c&version=GBx.md")
        assert url.startswith("https://dev.azure.com/org/Proj/_git/repo?")
        assert parse_qs(urlsplit(url).query) == {"path": ["/a#b?c&version=GBx.md"], "version": ["GBfeature/x"]}

    def test_an_old_chunk_has_no_repository_and_no_link(self):
        github = wrapper(EliteAGitHubAPIWrapper, github_base_url="https://api.github.com",
                         github_repository="org/current-default")
        info = source_info_of(github, {"file_path": "src/app.py", "filename": "src/app.py", "source": "src/app.py"})
        assert info == {"type": "github", "location": "src/app.py", "id": "src/app.py"}

    def test_a_stamped_chunk_names_repository_branch_and_link(self):
        github = wrapper(EliteAGitHubAPIWrapper, github_base_url="https://api.github.com")
        info = source_info_of(github, {"file_path": "a.py", "repository": "org/repo", "branch": "dev"})
        assert info == {"type": "github", "location": "org/repo@dev:a.py",
                        "url": "https://github.com/org/repo/blob/dev/a.py", "id": "a.py"}


class TestDocumentLinks:
    def test_confluence_page(self):
        confluence = wrapper(ConfluenceAPIWrapper, base_url="https://wiki.corp/confluence")
        info = source_info_of(confluence, {
            "id": "42", "title": "Runbook",
            "source": "https://wiki.corp/confluence/pages/viewpage.action?pageId=42&os_auth=x#top"})
        assert info == {"type": "confluence", "location": "Runbook", "id": "42",
                        "url": "https://wiki.corp/confluence/pages/viewpage.action?pageId=42"}

    def test_confluence_attachment_ignores_the_temp_file_source(self):
        confluence = wrapper(ConfluenceAPIWrapper, base_url="https://user:pw@wiki.corp")
        info = source_info_of(confluence, {
            "id": "att7", "name": "report.pdf", "title": "", "parent_id": "42", "source": "/tmp/tmpab12.pdf",
            "download_url": "https://wiki.corp/download/attachments/42/report.pdf?version=1&api=v2"})
        assert info == {"type": "confluence", "location": "42 / report.pdf", "id": "att7",
                        "url": "https://wiki.corp/download/attachments/42/report.pdf"}

    def test_jira_link_is_built_from_the_issue_key(self):
        jira = wrapper(JiraApiWrapper, base_url="https://svc:TOKEN@jira.corp")
        info = source_info_of(jira, {"id": "attach_1", "issue_key": "AB-1", "source": "/tmp/tmpx.pdf"})
        assert info == {"type": "jira", "location": "AB-1", "id": "attach_1",
                        "url": "https://jira.corp/browse/AB-1"}

    def test_sharepoint_link_only_on_the_site_host(self):
        sharepoint = wrapper(SharepointApiWrapper, site_url="https://tenant.sharepoint.com/sites/team")
        on_site = source_info_of(sharepoint, {"id": "1", "Path": "/sites/team/Docs/a.docx",
                                              "Link": "https://tenant.sharepoint.com/sites/team/Docs/a.docx"})
        elsewhere = source_info_of(sharepoint, {"id": "2", "Path": "/drive/root:/a.docx",
                                                "Link": "https://other.example/a.docx"})
        assert on_site["url"] == "https://tenant.sharepoint.com/sites/team/Docs/a.docx"
        assert "url" not in elsewhere

    def test_sharepoint_office_link_keeps_the_document_but_not_share_tokens(self):
        sharepoint = wrapper(SharepointApiWrapper, site_url="https://contoso.sharepoint.com/sites/Team")
        info = source_info_of(sharepoint, {"id": "1", "Path": "/drive/root:/a.docx", "Link": (
            "https://contoso.sharepoint.com/sites/Team/_layouts/15/Doc.aspx"
            "?sourcedoc=%7BAB-12%7D&file=a.docx&action=default&mobileredirect=true&e=ShareToken")})
        assert parse_qs(urlsplit(info["url"]).query) == {
            "sourcedoc": ["{AB-12}"], "file": ["a.docx"], "action": ["default"]}

    def test_artifact(self):
        artifact = wrapper(ArtifactWrapper, elitea=SimpleNamespace(public_url="https://elitea.corp", base_url="x"))
        info = source_info_of(artifact, {
            "id": "hash", "key": "docs/a.txt", "source": "/tmp/tmpq.txt",
            "link": "https://elitea.corp/artifacts/s3/my-bucket/docs/a.txt?project_id=2"})
        assert info == {"type": "artifact", "location": "my-bucket/docs/a.txt", "id": "docs/a.txt",
                        "url": "https://elitea.corp/artifacts/s3/my-bucket/docs/a.txt?project_id=2"}

    def test_an_artifact_link_from_the_cluster_host_is_dropped(self):
        artifact = wrapper(ArtifactWrapper, elitea=SimpleNamespace(public_url="https://elitea.corp", base_url="x"))
        info = source_info_of(artifact, {"key": "a.txt", "link": "http://pylon_main:8080/s3/b/a.txt?project_id=2"})
        assert info == {"type": "artifact", "location": "b/a.txt", "id": "a.txt"}

    def test_a_connector_without_a_template_has_no_link(self):
        assert source_info_of(wrapper(TestrailAPIWrapper), {"id": "C1", "title": "x"}) == {"type": "testrail", "id": "C1"}


class TestSearchResults:
    @pytest.fixture
    def found(self, monkeypatch):
        docs = [{"page_content": "x", "score": 0.9,
                 "metadata": {"id": "AB-1", "issue_key": "AB-1", "source": "https://jira.corp/browse/AB-1"}}]
        monkeypatch.setattr(VectorStoreWrapperBase, "search_documents", lambda self, *a, **kw: json.loads(json.dumps(docs)))
        monkeypatch.setattr(VectorStoreWrapperBase, "stepback_search", lambda self, *a, **kw: json.loads(json.dumps(docs)))
        monkeypatch.setattr(BaseIndexerToolkit, "list_indexes", lambda self: ["idx"])
        return wrapper(JiraApiWrapper, base_url="https://jira.corp")

    def test_search_index_adds_source_info_and_keeps_source(self, found):
        metadata = found.search_index("q", index_name="idx")[0]["metadata"]
        assert metadata["source"] == "https://jira.corp/browse/AB-1"
        assert metadata["source_info"] == {"type": "jira", "location": "AB-1", "id": "AB-1",
                                           "url": "https://jira.corp/browse/AB-1"}

    def test_source_info_is_selectable_through_output_fields(self, found):
        result = found.search_index("q", index_name="idx", output_fields=["metadata.source_info"])
        assert result == [{"metadata": {"source_info": {"type": "jira", "location": "AB-1", "id": "AB-1",
                                                        "url": "https://jira.corp/browse/AB-1"}}}]

    def test_stepback_search_index_adds_source_info(self, found):
        text = found.stepback_search_index("q", index_name="idx")
        assert json.loads(text.split("\n", 1)[1])[0]["metadata"]["source_info"]["type"] == "jira"


class TestRunReport:
    def report(self, source_info=None):
        return IndexingStats().build_report(status=IndexingStatus.OK, indexed_count=2,
                                            item_labels=("file", "files"), dependent_labels=("a", "b"),
                                            source_info=source_info)

    def test_a_report_without_source_is_unchanged(self):
        report = self.report()
        assert "source_info" not in report
        assert "Source" not in render_report_text(report)

    def test_the_source_line_follows_the_headline(self):
        lines = render_report_text(self.report({"type": "github", "location": "org/repo@main"})).splitlines()
        assert lines[1] == 'Source: github "org/repo@main"'

    def test_the_source_line_is_not_reformatted_by_the_older_ui(self):
        line = render_report_text(self.report(
            {"type": "sharepoint", "location": "host/sites/x / Shared Documents/2024 Reports"})).splitlines()[1]
        assert not LEGACY_UI_LINE.match(line)


class GitHubLikeToolkit(IdentityToolkit):
    index_source_type = "github"

    def _index_target_metadata(self, target):
        return {"repository": "org/repo"}

    def _index_branch(self, branch, target):
        return branch or "main"

    def _code_web_url(self, repository, branch, path=""):
        return f"https://github.com/{repository}/{'blob' if path else 'tree'}/{branch}{'/' + path if path else ''}"


class StampingAdapter(FakeStagingAdapter):
    def __init__(self):
        super().__init__()
        self.target_stamps = []

    def stamp_source_target(self, wrapper, row_ids, source_target):
        self.target_stamps.append((sorted(row_ids), dict(source_target)))
        return len(row_ids)


def stored_entry(filename, rows):
    """rows: (row_id, commit_hash, blob_sha, source_target) as the PGVector adapter reads them."""
    return {
        "metadata": {"collection": "x", "filename": filename},
        "commit_hashes": [row[1] for row in rows],
        "blob_shas": [row[2] for row in rows],
        "source_targets": [row[3] for row in rows],
        "ids": [row[0] for row in rows],
    }


def code_toolkit(monkeypatch, tree, indexed):
    toolkit = build_toolkit(monkeypatch, tree, indexed, toolkit_cls=GitHubLikeToolkit)
    object.__setattr__(toolkit, "vector_adapter", StampingAdapter())
    seed_completed_run(toolkit)
    return toolkit


class TestCodeRuns:
    def test_the_report_names_the_resolved_repository_and_branch(self, monkeypatch):
        result = run_index(code_toolkit(monkeypatch, {"a.py": "sha-new"}, {}))
        assert result["report"]["source_info"] == {"type": "github", "location": "org/repo@main",
                                                   "url": "https://github.com/org/repo/tree/main"}
        assert 'Source: github "org/repo@main"' in result["message"]

    def test_new_chunks_carry_repository_and_branch(self, monkeypatch):
        written = []
        toolkit = code_toolkit(monkeypatch, {"a.py": "sha-new", "b.md": "sha-md", "c.txt": "sha-txt"}, {})
        object.__setattr__(toolkit, "contents", {"b.md": "# Title\n\nBody text.\n", "c.txt": "plain words"})
        monkeypatch.setattr("elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents",
                            lambda vectorstore=None, documents=None, ids=None:
                            written.extend(d.metadata for d in documents) or [f"r{n}" for n in range(len(documents))])
        run_index(toolkit, branch="dev")
        chunks = [metadata for metadata in written if metadata.get("file_path") in ("a.py", "b.md", "c.txt")]
        assert {metadata["file_path"] for metadata in chunks} == {"a.py", "b.md", "c.txt"}
        assert all(m["repository"] == "org/repo" and m["branch"] == "dev" for m in chunks)

    def test_files_skipped_by_identity_are_backfilled(self, monkeypatch):
        indexed = {"a.py": stored_entry("a.py", [("r1", "h", "sha-a", {}), ("r2", "h", "sha-a", {})]),
                   "b.py": stored_entry("b.py", [("r3", "h", "sha-b", {"repository": "org/repo", "branch": "main"})])}
        toolkit = code_toolkit(monkeypatch, {"a.py": "sha-a", "b.py": "sha-b"}, indexed)
        run_index(toolkit)
        assert toolkit.reads == []
        assert toolkit.vector_adapter.target_stamps == [(["r1", "r2"], {"repository": "org/repo", "branch": "main"})]

    def test_files_matched_by_content_are_backfilled(self, monkeypatch):
        indexed = {"a.py": stored_entry("a.py", [("r1", PY_BODY_HASH, "sha-stale", {})])}
        toolkit = code_toolkit(monkeypatch, {"a.py": SHA_PY_BODY}, indexed)
        run_index(toolkit)
        assert toolkit.reads == ["a.py"]
        assert toolkit.vector_adapter.target_stamps == [(["r1"], {"repository": "org/repo", "branch": "main"})]

    def test_another_branch_restamps_unchanged_rows(self, monkeypatch):
        indexed = {"a.py": stored_entry("a.py", [("r1", "h", "sha-a", {"repository": "org/repo", "branch": "main"})])}
        toolkit = code_toolkit(monkeypatch, {"a.py": "sha-a"}, indexed)
        run_index(toolkit, branch="release")
        assert toolkit.vector_adapter.target_stamps == [(["r1"], {"repository": "org/repo", "branch": "release"})]

    def test_rows_already_carrying_the_target_are_not_rewritten(self, monkeypatch):
        indexed = {"a.py": stored_entry("a.py", [("r1", "h", "sha-a", {"repository": "org/repo", "branch": "main"})])}
        toolkit = code_toolkit(monkeypatch, {"a.py": "sha-a"}, indexed)
        run_index(toolkit)
        assert toolkit.vector_adapter.target_stamps == []
