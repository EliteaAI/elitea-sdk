from elitea_sdk.tools.github.github_client import GitHubClient
from elitea_sdk.tools.github.schemas import (
    ApplyGitPatch,
    ApplyGitPatchFromArtifact,
    CreateFile,
    DeleteFile,
    UpdateFile,
)


class FakeFile:
    def __init__(self, content='existing content', sha='sha-123'):
        self.decoded_content = content.encode('utf-8')
        self.sha = sha


class FakeRepo:
    def __init__(self):
        self.files = {}
        self.create_calls = []
        self.update_calls = []
        self.delete_calls = []

    def get_contents(self, path, ref=None):
        key = (path, ref)
        if key not in self.files:
            raise Exception(f'{path} missing on {ref}')
        return self.files[key]

    def create_file(self, path, message, content, branch):
        self.create_calls.append(
            {'path': path, 'message': message, 'content': content, 'branch': branch}
        )
        self.files[(path, branch)] = FakeFile(content=content)

    def update_file(self, path, message, content, sha, branch):
        self.update_calls.append(
            {
                'path': path,
                'message': message,
                'content': content,
                'sha': sha,
                'branch': branch,
            }
        )
        self.files[(path, branch)] = FakeFile(content=content, sha=sha)

    def delete_file(self, path, message, sha, branch):
        self.delete_calls.append(
            {'path': path, 'message': message, 'sha': sha, 'branch': branch}
        )
        self.files.pop((path, branch), None)


class FakeGitHubApi:
    def __init__(self, repo):
        self.repo = repo

    def get_repo(self, repo_name):
        return self.repo


def _make_client(repo):
    return GitHubClient.model_construct(
        github_repository='owner/repo',
        active_branch='main',
        github_base_branch='main',
        github_api=FakeGitHubApi(repo),
        elitea=None,
    )


def test_create_file_respects_explicit_branch_when_active_branch_is_main():
    repo = FakeRepo()
    client = _make_client(repo)

    result = client.create_file(
        file_path='docs/example.txt',
        file_contents='hello',
        branch='feature/docs',
    )

    assert result == 'Created file docs/example.txt'
    assert repo.create_calls[-1]['branch'] == 'feature/docs'


def test_update_file_passes_explicit_branch_to_edit_file(monkeypatch):
    repo = FakeRepo()
    client = _make_client(repo)
    captured = {}

    def fake_edit_file(self, file_path, file_query, branch=None, commit_message=None):
        captured['file_path'] = file_path
        captured['file_query'] = file_query
        captured['branch'] = branch
        captured['commit_message'] = commit_message
        return 'updated'

    monkeypatch.setattr(GitHubClient, 'edit_file', fake_edit_file)

    result = client.update_file(
        file_query='docs/example.txt\nOLD <<<<\nold\n>>>> OLD\nNEW <<<<\nnew\n>>>> NEW',
        branch='feature/docs',
        commit_message='custom commit',
    )

    assert result == 'updated'
    assert captured['file_path'] == 'docs/example.txt'
    assert captured['branch'] == 'feature/docs'
    assert captured['commit_message'] == 'custom commit'


def test_apply_git_patch_respects_explicit_branch_when_active_branch_is_main(monkeypatch):
    repo = FakeRepo()
    client = _make_client(repo)

    monkeypatch.setattr(
        GitHubClient,
        '_parse_git_patch',
        lambda self, patch_content: [
            {
                'operation': 'create',
                'file_path': 'docs/patched.txt',
                'new_content': 'patched content',
            }
        ],
    )

    result = client.apply_git_patch(
        patch_content='diff --git a/docs/patched.txt b/docs/patched.txt',
        branch='feature/docs',
    )

    assert result['failed_count'] == 0
    assert result['applied_changes'] == ['Created: docs/patched.txt']
    assert repo.create_calls[-1]['branch'] == 'feature/docs'


def test_delete_file_respects_explicit_branch_when_active_branch_is_main():
    repo = FakeRepo()
    repo.files[('docs/example.txt', 'feature/docs')] = FakeFile(content='hello')
    client = _make_client(repo)

    result = client.delete_file('docs/example.txt', branch='feature/docs')

    assert result == 'Deleted file docs/example.txt'
    assert repo.delete_calls[-1]['branch'] == 'feature/docs'


def test_github_write_schemas_expose_branch_parameter():
    assert 'branch' in CreateFile.model_fields
    assert 'branch' in UpdateFile.model_fields
    assert 'branch' in DeleteFile.model_fields
    assert 'branch' in ApplyGitPatch.model_fields
    assert 'branch' in ApplyGitPatchFromArtifact.model_fields
