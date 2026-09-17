"""Artifact list_files returns data, not a Python repr (#6532).

`return_as_string` defaulted to True and was never exposed in the tool's
args_schema, so a model could not turn it off: every call returned
`str({'total': 129, 'rows': [...]})` — single-quoted, unfenced, with each link
markdown-linkified in the UI. Same shape as the zephyr_scale kwargs bugs.
"""

import json
from types import SimpleNamespace

from elitea_sdk.runtime.tools.artifact import ArtifactWrapper


def _wrapper(rows):
    wrapper = ArtifactWrapper.model_construct()
    wrapper.__dict__['artifact'] = SimpleNamespace(
        list=lambda bucket_name, prefix, delimiter: {'total': len(rows), 'rows': rows}
    )
    wrapper.__dict__['bucket'] = 'e2e'
    return wrapper


def test_list_files_returns_native_data():
    rows = [
        {'name': 'notes.md', 'size': 12, 'type': 'file', 'link': 'https://example/notes.md'},
        {'name': 'docs', 'size': 0, 'type': 'folder'},
    ]

    result = _wrapper(rows).list_files()

    assert isinstance(result, dict)
    assert result['rows'] == rows
    assert json.loads(json.dumps(result)) == result


def test_missing_bucket_returns_an_empty_listing_not_a_string():
    wrapper = ArtifactWrapper.model_construct()
    wrapper.__dict__['artifact'] = SimpleNamespace(
        list=lambda bucket_name, prefix, delimiter: {'error': 'no such bucket'}
    )
    wrapper.__dict__['bucket'] = 'missing'

    assert wrapper.list_files() == {'total': 0, 'rows': []}


def test_a_failed_listing_raises_for_indexing():
    """Indexing decides whether to delete previously indexed content from this answer,
    so a bucket that could not be listed must never arrive as an empty one."""
    from langchain_core.tools import ToolException
    import pytest

    wrapper = ArtifactWrapper.model_construct()
    wrapper.__dict__['artifact'] = SimpleNamespace(
        list=lambda bucket_name, prefix, delimiter: {'error': 'connection reset'}
    )
    wrapper.__dict__['bucket'] = 'unreachable'

    with pytest.raises(ToolException, match="unreachable"):
        wrapper.list_files(strict=True)


def test_the_loader_attests_only_when_the_bucket_answered():
    rows = [{'name': 'a.md', 'size': 3, 'type': 'file', 'key': 'a.md', 'modified': '1'}]
    wrapper = _wrapper(rows)
    run = SimpleNamespace(loader_attested=False)
    wrapper.__dict__['_index_run'] = run

    list(wrapper._base_loader())

    assert run.loader_attested is True


def test_the_loader_does_not_attest_when_the_listing_failed():
    from langchain_core.tools import ToolException
    import pytest

    wrapper = ArtifactWrapper.model_construct()
    wrapper.__dict__['artifact'] = SimpleNamespace(
        list=lambda bucket_name, prefix, delimiter: {'error': 'connection reset'}
    )
    wrapper.__dict__['bucket'] = 'unreachable'
    run = SimpleNamespace(loader_attested=False)
    wrapper.__dict__['_index_run'] = run

    with pytest.raises(ToolException):
        list(wrapper._base_loader())

    assert run.loader_attested is False


def _client(s3_response):
    return SimpleNamespace(
        bucket_exists=lambda name: True,
        create_bucket=lambda name: None,
        list_artifacts_s3=lambda bucket_name, prefix, delimiter: s3_response,
    )


def test_a_truncated_listing_is_reported_as_such():
    from elitea_sdk.runtime.clients.artifact import Artifact

    result = Artifact(_client({'contents': [], 'isTruncated': True}), 'big').list()

    assert result['truncated'] is True


def _loader(rows, truncated=False, folder=None):
    wrapper = ArtifactWrapper.model_construct()
    wrapper.__dict__['artifact'] = SimpleNamespace(
        list=lambda bucket_name, prefix, delimiter: {
            'total': len(rows), 'rows': rows, 'truncated': truncated
        }
    )
    wrapper.__dict__['bucket'] = 'b'
    run = SimpleNamespace(loader_attested=False)
    wrapper.__dict__['_index_run'] = run
    list(wrapper._base_loader(**({'folder': folder} if folder else {})))
    return run


def test_a_truncated_listing_is_never_attested():
    """Attesting a first page would make every object past it an orphan at promote."""
    assert _loader([], truncated=True).loader_attested is False


def test_a_folder_scoped_listing_is_never_attested():
    """A renamed folder lists as an empty prefix on an existing bucket, so an empty
    result there says nothing about the source."""
    assert _loader([], folder='docs').loader_attested is False


def test_a_complete_bucket_listing_is_attested():
    assert _loader([]).loader_attested is True
