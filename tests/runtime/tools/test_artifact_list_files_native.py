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
