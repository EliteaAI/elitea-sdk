"""Regression tests for #6356.

The artifact branch of ``runtime.toolkits.tools.get_tools`` handed ``tool['settings']`` to
``_inject_toolkit_id``. Artifact payloads carry ``id`` at the top level only, so the injection
was skipped and ``api_wrapper.toolkit_id`` kept its class default 0 — which is then written into
``index_meta`` and onto the ``index_data_removed`` event, where the platform rejects it and drops
the schedule cleanup.

The tools built here are the real ``BaseAction``/``BaseToolApiWrapper`` pair the artifact toolkit
produces, so an attribute-shape change would fail these tests rather than pass against a
permissive stand-in.
"""
import logging
from unittest.mock import patch

from elitea_sdk.runtime.toolkits.tools import get_tools
from elitea_sdk.tools.base.tool import BaseAction
from elitea_sdk.tools.elitea_base import BaseToolApiWrapper


def _artifact_config(toolkit_id=77):
    return {
        'id': toolkit_id,
        'type': 'artifact',
        'name': 'My Artifacts',
        'toolkit_name': 'my_artifacts',
        'settings': {
            'bucket': 'b1',
            'selected_tools': ['index_data', 'remove_index'],
            'pgvector_configuration': {'connection_string': 'postgresql://x/y'},
            'embedding_model': 'amazon.titan-embed-text-v2:0',
        },
    }


def _load_artifact_tools(config):
    artifact_tool = BaseAction(
        api_wrapper=BaseToolApiWrapper(),
        name='index_data',
        description='index the bucket',
    )
    with patch('elitea_sdk.runtime.toolkits.tools.ArtifactToolkit') as artifact_toolkit:
        artifact_toolkit.get_toolkit.return_value.get_tools.return_value = [artifact_tool]
        tools = get_tools([config], elitea_client=object(), llm=object())
    return artifact_tool, tools, artifact_toolkit.get_toolkit.call_args.kwargs


class TestArtifactToolkitIdInjection:
    def test_toolkit_id_reaches_the_api_wrapper(self):
        artifact_tool, tools, _ = _load_artifact_tools(_artifact_config(77))

        assert tools == [artifact_tool]
        assert artifact_tool.api_wrapper.toolkit_id == 77

    def test_the_vector_schema_and_the_toolkit_id_agree(self):
        # Both name the toolkit id, and the platform resolves the vector schema from the
        # top-level id alone (`load_and_validate_toolkit_for_index`), so a config where they
        # diverge writes into a schema nothing reads. The wrong-dict bug split them exactly here.
        artifact_tool, _, toolkit_kwargs = _load_artifact_tools(_artifact_config(77))

        assert toolkit_kwargs['collection_schema'] == str(artifact_tool.api_wrapper.toolkit_id)
        assert toolkit_kwargs['collection_schema'] == '77'

    def test_a_decoy_settings_id_is_ignored_by_both(self):
        # The only config shape that tells the two id expressions apart. `collection_schema`
        # used to read `settings['id']` first while injection read the top level, so a config
        # like this wrote into a vector schema the platform never looks in — it resolves the
        # schema from the top-level id alone (`load_and_validate_toolkit_for_index`).
        config = _artifact_config(77)
        config['settings']['id'] = 4242

        artifact_tool, _, toolkit_kwargs = _load_artifact_tools(config)

        assert toolkit_kwargs['collection_schema'] == '77'
        assert artifact_tool.api_wrapper.toolkit_id == 77

    def test_nothing_is_warned_about_a_missing_id(self, caplog):
        with caplog.at_level(logging.DEBUG, logger='elitea_sdk.tools'):
            _load_artifact_tools(_artifact_config(77))

        assert not any(
            'Toolkit ID is missing or not an integer' in r.message for r in caplog.records
        )

    def test_display_name_is_still_injected(self):
        artifact_tool, _, _ = _load_artifact_tools(_artifact_config(77))

        assert artifact_tool.metadata['display_name'] == 'My Artifacts'
