"""Tests for the Confluence toolkit "space" -> "space_key" config field rename (#5997).

get_toolkit() must resolve the new `space_key` settings key, while still
honoring the legacy `space` key for backward compatibility with previously
saved toolkit configurations.
"""

from unittest.mock import patch

from elitea_sdk.tools.confluence import ConfluenceToolkit, get_toolkit


def _tool(settings):
    return {
        "settings": settings,
        "toolkit_name": "confluence",
    }


def _base_settings(**overrides):
    settings = {
        "confluence_configuration": {"hosting": "Cloud", "base_url": "https://example.atlassian.net"},
    }
    settings.update(overrides)
    return settings


class TestSpaceKeyResolution:
    @patch.object(ConfluenceToolkit, "get_toolkit")
    def test_space_key_setting_is_used(self, mock_get_toolkit):
        get_toolkit(_tool(_base_settings(space_key="MPS")))

        assert mock_get_toolkit.call_args.kwargs["space"] == "MPS"

    @patch.object(ConfluenceToolkit, "get_toolkit")
    def test_legacy_space_setting_still_works(self, mock_get_toolkit):
        get_toolkit(_tool(_base_settings(space="LEGACY")))

        assert mock_get_toolkit.call_args.kwargs["space"] == "LEGACY"

    @patch.object(ConfluenceToolkit, "get_toolkit")
    def test_space_key_takes_precedence_over_legacy_space(self, mock_get_toolkit):
        get_toolkit(_tool(_base_settings(space_key="NEW", space="OLD")))

        assert mock_get_toolkit.call_args.kwargs["space"] == "NEW"

    @patch.object(ConfluenceToolkit, "get_toolkit")
    def test_neither_setting_defaults_to_none(self, mock_get_toolkit):
        get_toolkit(_tool(_base_settings()))

        assert mock_get_toolkit.call_args.kwargs["space"] is None


class TestToolkitConfigSchema:
    def test_schema_field_is_named_space_key(self):
        schema = ConfluenceToolkit.toolkit_config_schema()

        assert "space_key" in schema.model_fields
        assert "space" not in schema.model_fields
        assert "Space Key" in schema.model_fields["space_key"].description
