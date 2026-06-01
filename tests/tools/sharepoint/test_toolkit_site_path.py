"""Tests for SharePoint toolkit site_path resolution."""

from unittest.mock import patch, MagicMock

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.sharepoint import SharepointToolkit


_MISSING = object()


def _sharepoint_config(site_url):
    return {
        "site_url": site_url,
        "client_id": "client-id",
        "client_secret": "client-secret",
    }


def _get_wrapper_call_kwargs(site_url, site_path=_MISSING):
    with patch("elitea_sdk.tools.sharepoint.SharepointApiWrapper") as wrapper_cls:
        mock_wrapper = MagicMock()
        mock_wrapper.site_url = site_url
        mock_wrapper.get_available_tools.return_value = []
        wrapper_cls.return_value = mock_wrapper

        kwargs = {
            "selected_tools": [],
            "sharepoint_configuration": _sharepoint_config(site_url),
        }
        if site_path is not _MISSING:
            kwargs["site_path"] = site_path

        SharepointToolkit.get_toolkit(**kwargs)

    wrapper_cls.assert_called_once()
    return wrapper_cls.call_args.kwargs


class TestToolkitSitePath:
    """Test SharePoint toolkit site_path field and URL resolution."""

    def test_toolkit_config_schema_contains_site_path(self):
        """Verify site_path field is in the toolkit schema."""
        schema = SharepointToolkit.toolkit_config_schema().model_json_schema()

        assert "site_path" in schema["properties"]
        properties = schema["properties"]["site_path"]
        assert "sites/site-name" in properties["description"]
        assert "teams/team-name" in properties["description"]
        # Verify it's optional (has default)
        assert properties.get("default") is None

    @pytest.mark.parametrize(("site_path", "expected_site_url"), [
        ("sites/my-site", "https://tenant.sharepoint.com/sites/my-site"),
        ("teams/my-team", "https://tenant.sharepoint.com/teams/my-team"),
    ])
    def test_toolkit_resolves_tenant_url_with_site_path(self, site_path, expected_site_url):
        """Tenant URL plus site_path resolves to a full SharePoint URL."""
        call_kwargs = _get_wrapper_call_kwargs("https://tenant.sharepoint.com", site_path)

        assert call_kwargs["site_url"] == expected_site_url

    @pytest.mark.parametrize("site_url", [
        "https://tenant.sharepoint.com/sites/my-site",
        "https://tenant.sharepoint.com/teams/my-team",
    ])
    def test_toolkit_keeps_legacy_full_url_when_site_path_missing(self, site_url):
        """Full /sites/ and /teams/ URLs still work without site_path."""
        call_kwargs = _get_wrapper_call_kwargs(site_url)

        assert call_kwargs["site_url"] == site_url

    def test_toolkit_site_path_with_leading_trailing_slashes(self):
        """Test site_path normalization with slashes."""
        call_kwargs = _get_wrapper_call_kwargs("https://tenant.sharepoint.com/", "/teams/my-team/")

        assert call_kwargs["site_url"] == "https://tenant.sharepoint.com/teams/my-team"

    def test_toolkit_blank_site_path_keeps_legacy_site_url(self):
        """Blank site_path is treated as missing for backwards compatibility."""
        call_kwargs = _get_wrapper_call_kwargs("https://tenant.sharepoint.com/sites/my-site", "   ")

        assert call_kwargs["site_url"] == "https://tenant.sharepoint.com/sites/my-site"

    def test_toolkit_site_path_overrides_full_url(self):
        """Test that site_path overrides when both full URL and site_path provided."""
        call_kwargs = _get_wrapper_call_kwargs("https://tenant.sharepoint.com/sites/old-site", "teams/new-team")

        assert call_kwargs["site_url"] == "https://tenant.sharepoint.com/teams/new-team"

    @pytest.mark.parametrize("site_path", [
        "team/my-team",
        "https://tenant.sharepoint.com/teams/my-team",
        "sites/",
    ])
    def test_toolkit_rejects_invalid_site_path(self, site_path):
        """Invalid site_path values fail before wrapper construction."""
        with patch("elitea_sdk.tools.sharepoint.SharepointApiWrapper") as wrapper_cls:
            with pytest.raises(ToolException) as exc_info:
                SharepointToolkit.get_toolkit(
                    selected_tools=[],
                    sharepoint_configuration={
                        "site_url": "https://tenant.sharepoint.com",
                        "client_id": "client-id",
                        "client_secret": "client-secret",
                    },
                    site_path=site_path,
                )

        wrapper_cls.assert_not_called()
        assert "site_path must be relative" in str(exc_info.value)
        assert "sites/my-site" in str(exc_info.value)
        assert "teams/my-team" in str(exc_info.value)
