"""Tests for SharePoint configuration validation."""

import builtins

import pytest

from elitea_sdk.configurations.sharepoint import SharepointConfiguration


class TestSharepointConfiguration:
    """Test SharePoint configuration validation for /teams/ URLs."""

    def test_validate_tenant_url_without_path(self):
        """Test validation of tenant-only URL."""
        site_url, error = SharepointConfiguration._validate_site_url({
            "site_url": "https://tenant.sharepoint.com/"
        })

        assert error is None
        assert site_url == "https://tenant.sharepoint.com"

    def test_validate_sites_url(self):
        """Test validation of full /sites/ URL."""
        site_url, error = SharepointConfiguration._validate_site_url({
            "site_url": "https://tenant.sharepoint.com/sites/my-site/"
        })

        assert error is None
        assert site_url == "https://tenant.sharepoint.com/sites/my-site"

    def test_validate_teams_url(self):
        """Test validation of full /teams/ URL."""
        site_url, error = SharepointConfiguration._validate_site_url({
            "site_url": "https://tenant.sharepoint.com/teams/my-team/"
        })

        assert error is None
        assert site_url == "https://tenant.sharepoint.com/teams/my-team"

    def test_validate_url_strips_trailing_slash(self):
        """Test that trailing slashes are removed."""
        site_url, error = SharepointConfiguration._validate_site_url({
            "site_url": "https://tenant.sharepoint.com/teams/my-team///"
        })

        assert error is None
        assert site_url == "https://tenant.sharepoint.com/teams/my-team"
        assert not site_url.endswith("/")

    def test_validate_empty_url_returns_error(self):
        """Test that empty URL is rejected."""
        site_url, error = SharepointConfiguration._validate_site_url({
            "site_url": ""
        })

        assert site_url is None
        assert error == "Site URL cannot be empty"

    def test_validate_missing_url_returns_error(self):
        """Test that missing URL is rejected."""
        site_url, error = SharepointConfiguration._validate_site_url({})

        assert site_url is None
        assert error == "Site URL is required"

    def test_validate_url_without_protocol_returns_error(self):
        """Test that URL without http:// or https:// is rejected."""
        site_url, error = SharepointConfiguration._validate_site_url({
            "site_url": "tenant.sharepoint.com/sites/my-site"
        })

        assert site_url is None
        assert error == "Site URL must start with http:// or https://"

    def test_validate_non_string_url_returns_error(self):
        """Test that non-string URL is rejected."""
        site_url, error = SharepointConfiguration._validate_site_url({
            "site_url": 12345
        })

        assert site_url is None
        assert error == "Site URL must be a string"

    def test_client_credentials_falls_back_to_graph_when_acs_unavailable(self, monkeypatch):
        """Azure AD app-only credentials still validate through Graph when ACS cannot be used."""
        original_import = builtins.__import__

        def import_with_missing_office365(name, *args, **kwargs):
            if name.startswith("office365"):
                raise ImportError("office365-rest-python-client is not installed")
            return original_import(name, *args, **kwargs)

        class GraphAuthHelper:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def generate_token_and_site_id(self, site_url):
                assert site_url == "https://tenant.sharepoint.com/teams/my-team"
                return "token", "site-id"

        monkeypatch.setattr(builtins, "__import__", import_with_missing_office365)
        monkeypatch.setattr(
            "elitea_sdk.tools.sharepoint.authorization_helper.SharepointAuthorizationHelper",
            GraphAuthHelper,
        )

        error = SharepointConfiguration._check_connection_client_credentials({
            "site_url": "https://tenant.sharepoint.com/teams/my-team",
            "client_id": "client-id",
            "client_secret": "client-secret",
        })

        assert error is None
