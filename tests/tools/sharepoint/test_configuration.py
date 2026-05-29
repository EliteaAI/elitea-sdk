"""Tests for SharePoint configuration validation."""

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

    def test_error_message_mentions_teams_format(self):
        """Verify error messages include /teams/ format examples."""
        error = SharepointConfiguration._check_connection_client_credentials({
            "client_id": "client-id",
            "client_secret": "client-secret",
            "site_url": "https:///sites/my-site",
        })

        assert "https://<tenant>.sharepoint.com/sites/<site>" in error
        assert "https://<tenant>.sharepoint.com/teams/<team>" in error
