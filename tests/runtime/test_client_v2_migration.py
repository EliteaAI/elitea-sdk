"""Tests that EliteAClient and SandboxClient use only /api/v2 endpoints."""
import pytest
from unittest.mock import MagicMock, patch


BASE = "https://platform.example.com"
PROJECT_ID = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_elitea_client(**kwargs):
    from elitea_sdk.runtime.clients.client import EliteAClient
    return EliteAClient(base_url=BASE, project_id=PROJECT_ID, auth_token="tok", **kwargs)


def make_sandbox_client(**kwargs):
    from elitea_sdk.runtime.clients.sandbox_client import SandboxClient
    return SandboxClient(base_url=BASE, project_id=PROJECT_ID, auth_token="tok", **kwargs)


# ---------------------------------------------------------------------------
# EliteAClient — URL attribute assertions
# ---------------------------------------------------------------------------

class TestEliteAClientV2Urls:
    def setup_method(self):
        self.client = make_elitea_client()

    def test_no_api_path_v1_attribute(self):
        assert not hasattr(self.client, 'api_path'), "api_path (v1) must not exist"

    def test_application_versions(self):
        assert "/api/v2/" in self.client.application_versions
        assert "elitea_core/version/prompt_lib" in self.client.application_versions
        assert "/api/v1/" not in self.client.application_versions

    def test_list_apps_url(self):
        assert "/api/v2/" in self.client.list_apps_url
        assert "elitea_core/applications/prompt_lib" in self.client.list_apps_url
        assert "/api/v1/" not in self.client.list_apps_url

    def test_secrets_url(self):
        assert "/api/v2/" in self.client.secrets_url
        assert "/api/v1/" not in self.client.secrets_url

    def test_models_url(self):
        assert "/api/v2/" in self.client.models_url
        assert "/api/v1/" not in self.client.models_url

    def test_mcp_tools_list(self):
        assert "/api/v2/" in self.client.mcp_tools_list
        assert "elitea_core/tools_list" in self.client.mcp_tools_list

    def test_mcp_tools_call(self):
        assert "/api/v2/" in self.client.mcp_tools_call
        assert "elitea_core/tools_call" in self.client.mcp_tools_call

    def test_toolkit_url(self):
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.ok = False
            mock_resp.text = "not found"
            mock_get.return_value = mock_resp
            try:
                self.client.toolkit(7)
            except ValueError:
                pass  # expected — mock returns not-ok
        called_url = mock_get.call_args[0][0]
        assert "/api/v2/" in called_url, f"toolkit() used wrong URL: {called_url}"
        assert "elitea_core/tool/prompt_lib" in called_url

    def test_removed_integration_details_attr(self):
        assert not hasattr(self.client, 'integration_details')

    def test_removed_configurations_url_attr(self):
        assert not hasattr(self.client, 'configurations_url')

    def test_removed_get_integration_details_method(self):
        assert not hasattr(self.client, 'get_integration_details')

    def test_removed_fetch_available_configurations_method(self):
        assert not hasattr(self.client, 'fetch_available_configurations')

    def test_no_configurations_attr(self):
        assert not hasattr(self.client, 'configurations')


class TestEliteAClientGetAppVersionDetails:
    def setup_method(self):
        self.client = make_elitea_client()

    def test_patches_v2_url(self):
        with patch("requests.patch") as mock_patch:
            mock_patch.return_value = MagicMock(ok=True, json=lambda: {"llm_settings": {}})
            self.client.get_app_version_details(1, 2)
        url = mock_patch.call_args[0][0]
        assert "/api/v2/" in url, f"Expected v2 URL, got: {url}"
        assert "elitea_core/version/prompt_lib" in url

    def test_uses_application_versions_attr(self):
        with patch("requests.patch") as mock_patch:
            mock_patch.return_value = MagicMock(ok=True, json=lambda: {"llm_settings": {}})
            self.client.get_app_version_details(10, 20)
        url = mock_patch.call_args[0][0]
        assert url.startswith(self.client.application_versions)

    def test_no_json_body_sent(self):
        with patch("requests.patch") as mock_patch:
            mock_patch.return_value = MagicMock(ok=True, json=lambda: {"llm_settings": {}})
            self.client.get_app_version_details(1, 2)
        # no json= kwarg should be passed
        kwargs = mock_patch.call_args[1]
        assert "json" not in kwargs


# ---------------------------------------------------------------------------
# SandboxClient — URL attribute assertions
# ---------------------------------------------------------------------------

class TestSandboxClientV2Urls:
    def setup_method(self):
        self.client = make_sandbox_client()

    def test_no_api_path_v1_attribute(self):
        assert not hasattr(self.client, 'api_path'), "api_path (v1) must not exist"

    def test_app_url(self):
        assert "/api/v2/" in self.client.app
        assert "elitea_core/application/prompt_lib" in self.client.app

    def test_application_versions(self):
        assert "/api/v2/" in self.client.application_versions
        assert "elitea_core/version/prompt_lib" in self.client.application_versions

    def test_list_apps_url(self):
        assert "/api/v2/" in self.client.list_apps_url
        assert "elitea_core/applications/prompt_lib" in self.client.list_apps_url

    def test_mcp_tools_list_no_user_id(self):
        url = self.client.mcp_tools_list
        assert "/api/v2/" in url
        assert "elitea_core/tools_list" in url
        assert url == f"{BASE}/api/v2/elitea_core/tools_list/{PROJECT_ID}"

    def test_mcp_tools_call_no_user_id(self):
        url = self.client.mcp_tools_call
        assert "/api/v2/" in url
        assert "elitea_core/tools_call" in url
        assert url == f"{BASE}/api/v2/elitea_core/tools_call/{PROJECT_ID}"

    def test_secrets_url(self):
        assert "/api/v2/" in self.client.secrets_url

    def test_artifacts_url(self):
        assert "/api/v2/" in self.client.artifacts_url

    def test_artifact_url(self):
        assert "/api/v2/" in self.client.artifact_url

    def test_bucket_url(self):
        assert "/api/v2/" in self.client.bucket_url

    def test_removed_predict_url(self):
        assert not hasattr(self.client, 'predict_url')

    def test_removed_prompt_versions(self):
        assert not hasattr(self.client, 'prompt_versions')

    def test_removed_prompts(self):
        assert not hasattr(self.client, 'prompts')

    def test_auth_user_url(self):
        assert hasattr(self.client, 'auth_user_url')
        assert "/api/v2/" in self.client.auth_user_url
        assert "/api/v1/" not in self.client.auth_user_url

    def test_removed_integration_details(self):
        assert not hasattr(self.client, 'integration_details')

    def test_removed_configurations_url(self):
        assert not hasattr(self.client, 'configurations_url')

    def test_get_user_data_exists(self):
        assert hasattr(self.client, 'get_user_data')
        assert callable(self.client.get_user_data)

    def test_removed_get_real_user_id(self):
        assert not hasattr(self.client, '_get_real_user_id')

    def test_removed_get_integration_details(self):
        assert not hasattr(self.client, 'get_integration_details')

    def test_removed_fetch_available_configurations(self):
        assert not hasattr(self.client, 'fetch_available_configurations')

    def test_no_configurations_attr(self):
        assert not hasattr(self.client, 'configurations')


class TestSandboxClientMcpV2:
    def setup_method(self):
        self.client = make_sandbox_client()

    def test_get_mcp_toolkits_calls_v2_url_directly(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(json=lambda: [{"name": "tool1"}])
            result = self.client.get_mcp_toolkits()
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert called_url == self.client.mcp_tools_list
        assert "/api/v2/" in called_url

    def test_mcp_tool_call_posts_to_v2_url_directly(self):
        params = {"params": {"arguments": {}}}
        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(json=lambda: {"result": "ok"})
            self.client.mcp_tool_call(params)
        called_url = mock_post.call_args[0][0]
        assert called_url == self.client.mcp_tools_call
        assert "/api/v2/" in called_url


class TestSandboxClientGetAppVersionDetails:
    def setup_method(self):
        self.client = make_sandbox_client()

    def test_patches_v2_url(self):
        with patch("requests.patch") as mock_patch:
            mock_patch.return_value = MagicMock(ok=True, json=lambda: {"llm_settings": {}})
            self.client.get_app_version_details(3, 4)
        url = mock_patch.call_args[0][0]
        assert "/api/v2/" in url
        assert "elitea_core/version/prompt_lib" in url

    def test_no_json_body_sent(self):
        with patch("requests.patch") as mock_patch:
            mock_patch.return_value = MagicMock(ok=True, json=lambda: {"llm_settings": {}})
            self.client.get_app_version_details(3, 4)
        kwargs = mock_patch.call_args[1]
        assert "json" not in kwargs
