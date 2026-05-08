"""
Tests for ADO project configuration at toolkit level (not credentials level).

Issue #3620: Move ADO project id from Credentials to Toolkit for Universal Applicability
"""
import pytest
from unittest.mock import patch, MagicMock
from urllib.parse import urlparse
from pydantic import SecretStr

from elitea_sdk.configurations.ado import AdoConfiguration
from elitea_sdk.tools.ado.work_item import AzureDevOpsWorkItemsToolkit
from elitea_sdk.tools.ado.repos import AzureDevOpsReposToolkit
from elitea_sdk.tools.ado.wiki import AzureDevOpsWikiToolkit
from elitea_sdk.tools.ado.test_plan import AzureDevOpsPlansToolkit


class TestAdoConfigurationSchema:
    """Test that AdoConfiguration does NOT have project field."""

    def test_ado_configuration_has_no_project_field(self):
        """AdoConfiguration should only have organization_url and token."""
        schema = AdoConfiguration.model_json_schema()
        properties = schema.get('properties', {})

        assert 'organization_url' in properties
        assert 'token' in properties
        assert 'project' not in properties, "project should NOT be in AdoConfiguration"

    def test_ado_configuration_can_be_created_without_project(self):
        """AdoConfiguration should be creatable without project field."""
        config = AdoConfiguration(
            organization_url="https://dev.azure.com/myorg",
            token=SecretStr("test-token")
        )
        assert config.organization_url == "https://dev.azure.com/myorg"
        assert config.token.get_secret_value() == "test-token"


class TestToolkitSchemaHasProject:
    """Test that all ADO toolkits have project field in their schema."""

    @pytest.mark.parametrize("toolkit_class,expected_project", [
        (AzureDevOpsWorkItemsToolkit, True),
        (AzureDevOpsReposToolkit, True),
        (AzureDevOpsWikiToolkit, True),
        (AzureDevOpsPlansToolkit, True),
    ])
    def test_toolkit_schema_has_project_field(self, toolkit_class, expected_project):
        """Each ADO toolkit should have project field in its config schema."""
        schema_model = toolkit_class.toolkit_config_schema()
        schema = schema_model.model_json_schema()
        properties = schema.get('properties', {})

        if expected_project:
            assert 'project' in properties, f"{toolkit_class.__name__} should have project field"
            assert properties['project']['type'] == 'string'


class TestAdoConfigurationCheckConnection:
    """Test AdoConfiguration.check_connection validates only org_url and token."""

    def test_check_connection_validates_organization_url_format(self):
        """check_connection should validate organization URL format."""
        error = AdoConfiguration.check_connection({
            'organization_url': 'invalid-url',
            'token': 'test-token'
        })
        assert error is not None
        assert 'Organization URL' in error

    def test_check_connection_requires_organization_url(self):
        """check_connection should require organization_url."""
        error = AdoConfiguration.check_connection({
            'token': 'test-token'
        })
        assert error is not None
        assert 'Organization URL is required' in error

    def test_check_connection_empty_organization_url(self):
        """check_connection should reject empty organization_url."""
        error = AdoConfiguration.check_connection({
            'organization_url': '',
            'token': 'test-token'
        })
        assert error is not None
        assert 'empty' in error.lower()

    @patch('requests.get')
    def test_check_connection_with_valid_token(self, mock_get):
        """check_connection should validate token against profile endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        error = AdoConfiguration.check_connection({
            'organization_url': 'https://dev.azure.com/myorg',
            'token': 'valid-token'
        })

        assert error is None
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        parsed_url = urlparse(call_url)
        assert parsed_url.hostname == 'vssps.dev.azure.com'
        assert 'profile' in call_url

    @patch('requests.get')
    def test_check_connection_with_invalid_token(self, mock_get):
        """check_connection should return error for invalid token."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        error = AdoConfiguration.check_connection({
            'organization_url': 'https://dev.azure.com/myorg',
            'token': 'invalid-token'
        })

        assert error is not None
        assert 'Invalid' in error or 'expired' in error


class TestToolkitCheckConnection:
    """Test that toolkit check_connection uses project from toolkit config."""

    @patch('requests.get')
    def test_work_items_toolkit_check_connection_uses_toolkit_project(self, mock_get):
        """ADO boards toolkit check_connection should use self.project."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        schema_model = AzureDevOpsWorkItemsToolkit.toolkit_config_schema()
        instance = schema_model(
            ado_configuration=AdoConfiguration(
                organization_url="https://dev.azure.com/myorg",
                token=SecretStr("test-token")
            ),
            project="MyProject",
            selected_tools=[]
        )

        instance.check_connection()

        call_url = mock_get.call_args[0][0]
        assert 'MyProject' in call_url
        assert '_apis/wit/workitemtypes' in call_url

    @patch('requests.get')
    def test_repos_toolkit_check_connection_uses_toolkit_project(self, mock_get):
        """ADO repos toolkit check_connection should use self.project."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        schema_model = AzureDevOpsReposToolkit.toolkit_config_schema()
        instance = schema_model(
            ado_configuration=AdoConfiguration(
                organization_url="https://dev.azure.com/myorg",
                token=SecretStr("test-token")
            ),
            project="MyProject",
            repository_id="my-repo",
            selected_tools=[]
        )

        instance.check_connection()

        call_url = mock_get.call_args[0][0]
        assert 'MyProject' in call_url
        assert '_apis/git/repositories' in call_url

    @patch('requests.get')
    def test_wiki_toolkit_check_connection_uses_toolkit_project(self, mock_get):
        """ADO wiki toolkit check_connection should use self.project."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        schema_model = AzureDevOpsWikiToolkit.toolkit_config_schema()
        instance = schema_model(
            ado_configuration=AdoConfiguration(
                organization_url="https://dev.azure.com/myorg",
                token=SecretStr("test-token")
            ),
            project="MyProject",
            selected_tools=[]
        )

        instance.check_connection()

        call_url = mock_get.call_args[0][0]
        assert 'MyProject' in call_url
        assert '_apis/wiki/wikis' in call_url

    @patch('requests.get')
    def test_plans_toolkit_check_connection_uses_toolkit_project(self, mock_get):
        """ADO plans toolkit check_connection should use self.project."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        schema_model = AzureDevOpsPlansToolkit.toolkit_config_schema()
        instance = schema_model(
            ado_configuration=AdoConfiguration(
                organization_url="https://dev.azure.com/myorg",
                token=SecretStr("test-token")
            ),
            project="MyProject",
            selected_tools=[]
        )

        instance.check_connection()

        call_url = mock_get.call_args[0][0]
        assert 'MyProject' in call_url
        assert '_apis/testplan/plans' in call_url


class TestToolkitProjectRequired:
    """Test that project is required at toolkit level."""

    def test_work_items_toolkit_requires_project(self):
        """ADO boards toolkit should require project field."""
        schema_model = AzureDevOpsWorkItemsToolkit.toolkit_config_schema()
        schema = schema_model.model_json_schema()
        required = schema.get('required', [])

        assert 'project' in required

    def test_repos_toolkit_requires_project(self):
        """ADO repos toolkit should require project field."""
        schema_model = AzureDevOpsReposToolkit.toolkit_config_schema()
        schema = schema_model.model_json_schema()
        required = schema.get('required', [])

        assert 'project' in required

    def test_wiki_toolkit_requires_project(self):
        """ADO wiki toolkit should require project field."""
        schema_model = AzureDevOpsWikiToolkit.toolkit_config_schema()
        schema = schema_model.model_json_schema()
        required = schema.get('required', [])

        assert 'project' in required

    def test_plans_toolkit_requires_project(self):
        """ADO plans toolkit should require project field."""
        schema_model = AzureDevOpsPlansToolkit.toolkit_config_schema()
        schema = schema_model.model_json_schema()
        required = schema.get('required', [])

        assert 'project' in required
