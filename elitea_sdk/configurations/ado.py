import re
from typing import Optional

import requests
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class AdoConfiguration(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "metadata": {
                "label": "Ado",
                "icon_url": None,
                "section": "credentials",
                "type": "ado",
                "categories": ["project management"],
            }
        }
    )
    organization_url: str = Field(description="Base API URL")
    token: Optional[SecretStr] = Field(description="ADO Token")

    @staticmethod
    def check_connection(settings: dict) -> str | None:
        """
        Test the connection to Azure DevOps API.

        Args:
            settings: Dictionary containing 'organization_url' and optionally 'token'

        Returns:
            None if connection is successful, error message string otherwise
        """
        organization_url = settings.get("organization_url")
        if organization_url is None or organization_url == "":
            if organization_url == "":
                return "Organization URL cannot be empty"
            return "Organization URL is required"

        # Validate organization URL format
        if not isinstance(organization_url, str):
            return "Organization URL must be a string"

        organization_url = organization_url.strip()
        if not organization_url:
            return "Organization URL cannot be empty"

        if not organization_url.startswith(("http://", "https://")):
            return "Organization URL must start with http:// or https://"

        # Remove trailing slash for consistency
        organization_url = organization_url.rstrip("/")

        token = settings.get("token")

        # Extract secret value if it's a SecretStr
        if token is not None and hasattr(token, "get_secret_value"):
            token = token.get_secret_value()

        # Validate token if provided
        if token is not None and (not token or not token.strip()):
            return "Token cannot be empty if provided"

        # Strictly require a canonical organization URL so we can build reliable API URLs.
        # Supported formats:
        # - https://dev.azure.com/<org>
        # - https://<org>.visualstudio.com
        org_name: str | None = None
        org_url_kind: str | None = None  # 'dev.azure.com' | '*.visualstudio.com'
        m = re.match(r"^https?://dev\.azure\.com/(?P<org>[^/]+)$", organization_url, flags=re.IGNORECASE)
        if m:
            org_name = m.group('org')
            org_url_kind = 'dev.azure.com'
        else:
            m = re.match(r"^https?://(?P<org>[^/.]+)\.visualstudio\.com$", organization_url, flags=re.IGNORECASE)
            if m:
                org_name = m.group('org')
                org_url_kind = '*.visualstudio.com'

        if org_name is None:
            return (
                "Organization URL format is invalid. Use 'https://dev.azure.com/<org>' "
                "(recommended) or 'https://<org>.visualstudio.com'."
            )

        # Auth-required endpoint to validate PAT (works regardless of project visibility)
        if org_url_kind == 'dev.azure.com':
            base_url = f"https://dev.azure.com/{org_name}"
        else:
            base_url = f"https://{org_name}.visualstudio.com"

        # Append the organization-level Projects API endpoint
        # We add top=1 so the response is tiny, making it incredibly fast
        test_url = f"{base_url}/_apis/projects?api-version=7.1&$top=1"

        try:
            if token:
                # Use Basic Auth with PAT token (username can be empty)
                from requests.auth import HTTPBasicAuth
                auth = HTTPBasicAuth("", token)

                # Validate token against projects endpoint
                project_resp = requests.get(test_url, auth=auth, timeout=10)
                if project_resp.status_code == 200:
                    return None  # Connection successful
                elif project_resp.status_code == 401:
                    return "Invalid or expired token (PAT). Please generate a new token and try again."
                elif project_resp.status_code == 403:
                    return "Token is valid but lacks permission to access profile. Check PAT scopes/permissions."
                elif project_resp.status_code == 404:
                    return "Organization not found. Verify the Organization URL."
                else:
                    return f"Token validation failed (HTTP {project_resp.status_code})."
            else:
                # Without token, just verify the organization URL is reachable
                # Try to access the projects list endpoint (may work for public orgs)
                projects_url = f"{organization_url}/_apis/projects?api-version=7.0&$top=1"
                response = requests.get(projects_url, timeout=10)
                if response.status_code == 200:
                    return None  # Connection successful
                elif response.status_code == 401:
                    return "Authentication required - please provide a token"
                elif response.status_code == 404:
                    return "Organization not found. Verify the Organization URL."
                else:
                    return f"Connection failed (HTTP {response.status_code})."

        except requests.exceptions.Timeout:
            return "Connection timeout - Azure DevOps did not respond within 10 seconds"
        except requests.exceptions.ConnectionError:
            return "Connection error - unable to reach Azure DevOps. Check the Organization URL and your network."
        except requests.exceptions.RequestException as e:
            return f"Request failed: {str(e)}"
        except Exception:
            return "Unexpected error during Azure DevOps connection check"
