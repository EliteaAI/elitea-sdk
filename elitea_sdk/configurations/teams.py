import requests
import logging
from typing import Optional, List, Dict
from pydantic import BaseModel, ConfigDict, Field, SecretStr

log = logging.getLogger(__name__)


class TeamsConfiguration(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "metadata": {
                "label": "Teams",
                "icon_url": "teams.svg",
                "sections": {
                    "auth": {
                        "required": True,
                        "subsections": [
                            {
                                "name": "Delegated",
                                "fields": ["oauth_discovery_endpoint", "scopes", "auto_refresh_token"]
                            }
                        ]
                    }
                },
                "section": "credentials",
                "type": "teams",
                "categories": ["office"],
                "extra_categories": ["teams", "microsoft", "chat", "messaging"],
            }
        }
    )

    client_id: str = Field(description="Azure AD Application (client) ID")
    client_secret: SecretStr = Field(description="Azure AD Client Secret")

    oauth_discovery_endpoint: Optional[str] = Field(
        default=None,
        description="OAuth Discovery Endpoint. Format: https://login.microsoftonline.com/{tenant_id}"
    )
    scopes: Optional[List[str]] = Field(
        default=None,
        description=(
            "OAuth Scopes (e.g., User.Read, User.ReadBasic.All, Chat.Read, ChatMessage.Send, Chat.Create, "
            "Team.ReadBasic.All, Channel.ReadBasic.All, ChannelMessage.Send)"
        )
    )
    auto_refresh_token: Optional[bool] = Field(
        default=True,
        description="Automatically refresh the access token using the offline_access scope."
    )

    @staticmethod
    def check_connection(settings: dict) -> str | None:
        """Test the connection to Microsoft Graph Teams API.

        Uses delegated OAuth flow - verifies the access token by calling
        the /me/joinedTeams endpoint.

        Returns:
            None if connection is successful, error message string otherwise.

        Raises:
            McpAuthorizationRequired: when token is absent or invalid/expired.
        """
        safe_settings = {
            k: "***" if k in ("client_secret", "access_token", "refresh_token") else v
            for k, v in settings.items()
        }
        log.debug(f"Checking Teams connection with settings: {safe_settings}")

        oauth_discovery_endpoint = settings.get("oauth_discovery_endpoint")
        if not oauth_discovery_endpoint:
            return "OAuth discovery endpoint is required for Teams"

        return TeamsConfiguration._check_connection_delegated(settings, oauth_discovery_endpoint)

    @staticmethod
    def _check_connection_delegated(settings: dict, oauth_discovery_endpoint: str) -> str | None:
        """Delegated flow - verify access token via Graph API."""
        scopes = settings.get("scopes")
        configuration_uuid = settings.get("configuration_uuid")
        access_token = settings.get("access_token")

        if not access_token:
            raise TeamsConfiguration._build_mcp_authorization_required(
                message="Teams requires OAuth authorization. Please complete the OAuth flow to obtain an access token.",
                oauth_discovery_endpoint=oauth_discovery_endpoint,
                scopes=scopes,
                configuration_uuid=configuration_uuid,
            )

        return TeamsConfiguration._call_graph_api(
            access_token, oauth_discovery_endpoint, scopes, configuration_uuid
        )

    @staticmethod
    def _call_graph_api(
        access_token: str,
        oauth_discovery_endpoint: str,
        scopes: Optional[List[str]] = None,
        configuration_uuid: Optional[str] = None
    ) -> str | None:
        """Health-check access token using Microsoft Graph Teams API."""
        try:
            graph_url = "https://graph.microsoft.com/v1.0/me/joinedTeams"
            resp = requests.get(
                graph_url,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )

            if resp.status_code == 200:
                return None
            elif resp.status_code == 401:
                api_message = TeamsConfiguration._extract_api_error_message(resp)
                raise TeamsConfiguration._build_mcp_authorization_required(
                    message=f"Teams access token is invalid or expired: {api_message}. Please re-authorize.",
                    oauth_discovery_endpoint=oauth_discovery_endpoint,
                    scopes=scopes,
                    status=401,
                    configuration_uuid=configuration_uuid,
                )
            elif resp.status_code == 403:
                return "Access forbidden - token lacks required Microsoft Graph Teams permissions (Team.ReadBasic.All)"
            else:
                return f"Microsoft Graph API request failed with status {resp.status_code}"

        except requests.exceptions.Timeout:
            return "Connection timeout - Microsoft Graph is not responding"
        except requests.exceptions.ConnectionError:
            return "Connection error - unable to reach Microsoft Graph"
        except requests.exceptions.RequestException as exc:
            return f"Request failed: {str(exc)}"
        except Exception as exc:
            if type(exc).__name__ == "McpAuthorizationRequired":
                raise
            log.error(f"Error calling Microsoft Graph API: {exc}")
            return f"Unexpected error: {str(exc)}"

    @staticmethod
    def _extract_api_error_message(response) -> str:
        """Extract human-readable error message from Graph API response."""
        try:
            body = response.json()
            if "error" in body and isinstance(body["error"], dict):
                return body["error"].get("message") or str(body["error"])
            if "message" in body:
                return body["message"]
            return str(body)
        except Exception:
            return response.text or "unknown error"

    @staticmethod
    def _build_mcp_authorization_required(
        message: str,
        oauth_discovery_endpoint: str,
        scopes: Optional[List[str]],
        status: Optional[int] = None,
        configuration_uuid: Optional[str] = None,
        toolkit_id: Optional[int] = None,
        toolkit_name: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[SecretStr | str] = None,
    ) -> "McpAuthorizationRequired":
        """Build McpAuthorizationRequired exception for OAuth flow."""
        from ..runtime.utils.mcp_oauth import (
            McpAuthorizationRequired,
            fetch_oauth_authorization_server_metadata,
        )

        effective_scopes = list(scopes or [])
        if "offline_access" not in effective_scopes:
            effective_scopes.insert(0, "offline_access")

        base_discovery = oauth_discovery_endpoint.rstrip("/")
        azure_v2_endpoint = f"{base_discovery}/v2.0/.well-known/openid-configuration"

        openid_meta = fetch_oauth_authorization_server_metadata(
            base_discovery,
            extra_endpoints=[azure_v2_endpoint],
        )

        resource_metadata_url = azure_v2_endpoint

        authorization_endpoint = (openid_meta or {}).get(
            "authorization_endpoint",
            f"{base_discovery}/v2.0/oauth2/authorize",
        )
        token_endpoint = (openid_meta or {}).get(
            "token_endpoint",
            f"{base_discovery}/v2.0/oauth2/token",
        )
        jwks_uri = (openid_meta or {}).get("jwks_uri")
        issuer = (openid_meta or {}).get("issuer", base_discovery)
        scopes_supported = list((openid_meta or {}).get("scopes_supported") or [])

        if effective_scopes:
            for s in effective_scopes:
                if s not in scopes_supported:
                    scopes_supported.append(s)

        www_authenticate = (
            f'Bearer error="unauthorized_client", '
            f'error_description="No access token was provided", '
            f'resource_metadata="{resource_metadata_url}", '
            f'authorization_uri="{authorization_endpoint}"'
        )

        oauth_authorization_server: Dict = {
            "issuer": issuer,
            "authorization_endpoint": authorization_endpoint,
            "token_endpoint": token_endpoint,
        }
        if jwks_uri:
            oauth_authorization_server["jwks_uri"] = jwks_uri
        if scopes_supported:
            oauth_authorization_server["scopes_supported"] = scopes_supported
        if openid_meta:
            for key in ("response_types_supported", "claims_supported",
                        "id_token_signing_alg_values_supported"):
                if key in openid_meta:
                    oauth_authorization_server[key] = openid_meta[key]

        resource_metadata: Dict = {
            "resource_name": "Teams",
            "resource": "https://graph.microsoft.com/Teams",
            "authorization_servers": [base_discovery],
            "bearer_methods_supported": ["header"],
            "oauth_authorization_server": oauth_authorization_server,
        }
        if effective_scopes:
            resource_metadata["scopes_supported"] = effective_scopes
        if configuration_uuid:
            resource_metadata["configuration_uuid"] = configuration_uuid
        if toolkit_id is not None:
            resource_metadata["toolkit_id"] = toolkit_id

        provided_settings: Dict = {}
        if client_id:
            provided_settings["mcp_client_id"] = client_id
        if client_secret:
            secret_value = (
                client_secret.get_secret_value()
                if hasattr(client_secret, "get_secret_value")
                else str(client_secret)
            )
            if secret_value:
                from ..runtime.utils.utils import mask_secret
                provided_settings["mcp_client_secret"] = mask_secret(secret_value)
        if effective_scopes:
            provided_settings["scopes"] = effective_scopes
        if provided_settings:
            resource_metadata["provided_settings"] = provided_settings

        auth_error = McpAuthorizationRequired(
            message=message,
            server_url="https://graph.microsoft.com",
            resource_metadata_url=resource_metadata_url,
            www_authenticate=www_authenticate,
            resource_metadata=resource_metadata,
            status=status,
            tool_name="Teams",
            toolkit_type="teams",
            toolkit_name=toolkit_name,
        )
        if toolkit_id is not None:
            auth_error.toolkit_id = toolkit_id
        return auth_error
