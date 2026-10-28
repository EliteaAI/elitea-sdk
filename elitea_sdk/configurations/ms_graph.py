import logging
import re
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from .microsoft_graph_scopes import GRAPH_RESOURCE_PREFIX
from .oauth_app_mode import OAUTH_APP_ID_FIELD, require_app_client

log = logging.getLogger(__name__)

MS_GRAPH_DEFAULT_SCOPES = ["User.Read"]
GRAPH_DEFAULT_SCOPE = f"{GRAPH_RESOURCE_PREFIX}.default"

# OIDC scopes the sign-in may carry next to Graph permissions
_OIDC_SCOPES = {"openid", "profile", "email", "offline_access"}
_SPLIT_RE = re.compile(r"[,\s]+")
# A Graph permission name: Mail.Read, Sites.ReadWrite.All, User.Read ...
_PERMISSION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*(\.[A-Za-z0-9_-]+)+$")

MS_GRAPH_SCOPES_NOTE = (
    "Microsoft Graph delegated permissions requested at sign-in, e.g. Mail.Read, Files.ReadWrite, Sites.Read.All. "
    "ELITEA passes them to Microsoft as-is; empty means User.Read.\n\n"
    "- Only Microsoft Graph permissions: a bare name or https://graph.microsoft.com/<name>. "
    "https://graph.microsoft.com/.default requests every permission the app was granted.\n"
    "- The token always holds every permission you already approved for this Entra app.\n"
    "- After adding a permission, sign in again so the token gets it.\n"
    "- offline_access is always added so the token can be refreshed."
)


def normalize_ms_graph_scopes(value: Any) -> List[str]:
    """Free-text Microsoft Graph permissions from a list or a comma / space separated string.

    Strips the https://graph.microsoft.com/ prefix (keeps the full ``.default`` scope), drops OIDC
    scopes and duplicates (case-insensitive, first spelling wins) and rejects other resources: the
    token is for Microsoft Graph only. Empty means User.Read.
    """
    if value is None:
        items: List[Any] = []
    elif isinstance(value, str):
        items = _SPLIT_RE.split(value)
    elif isinstance(value, (list, tuple, set)):
        items = [part for item in value for part in _SPLIT_RE.split(str(item))]
    else:
        items = [value]

    result: List[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text.lower() in _OIDC_SCOPES:
            continue
        if text.lower().startswith(GRAPH_RESOURCE_PREFIX):
            text = text[len(GRAPH_RESOURCE_PREFIX):]
        if text.lower() == ".default":
            text = GRAPH_DEFAULT_SCOPE
        elif not _PERMISSION_RE.match(text):
            raise ValueError(
                f"'{item}' is not a Microsoft Graph permission. "
                "Use names like Mail.Read or https://graph.microsoft.com/Mail.Read"
            )
        if text.lower() not in seen:
            seen.add(text.lower())
            result.append(text)
    if GRAPH_DEFAULT_SCOPE in result and len(result) > 1:
        raise ValueError(f"{GRAPH_DEFAULT_SCOPE} cannot be combined with other permissions")
    return result or list(MS_GRAPH_DEFAULT_SCOPES)


class MsGraphConfiguration(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "metadata": {
                "label": "Microsoft Graph",
                "icon_url": "ms_graph.svg",
                "sections": {
                    "auth": {
                        "required": True,
                        "subsections": [
                            {
                                "name": "Default app",
                                "key_fields": ["oauth_app_id"],
                                "fields": ["oauth_app_id"]
                            },
                            {
                                "name": "Custom app",
                                "key_fields": ["client_id"],
                                "fields": ["client_id", "client_secret", "oauth_discovery_endpoint", "scopes",
                                           "auto_refresh_token"]
                            }
                        ]
                    }
                },
                "section": "credentials",
                "type": "ms_graph",
                "categories": ["office"],
                "extra_categories": ["microsoft", "graph", "office 365", "microsoft 365", "entra"],
            }
        }
    )

    oauth_app_id: Optional[str] = OAUTH_APP_ID_FIELD
    client_id: Optional[str] = Field(default=None, description="Azure AD Application (client) ID")
    client_secret: Optional[SecretStr] = Field(default=None, description="Azure AD Client Secret")

    oauth_discovery_endpoint: Optional[str] = Field(
        default=None,
        description="OAuth Discovery Endpoint. Format: https://login.microsoftonline.com/{tenant_id}"
    )
    scopes: Optional[List[str]] = Field(
        default=list(MS_GRAPH_DEFAULT_SCOPES),
        description=MS_GRAPH_SCOPES_NOTE,
    )
    auto_refresh_token: Optional[bool] = Field(
        default=True,
        description="Automatically refresh the access token using the offline_access scope."
    )
    # Set by the Login button on the toolkit page; lets scheduled runs work without a browser login
    refresh_token: Optional[SecretStr] = Field(
        default=None, description="Refresh token of the signed-in user (set by Login)",
        json_schema_extra={"hidden": True},
    )

    @field_validator("oauth_discovery_endpoint")
    @classmethod
    def _entra_authority_only(cls, value):
        # The client secret and the stored login are posted to this authority's token endpoint
        from ..runtime.utils.mcp_oauth import is_entra_url
        if value and not is_entra_url(value):
            raise ValueError("Must be https://login.microsoftonline.com/<tenant_id>")
        return value

    @model_validator(mode="after")
    def _app_or_client(self):
        require_app_client(self)
        return self

    @field_validator("scopes", mode="before")
    @classmethod
    def _normalize_scopes(cls, value):
        return normalize_ms_graph_scopes(value)

    @staticmethod
    def check_connection(settings: dict) -> str | None:
        """Test the connection to Microsoft Graph with the signed-in user's token (GET /me).

        Returns:
            None if connection is successful, error message string otherwise.

        Raises:
            McpAuthorizationRequired: when token is absent or invalid/expired.
        """
        oauth_discovery_endpoint = settings.get("oauth_discovery_endpoint")
        if not oauth_discovery_endpoint:
            return "OAuth discovery endpoint is required for Microsoft Graph"

        oauth_app_id = settings.get("oauth_app_id")
        auth_kwargs = dict(
            oauth_discovery_endpoint=oauth_discovery_endpoint,
            scopes=settings.get("scopes"),
            configuration_uuid=settings.get("configuration_uuid"),
            oauth_app_id=oauth_app_id,
            # Default app mode: the client ID is public and needed for the sign-in URL
            client_id=settings.get("client_id") if oauth_app_id else None,
        )
        from ..runtime.utils.mcp_oauth import access_token_from_stored_login
        access_token = access_token_from_stored_login(settings)
        if not access_token:
            raise MsGraphConfiguration._build_mcp_authorization_required(
                message="Microsoft Graph requires OAuth authorization. "
                        "Please complete the OAuth flow to obtain an access token.",
                **auth_kwargs,
            )

        try:
            resp = requests.get(
                "https://graph.microsoft.com/v1.0/me",
                params={"$select": "id,displayName"},
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
        except requests.exceptions.Timeout:
            return "Connection timeout - Microsoft Graph is not responding"
        except requests.exceptions.ConnectionError:
            return "Connection error - unable to reach Microsoft Graph"
        except requests.exceptions.RequestException as exc:
            return f"Request failed: {str(exc)}"

        if resp.status_code == 200:
            return None
        if resp.status_code == 401:
            api_message = MsGraphConfiguration._extract_api_error_message(resp)
            raise MsGraphConfiguration._build_mcp_authorization_required(
                message=f"Microsoft Graph access token is invalid or expired: {api_message}. Please re-authorize.",
                status=401,
                **auth_kwargs,
            )
        if resp.status_code == 403:
            return (
                "Access forbidden - the token lacks the Microsoft Graph permission User.Read. "
                "Add it to the credential scopes and sign in again."
            )
        return f"Microsoft Graph API request failed with status {resp.status_code}"

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
        oauth_app_id: Optional[str] = None,
    ) -> "McpAuthorizationRequired":
        """Build McpAuthorizationRequired exception for OAuth flow (same shape as Teams / Outlook)."""
        from ..runtime.utils.mcp_oauth import (
            McpAuthorizationRequired,
            fetch_oauth_authorization_server_metadata,
        )

        effective_scopes = ["offline_access", *normalize_ms_graph_scopes(scopes)]

        base_discovery = oauth_discovery_endpoint.rstrip("/")
        azure_v2_endpoint = f"{base_discovery}/v2.0/.well-known/openid-configuration"
        openid_meta = fetch_oauth_authorization_server_metadata(
            base_discovery,
            extra_endpoints=[azure_v2_endpoint],
        ) or {}

        authorization_endpoint = openid_meta.get("authorization_endpoint", f"{base_discovery}/v2.0/oauth2/authorize")
        scopes_supported = list(openid_meta.get("scopes_supported") or [])
        scopes_supported += [s for s in effective_scopes if s not in scopes_supported]

        oauth_authorization_server: Dict = {
            "issuer": openid_meta.get("issuer", base_discovery),
            "authorization_endpoint": authorization_endpoint,
            "token_endpoint": openid_meta.get("token_endpoint", f"{base_discovery}/v2.0/oauth2/token"),
            "scopes_supported": scopes_supported,
        }
        for key in ("jwks_uri", "response_types_supported", "claims_supported",
                    "id_token_signing_alg_values_supported"):
            if key in openid_meta:
                oauth_authorization_server[key] = openid_meta[key]

        resource_metadata: Dict = {
            "resource_name": "Microsoft Graph",
            "resource": "https://graph.microsoft.com",
            "authorization_servers": [base_discovery],
            "bearer_methods_supported": ["header"],
            "oauth_authorization_server": oauth_authorization_server,
            "scopes_supported": effective_scopes,
        }
        if configuration_uuid:
            resource_metadata["configuration_uuid"] = configuration_uuid
        if toolkit_id is not None:
            resource_metadata["toolkit_id"] = toolkit_id

        provided_settings: Dict = {"scopes": effective_scopes}
        if client_id:
            provided_settings["mcp_client_id"] = client_id
        secret_value = (
            client_secret.get_secret_value() if hasattr(client_secret, "get_secret_value") else client_secret
        )
        if secret_value:
            from ..runtime.utils.utils import mask_secret
            provided_settings["mcp_client_secret"] = mask_secret(str(secret_value))
        if oauth_app_id:
            # Default app mode: the platform holds the secret; sign-in sends this reference instead
            provided_settings["oauth_app_id"] = oauth_app_id
            provided_settings["credential_type"] = "ms_graph"
            provided_settings["has_mcp_client_secret"] = True
        resource_metadata["provided_settings"] = provided_settings

        auth_error = McpAuthorizationRequired(
            message=message,
            server_url="https://graph.microsoft.com",
            resource_metadata_url=azure_v2_endpoint,
            www_authenticate=(
                f'Bearer error="unauthorized_client", '
                f'error_description="No access token was provided", '
                f'resource_metadata="{azure_v2_endpoint}", '
                f'authorization_uri="{authorization_endpoint}"'
            ),
            resource_metadata=resource_metadata,
            status=status,
            tool_name="Microsoft Graph",
            toolkit_type="ms_graph",
            toolkit_name=toolkit_name,
        )
        if toolkit_id is not None:
            auth_error.toolkit_id = toolkit_id
        return auth_error
