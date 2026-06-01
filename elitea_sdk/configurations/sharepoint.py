import re
import requests
import logging
from typing import Optional, List, Tuple, Dict
from urllib.parse import urlparse
from pydantic import BaseModel, ConfigDict, Field, SecretStr

log = logging.getLogger(__name__)

SITE_URL_FORMAT_ERROR = (
    "Failed to parse SharePoint URL - use https://<tenant>.sharepoint.com, "
    "https://<tenant>.sharepoint.com/sites/<site>, or "
    "https://<tenant>.sharepoint.com/teams/<team>"
)


class SharepointConfiguration(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "metadata": {
                "label": "SharePoint",
                "icon_url": "sharepoint.svg",
                "sections": {
                    "auth": {
                        "required": True,
                        "subsections": [
                            {
                                "name": "App-only",
                                "fields": []
                            },
                            {
                                "name": "Delegated",
                                "fields": ["oauth_discovery_endpoint", "scopes", "auto_refresh_token"]
                            }
                        ]
                    }
                },
                "section": "credentials",
                "type": "sharepoint",
                "categories": ["office"],
                "extra_categories": ["sharepoint", "microsoft", "documents", "collaboration"],
            }
        }
    )
    # Client credentials (shared by both app_auth and delegated flows)
    client_id: str = Field(description="SharePoint Client ID")
    client_secret: SecretStr = Field(description="SharePoint Client Secret")
    site_url: str = Field(description="SharePoint tenant or site URL")

    # Additional fields for delegated/OAuth flows
    oauth_discovery_endpoint: Optional[str] = Field(default=None, description="OAuth Discovery Endpoint. Usually in format: https://login.microsoftonline.com/{tenant_id}")
    scopes: Optional[List[str]] = Field(default=None, description="OAuth Scopes")
    auto_refresh_token: Optional[bool] = Field(
        default=False,
        description="Automatically refresh the access token using the offline_access scope. "
                    "When enabled, the OAuth authorization request will include offline_access, "
                    "allowing the system to silently renew the access token without user interaction."
    )

    @staticmethod
    def check_connection(settings: dict) -> str | None:
        """
        Test the connection to SharePoint API.

        Two authentication flows are supported, selected by the presence of
        ``oauth_discovery_endpoint``:

        **SharePoint ACS / app-only** (no ``oauth_discovery_endpoint``):
            Uses ``office365-rest-python-client`` with ``ClientCredential``
            (the same path as the toolkit).  No Azure AD tenant discovery is
            needed — the library handles ACS token acquisition internally.

        **Delegated flow** (``oauth_discovery_endpoint`` present):
            Verifies the existing ``access_token`` via Microsoft Graph Sites
            API.  If the token is absent or expired, ``McpAuthorizationRequired``
            is raised so the caller can restart the OAuth dance.

        Args:
            settings: Dictionary that may contain:
                - site_url (required for both flows)
                - oauth_discovery_endpoint: activates the delegated flow
                - access_token: pre-obtained OAuth bearer token (delegated flow)
                - client_id / client_secret: app credentials

        Returns:
            None if connection is successful, error message string otherwise.

        Raises:
            McpAuthorizationRequired: delegated flow only, when token is absent
                or invalid/expired.
        """
        safe_settings = {k: "***" if k in ("client_secret", "access_token", "refresh_token") else v for k, v in settings.items()}
        log.debug(f"Checking SharePoint connection with settings: {safe_settings}")
        oauth_discovery_endpoint = settings.get("oauth_discovery_endpoint")
        if oauth_discovery_endpoint:
            log.info("Using delegated flow (oauth_discovery_endpoint present)")
            return SharepointConfiguration._check_connection_delegated(settings, oauth_discovery_endpoint)
        log.info("Using SharePoint ACS app-only flow (client_id + client_secret)")
        return SharepointConfiguration._check_connection_client_credentials(settings)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_site_url(settings: dict) -> Tuple[Optional[str], Optional[str]]:
        """Validate and normalise ``site_url`` from settings.

        Returns:
            ``(site_url, None)`` on success or ``(None, error_message)`` on failure.
        """
        site_url = settings.get("site_url")
        if site_url is None or site_url == "":
            return None, ("Site URL cannot be empty" if site_url == "" else "Site URL is required")
        if not isinstance(site_url, str):
            return None, "Site URL must be a string"
        site_url = site_url.strip()
        if not site_url:
            return None, "Site URL cannot be empty"
        if not site_url.startswith(("http://", "https://")):
            return None, "Site URL must start with http:// or https://"
        if not urlparse(site_url).netloc:
            return None, SITE_URL_FORMAT_ERROR
        return site_url.rstrip("/"), None

    @staticmethod
    def _extract_api_error_message(response) -> str:
        """Extract a human-readable error message from a SharePoint / Graph API error response.

        SharePoint can return the error in two shapes:
            ``{'code': '...', 'message': '...', 'innerError': {...}}``
            ``{'error': {'code': '...', 'message': '...', 'innerError': {...}}}``
        Falls back to the raw response text when the body cannot be parsed.
        """
        try:
            body = response.json()
            # Graph-style: top-level 'error' wrapper
            if "error" in body and isinstance(body["error"], dict):
                return body["error"].get("message") or str(body["error"])
            # SharePoint-style: 'message' at top level
            if "message" in body:
                return body["message"]
            return str(body)
        except Exception:
            return response.text or "unknown error"

    @staticmethod
    def _build_mcp_authorization_required(
        message: str,
        site_url: str,
        oauth_discovery_endpoint: str,
        scopes: Optional[List[str]],
        status: Optional[int] = None,
        configuration_uuid: Optional[str] = None,
        auto_refresh_token: bool = False,
    ) -> "McpAuthorizationRequired":
        """Build a ``McpAuthorizationRequired`` exception with the same rich metadata
        shape that the MCP OAuth flow produces, so upstream handlers can treat
        SharePoint delegated auth identically to MCP server auth.

        Delegates discovery to ``fetch_oauth_authorization_server_metadata`` from
        ``mcp_oauth.py``, passing the Azure AD v2.0 URL as an ``extra_endpoint``
        so it is tried first before the standard candidates — no duplicated fetch
        logic in this file.
        """
        log.debug(f"build_mcp_authorization_request for {message}, site_url={site_url}, oauth_discovery_endpoint={oauth_discovery_endpoint}, scopes={scopes}")
        from ..runtime.utils.mcp_oauth import (
            McpAuthorizationRequired,
            fetch_oauth_authorization_server_metadata,
        )

        # Always inject offline_access in delegated OAuth mode so Azure AD returns
        # a refresh_token in the token exchange response. The auto_refresh_token
        # flag previously gated this, but the feature spec requires it by default.
        effective_scopes = list(scopes or [])
        if "offline_access" not in effective_scopes:
            effective_scopes.insert(0, "offline_access")

        base_discovery = oauth_discovery_endpoint.rstrip("/")
        # Azure AD v2.0 well-known URL — injected as an extra candidate so the
        # shared helper tries it first, then falls back to the standard ones.
        azure_v2_endpoint = f"{base_discovery}/v2.0/.well-known/openid-configuration"

        openid_meta = fetch_oauth_authorization_server_metadata(
            base_discovery,
            extra_endpoints=[azure_v2_endpoint],
        )

        # resource_metadata_url points to whichever endpoint actually responded,
        # or to the v2.0 URL as a sensible default when discovery fails.
        resource_metadata_url = azure_v2_endpoint
        log.debug(f"Fetched OpenID metadata for SharePoint OAuth discovery: {openid_meta}")

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
            f'error_description="No access token was provided in this request", '
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
            "resource_name": "SharePoint",
            "resource": site_url,
            "authorization_servers": [base_discovery],
            "bearer_methods_supported": ["header"],
            "oauth_authorization_server": oauth_authorization_server,
        }
        if effective_scopes:
            resource_metadata["scopes_supported"] = effective_scopes
        if configuration_uuid:
            resource_metadata["configuration_uuid"] = configuration_uuid
        log.debug(f"SharePoint resource_metadata: {resource_metadata}")
        return McpAuthorizationRequired(
            message=message,
            server_url=site_url,
            resource_metadata_url=resource_metadata_url,
            www_authenticate=www_authenticate,
            resource_metadata=resource_metadata,
            status=status,
            tool_name=site_url,
        )

    @staticmethod
    def _call_graph_api(site_url: str, access_token: str, oauth_discovery_endpoint: Optional[str], scopes: Optional[List[str]] = None, configuration_uuid: Optional[str] = None) -> str | None:
        """Health-check an access token using the Microsoft Graph API.

        Used by both the delegated flow (Azure AD delegated tokens) and the
        app-only flow (Graph-scoped client-credentials tokens).

        When ``oauth_discovery_endpoint`` is supplied (delegated flow), a 401
        raises ``McpAuthorizationRequired`` so the caller can restart the OAuth
        dance.  When it is ``None`` (app-only flow) a 401 returns a plain error
        string — no re-authorization dance is needed.

        Args:
            site_url: Normalised SharePoint site URL.
            access_token: Bearer token to verify.
            oauth_discovery_endpoint: Azure AD base URL for delegated flow, or
                ``None`` for the app-only flow.
            scopes: OAuth scopes forwarded into McpAuthorizationRequired metadata.

        Returns:
            ``None`` on success, error message string otherwise.

        Raises:
            McpAuthorizationRequired: delegated flow only, when token is invalid
                or expired (401).
        """
        try:
            parsed = urlparse(site_url)
            hostname = parsed.netloc
            path = parsed.path or "/"

            graph_url = f"https://graph.microsoft.com/v1.0/sites/{hostname}:{path}"
            resp = requests.get(
                graph_url,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )

            if resp.status_code == 200:
                return None
            elif resp.status_code == 401:
                api_message = SharepointConfiguration._extract_api_error_message(resp)
                if not oauth_discovery_endpoint:
                    # App-only flow — return plain error, no re-auth dance.
                    return (
                        f"SharePoint access token was rejected by Microsoft Graph: {api_message}. "
                        "Please verify that the Azure AD app has the required Graph permissions "
                        "(e.g. Sites.Read.All) and that admin consent has been granted."
                    )
                # Delegated flow — import deferred so running the file directly as
                # __main__ doesn't fail on the relative import on the happy path.
                from ..runtime.utils.mcp_oauth import McpAuthorizationRequired  # noqa: PLC0415
                raise SharepointConfiguration._build_mcp_authorization_required(
                    message=(
                        f"SharePoint delegated access token is invalid or expired: {api_message}. "
                        "Please re-authorize to obtain a fresh access token."
                    ),
                    site_url=site_url,
                    oauth_discovery_endpoint=oauth_discovery_endpoint,
                    scopes=scopes,
                    status=401,
                    configuration_uuid=configuration_uuid,
                )
            elif resp.status_code == 403:
                return "Access forbidden - token lacks required Microsoft Graph permissions for this site"
            elif resp.status_code == 404:
                return f"Site not found in Microsoft Graph: {site_url}"
            else:
                return f"Microsoft Graph API request failed with status {resp.status_code}"

        except requests.exceptions.Timeout:
            return "Connection timeout - Microsoft Graph is not responding"
        except requests.exceptions.ConnectionError:
            return "Connection error - unable to reach Microsoft Graph"
        except requests.exceptions.RequestException as exc:
            return f"Request failed: {str(exc)}"
        except Exception as exc:
            # Re-raise McpAuthorizationRequired (delegated flow).
            # Checked by name to avoid a top-level import that would break
            # running this file directly as __main__.
            if type(exc).__name__ == "McpAuthorizationRequired":
                raise
            log.error(f"Error calling Microsoft Graph API: {exc}")
            return f"Unexpected error: {str(exc)}"

    @staticmethod
    def _call_sharepoint_api(site_url: str, access_token: str) -> str | None:
        """Call ``<site_url>/_api/web`` with *access_token*.

        Used exclusively by the **client-credentials flow**.  ACS tokens obtained
        via the legacy client-credentials grant are SharePoint tokens (not Graph
        tokens) and must be verified against the SharePoint REST endpoint.

        On HTTP 401 a plain error string is returned — this flow does not use
        delegated OAuth, so ``McpAuthorizationRequired`` is never raised here.
        """
        try:
            resp = requests.get(
                f"{site_url}/_api/web",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            if resp.status_code == 200:
                return None
            elif resp.status_code == 401:
                api_message = SharepointConfiguration._extract_api_error_message(resp)
                return (
                    f"SharePoint access token is invalid or expired: {api_message}. "
                    "Please check your client credentials and try again."
                )
            elif resp.status_code == 403:
                return "Access forbidden - client may lack required permissions for this site"
            elif resp.status_code == 404:
                return f"Site not found or not accessible: {site_url}"
            else:
                return f"SharePoint API request failed with status {resp.status_code}"
        except requests.exceptions.Timeout:
            return "Connection timeout - SharePoint is not responding"
        except requests.exceptions.ConnectionError:
            return "Connection error - unable to reach SharePoint"
        except requests.exceptions.RequestException as e:
            return f"Request failed: {str(e)}"

    @staticmethod
    def _check_connection_delegated(settings: dict, oauth_discovery_endpoint: str) -> str | None:
        """Delegated flow.

        Uses the ``access_token`` from *settings* to probe the site via the
        Microsoft Graph API (the correct endpoint for Azure AD delegated tokens).

        Raises ``McpAuthorizationRequired`` when:
        - no ``access_token`` is present (authorization dance not yet started), or
        - the token is invalid / expired (401 from Graph API).

        In both cases the exception carries the same rich metadata shape as a
        real MCP OAuth challenge, including ``resource_metadata``,
        ``www_authenticate``, and ``resource_metadata_url``, so upstream
        handlers need no special-casing for SharePoint.
        """
        site_url, err = SharepointConfiguration._validate_site_url(settings)
        if err:
            return err

        scopes = settings.get("scopes")
        configuration_uuid = settings.get("configuration_uuid")
        auto_refresh_token = settings.get("auto_refresh_token", False)
        access_token = settings.get("access_token")
        if not access_token:
            raise SharepointConfiguration._build_mcp_authorization_required(
                message=(
                    f"SharePoint site {site_url} requires OAuth authorization. "
                    "Please complete the OAuth flow to obtain an access token."
                ),
                site_url=site_url,
                oauth_discovery_endpoint=oauth_discovery_endpoint,
                scopes=scopes,
                configuration_uuid=configuration_uuid,
                auto_refresh_token=auto_refresh_token,
            )

        # Use Graph API for health-check — delegated tokens are Graph tokens
        return SharepointConfiguration._call_graph_api(
            site_url, access_token, oauth_discovery_endpoint, scopes, configuration_uuid
        )

    @staticmethod
    def _check_connection_client_credentials(settings: dict) -> str | None:
        """SharePoint ACS app-only flow.

        Uses ``office365-rest-python-client`` with ``ClientCredential``,
        identical to the toolkit's primary authentication path when
        ``client_id`` + ``client_secret`` are supplied without an
        ``oauth_discovery_endpoint``.
        """
        client_id = settings.get("client_id")
        if client_id is None or client_id == "":
            return "Client ID cannot be empty" if client_id == "" else "Client ID is required"
        if not isinstance(client_id, str):
            return "Client ID must be a string"
        client_id = client_id.strip()
        if not client_id:
            return "Client ID cannot be empty"

        client_secret = settings.get("client_secret")
        if client_secret is None:
            return "Client secret is required"
        if hasattr(client_secret, "get_secret_value"):
            client_secret = client_secret.get_secret_value()
        if not client_secret or not client_secret.strip():
            return "Client secret cannot be empty"

        site_url, err = SharepointConfiguration._validate_site_url(settings)
        if err:
            return err

        try:
            from office365.runtime.auth.client_credential import ClientCredential
            from office365.sharepoint.client_context import ClientContext

            credentials = ClientCredential(client_id, client_secret)
            ctx = ClientContext(site_url).with_credentials(credentials)
            ctx.load(ctx.web)
            ctx.execute_query()
            return None

        except ImportError:
            return (
                "office365-rest-python-client is not installed. "
                "Run: pip install office365-rest-python-client"
            )
        except Exception as e:
            error_msg = str(e)
            error_lower = error_msg.lower()
            if "401" in error_msg or "unauthorized" in error_lower:
                return (
                    "Authentication failed - invalid client ID or client secret. "
                    "Verify the app is registered in SharePoint with the correct permissions."
                )
            if "403" in error_msg or "forbidden" in error_lower:
                return (
                    "Access denied - the app does not have permission to access this site. "
                    "Ensure the SharePoint app principal has been granted site collection access."
                )
            if "timeout" in error_lower:
                return "Connection timeout - SharePoint is not responding"
            if "connection" in error_lower or "name or service not known" in error_lower:
                return "Connection error - unable to reach SharePoint"
            return f"SharePoint ACS connection failed: {error_msg}"
