import base64
import logging
from typing import Any, Dict, Literal, Optional

import requests
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

log = logging.getLogger(__name__)


class OpenApiConfiguration(BaseModel):
    """
    OpenAPI configuration for authentication.

    Supports four authentication modes:
    - Anonymous: No authentication (all fields empty)
    - API Key: Static key sent via header (Bearer, Basic, or Custom)
    - OAuth2 Client Credentials: Machine-to-machine authentication flow
    - OAuth2 Authorization Code (Browser): Delegated user-context flow via browser popup
      (activated when oauth_discovery_endpoint is set)
    """
    
    model_config = ConfigDict(
        extra='allow',
        json_schema_extra={
            "metadata": {
                "label": "OpenAPI",
                "icon_url": "openapi.svg",
                "categories": ["integrations"],
                "type": "openapi",
                "extra_categories": ["api", "openapi", "swagger"],
                "check_connection": {
                    "enabled_when": {
                        "any_of": [
                            {"all_fields_set": ["client_id", "client_secret", "method", "token_url"]},
                            {"all_fields_set": ["client_id", "client_secret", "oauth_discovery_endpoint"]},
                        ],
                    },
                    "disabled_tooltip": "Available for OAuth. Please setup client_id, client_secret, and either token_url (Client Credentials) or oauth_discovery_endpoint (Delegated) to activate.",
                },
                "sections": {
                    "auth": {
                        "required": False,
                        "subsections": [
                            {
                                "name": "API Key",
                                "fields": ["api_key", "auth_type", "custom_header_name"],
                            },
                            {
                                "name": "OAuth (Delegated)",
                                "fields": [
                                    "client_id",
                                    "client_secret",
                                    "oauth_discovery_endpoint",
                                    "scope",
                                ],
                            },
                            {
                                "name": "OAuth (Client Credentials)",
                                "fields": [
                                    "client_id",
                                    "client_secret",
                                    "token_url",
                                    "scope",
                                    "method",
                                ],
                            },
                        ],
                    },
                },
                "section": "credentials",
            }
        }
    )

    # =========================================================================
    # API Key Authentication Fields
    # =========================================================================
    
    api_key: Optional[SecretStr] = Field(
        default=None,
        description=(
            "API key value (stored as a secret). Used when selecting 'API Key' authentication subsection."
        ),
    )
    auth_type: Optional[Literal['Basic', 'Bearer', 'Custom']] = Field(
        default=None,
        description=(
            "How to apply the API key. "
            "- 'Bearer': sets 'Authorization: Bearer <api_key>' "
            "- 'Basic': sets 'Authorization: Basic <api_key>' "
            "- 'custom': sets '<custom_header_name>: <api_key>'"
        ),
    )
    custom_header_name: Optional[str] = Field(
        default=None,
        description="Custom header name to use when auth_type='custom' (e.g. 'X-Api-Key').",
        json_schema_extra={'visible_when': {'field': 'auth_type', 'value': 'custom'}},
    )

    # =========================================================================
    # OAuth2 Client Credentials Flow Fields
    # =========================================================================
    
    client_id: Optional[str] = Field(
        default=None, 
        description='OAuth2 client ID (also known as Application ID or App ID)'
    )
    client_secret: Optional[SecretStr] = Field(
        default=None, 
        description='OAuth2 client secret (stored securely)'
    )
    token_url: Optional[str] = Field(
        default=None, 
        description=(
            'OAuth2 token endpoint URL for obtaining access tokens. '
            'Examples: '
            'Azure AD: https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token, '
            'Google: https://oauth2.googleapis.com/token, '
            'Auth0: https://{domain}/oauth/token, '
            'Spotify: https://accounts.spotify.com/api/token'
        )
    )
    scope: Optional[str] = Field(
        default=None, 
        description=(
            'OAuth2 scope(s), space-separated if multiple (per OAuth2 RFC 6749). '
            'Examples: "user-read-private user-read-email" (Spotify), '
            '"api://app-id/.default" (Azure), '
            '"https://www.googleapis.com/auth/cloud-platform" (Google)'
        )
    )
    method: Optional[Literal['default', 'Basic']] = Field(
        default=None,
        description=(
            "Token exchange method for client credentials flow. "
            "'default': Sends client_id and client_secret in POST body (Azure AD, Auth0, most providers). "
            "'Basic': Sends credentials via HTTP Basic auth header - required by Spotify, some AWS services, and certain OAuth providers."
        ),
    )

    # OAuth2 Authorization Code Flow (Delegated/Browser) Fields
    oauth_discovery_endpoint: Optional[str] = Field(
        default=None,
        description=(
            "OAuth2 discovery endpoint (identity provider base URL). When set, enables "
            "browser-based Authorization Code flow instead of Client Credentials. "
            "The SDK discovers authorization and token endpoints from .well-known/openid-configuration. "
            "Examples: "
            "Azure AD: https://login.microsoftonline.com/{tenant_id}, "
            "Google: https://accounts.google.com, "
            "Okta: https://{domain}.okta.com"
        ),
    )
    configuration_uuid: Optional[str] = Field(
        default=None,
        description=(
            "Unique identifier for this credential record, injected at runtime by the platform. "
            "Used as the token storage key to isolate OAuth sessions across multiple configurations."
        ),
        json_schema_extra={'hidden': True},
    )

    @model_validator(mode='before')
    @classmethod
    def _validate_auth_consistency(cls, values):
        if not isinstance(values, dict):
            return values

        is_delegated = bool(values.get('oauth_discovery_endpoint'))

        # OAuth (Delegated): oauth_discovery_endpoint requires client_id and client_secret
        if is_delegated:
            missing = []
            if not values.get('client_id'):
                missing.append('client_id')
            if not values.get('client_secret'):
                missing.append('client_secret')

            if missing:
                raise ValueError(
                    f"OAuth (Delegated) is misconfigured; missing: {', '.join(missing)}"
                )
        else:
            # OAuth (Client Credentials): if any OAuth field is provided, require the essential ones
            has_any_oauth = any(
                (values.get('client_id'), values.get('client_secret'), values.get('token_url'))
            )
            if has_any_oauth:
                missing = []
                if not values.get('client_id'):
                    missing.append('client_id')
                if not values.get('client_secret'):
                    missing.append('client_secret')
                if not values.get('token_url'):
                    missing.append('token_url')
                if missing:
                    raise ValueError(
                        f"OAuth (Client Credentials) is misconfigured; missing: {', '.join(missing)}"
                    )

        # API key: if auth_type is custom, custom_header_name must be present
        auth_type = values.get('auth_type')
        if isinstance(auth_type, str) and auth_type.strip().lower() == 'custom' and values.get('api_key'):
            if not values.get('custom_header_name'):
                raise ValueError("custom_header_name is required when auth_type='custom'")

        return values

    @staticmethod
    def check_connection(settings: dict) -> str | None:
        """
        Validate the OpenAPI configuration by testing connectivity where possible.
        
        Validation behavior by authentication type:
        
        1. ANONYMOUS (no auth fields configured):
           - Cannot validate without making actual API calls
           - Returns None (success) - validation skipped
           
        2. API KEY (api_key field configured):
           - Cannot validate without knowing which endpoint to call
           - The OpenAPI spec is not available at configuration time
           - Returns None (success) - validation skipped
           
        3. OAUTH2 CLIENT CREDENTIALS (client_id, client_secret, token_url configured):
           - CAN validate by attempting token exchange with the OAuth provider
           - Makes a real HTTP request to token_url
           - Returns None on success, error message on failure
        
        Args:
            settings: Dictionary containing OpenAPI configuration fields
        
        Returns:
            None: Configuration is valid (or cannot be validated for this auth type)
            str: Error message describing the validation failure
        """
        
        # Determine authentication type from configured fields
        client_id = settings.get('client_id')
        client_secret = settings.get('client_secret')
        token_url = settings.get('token_url')
        oauth_discovery_endpoint = settings.get('oauth_discovery_endpoint')

        has_oauth_fields = client_id or client_secret or token_url

        # DELEGATED (Authorization Code flow): Raise McpAuthorizationRequired
        if oauth_discovery_endpoint:
            # If a valid access_token is already in settings (injected by the platform after
            # the user completes the OAuth popup), skip the authorization challenge and
            # treat the connection as successful — mirrors SharePoint's check_connection behavior.
            access_token = settings.get('access_token')
            if access_token:
                return None

            from ..runtime.utils.mcp_oauth import (
                McpAuthorizationRequired,
                fetch_oauth_authorization_server_metadata,
            )
            base_discovery = oauth_discovery_endpoint.rstrip("/")
            azure_v2_endpoint = f"{base_discovery}/v2.0/.well-known/openid-configuration"

            openid_meta = fetch_oauth_authorization_server_metadata(
                base_discovery,
                extra_endpoints=[azure_v2_endpoint],
            )

            resource_metadata_url = f"{base_discovery}/.well-known/openid-configuration"
            authorization_endpoint = (openid_meta or {}).get(
                "authorization_endpoint",
                f"{base_discovery}/oauth2/v2.0/authorize",
            )
            token_endpoint = (openid_meta or {}).get(
                "token_endpoint",
                token_url or f"{base_discovery}/oauth2/v2.0/token",
            )

            scope = settings.get('scope')
            # All scopes the IdP supports — used only for oauth_authorization_server metadata
            idp_scopes_supported = list((openid_meta or {}).get("scopes_supported") or [])
            # Scopes the user explicitly configured for this resource
            configured_scopes = scope.split() if scope else []
            # Merged list for auth-server metadata
            all_scopes = list(idp_scopes_supported)
            for s in configured_scopes:
                if s not in all_scopes:
                    all_scopes.append(s)

            www_authenticate = (
                f'Bearer error="unauthorized_client", '
                f'error_description="No access token was provided in this request", '
                f'resource_metadata="{resource_metadata_url}", '
                f'authorization_uri="{authorization_endpoint}"'
            )

            oauth_authorization_server: Dict = {
                "issuer": (openid_meta or {}).get("issuer", base_discovery),
                "authorization_endpoint": authorization_endpoint,
                "token_endpoint": token_endpoint,
            }
            jwks_uri = (openid_meta or {}).get("jwks_uri")
            if jwks_uri:
                oauth_authorization_server["jwks_uri"] = jwks_uri
            if all_scopes:
                oauth_authorization_server["scopes_supported"] = all_scopes
            if openid_meta:
                for key in (
                    "response_types_supported",
                    "claims_supported",
                    "id_token_signing_alg_values_supported",
                    "userinfo_endpoint",
                    "code_challenge_methods_supported",
                    "grant_types_supported",
                    "token_endpoint_auth_methods_supported",
                ):
                    if key in openid_meta:
                        oauth_authorization_server[key] = openid_meta[key]

            resource_metadata: Dict = {
                "resource_name": "OpenAPI",
                "resource": settings.get('base_url', ''),
                "authorization_servers": [base_discovery],
                "bearer_methods_supported": ["header"],
                "oauth_authorization_server": oauth_authorization_server,
            }
            # resource_metadata.scopes_supported = only user-configured scopes (what this resource needs).
            # Do NOT include all IdP scopes here — the UI uses this field as the initial scope to request.
            if configured_scopes:
                resource_metadata["scopes_supported"] = configured_scopes

            configuration_uuid = settings.get('configuration_uuid')
            if configuration_uuid:
                resource_metadata["configuration_uuid"] = configuration_uuid

            provided_settings = {}
            if client_id:
                provided_settings['mcp_client_id'] = client_id
            # NOTE: mcp_client_secret is intentionally NOT included here.
            # In the check_connection path there is no toolkit DB record, so the proxy
            # cannot recover a masked secret via toolkit_id lookup.
            # The frontend falls back to formClientSecret (the vault reference
            # returned by the credential GET API), which the proxy resolves via
            # VaultClient.unsecret() — that path works correctly.
            if scope:
                provided_settings['scopes'] = scope
            if provided_settings:
                resource_metadata['provided_settings'] = provided_settings

            raise McpAuthorizationRequired(
                message="OpenAPI endpoint requires OAuth authorization. Please log in to continue.",
                server_url=settings.get('base_url', ''),
                resource_metadata_url=resource_metadata_url,
                www_authenticate=www_authenticate,
                resource_metadata=resource_metadata,
            )

        # =====================================================================
        # ANONYMOUS or API KEY: Cannot validate, return success
        # =====================================================================

        if not has_oauth_fields:
            # No OAuth fields configured - this is either:
            # - Anonymous authentication (no auth at all)
            # - API Key authentication (api_key field may be set)
            #
            # Neither can be validated without making actual API calls to the
            # target service, and we don't have the OpenAPI spec available here.
            return None
        
        # =====================================================================
        # OAUTH2: Validate by attempting token exchange
        # =====================================================================
        
        # Check for required OAuth fields
        if not client_id:
            return "OAuth client_id is required when using OAuth authentication"
        if not client_secret:
            return "OAuth client_secret is required when using OAuth authentication"
        if not token_url:
            return "OAuth token_url is required when using OAuth authentication"
        
        # Extract secret value if it's a SecretStr
        if hasattr(client_secret, 'get_secret_value'):
            client_secret = client_secret.get_secret_value()
        
        if not client_secret or not str(client_secret).strip():
            return "OAuth client_secret cannot be empty"
        
        # Validate token_url format
        token_url = token_url.strip()
        if not token_url.startswith(('http://', 'https://')):
            return "OAuth token_url must start with http:// or https://"
        
        # Get optional OAuth settings
        scope = settings.get('scope')
        method = settings.get('method', 'default') or 'default'
        
        # ---------------------------------------------------------------------
        # Attempt OAuth2 Client Credentials token exchange
        # ---------------------------------------------------------------------
        
        try:
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json',
            }
            
            data = {
                'grant_type': 'client_credentials',
            }
            
            # Apply credentials based on method
            if method == 'Basic':
                # Basic method: credentials in Authorization header (Spotify, some AWS)
                credentials = f"{client_id}:{client_secret}"
                encoded = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
                headers['Authorization'] = f'Basic {encoded}'
            else:
                # Default method: credentials in POST body (Azure AD, Auth0, most providers)
                data['client_id'] = client_id
                data['client_secret'] = str(client_secret)
            
            if scope:
                data['scope'] = scope
            
            response = requests.post(
                token_url,
                headers=headers,
                data=data,
                timeout=30,
            )
            
            # ---------------------------------------------------------------------
            # Handle response
            # ---------------------------------------------------------------------
            
            if response.status_code == 200:
                try:
                    token_data = response.json()
                    if 'access_token' in token_data:
                        return None  # Success - token obtained
                    return "OAuth response did not contain 'access_token'"
                except Exception:
                    return "Failed to parse OAuth token response"
            
            # Handle common error status codes with helpful messages
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    error = error_data.get('error', 'bad_request')
                    error_desc = error_data.get('error_description', '')
                    if error_desc:
                        return f"OAuth error: {error} - {error_desc}"
                    return f"OAuth error: {error}"
                except Exception:
                    return "OAuth request failed: bad request (400)"
            
            if response.status_code == 401:
                return "OAuth authentication failed: invalid client_id or client_secret"
            
            if response.status_code == 403:
                return "OAuth access forbidden: client may lack required permissions"
            
            if response.status_code == 404:
                return f"OAuth token endpoint not found: {token_url}"
            
            return f"OAuth token request failed with status {response.status_code}"
            
        except requests.exceptions.SSLError as e:
            error_str = str(e).lower()
            if 'hostname mismatch' in error_str:
                return "OAuth token_url hostname does not match SSL certificate - verify the URL is correct"
            if 'certificate verify failed' in error_str:
                return "SSL certificate verification failed for OAuth endpoint - the server may have an invalid or self-signed certificate"
            if 'certificate has expired' in error_str:
                return "SSL certificate has expired for OAuth endpoint"
            return "SSL error connecting to OAuth endpoint - verify the token_url is correct"
        except requests.exceptions.ConnectionError as e:
            error_str = str(e).lower()
            if 'name or service not known' in error_str or 'nodename nor servname provided' in error_str:
                return "OAuth token_url hostname could not be resolved - verify the URL is correct"
            if 'connection refused' in error_str:
                return "Connection refused by OAuth endpoint - verify the token_url and port are correct"
            return "Cannot connect to OAuth token endpoint - verify the token_url is correct"
        except requests.exceptions.Timeout:
            return "OAuth token request timed out - the endpoint may be unreachable"
        except requests.exceptions.RequestException:
            return "OAuth request failed - verify the token_url is correct and accessible"
        except Exception:
            return "Unexpected error during OAuth configuration validation"
