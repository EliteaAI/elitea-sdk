from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import BaseModel, ConfigDict, Field, create_model
import requests
import yaml

from .api_wrapper import _get_base_url_from_spec, build_wrapper
from .tool import OpenApiAction
from ..elitea_base import filter_missconfigured_index_tools
from ...configurations.openapi import OpenApiConfiguration
from ...runtime.utils.constants import TOOLKIT_NAME_META, TOOL_NAME_META, TOOLKIT_TYPE_META

logger = logging.getLogger(__name__)

name = 'openapi'

# Module-level token cache: {cache_key: (access_token, expires_at_timestamp)}
# Protected by _oauth_token_cache_lock for thread-safe access
_oauth_token_cache: Dict[str, Tuple[str, float]] = {}
_oauth_token_cache_lock = threading.Lock()

# Token expiry buffer in seconds (refresh 60 seconds before actual expiry)
_TOKEN_EXPIRY_BUFFER = 60


def _get_oauth_cache_key(client_id: str, token_url: str, scope: Optional[str]) -> str:
    """Generate a cache key for OAuth tokens."""
    return f"{client_id}:{token_url}:{scope or ''}"


def _get_cached_token(cache_key: str) -> Optional[str]:
    """Get a cached token if it exists and is not expired. Thread-safe."""
    with _oauth_token_cache_lock:
        if cache_key not in _oauth_token_cache:
            return None
        token, expires_at = _oauth_token_cache[cache_key]
        if time.time() >= expires_at - _TOKEN_EXPIRY_BUFFER:
            # Token expired or about to expire
            del _oauth_token_cache[cache_key]
            return None
        return token


def _cache_token(cache_key: str, token: str, expires_in: Optional[int]) -> None:
    """Cache a token with its expiry time. Thread-safe."""
    # Default to 1 hour if expires_in not provided
    expires_in = expires_in or 3600
    expires_at = time.time() + expires_in
    with _oauth_token_cache_lock:
        _oauth_token_cache[cache_key] = (token, expires_at)


def _obtain_oauth_token(
    client_id: str,
    client_secret: str,
    token_url: str,
    scope: Optional[str] = None,
    method: str = 'default',
    timeout: int = 30,
) -> Tuple[str, Optional[str]]:
    """
    Obtain an OAuth2 access token using client credentials flow.
    
    Args:
        client_id: OAuth client ID
        client_secret: OAuth client secret
        token_url: OAuth token endpoint URL
        scope: Optional OAuth scope(s), space-separated if multiple
        method: Token exchange method - 'default' (POST body) or 'Basic' (Basic auth header)
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (access_token, error_message)
        On success: (token, None)
        On failure: (None, error_message)
    """
    try:
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
        }
        
        # Build form data
        data: Dict[str, str] = {
            'grant_type': 'client_credentials',
        }
        
        if method == 'Basic':
            # Use Basic auth header for client credentials
            credentials = f"{client_id}:{client_secret}"
            encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
            headers['Authorization'] = f'Basic {encoded_credentials}'
        else:
            # Default: include credentials in POST body
            data['client_id'] = client_id
            data['client_secret'] = client_secret
        
        if scope:
            data['scope'] = scope
        
        # Log only the domain to avoid exposing sensitive path parameters (e.g., tenant IDs)
        token_domain = urlparse(token_url).netloc or 'unknown'
        logger.debug(f"OAuth token request to {token_domain} using method '{method}'")
        
        response = requests.post(
            token_url,
            headers=headers,
            data=data,
            timeout=timeout,
        )
        
        if response.status_code == 200:
            try:
                token_data = response.json()
                access_token = token_data.get('access_token')
                if not access_token:
                    return None, "OAuth response did not contain 'access_token'"
                
                # Cache the token
                cache_key = _get_oauth_cache_key(client_id, token_url, scope)
                expires_in = token_data.get('expires_in')
                _cache_token(cache_key, access_token, expires_in)
                
                logger.debug(f"OAuth token obtained successfully (expires_in: {expires_in})")
                return access_token, None
            except json.JSONDecodeError as e:
                return None, f"Failed to parse OAuth token response as JSON: {e}"
        
        # Handle error responses
        error_msg = f"OAuth token request failed with status {response.status_code}"
        try:
            error_data = response.json()
            if 'error' in error_data:
                error_msg = f"{error_msg}: {error_data.get('error')}"
                if 'error_description' in error_data:
                    error_msg = f"{error_msg} - {error_data.get('error_description')}"
        except Exception:
            if response.text:
                error_msg = f"{error_msg}: {response.text[:500]}"
        
        return None, error_msg
        
    except requests.exceptions.Timeout:
        return None, f"OAuth token request to {token_url} timed out"
    except requests.exceptions.ConnectionError as e:
        return None, f"Failed to connect to OAuth token endpoint {token_url}: {e}"
    except requests.exceptions.RequestException as e:
        return None, f"OAuth token request failed: {e}"
    except Exception as e:
        return None, f"Unexpected error during OAuth token exchange: {e}"


def _secret_to_str(value: Any) -> Optional[str]:
    """Convert a secret value to string, handling SecretStr and other types."""
    if value is None:
        return None
    if hasattr(value, 'get_secret_value'):
        try:
            value = value.get_secret_value()
        except Exception:
            pass
    if isinstance(value, str):
        return value
    return str(value)


def _get_oauth_access_token(settings: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Get an OAuth access token from settings, using cache if available.
    
    Args:
        settings: Dictionary containing OAuth configuration
    
    Returns:
        Tuple of (access_token, error_message)
        On success: (token, None)
        On failure: (None, error_message)
        If OAuth not configured: (None, None)
    """
    client_id = settings.get('client_id')
    client_secret = _secret_to_str(settings.get('client_secret'))
    token_url = settings.get('token_url')
    
    # Check if OAuth is configured
    if not client_id or not client_secret or not token_url:
        return None, None  # OAuth not configured
    
    scope = settings.get('scope')
    method = settings.get('method', 'default') or 'default'
    
    # Try to get cached token
    cache_key = _get_oauth_cache_key(client_id, token_url, scope)
    cached_token = _get_cached_token(cache_key)
    if cached_token:
        logger.debug("Using cached OAuth token")
        return cached_token, None
    
    # Obtain new token
    return _obtain_oauth_token(
        client_id=client_id,
        client_secret=client_secret,
        token_url=token_url,
        scope=scope,
        method=method,
    )


def _build_openapi_mcp_authorization_required(
    oauth_discovery_endpoint: str,
    scope: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[Any] = None,
    configuration_uuid: Optional[str] = None,
    toolkit_id: Optional[int] = None,
    base_url: str = '',
):
    """Build a McpAuthorizationRequired exception for the OpenAPI toolkit.

    Mirrors SharepointConfiguration._build_mcp_authorization_required() —
    discovers OAuth endpoints from .well-known and constructs resource_metadata
    so the frontend can trigger the browser login popup.
    """
    from ...runtime.utils.mcp_oauth import (
        McpAuthorizationRequired,
        fetch_oauth_authorization_server_metadata,
    )
    from ...runtime.utils.utils import mask_secret

    base_discovery = oauth_discovery_endpoint.rstrip("/")
    azure_v2_endpoint = f"{base_discovery}/v2.0/.well-known/openid-configuration"

    openid_meta = fetch_oauth_authorization_server_metadata(
        base_discovery,
        extra_endpoints=[azure_v2_endpoint],
    )
    logger.debug("OpenAPI OAuth discovery metadata fetched successfully")

    resource_metadata_url = f"{base_discovery}/.well-known/openid-configuration"
    authorization_endpoint = (openid_meta or {}).get(
        "authorization_endpoint",
        f"{base_discovery}/oauth2/v2.0/authorize",
    )
    token_endpoint = (openid_meta or {}).get(
        "token_endpoint",
        f"{base_discovery}/oauth2/v2.0/token",
    )

    # All scopes the IdP supports — used only for oauth_authorization_server metadata
    idp_scopes_supported = list((openid_meta or {}).get("scopes_supported") or [])
    # Scopes the user explicitly configured for this resource
    configured_scopes = scope.split() if scope else []
    # Merged list for auth-server metadata (IdP list + any configured scopes not already present)
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
        "resource": base_url,
        "authorization_servers": [base_discovery],
        "bearer_methods_supported": ["header"],
        "oauth_authorization_server": oauth_authorization_server,
    }
    # resource_metadata.scopes_supported = only user-configured scopes (what this resource needs).
    # Do NOT include all IdP scopes here — the UI uses this field as the initial scope to request.
    if configured_scopes:
        resource_metadata["scopes_supported"] = configured_scopes
    if configuration_uuid:
        resource_metadata["configuration_uuid"] = configuration_uuid
    if toolkit_id is not None:
        resource_metadata["toolkit_id"] = toolkit_id

    provided_settings: Dict = {}
    if client_id:
        provided_settings['mcp_client_id'] = client_id
    if client_secret:
        secret_val = client_secret
        if hasattr(secret_val, 'get_secret_value'):
            secret_val = secret_val.get_secret_value()
        if secret_val:
            provided_settings['mcp_client_secret'] = mask_secret(str(secret_val))
    if scope:
        provided_settings['scopes'] = scope
    if provided_settings:
        resource_metadata['provided_settings'] = provided_settings

    return McpAuthorizationRequired(
        message=(
            f"OpenAPI endpoint requires OAuth authorization. "
            "Please log in to continue."
        ),
        server_url=base_url,
        resource_metadata_url=resource_metadata_url,
        www_authenticate=www_authenticate,
        resource_metadata=resource_metadata,
        tool_name=base_url,
    )


def get_toolkit(tool) -> BaseToolkit:
    settings = tool.get('settings', {}) or {}
    # Extract selected_tools separately to avoid duplicate keyword argument when unpacking **settings
    selected_tools = settings.get('selected_tools', [])
    # Filter out selected_tools from settings to prevent "got multiple values for keyword argument"
    filtered_settings = {k: v for k, v in settings.items() if k != 'selected_tools'}
    return EliteAOpenAPIToolkit.get_toolkit(
        selected_tools=selected_tools,
        toolkit_name=tool.get('toolkit_name'),
        toolkit_id=tool.get('id'),
        **filtered_settings,
    )


def get_tools(tool):
    return get_toolkit(tool).get_tools()


def get_toolkit_available_tools(settings: dict) -> dict:
    """Return instance-dependent tool list + per-tool args JSON schemas.

    This is used by backend services when the UI needs spec-derived tool names
    and input schemas (one tool per operationId). It must be JSON-serializable.
    """
    if not isinstance(settings, dict):
        settings = {}

    # Extract and merge openapi_configuration if present (same pattern as get_toolkit)
    openapi_configuration = settings.get('openapi_configuration') or {}
    if hasattr(openapi_configuration, 'model_dump'):
        openapi_configuration = openapi_configuration.model_dump(mode='json')
    if not isinstance(openapi_configuration, dict):
        openapi_configuration = {}

    # Merge settings with openapi_configuration so api_key, auth_type etc. are at root level
    merged_settings: Dict[str, Any] = {
        **settings,
        **openapi_configuration,
    }

    spec = merged_settings.get('spec') or merged_settings.get('schema_settings') or merged_settings.get('openapi_spec')
    base_url_override = merged_settings.get('base_url') or merged_settings.get('base_url_override')

    if not spec or not isinstance(spec, (str, dict)):
        return {"tools": [], "args_schemas": {}, "error": "OpenAPI spec is missing"}

    try:
        # For tool listing, delegated OAuth does not block — we just need
        # to parse the spec. Auth headers are only needed for actual API calls.
        oauth_discovery_endpoint = merged_settings.get('oauth_discovery_endpoint')
        if oauth_discovery_endpoint:
            headers = {}
        else:
            headers = _build_headers_from_settings(merged_settings)
        api_wrapper = build_wrapper(
            openapi_spec=spec,
            base_headers=headers,
            base_url_override=base_url_override,
        )

        tool_defs = api_wrapper.get_available_tools(selected_tools=None)

        tools = []
        args_schemas = {}

        for tool_def in tool_defs:
            name_val = tool_def.get('name')
            if not isinstance(name_val, str) or not name_val:
                continue

            desc_val = tool_def.get('description')
            if not isinstance(desc_val, str):
                desc_val = ''

            tools.append({"name": name_val, "description": desc_val})

            args_schema = tool_def.get('args_schema')
            if args_schema is None:
                args_schemas[name_val] = {"type": "object", "properties": {}, "required": []}
                continue

            try:
                if hasattr(args_schema, 'model_json_schema'):
                    args_schemas[name_val] = args_schema.model_json_schema()
                elif hasattr(args_schema, 'schema'):
                    args_schemas[name_val] = args_schema.schema()
                else:
                    args_schemas[name_val] = {"type": "object", "properties": {}, "required": []}
            except Exception:
                args_schemas[name_val] = {"type": "object", "properties": {}, "required": []}

        # Ensure stable JSON-serializability.
        try:
            json.dumps({"tools": tools, "args_schemas": args_schemas})
        except Exception:
            return {"tools": tools, "args_schemas": {}}

        return {"tools": tools, "args_schemas": args_schemas}

    except Exception as e:  # pylint: disable=W0718
        return {"tools": [], "args_schemas": {}, "error": str(e)}

class EliteAOpenAPIToolkit(BaseToolkit):
    request_session: Any  #: :meta private:
    tools: List[BaseTool] = []

    @staticmethod
    def toolkit_config_schema() -> BaseModel:
        # OpenAPI tool names + per-tool args schemas depend on the user-provided spec,
        # so `selected_tools` cannot be an enum here (unlike most toolkits).

        model = create_model(
            name,
            __config__=ConfigDict(
                extra='ignore',
                json_schema_extra={
                    'metadata': {
                        'label': 'OpenAPI',
                        'icon_url': 'openapi.svg',
                        'categories': ['integrations'],
                        'extra_categories': ['api', 'openapi', 'swagger'],
                    }
                }
            ),
            openapi_configuration=(
                OpenApiConfiguration,
                Field(
                    description='OpenAPI credentials configuration',
                    json_schema_extra={'configuration_types': ['openapi']},
                ),
            ),
            base_url=(
                Optional[str],
                Field(
                    default=None,
                    description=(
                        "Optional base URL override (absolute, starting with http:// or https://). "
                        "Use this when your OpenAPI spec has no `servers` entry, or when `servers[0].url` "
                        "is not absolute (e.g. '/api/v3'). Example: 'https://petstore3.swagger.io'."
                    ),
                ),
            ),
            spec=(
                str,
                Field(
                    description=(
                        'OpenAPI specification (URL or raw JSON/YAML text). '
                        'Used to generate per-operation tools (one tool per operationId).'
                    ),
                    json_schema_extra={'ui_component': 'openapi_spec'},
                ),
            ),
            selected_tools=(
                List[str],
                Field(
                    default=[],
                    description='Optional list of operationIds to enable. If empty, all operations are enabled.',
                    json_schema_extra={'args_schemas': {}},
                ),
            ),
        )
        return model

    @classmethod
    @filter_missconfigured_index_tools
    def get_toolkit(
        cls,
        selected_tools: list[str] | None = None,
        toolkit_name: Optional[str] = None,
        **kwargs,
    ):
        if selected_tools is None:
            selected_tools = []

        tool_names = _coerce_selected_tool_names(selected_tools)

        openapi_configuration = kwargs.get('openapi_configuration') or {}
        if hasattr(openapi_configuration, 'model_dump'):
            openapi_configuration = openapi_configuration.model_dump(mode='json')
        if not isinstance(openapi_configuration, dict):
            openapi_configuration = {}

        merged_settings: Dict[str, Any] = {
            **kwargs,
            **openapi_configuration,
        }

        openapi_spec = merged_settings.get('spec') or merged_settings.get('schema_settings') or merged_settings.get('openapi_spec')
        base_url_override = merged_settings.get('base_url') or merged_settings.get('base_url_override')

        # --- Delegated OAuth (Authorization Code flow) ---
        oauth_discovery_endpoint = merged_settings.get('oauth_discovery_endpoint')
        tokens = kwargs.get('tokens') or {}
        toolkit_id = kwargs.get('toolkit_id')

        if oauth_discovery_endpoint:
            config_uuid = merged_settings.get('configuration_uuid')
            logger.debug("[OpenAPI OAuth] delegated flow active")
            token = None
            if config_uuid:
                token = tokens.get(f"{config_uuid}:{oauth_discovery_endpoint}")
            if token is None:
                token = tokens.get(oauth_discovery_endpoint)

            if token is not None:
                access_token = token.get('access_token') if isinstance(token, dict) else token
                headers = {'Authorization': f'Bearer {access_token}'}
                logger.debug("Using delegated OAuth token for OpenAPI authentication")
            else:
                logger.debug("OpenAPI OAuth mode active but no token found — raising McpAuthorizationRequired.")
                raise _build_openapi_mcp_authorization_required(
                    oauth_discovery_endpoint=oauth_discovery_endpoint,
                    scope=merged_settings.get('scope'),
                    client_id=merged_settings.get('client_id'),
                    client_secret=merged_settings.get('client_secret'),
                    configuration_uuid=config_uuid,
                    toolkit_id=toolkit_id,
                    base_url=base_url_override or '',
                )
        else:
            headers = _build_headers_from_settings(merged_settings)

        api_wrapper = build_wrapper(
            openapi_spec=openapi_spec,
            base_headers=headers,
            base_url_override=base_url_override,
        )
        base_url = _get_base_url_from_spec(api_wrapper.spec)

        tools: List[BaseTool] = []
        for tool_def in api_wrapper.get_available_tools(selected_tools=tool_names):
            description = tool_def.get('description') or ''
            if toolkit_name:
                description = f"{description}\nToolkit: {toolkit_name}"
            if base_url:
                description = f"{description}\nBase URL: {base_url}"
            description = description[:1000]

            tools.append(
                OpenApiAction(
                    api_wrapper=api_wrapper,
                    name=tool_def['name'],
                    description=description,
                    args_schema=tool_def.get('args_schema'),
                    metadata={TOOLKIT_NAME_META: toolkit_name, TOOLKIT_TYPE_META: name, TOOL_NAME_META: tool_def["name"]} if toolkit_name else {TOOL_NAME_META: tool_def["name"]},
                )
            )

        return cls(request_session=api_wrapper, tools=tools)

    def get_tools(self):
        return self.tools


def _coerce_selected_tool_names(selected_tools: Any) -> list[str]:
    if not selected_tools:
        return []

    if isinstance(selected_tools, list):
        tool_names: List[str] = []
        for item in selected_tools:
            if isinstance(item, str):
                tool_names.append(item)
            elif isinstance(item, dict):
                name_val = item.get('name')
                if isinstance(name_val, str) and name_val.strip():
                    tool_names.append(name_val)
        return [t for t in tool_names if t]

    return []


def _build_headers_from_settings(settings: Dict[str, Any]) -> Dict[str, str]:
    """
    Build HTTP headers from settings, supporting API key and OAuth authentication.
    
    Authentication priority:
    1. OAuth (client credentials flow) - if client_id, client_secret, and token_url are provided
    2. API Key - if api_key is provided
    3. Legacy authentication structure (for backward compatibility)
    
    Args:
        settings: Dictionary containing authentication settings
        
    Returns:
        Dictionary of HTTP headers to include in requests
    """
    headers: Dict[str, str] = {}

    # First, try OAuth authentication (client credentials flow)
    # This takes priority because it's more secure and commonly used with modern APIs
    oauth_token, oauth_error = _get_oauth_access_token(settings)
    if oauth_token:
        headers['Authorization'] = f'Bearer {oauth_token}'
        logger.debug("Using OAuth Bearer token for authentication")
        return headers
    elif oauth_error:
        # OAuth was configured but failed - log the error
        # We'll still try API key auth as fallback
        logger.warning(f"OAuth token exchange failed: {oauth_error}")

    # Legacy structure used by the custom OpenAPI UI
    auth = settings.get('authentication')
    if isinstance(auth, dict) and auth.get('type') == 'api_key':
        auth_settings = auth.get('settings') or {}
        if isinstance(auth_settings, dict):
            auth_type = str(auth_settings.get('auth_type', '')).strip().lower()
            api_key = _secret_to_str(auth_settings.get('api_key'))
            if api_key:
                if auth_type == 'bearer':
                    headers['Authorization'] = f'Bearer {api_key}'
                elif auth_type == 'basic':
                    headers['Authorization'] = f'Basic {api_key}'
                elif auth_type == 'custom':
                    header_name = auth_settings.get('custom_header_name')
                    if header_name:
                        headers[str(header_name)] = f'{api_key}'

    # New regular-schema structure (GitHub-style sections) uses flattened fields
    if not headers:
        api_key = _secret_to_str(settings.get('api_key'))
        if api_key:
            auth_type = str(settings.get('auth_type', 'Bearer'))
            auth_type_norm = auth_type.strip().lower()
            if auth_type_norm == 'bearer':
                headers['Authorization'] = f'Bearer {api_key}'
            elif auth_type_norm == 'basic':
                headers['Authorization'] = f'Basic {api_key}'
            elif auth_type_norm == 'custom':
                header_name = settings.get('custom_header_name')
                if header_name:
                    headers[str(header_name)] = f'{api_key}'

    return headers
