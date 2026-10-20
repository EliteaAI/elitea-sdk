import logging
import re
import threading
import time
from hashlib import sha256
from typing import Optional
from urllib.parse import quote, urlencode, urlparse

import requests
from pydantic import BaseModel, ConfigDict, Field, SecretStr

log = logging.getLogger(__name__)

API_PATH = "/api/data/v9.2/"
TOKEN_SKEW_SECONDS = 120
_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,99}$")
_token_lock = threading.Lock()
_token_cache: dict = {}


class DataverseError(Exception):
    """Dataverse auth or request failure with a message safe to show to the user."""


class DataverseLoginRequired(DataverseError):
    """The stored refresh token is missing, expired or revoked: the user must sign in again."""


class DataverseConfiguration(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "metadata": {
                "label": "Dataverse (OneCRM)",
                "icon_url": "dataverse.svg",
                "section": "credentials",
                "type": "dataverse",
                "categories": ["other"],
                "extra_categories": ["dataverse", "dynamics", "crm", "onecrm", "microsoft"],
            }
        }
    )
    url: str = Field(description="Dataverse environment URL, e.g. https://onecrmepam.crm4.dynamics.com")
    tenant_id: str = Field(description="Entra tenant ID (GUID) or verified domain")
    client_id: str = Field(description="Entra app (client) ID")
    client_secret: SecretStr = Field(description="Entra app client secret")
    # Set by the Login button on the toolkit page, never typed by the user; stored in the vault like other secrets
    refresh_token: Optional[SecretStr] = Field(
        default=None, description="Refresh token of the signed-in user (set by Login)",
        json_schema_extra={"hidden": True},
    )

    @staticmethod
    def check_connection(settings: dict) -> str | None:
        """Refresh the token and call WhoAmI; asks the UI to run the Login flow when no token is stored."""
        if not _secret(settings.get("refresh_token")):
            raise build_login_required(settings, "Sign in to Dataverse to connect this credential.")
        try:
            dataverse_get(settings, "WhoAmI")
        except DataverseLoginRequired as e:
            raise build_login_required(settings, str(e)) from None
        except DataverseError as e:
            return str(e)
        return None


def normalize_org_url(url) -> str:
    url = (url or "").strip().rstrip("/")
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        raise DataverseError("Dataverse URL is not valid") from None
    host = (parsed.hostname or "").lower()
    if (parsed.scheme != "https" or not host.endswith(".dynamics.com") or parsed.path
            or parsed.username or port not in (None, 443)):
        raise DataverseError("Dataverse URL must look like https://<org>.crm.dynamics.com")
    return f"https://{host}"


def _secret(value) -> str:
    return value.get_secret_value() if hasattr(value, "get_secret_value") else (value or "")


def _refresh_error(error: str) -> DataverseError:
    if "AADSTS7000215" in error or "AADSTS7000222" in error:
        return DataverseError("The client secret is wrong or expired.")
    if "AADSTS65001" in error:
        return DataverseLoginRequired("The app has no consent for Dataverse: log in and approve the permission.")
    if any(code in error for code in ("AADSTS700082", "AADSTS700084", "AADSTS50173", "invalid_grant")):
        return DataverseLoginRequired("The Dataverse login expired or was revoked: log in again.")
    return DataverseError(f"Token refresh failed: {error[:200]}")


def get_access_token(settings: dict) -> str:
    """Exchange the stored refresh token for a Dataverse access token, cached until near expiry."""
    from ..runtime.utils.mcp_oauth import refresh_oauth_token

    org = normalize_org_url(settings.get("url"))
    # Token injected by the platform's stored-login service (scheduled runs); no refresh here
    if _secret(settings.get("access_token")):
        return _secret(settings.get("access_token"))
    tenant = (settings.get("tenant_id") or "").strip()
    client_id = (settings.get("client_id") or "").strip()
    client_secret = _secret(settings.get("client_secret"))
    refresh_token = _secret(settings.get("refresh_token"))
    if not (tenant and client_id and client_secret):
        raise DataverseError("tenant_id, client_id and client_secret are all required")
    if not refresh_token:
        raise DataverseLoginRequired("Not logged in to Dataverse: use the Login button on the credential.")
    if not _TENANT_RE.match(tenant):
        raise DataverseError("tenant_id must be a tenant GUID or domain name")

    key = (org, tenant, client_id, sha256(refresh_token.encode()).hexdigest())
    with _token_lock:
        cached = _token_cache.get(key)
        if cached and cached[1] - TOKEN_SKEW_SECONDS > time.time():
            return cached[0]
    try:
        data = refresh_oauth_token(
            token_endpoint=f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
            scope=f"{org}/user_impersonation offline_access",
        )
    except (ValueError, requests.RequestException) as e:
        raise _refresh_error(str(e)) from None
    token = data.get("access_token")
    if not token:
        raise DataverseError("Microsoft returned no access token")
    with _token_lock:
        _token_cache[key] = (token, time.time() + int(data.get("expires_in") or 3600))
    return token


def dataverse_get(settings: dict, path: str, params: Optional[dict] = None) -> dict:
    return _dataverse_request(settings, "GET", path, params=params)


def dataverse_patch(settings: dict, path: str, fields: dict) -> dict:
    """Update the given columns of an existing record; If-Match: * stops PATCH from creating a new one."""
    return _dataverse_request(settings, "PATCH", path, json_body=fields, extra_headers={
        "If-Match": "*",
        "Prefer": "return=representation",
    })


def _send(settings: dict, method: str, url: str, token: str, json_body, extra_headers) -> requests.Response:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0",
        "Prefer": 'odata.include-annotations="OData.Community.Display.V1.FormattedValue"',
        **(extra_headers or {}),
    }
    try:
        return requests.request(method, url, headers=headers, json=json_body, timeout=30)
    except requests.RequestException as e:
        raise DataverseError(f"Could not reach Dataverse ({type(e).__name__})") from None


def _dataverse_request(settings: dict, method: str, path: str, params: Optional[dict] = None,
                       json_body: Optional[dict] = None, extra_headers: Optional[dict] = None) -> dict:
    org = normalize_org_url(settings.get("url"))
    url = f"{org}{API_PATH}{path}"
    if params:
        url += "?" + urlencode(params, quote_via=quote)
    resp = _send(settings, method, url, get_access_token(settings), json_body, extra_headers)
    # A platform-injected token may expire mid-run; the stored login can mint a fresh one
    if resp.status_code == 401 and _secret(settings.get("access_token")) and _secret(settings.get("refresh_token")):
        fresh = {**settings, "access_token": None}
        resp = _send(fresh, method, url, get_access_token(fresh), json_body, extra_headers)
    if resp.status_code == 200:
        return resp.json()
    if resp.status_code == 204:
        return {}
    if resp.status_code == 401:
        raise DataverseError("Dataverse rejected the access token")
    if resp.status_code == 403:
        raise DataverseError("Your Dataverse user has no permission for this operation")
    if resp.status_code == 404:
        raise DataverseError("Record or table not found")
    if resp.status_code == 412:
        raise DataverseError("Record not found (updates never create new records)")
    if resp.status_code == 429:
        raise DataverseError("Dataverse rate limit reached, try again later")
    try:
        detail = resp.json().get("error", {}).get("message", "")
    except ValueError:
        detail = resp.text
    raise DataverseError(f"Dataverse returned HTTP {resp.status_code}: {detail[:300]}")


def build_login_required(settings: dict, message: str):
    """McpAuthorizationRequired carrying the Entra endpoints, so the UI opens its Login popup."""
    from ..runtime.utils.mcp_oauth import McpAuthorizationRequired

    try:
        org = normalize_org_url(settings.get("url"))
    except DataverseError as e:
        return e
    tenant = (settings.get("tenant_id") or "").strip()
    if not _TENANT_RE.match(tenant):
        return DataverseError("tenant_id must be a tenant GUID or domain name")
    authority = f"https://login.microsoftonline.com/{tenant}"
    scopes = [f"{org}/user_impersonation", "offline_access"]
    resource_metadata = {
        "resource_name": "Dataverse",
        "resource": org,
        "authorization_servers": [authority],
        "bearer_methods_supported": ["header"],
        "oauth_authorization_server": {
            "issuer": f"{authority}/v2.0",
            "authorization_endpoint": f"{authority}/oauth2/v2.0/authorize",
            "token_endpoint": f"{authority}/oauth2/v2.0/token",
            "scopes_supported": scopes,
        },
        "scopes_supported": scopes,
    }
    if settings.get("configuration_uuid"):
        resource_metadata["configuration_uuid"] = settings["configuration_uuid"]
    client_id = (settings.get("client_id") or "").strip()
    if client_id:
        resource_metadata["provided_settings"] = {"mcp_client_id": client_id, "scopes": scopes}
    return McpAuthorizationRequired(
        message=message,
        server_url=authority,
        resource_metadata_url=f"{authority}/v2.0/.well-known/openid-configuration",
        resource_metadata=resource_metadata,
        tool_name="Dataverse",
        toolkit_type="dataverse",
    )


def entra_authority(settings: dict) -> str:
    """Authority the platform keys injected tokens by, same as the Login flow's server_url."""
    return f"https://login.microsoftonline.com/{(settings.get('tenant_id') or '').strip()}"
