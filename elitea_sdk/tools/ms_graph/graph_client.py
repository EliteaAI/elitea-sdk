"""HTTP client for arbitrary Microsoft Graph requests with a delegated (user) token.

The bearer token is sent only to https://graph.microsoft.com. Redirects (file downloads
answer 302 to a pre-authenticated storage URL) are followed by hand: to Graph with the
token, anywhere else without it. A 401 refreshes the token once with the stored login;
429 / 503 are retried honouring Retry-After.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from langchain_core.tools import ToolException

from ...runtime.utils.mcp_oauth import is_entra_url
from .rules import GRAPH_HOST, split_graph_url

log = logging.getLogger(__name__)

GRAPH_ROOT = f"https://{GRAPH_HOST}"

_MAX_RETRIES = 3
_MAX_RETRY_AFTER = 30
_MAX_REDIRECTS = 5
_TIMEOUT = 60
_CHUNK = 64 * 1024

# Headers the caller may not set: auth and transport are the client's job
_RESERVED_HEADERS = {"authorization", "host", "content-length", "transfer-encoding", "cookie"}
# Response headers worth showing to the LLM
RESPONSE_HEADERS = ("content-type", "location", "request-id", "retry-after", "etag", "odata-version")


class GraphError(ToolException):
    """Graph API error carrying the HTTP status."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status
        if status == 404:
            # Read by runtime.tool_outcome so a missing resource is classed as INPUT, not unclassified.
            self.provider_error_category = "resource_not_found"


class GraphResponse:
    """Status, selected headers and the body of a Graph response (bytes, already size-checked)."""

    def __init__(self, status: int, headers: Dict[str, str], content: bytes, url: str):
        self.status = status
        self.headers = headers
        self.content = content
        self.url = url

    @property
    def content_type(self) -> str:
        return (self.headers.get("content-type") or "").split(";", 1)[0].strip().lower()

    @property
    def is_json(self) -> bool:
        return self.content_type == "application/json" or self.content_type.endswith("+json")

    @property
    def is_text(self) -> bool:
        ct = self.content_type
        return ct.startswith("text/") or ct in ("application/xml", "application/xhtml+xml") or ct.endswith("+xml")

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8")) if self.content else None

    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


def _is_graph_url(url: str) -> bool:
    parsed = urlparse(url)
    return (parsed.scheme == "https" and (parsed.hostname or "").lower() == GRAPH_HOST
            and not parsed.username and parsed.port in (None, 443))


def _error_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            code, message = error.get("code"), error.get("message")
            return f"{code}: {message}" if code and message else (message or code or str(error))
        return str(body)
    except Exception:
        return (resp.text or "").strip()[:500] or "no details"


class GraphClient:
    """Sends Graph requests as the signed-in user.

    Args:
        token: delegated access token.
        scopes: scopes of the credential, used to refresh the token.
        refresh_token / client_id / client_secret / oauth_token_endpoint: stored login for one refresh on 401.
        on_unauthorized: called with the 401 response once refreshing failed; must raise.
        max_download_bytes: largest response body read.
    """

    def __init__(
        self,
        token: str,
        scopes: Optional[list] = None,
        refresh_token: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        oauth_token_endpoint: Optional[str] = None,
        on_unauthorized=None,
        max_download_bytes: int = 50 * 1024 * 1024,
    ):
        self._token = token
        self._scopes = list(scopes or [])
        self._refresh_token = refresh_token
        self._client_id = client_id
        self._client_secret = client_secret
        self._oauth_token_endpoint = oauth_token_endpoint
        self._on_unauthorized = on_unauthorized
        self.max_download_bytes = max_download_bytes

    # ------------------------------------------------------------------ #

    def build_url(self, path: str, api_version: str = "v1.0") -> str:
        """Absolute Graph URL for a relative path or a Graph URL (whose own version wins)."""
        version, clean = split_graph_url(path)
        return f"{GRAPH_ROOT}/{version or api_version}{clean}"

    def _try_refresh_token(self) -> bool:
        """Attempt to refresh the access token using the stored refresh token."""
        if not self._refresh_token or not self._oauth_token_endpoint:
            return False
        # Never send the refresh token and client secret to a non-Entra host
        if not is_entra_url(self._oauth_token_endpoint):
            log.warning("MS Graph: token refresh skipped, token endpoint is not an Entra host")
            return False
        try:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "scope": " ".join(self._scopes) if self._scopes else f"{GRAPH_ROOT}/.default",
            }
            if self._client_id:
                data["client_id"] = self._client_id
            if self._client_secret:
                data["client_secret"] = self._client_secret
            resp = requests.post(self._oauth_token_endpoint, data=data, timeout=10)
            if resp.status_code == 200:
                token_data = resp.json()
                self._token = token_data["access_token"]
                if token_data.get("refresh_token"):
                    self._refresh_token = token_data["refresh_token"]
                log.info("MS Graph: access token refreshed successfully")
                return True
            log.warning("MS Graph: token refresh failed with HTTP %s", resp.status_code)
            return False
        except Exception as e:
            log.warning("MS Graph: token refresh error: %s", type(e).__name__)  # never the request (secret, token)
            return False

    @staticmethod
    def _retry_delay(resp: requests.Response, attempt: int) -> float:
        try:
            delay = float((resp.headers or {}).get("Retry-After"))
        except (TypeError, ValueError):
            delay = 2 ** attempt
        return max(0.0, min(delay, _MAX_RETRY_AFTER))

    def _read_body(self, resp: requests.Response) -> bytes:
        length = resp.headers.get("Content-Length")
        if length and length.isdigit() and int(length) > self.max_download_bytes:
            resp.close()
            raise ToolException(
                f"Response is {int(length)} bytes, more than the toolkit limit of {self.max_download_bytes} bytes"
            )
        chunks, size = [], 0
        for chunk in resp.iter_content(_CHUNK):
            size += len(chunk)
            if size > self.max_download_bytes:
                resp.close()
                raise ToolException(f"Response is larger than the toolkit limit of {self.max_download_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    def _send(self, method: str, url: str, headers: Dict[str, str], params, data) -> requests.Response:
        with_token = _is_graph_url(url)
        send_headers = dict(headers)
        if with_token:
            send_headers["Authorization"] = f"Bearer {self._token}"
        return requests.request(method, url, headers=send_headers, params=params, data=data,
                                timeout=_TIMEOUT, allow_redirects=False, stream=True)

    def request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        body: Any = None,
        headers: Optional[Dict[str, str]] = None,
        content: Optional[bytes] = None,
        content_type: Optional[str] = None,
    ) -> GraphResponse:
        """Send one request; JSON-encode dict / list bodies, send ``content`` bytes as is."""
        if not _is_graph_url(url):
            raise ToolException(f"Only {GRAPH_ROOT} URLs are allowed")
        method = method.upper()
        send_headers = {"Accept": "application/json"}
        for key, value in (headers or {}).items():
            if str(key).strip().lower() in _RESERVED_HEADERS:
                log.debug("MS Graph: header %s is set by the client, ignored", key)
                continue
            send_headers[str(key)] = str(value)

        data = None
        if content is not None:
            data = content
            send_headers["Content-Type"] = content_type or send_headers.get("Content-Type") or "application/octet-stream"
        elif body is not None:
            if isinstance(body, (dict, list)):
                data = json.dumps(body).encode("utf-8")
                send_headers.setdefault("Content-Type", "application/json")
            else:
                data = str(body).encode("utf-8")
                send_headers.setdefault("Content-Type", "text/plain")

        refreshed = False
        attempt = 0
        redirects = 0
        current_url, current_method, current_params, current_data = url, method, params, data
        while True:
            resp = self._send(current_method, current_url, send_headers, current_params, current_data)
            on_graph = _is_graph_url(current_url)
            if resp.status_code == 401 and on_graph and not refreshed:
                refreshed = True
                if self._try_refresh_token():
                    resp.close()
                    continue
            if resp.status_code in (429, 503) and attempt < _MAX_RETRIES:
                delay = self._retry_delay(resp, attempt)
                attempt += 1
                resp.close()
                log.warning("MS Graph: throttled (HTTP %s), retrying in %.0fs", resp.status_code, delay)
                time.sleep(delay)
                continue
            if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                redirects += 1
                if redirects > _MAX_REDIRECTS:
                    raise ToolException("Too many redirects")
                target = urljoin(current_url, resp.headers["Location"])
                resp.close()
                if urlparse(target).scheme != "https":
                    raise ToolException("Redirect to a non-https URL refused")
                if resp.status_code == 303 or (resp.status_code in (301, 302) and current_method != "GET"):
                    current_method, current_data = "GET", None
                current_url, current_params = target, None
                if not _is_graph_url(target):
                    # Pre-authenticated storage URL: no token and none of the caller's headers
                    send_headers = {"Accept": "*/*"}
                continue
            if resp.status_code == 401 and on_graph and self._on_unauthorized:
                self._on_unauthorized(resp)
            if not resp.ok:
                message = _error_message(resp)
                resp.close()
                raise GraphError(f"Microsoft Graph returned HTTP {resp.status_code}: {message}", resp.status_code)
            content_bytes = self._read_body(resp)
            selected = {k: resp.headers[k] for k in RESPONSE_HEADERS if k in resp.headers}
            return GraphResponse(resp.status_code, {k.lower(): v for k, v in selected.items()},
                                 content_bytes, resp.url or current_url)
