"""Microsoft Graph API wrapper: arbitrary Graph requests as the signed-in user, limited by toolkit rules."""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from ...configurations.ms_graph import MsGraphConfiguration, normalize_ms_graph_scopes
from ..utils.tool_groups import tool_group, with_tool_groups
from .graph_client import GraphClient, GraphError, GraphResponse
from .models import GraphBatch, GraphRequest
from .rules import DEFAULT_METHODS, RequestRules, normalize_path

log = logging.getLogger(__name__)

DEFAULT_MAX_RESPONSE_CHARS = 20000
DEFAULT_MAX_UPLOAD_MB = 25
DEFAULT_MAX_DOWNLOAD_MB = 50
DEFAULT_DOWNLOAD_BUCKET = "ms-graph"

_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

_REQUEST_DESCRIPTION = (
    "Call any Microsoft Graph API endpoint as the signed-in user (delegated access): method, path, OData query, "
    "JSON body and headers. Use $select to keep responses small and $top / follow_next_link for paging. "
    "Binary responses (file content, photos) are saved to artifact storage and returned as a file reference; "
    "upload_from_artifact sends a stored file as the body. The token holds only the credential's Graph "
    "permissions: HTTP 403 means a permission is missing. Examples: GET /me; GET /me/messages with query "
    "{'$top': 5, '$select': 'subject,from'}; POST /me/sendMail with a message body; "
    "GET /me/drive/root:/report.pdf:/content."
)
_BATCH_DESCRIPTION = (
    "Send up to 20 Microsoft Graph requests in one $batch call; each request is checked against the toolkit "
    "rules and returns its own status and body, keyed by id. Use for many independent lookups, e.g. several "
    "users or messages by id."
)


def _secret(value) -> Optional[str]:
    if value is None:
        return None
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


class MsGraphApiWrapper(BaseModel):
    """Arbitrary Microsoft Graph requests with delegated OAuth."""

    client_id: Optional[str] = Field(default=None, description="Azure AD Client ID")
    client_secret: Optional[SecretStr] = Field(default=None, description="Azure AD Client Secret")

    token: Optional[str] = Field(default=None, description="OAuth access token")
    refresh_token: Optional[str] = Field(default=None, description="OAuth refresh token")
    scopes: Optional[List[str]] = Field(default=None, description="OAuth scopes")

    oauth_discovery_endpoint: Optional[str] = Field(default=None, description="OAuth discovery endpoint")
    oauth_token_endpoint: Optional[str] = Field(default=None, description="OAuth token endpoint")
    oauth_app_id: Optional[str] = Field(default=None, description="Admin-managed OAuth app reference")

    allowed_methods: List[str] = Field(default_factory=lambda: list(DEFAULT_METHODS))
    allowed_paths: List[str] = Field(default_factory=list)
    denied_paths: List[str] = Field(default_factory=list)
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB
    max_download_mb: int = DEFAULT_MAX_DOWNLOAD_MB

    configuration_uuid: Optional[str] = Field(default=None, description="Configuration UUID")
    toolkit_id: Optional[int] = Field(default=None, description="Toolkit ID")
    toolkit_name: Optional[str] = Field(default=None, description="Toolkit name")
    elitea: Optional[Any] = Field(default=None, exclude=True)

    _client: Optional[GraphClient] = None
    _rules: Optional[RequestRules] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @field_validator("scopes", mode="before")
    @classmethod
    def _normalize_scopes(cls, value):
        return normalize_ms_graph_scopes(value)

    @field_validator("allowed_methods", "allowed_paths", "denied_paths", mode="before")
    @classmethod
    def _none_as_default(cls, value, info):
        if value is None:
            return list(DEFAULT_METHODS) if info.field_name == "allowed_methods" else []
        return value

    @model_validator(mode="after")
    def _build(self) -> "MsGraphApiWrapper":
        self._rules = RequestRules(self.allowed_methods, self.allowed_paths, self.denied_paths)
        if not self.token:
            log.warning("No access token provided - OAuth authorization required")
            return self
        token_endpoint = self.oauth_token_endpoint
        if not token_endpoint and self.oauth_discovery_endpoint:
            token_endpoint = f"{self.oauth_discovery_endpoint.rstrip('/')}/oauth2/v2.0/token"
        self._client = GraphClient(
            token=self.token,
            scopes=["offline_access", *(self.scopes or [])],
            refresh_token=self.refresh_token,
            client_id=self.client_id,
            client_secret=_secret(self.client_secret),
            oauth_token_endpoint=token_endpoint,
            on_unauthorized=self._raise_authorization_required,
            max_download_bytes=self.max_download_mb * 1024 * 1024,
        )
        return self

    # ------------------------------------------------------------------ #

    def _auth_required(self, message: str, status: Optional[int] = None):
        return MsGraphConfiguration._build_mcp_authorization_required(
            message=message,
            oauth_discovery_endpoint=self.oauth_discovery_endpoint or "https://login.microsoftonline.com/common",
            scopes=self.scopes,
            status=status,
            configuration_uuid=self.configuration_uuid,
            toolkit_id=self.toolkit_id,
            toolkit_name=self.toolkit_name,
            client_id=self.client_id,
            # An admin-managed app's secret stays server-side: not even a masked copy
            client_secret=None if self.oauth_app_id else self.client_secret,
            oauth_app_id=self.oauth_app_id,
        )

    def _raise_authorization_required(self, resp) -> None:
        api_message = MsGraphConfiguration._extract_api_error_message(resp)
        resp.close()
        if not self.oauth_discovery_endpoint:
            raise GraphError(f"Microsoft Graph access token is invalid or expired: {api_message}", 401)
        auth_error = self._auth_required(
            f"Microsoft Graph access token is invalid or expired: {api_message}. Please re-authorize.", 401
        )
        auth_error.tool_name = None
        raise auth_error

    def _ensure_client(self) -> GraphClient:
        if self._client is None:
            raise self._auth_required(
                "Microsoft Graph requires OAuth authorization. Please complete the OAuth flow to obtain an access token."
            )
        return self._client

    @staticmethod
    def _with_hint(exc: GraphError) -> GraphError:
        if exc.status == 403:
            exc.args = (f"{exc.args[0]}. The signed-in user or the token lacks a Microsoft Graph permission this "
                        "request needs: add it to the credential scopes and sign in again (an Entra admin may "
                        "need to grant it).",)
        return exc

    def _limit(self, max_chars: Optional[int]) -> int:
        return min(max_chars, self.max_response_chars) if max_chars else self.max_response_chars

    def _check(self, method: str, path: str) -> None:
        if normalize_path(path) == "/$batch":
            raise ToolException("Use the graph_batch tool for $batch requests")
        self._rules.check(method, path)

    # ------------------------------------------------------------------ #
    #  Artifacts                                                           #
    # ------------------------------------------------------------------ #

    def _read_artifact(self, filepath: str) -> tuple[bytes, str]:
        if not self.elitea:
            raise ToolException("Artifact storage is not available in this context")
        from ..utils import get_file_bytes_from_artifact
        try:
            file_bytes, file_name = get_file_bytes_from_artifact(self.elitea, filepath)
        except Exception as e:
            raise ToolException(str(e)) from e
        if not isinstance(file_bytes, (bytes, bytearray)):
            raise ToolException(f"Artifact '{filepath}' could not be read")
        limit = self.max_upload_mb * 1024 * 1024
        if len(file_bytes) > limit:
            raise ToolException(
                f"Artifact '{filepath}' is {len(file_bytes)} bytes, more than the toolkit upload limit of "
                f"{self.max_upload_mb} MB"
            )
        return bytes(file_bytes), file_name

    @staticmethod
    def _download_name(path: str, resp: GraphResponse, explicit: Optional[str]) -> str:
        if explicit:
            name = explicit
        else:
            segments = [s for s in unquote(normalize_path(path)).split("/") if s and s != "content"]
            name = segments[-1] if segments else "graph"
            # /drive/root:/folder/report.pdf:/content -> report.pdf
            name = name.rstrip(":").split(":")[-1] or "graph"
            if "." not in name:
                name += mimetypes.guess_extension(resp.content_type or "") or ".bin"
            name = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
        name = _UNSAFE_NAME_RE.sub("_", name).strip("._") or "graph.bin"
        return name[:200]

    def _save_artifact(self, resp: GraphResponse, path: str, bucket: Optional[str], file_name: Optional[str]) -> dict:
        if not self.elitea:
            raise ToolException(
                f"The response is binary ({resp.content_type or 'unknown type'}, {len(resp.content)} bytes) and "
                "artifact storage is not available to save it"
            )
        bucket = bucket or DEFAULT_DOWNLOAD_BUCKET
        name = self._download_name(path, resp, file_name)
        result = self.elitea.artifact(bucket).client.upload_artifact_s3(
            bucket, name, resp.content, content_type=resp.content_type or None
        )
        if isinstance(result, dict) and result.get("error"):
            raise ToolException(f"Could not save the response to artifact storage: {result['error']}")
        return {
            "filepath": result.get("filepath", f"/{bucket}/{name}"),
            "bucket": bucket,
            "file_name": result.get("filename", name),
            "size": len(resp.content),
            "content_type": resp.content_type,
        }

    # ------------------------------------------------------------------ #
    #  Tools                                                               #
    # ------------------------------------------------------------------ #

    def _body_result(self, resp: GraphResponse, limit: int) -> tuple[Any, bool]:
        if not resp.content:
            return None, False
        if resp.is_json:
            data = resp.json()
            text = json.dumps(data, ensure_ascii=False, default=str)
            if len(text) <= limit:
                return data, False
            return _truncate(text, limit)
        return _truncate(resp.text(), limit)

    @tool_group('execute')
    def graph_request(
        self,
        path: str,
        method: str = "GET",
        query: Optional[Dict[str, Any]] = None,
        body: Any = None,
        headers: Optional[Dict[str, str]] = None,
        api_version: str = "v1.0",
        follow_next_link: int = 0,
        max_chars: Optional[int] = None,
        upload_from_artifact: Optional[str] = None,
        upload_content_type: Optional[str] = None,
        download_bucket: Optional[str] = None,
        download_file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send one Microsoft Graph request."""
        method = (method or "GET").upper()
        self._check(method, path)
        client = self._ensure_client()
        url = client.build_url(path, api_version)

        content = None
        content_type = upload_content_type
        if upload_from_artifact:
            if body is not None:
                raise ToolException("Pass either body or upload_from_artifact, not both")
            content, file_name = self._read_artifact(upload_from_artifact)
            if not content_type:
                from ..utils import detect_mime_type
                content_type = detect_mime_type(content, file_name)

        try:
            resp = client.request(method, url, params=query, body=body, headers=headers,
                                  content=content, content_type=content_type)
        except GraphError as e:
            raise self._with_hint(e)

        result: Dict[str, Any] = {"status": resp.status, "headers": resp.headers}
        binary = resp.content and not (resp.is_json or resp.is_text)
        if binary or (download_bucket and resp.content and not resp.is_json):
            result["artifact"] = self._save_artifact(resp, path, download_bucket, download_file_name)
            return result

        limit = self._limit(max_chars)
        data, truncated = self._body_result(resp, limit)
        if isinstance(data, dict) and follow_next_link and isinstance(data.get("value"), list):
            data, truncated = self._follow_pages(client, method, data, follow_next_link, limit)
        if isinstance(data, dict) and data.get("@odata.nextLink"):
            result["next_link"] = data["@odata.nextLink"]
        result["body"] = data
        if truncated:
            result["truncated"] = True
            result["note"] = (f"Response cut to {limit} characters; use $select / $top or a smaller max_chars "
                              "page to get less data")
        return result

    def _follow_pages(self, client: GraphClient, method: str, data: dict, pages: int, limit: int):
        items = list(data.get("value") or [])
        next_link = data.get("@odata.nextLink")
        for _ in range(pages):
            if not next_link:
                break
            self._check(method, next_link)
            try:
                # nextLink is absolute and carries its own $skiptoken: send it unchanged
                page = client.request(method, next_link).json() or {}
            except GraphError as e:
                raise self._with_hint(e)
            items.extend(page.get("value") or [])
            next_link = page.get("@odata.nextLink")
            if len(json.dumps(items, default=str)) > limit:
                break
        merged = {k: v for k, v in data.items() if k not in ("value", "@odata.nextLink")}
        merged["value"] = items
        if next_link:
            merged["@odata.nextLink"] = next_link
        text = json.dumps(merged, ensure_ascii=False, default=str)
        if len(text) <= limit:
            return merged, False
        return _truncate(text, limit)

    @tool_group('execute')
    def graph_batch(self, requests: List[Any], api_version: str = "v1.0",
                    max_chars: Optional[int] = None) -> Dict[str, Any]:
        """Send up to 20 requests in one $batch call; every request is checked against the rules."""
        items = [dict(r) for r in requests]
        if not items or len(items) > 20:
            raise ToolException("graph_batch takes 1 to 20 requests")
        ids = [str(item.get("id")) for item in items]
        if len(set(ids)) != len(ids):
            raise ToolException("Request ids in a batch must be unique")
        batch_requests = []
        for item in items:
            method = (item.get("method") or "GET").upper()
            url = item.get("url") or ""
            if "://" in url:
                raise ToolException("Batch request urls are relative to the API version, e.g. /me/messages")
            self._check(method, url)
            entry: Dict[str, Any] = {"id": str(item["id"]), "method": method,
                                     "url": "/" + url.lstrip("/")}
            headers = dict(item.get("headers") or {})
            if item.get("body") is not None:
                entry["body"] = item["body"]
                headers.setdefault("Content-Type", "application/json")
            if headers:
                entry["headers"] = headers
            if item.get("depends_on"):
                entry["dependsOn"] = [str(d) for d in item["depends_on"]]
            batch_requests.append(entry)

        client = self._ensure_client()
        try:
            resp = client.request("POST", client.build_url("/$batch", api_version),
                                  body={"requests": batch_requests})
        except GraphError as e:
            raise self._with_hint(e)
        data = resp.json() or {}
        limit = self._limit(max_chars)
        responses = {str(r.get("id")): {"status": r.get("status"), "body": r.get("body")}
                     for r in data.get("responses") or []}
        text = json.dumps(responses, ensure_ascii=False, default=str)
        if len(text) <= limit:
            return {"responses": responses}
        cut, _ = _truncate(text, limit)
        return {"responses": cut, "truncated": True,
                "note": f"Response cut to {limit} characters; use $select or fewer requests"}

    # ------------------------------------------------------------------ #

    def run(self, action: str, **kwargs) -> Any:
        """Run a tool action by name."""
        action_map = {tool["name"]: tool["ref"] for tool in self.get_available_tools()}
        if action not in action_map:
            raise ValueError(f"Unknown action: {action}")
        return action_map[action](**kwargs)

    @with_tool_groups
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools with their schemas."""
        rules = f" {self._rules.describe()}" if self._rules else ""
        return [
            {
                "name": "graph_request",
                "description": _REQUEST_DESCRIPTION + rules,
                "args_schema": GraphRequest,
                "ref": self.graph_request,
            },
            {
                "name": "graph_batch",
                "description": _BATCH_DESCRIPTION + rules,
                "args_schema": GraphBatch,
                "ref": self.graph_batch,
            },
        ]
