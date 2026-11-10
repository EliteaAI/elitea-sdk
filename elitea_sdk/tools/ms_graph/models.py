"""Pydantic models for the Microsoft Graph toolkit input schemas."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, create_model

_PATH = (
    "Graph path relative to the API version, e.g. '/me/messages', '/users/{id}/events', "
    "'/sites/{site-id}/drive/root/children'; or a full https://graph.microsoft.com/... URL "
    "(e.g. an @odata.nextLink), whose own version is used. Query parameters belong in query"
)
_QUERY = (
    "OData query parameters, e.g. {'$select': 'id,subject', '$top': 10, '$filter': \"isRead eq false\", "
    "'$orderby': 'receivedDateTime desc'}. $count and $search on directory objects also need the header "
    "ConsistencyLevel: eventual"
)
_BODY = "Request body: a JSON object / array (sent as application/json) or a string. Omit for GET and DELETE"
_HEADERS = (
    "Extra request headers, e.g. {'ConsistencyLevel': 'eventual'}, {'Prefer': 'outlook.body-content-type=\"text\"'}, "
    "{'If-Match': '<etag>'}. Authorization is set by the toolkit"
)
_API_VERSION = "Graph API version: v1.0 (stable, default) or beta (preview APIs that can change)"
_FOLLOW = (
    "How many more pages to fetch by following @odata.nextLink (0 = only the first page, max 20). "
    "Items of all pages are merged into 'value'; next_link is returned when more remain"
)
_MAX_CHARS = "Cut the returned JSON / text to this many characters (default: the toolkit's limit)"
_UPLOAD = (
    "Send a file from artifact storage as the request body, e.g. PUT /me/drive/root:/report.pdf:/content. "
    "Format: /{bucket}/{file_name}. Use instead of body"
)
_UPLOAD_CONTENT_TYPE = "Content-Type of the uploaded file; detected from the file when omitted"
_DOWNLOAD_BUCKET = (
    "Save a binary response (file content, photo) to this artifact bucket. Binary responses are always saved "
    "to an artifact, by default to the toolkit's bucket"
)
_DOWNLOAD_NAME = "File name for the saved binary response; derived from the path when omitted"

Method = Literal["GET", "POST", "PATCH", "PUT", "DELETE"]
ApiVersion = Literal["v1.0", "beta"]

GraphRequest = create_model(
    "GraphRequest",
    method=(Method, Field(default="GET", description="HTTP method")),
    path=(str, Field(description=_PATH)),
    query=(Optional[Dict[str, Any]], Field(default=None, description=_QUERY)),
    body=(Optional[Any], Field(default=None, description=_BODY)),
    headers=(Optional[Dict[str, str]], Field(default=None, description=_HEADERS)),
    api_version=(ApiVersion, Field(default="v1.0", description=_API_VERSION)),
    follow_next_link=(int, Field(default=0, ge=0, le=20, description=_FOLLOW)),
    max_chars=(Optional[int], Field(default=None, ge=500, description=_MAX_CHARS)),
    upload_from_artifact=(Optional[str], Field(default=None, description=_UPLOAD)),
    upload_content_type=(Optional[str], Field(default=None, description=_UPLOAD_CONTENT_TYPE)),
    download_bucket=(Optional[str], Field(default=None, description=_DOWNLOAD_BUCKET)),
    download_file_name=(Optional[str], Field(default=None, description=_DOWNLOAD_NAME)),
)


_BATCH_REQUESTS = (
    "Up to 20 requests sent in one Graph $batch call. Each: {'id': '1', 'method': 'GET', "
    "'url': '/me/messages?$top=5', 'body': {...} (POST / PATCH / PUT only), 'headers': {...}, "
    "'depends_on': ['ids that must run first']}. url is relative to the API version; query is allowed in it"
)


GraphBatch = create_model(
    "GraphBatch",
    requests=(List[Dict[str, Any]], Field(min_length=1, max_length=20, description=_BATCH_REQUESTS)),
    api_version=(ApiVersion, Field(default="v1.0", description=_API_VERSION)),
    max_chars=(Optional[int], Field(default=None, ge=500, description=_MAX_CHARS)),
)
