"""Aha! API wrapper.

Talks to Aha! via two transports:
- REST v1 at ``{base_url}/api/v1``
- GraphQL v2 at ``{base_url}/api/v2/graphql``

Both use the same ``Authorization: Bearer <api_key>`` header.

Tool surface tracks Aha's own remote MCP server (``find_project``,
``search_records``, ``read_records``, ``manage_record``, ``add_comment``,
``copy_record``, ``create_record_link``, ``fields_metadata``,
``field_options_metadata``) plus the type-specific read/list tools used
under the hood.
"""

from __future__ import annotations

import html
import io
import logging
import re
from typing import Annotated, Any, Dict, Iterable, Iterator, List, Literal, Optional

import requests
from langchain_core.tools import ToolException
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SecretStr,
    create_model,
    model_validator,
)
from pydantic.fields import PrivateAttr

from ..elitea_base import BaseToolApiWrapper
from ..utils import get_file_bytes_from_artifact

logger = logging.getLogger(__name__)

_REST_PREFIX = "/api/v1"
_GRAPHQL_PATH = "/api/v2/graphql"

# Reference-number formats — copied verbatim from aha-mcp so bad input fails
# fast with a clear message before hitting Aha.
_FEATURE_REF_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d+$")
_REQUIREMENT_REF_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d+-\d+$")
_PAGE_REF_RE = re.compile(r"^[A-Z][A-Z0-9]*-N-\d+$")

# Aha REST resource plural mapping — used by comment/attachment/link tools.
_RESOURCE_PLURAL: Dict[str, str] = {
    "feature": "features",
    "requirement": "requirements",
    "idea": "ideas",
    "release": "releases",
    "release_phase": "release_phases",
    "epic": "epics",
    "initiative": "initiatives",
    "product": "products",
    "goal": "goals",
    "page": "pages",
    "to_do": "tasks",
}


def _format_allowed_values(values: Iterable[str]) -> str:
    quoted = [f"`{value}`" for value in values]
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])}, or {quoted[-1]}"


_COMMENTABLE_RESOURCE_TYPES = (
    "feature",
    "requirement",
    "idea",
    "release",
    "release_phase",
    "epic",
    "initiative",
    "goal",
    "page",
    "to_do",
)
_COMMENTABLE_RESOURCE_VALUES = _format_allowed_values(_COMMENTABLE_RESOURCE_TYPES)
_COMMENT_RESOURCE_ALIASES = {
    "todo": "to_do",
    "to-do": "to_do",
    "to-dos": "to_do",
    "to_dos": "to_do",
    "task": "to_do",
    "tasks": "to_do",
}
_CommentResourceType = Literal[
    "feature",
    "requirement",
    "idea",
    "release",
    "release_phase",
    "epic",
    "initiative",
    "goal",
    "page",
    "to_do",
]

# Record types that ``manage_record`` can create/update/delete via REST.
# Keep the whitelist tight so the tool cannot silently hit unsupported
# endpoints. Pages are Aha's "notes" resource and expose full CRUD via REST.
_MANAGEABLE_RECORD_TYPES = (
    "feature",
    "requirement",
    "idea",
    "release",
    "initiative",
    "epic",
    "page",
)
_MANAGEABLE_RECORD_VALUES = _format_allowed_values(_MANAGEABLE_RECORD_TYPES)

_MANAGE_ACTIONS = ("create", "update", "delete")
_MANAGE_ACTION_VALUES = _format_allowed_values(_MANAGE_ACTIONS)

_OUTPUT_FORMATS = ("json", "csv", "markdown")
_OUTPUT_FORMAT_VALUES = _format_allowed_values(_OUTPUT_FORMATS)

_SEARCHABLE_RECORD_TYPES = (
    "feature",
    "requirement",
    "release",
    "idea",
    "epic",
    "initiative",
    "product",
)
_SEARCHABLE_RECORD_VALUES = _format_allowed_values(_SEARCHABLE_RECORD_TYPES)

_READABLE_RECORD_VALUES = _format_allowed_values(_SEARCHABLE_RECORD_TYPES + ("page",))

_RECORD_LINK_TARGET_TYPES = (
    "feature",
    "release",
    "idea",
    "epic",
    "release_phase",
    "initiative",
    "page",
    "goal",
)
_RECORD_LINK_SOURCE_TYPES = _RECORD_LINK_TARGET_TYPES + ("requirement",)
_RECORD_LINK_SOURCE_VALUES = _format_allowed_values(_RECORD_LINK_SOURCE_TYPES)
_RECORD_LINK_TARGET_VALUES = _format_allowed_values(_RECORD_LINK_TARGET_TYPES)
_RECORD_LINK_TYPE_LABELS = {
    10: "relates to",
    20: "depends on",
    30: "duplicated by",
    40: "contained by",
    50: "impacted by",
    60: "blocked by",
    80: "research for",
}
_RECORD_LINK_TYPE_VALUES = ", ".join(
    f"`{code}` ({label})" for code, label in _RECORD_LINK_TYPE_LABELS.items()
)
_RecordLinkSourceType = Literal[
    "feature",
    "release",
    "idea",
    "epic",
    "release_phase",
    "initiative",
    "page",
    "goal",
    "requirement",
]
_RecordLinkTargetType = Literal[
    "feature",
    "release",
    "idea",
    "epic",
    "release_phase",
    "initiative",
    "page",
    "goal",
]
_RecordLinkType = Literal[10, 20, 30, 40, 50, 60, 80]

# GraphQL query strings — copied from aha-mcp v1.1.0.
_QUERY_GET_PAGE = """
query GetPage($id: ID!, $includeParent: Boolean!) {
  page(id: $id) {
    id
    referenceNum
    name
    description { markdownBody }
    children { id referenceNum name }
    parent @include(if: $includeParent) { id referenceNum name }
  }
}
"""

_QUERY_GET_FEATURE = """
query GetFeature($id: ID!) {
  feature(id: $id) {
    id
    referenceNum
    name
    description { markdownBody }
    workflowStatus { name }
  }
}
"""

_QUERY_GET_REQUIREMENT = """
query GetRequirement($id: ID!) {
  requirement(id: $id) {
    id
    referenceNum
    name
    description { markdownBody }
    workflowStatus { name }
  }
}
"""

_QUERY_SEARCH_DOCUMENTS = """
query SearchDocuments($query: String!, $searchableType: [String!]) {
  searchDocuments(filters: { query: $query, searchableType: $searchableType }) {
    nodes { name url searchableId searchableType }
  }
}
"""


# ---------------------------------------------------------------------------
# Args schemas
# ---------------------------------------------------------------------------

OUTPUT_FORMAT_FIELD = (
    Optional[str],
    Field(
        default="json",
        description=f"Response format: {_OUTPUT_FORMAT_VALUES}. Defaults to `json`.",
    ),
)
FIELDS_FIELD = (
    Optional[List[str]],
    Field(
        default=None,
        description=(
            "Optional allowlist of top-level record fields to include in the "
            "response. Reduces token usage for large payloads. Common fields: "
            "`id`, `reference_num`, `name`, `created_at`, `updated_at`."
        ),
    ),
)
PER_PAGE_FIELD = (
    Optional[int],
    Field(
        default=25,
        ge=1,
        le=200,
        description="Number of records per Aha page (max 200).",
    ),
)
MAX_RECORDS_FIELD = (
    Optional[int],
    Field(
        default=100,
        ge=1,
        le=2000,
        description="Total record cap across pagination (stops early once reached).",
    ),
)
RESOURCE_TYPE_FIELD = (
    str,
    Field(description=f"Aha resource type: {_COMMENTABLE_RESOURCE_VALUES}."),
)
COMMENT_RESOURCE_TYPE_FIELD = (
    _CommentResourceType,
    Field(
        description=(
            f"Canonical Aha comment resource type: {_COMMENTABLE_RESOURCE_VALUES}. "
            "Use `to_do` for an Aha to-do; the REST API addresses it as a task."
        )
    ),
)

def _create_reference_input(model_name: str, example: str) -> type[BaseModel]:
    return create_model(
        model_name,
        reference_or_id=(
            str,
            Field(
                description=(
                    f"Aha reference number (e.g. `{example}`) or numeric record ID."
                ),
            ),
        ),
        output_format=OUTPUT_FORMAT_FIELD,
        fields=FIELDS_FIELD,
    )


AhaFeatureReferenceInput = _create_reference_input(
    "AhaFeatureReferenceInput", "DEVELOP-123"
)
AhaRequirementReferenceInput = _create_reference_input(
    "AhaRequirementReferenceInput", "PROD-5-1"
)
AhaReleaseReferenceInput = _create_reference_input(
    "AhaReleaseReferenceInput", "PROD-R-4"
)
AhaInitiativeReferenceInput = _create_reference_input(
    "AhaInitiativeReferenceInput", "PROD-I-1"
)
AhaEpicReferenceInput = _create_reference_input(
    "AhaEpicReferenceInput", "PROD-E-1"
)
AhaIdeaReferenceInput = _create_reference_input(
    "AhaIdeaReferenceInput", "PROD-I-1"
)
AhaProductReferenceInput = _create_reference_input(
    "AhaProductReferenceInput", "PROD"
)

AhaListFeaturesInput = create_model(
    "AhaListFeaturesInput",
    product_id=(Optional[str], Field(default=None, description="Filter by product reference/ID.")),
    release_id=(Optional[str], Field(default=None, description="Filter by release reference/ID.")),
    q=(Optional[str], Field(default=None, description="Free-text search filter.")),
    updated_since=(Optional[str], Field(default=None, description="ISO-8601 timestamp filter.")),
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaListRequirementsInput = create_model(
    "AhaListRequirementsInput",
    feature_id=(
        str,
        Field(description="Feature reference/ID that owns the requirements."),
    ),
    q=(Optional[str], Field(default=None, description="Free-text search filter.")),
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaListReleasesInput = create_model(
    "AhaListReleasesInput",
    product_id=(Optional[str], Field(default=None, description="Filter by product reference/ID.")),
    parking_lot=(Optional[bool], Field(default=None, description="Filter parking-lot releases.")),
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaListInitiativesInput = create_model(
    "AhaListInitiativesInput",
    product_id=(Optional[str], Field(default=None, description="Filter by product reference/ID.")),
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaListEpicsInput = create_model(
    "AhaListEpicsInput",
    product_id=(Optional[str], Field(default=None, description="Filter by product reference/ID.")),
    release_id=(Optional[str], Field(default=None, description="Filter by release reference/ID.")),
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaListIdeasInput = create_model(
    "AhaListIdeasInput",
    product_id=(Optional[str], Field(default=None, description="Filter by product reference/ID.")),
    q=(Optional[str], Field(default=None, description="Free-text search filter.")),
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaListProductsInput = create_model(
    "AhaListProductsInput",
    updated_since=(Optional[str], Field(default=None, description="ISO-8601 timestamp filter.")),
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaSearchInput = create_model(
    "AhaSearchInput",
    q=(str, Field(description="Free-text search query (searches across Aha records).")),
    type=(
        Optional[str],
        Field(
            default=None,
            description=(
                "Optional record-type filter, e.g. `feature`, `requirement`, "
                "`release`, `idea`, `epic`."
            ),
        ),
    ),
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaGetPageInput = create_model(
    "AhaGetPageInput",
    reference=(
        str,
        Field(
            description="Aha page reference number, e.g. `ABC-N-213`.",
        ),
    ),
    include_parent=(
        Optional[bool],
        Field(default=False, description="Whether to include the parent page in the response."),
    ),
)

AhaSearchDocumentsInput = create_model(
    "AhaSearchDocumentsInput",
    query=(str, Field(description="Free-text query passed to Aha document search.")),
    searchable_type=(
        Optional[str],
        Field(
            default="Page",
            description="Document type filter, defaults to `Page`.",
        ),
    ),
)

AhaGetFeatureGqlInput = create_model(
    "AhaGetFeatureGqlInput",
    reference=(str, Field(description="Feature reference number, e.g. `DEVELOP-123`.")),
)

AhaGetRequirementGqlInput = create_model(
    "AhaGetRequirementGqlInput",
    reference=(str, Field(description="Requirement reference number, e.g. `ADT-123-1`.")),
)

# ----- M3 write / dispatcher schemas -----

def _normalize_properties_input(value: Any) -> Any:
    """Accept the empty array emitted by older toolkit forms as an empty map."""
    if isinstance(value, list) and not value:
        return {}
    return value


AHA_PROPERTIES_TYPE = Annotated[
    Dict[str, Any],
    BeforeValidator(_normalize_properties_input),
]

AhaAddCommentInput = create_model(
    "AhaAddCommentInput",
    resource_type=COMMENT_RESOURCE_TYPE_FIELD,
    resource_id=(str, Field(description="Aha reference number or numeric ID of the target record.")),
    body=(str, Field(description="Comment body (HTML or plain text).")),
)

AhaListCommentsInput = create_model(
    "AhaListCommentsInput",
    resource_type=COMMENT_RESOURCE_TYPE_FIELD,
    resource_id=(str, Field(description="Aha reference number or numeric ID of the target record.")),
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaManageRecordInput = create_model(
    "AhaManageRecordInput",
    action=(
        str,
        Field(
            description=(
                "`create` to insert a new record, `update` to modify an existing "
                "one, `delete` to remove one."
            )
        ),
    ),
    record_type=(
        str,
        Field(
            description=f"Record type. Accepted: {_MANAGEABLE_RECORD_VALUES}."
        ),
    ),
    record_id=(
        Optional[str],
        Field(
            default=None,
            description=(
                "Existing record reference/ID. Required for `action='update'` "
                "and `action='delete'`. For `action='create'` on a requirement, "
                "pass the parent feature reference here."
            ),
        ),
    ),
    parent_id=(
        Optional[str],
        Field(
            default=None,
            description=(
                "For `action='create'`: parent scope — release ref for features "
                "and epics, product ref for ideas/releases/initiatives/pages, "
                "feature ref for requirements. Ignored for updates and deletes."
            ),
        ),
    ),
    properties=(
        AHA_PROPERTIES_TYPE,
        Field(
            default_factory=dict,
            description=(
                "Field/value map to set on the record. See Aha REST docs for the "
                "specific fields accepted by each record type."
            ),
        ),
    ),
)

AhaCreateRecordInput = create_model(
    "AhaCreateRecordInput",
    record_type=(
        str,
        Field(description=f"Record type to create. Accepted: {_MANAGEABLE_RECORD_VALUES}."),
    ),
    parent_id=(
        str,
        Field(
            description=(
                "Parent scope: release ref for features and epics, product ref "
                "for ideas/releases/initiatives/pages, or feature ref for requirements."
            ),
        ),
    ),
    properties=(
        AHA_PROPERTIES_TYPE,
        Field(
            description=(
                "Field/value map for the new record. Include the fields required "
                "by Aha for the selected record type, usually at least `name`."
            ),
        ),
    ),
)

AhaUpdateRecordInput = create_model(
    "AhaUpdateRecordInput",
    record_type=(
        str,
        Field(description=f"Record type to update. Accepted: {_MANAGEABLE_RECORD_VALUES}."),
    ),
    record_id=(
        str,
        Field(description="Existing record reference or numeric ID."),
    ),
    parent_id=(
        Optional[str],
        Field(
            default=None,
            description=(
                "Product reference/ID. Required when updating a release or initiative; "
                "not used for other record types."
            ),
        ),
    ),
    properties=(
        AHA_PROPERTIES_TYPE,
        Field(description="Field/value map containing the Aha fields to update."),
    ),
)

AhaDeleteRecordInput = create_model(
    "AhaDeleteRecordInput",
    record_type=(
        str,
        Field(description=f"Record type to delete. Accepted: {_MANAGEABLE_RECORD_VALUES}."),
    ),
    record_id=(
        str,
        Field(description="Existing record reference or numeric ID."),
    ),
    parent_id=(
        Optional[str],
        Field(
            default=None,
            description=(
                "Product reference/ID. Required when deleting a release or initiative; "
                "not used for other record types."
            ),
        ),
    ),
)

AhaCreateRecordLinkInput = create_model(
    "AhaCreateRecordLinkInput",
    from_record_type=(
        _RecordLinkSourceType,
        Field(
            description=(
                f"Source record type. Accepted: {_RECORD_LINK_SOURCE_VALUES}."
            )
        ),
    ),
    from_id=(
        str,
        Field(
            description=(
                "Source record reference or numeric ID. References are resolved "
                "to Aha's internal numeric ID before creating the link. Release "
                "phases require numeric IDs."
            )
        ),
    ),
    to_record_type=(
        _RecordLinkTargetType,
        Field(
            description=(
                f"Target record type. Accepted: {_RECORD_LINK_TARGET_VALUES}."
            )
        ),
    ),
    to_id=(
        str,
        Field(
            description=(
                "Target record reference or numeric ID. References are resolved "
                "to Aha's internal numeric ID before creating the link. Release "
                "phases require numeric IDs."
            )
        ),
    ),
    link_type=(
        _RecordLinkType,
        Field(
            description=(
                f"Required Aha relationship code: {_RECORD_LINK_TYPE_VALUES}."
            ),
        ),
    ),
)

AhaCopyRecordInput = create_model(
    "AhaCopyRecordInput",
    record_type=(str, Field(description="Record type. Only `release` is currently supported by Aha.")),
    record_id=(str, Field(description="Record reference or numeric ID to duplicate.")),
)

AhaFieldsMetadataInput = create_model(
    "AhaFieldsMetadataInput",
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaFieldOptionsInput = create_model(
    "AhaFieldOptionsInput",
    field_id=(
        str,
        Field(
            description=(
                "Required numeric custom-field definition ID returned by "
                "`fields_metadata`."
            )
        ),
    ),
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaAttachFileInput = create_model(
    "AhaAttachFileInput",
    resource_type=RESOURCE_TYPE_FIELD,
    resource_id=(str, Field(description="Aha reference number or numeric ID of the target record.")),
    filepath=(
        str,
        Field(
            description=(
                "File path in `/{bucket}/{filename}` format pointing to the "
                "artifact to attach. Get this from a file/image generation or "
                "upload tool response."
            ),
        ),
    ),
    filename=(
        Optional[str],
        Field(default=None, description="Override filename sent to Aha (defaults to basename of `filepath`)."),
    ),
)

AhaFindProjectInput = create_model(
    "AhaFindProjectInput",
    q=(
        Optional[str],
        Field(default=None, description="Free-text search filter matched against product name."),
    ),
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaSearchRecordsInput = create_model(
    "AhaSearchRecordsInput",
    record_type=(
        str,
        Field(
            description=f"Record type to search: {_SEARCHABLE_RECORD_VALUES}.",
        ),
    ),
    q=(Optional[str], Field(default=None, description="Free-text search filter.")),
    feature_id=(
        Optional[str],
        Field(
            default=None,
            description="Feature reference/ID. Required when record_type is `requirement`.",
        ),
    ),
    product_id=(Optional[str], Field(default=None, description="Scope search to a product reference/ID.")),
    release_id=(Optional[str], Field(default=None, description="Scope search to a release reference/ID (features/epics).")),
    updated_since=(Optional[str], Field(default=None, description="ISO-8601 timestamp filter.")),
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaReadRecordsInput = create_model(
    "AhaReadRecordsInput",
    record_type=(
        str,
        Field(
            description=f"Record type: {_READABLE_RECORD_VALUES}.",
        ),
    ),
    reference_or_id=(
        str,
        Field(description="Aha reference number or numeric ID of the record."),
    ),
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)


class AhaApiWrapper(BaseToolApiWrapper):
    """Aha! transport wrapper.

    Fields mirror :class:`AhaConfiguration` so a config dict can be unpacked
    directly into the constructor via ``AhaApiWrapper(**config)``.
    """

    base_url: str
    api_key: SecretStr
    elitea: Any = Field(default=None, exclude=True)

    _session: Optional[requests.Session] = PrivateAttr(default=None)
    _rest_url: str = PrivateAttr(default="")
    _graphql_url: str = PrivateAttr(default="")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        base_url = (values.get("base_url") or "").strip()
        if not base_url:
            raise ToolException("Aha! base_url is required")
        if not base_url.startswith(("http://", "https://")):
            raise ToolException("Aha! base_url must start with http:// or https://")
        values["base_url"] = base_url.rstrip("/")

        if not values.get("api_key"):
            raise ToolException("Aha! api_key is required")
        return values

    def model_post_init(self, __context: Any) -> None:
        # ``model_construct()`` is used for schema introspection with no field
        # values populated — skip session setup in that case.
        if "base_url" not in self.__dict__ or "api_key" not in self.__dict__:
            return

        self._rest_url = f"{self.base_url}{_REST_PREFIX}"
        self._graphql_url = f"{self.base_url}{_GRAPHQL_PATH}"

        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        self._session = session

    # ----- REST helpers -----

    def _rest_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        url = f"{self._rest_url}/{path.lstrip('/')}"
        try:
            response = self._session.request(
                method,
                url,
                params=params,
                json=json,
                files=files,
                timeout=timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise ToolException(f"Aha! REST {method} {path} network error: {exc}") from exc

        if not response.ok:
            body = self._rest_error_excerpt(response.text or "")
            raise ToolException(
                f"Aha! REST {method} {path} failed ({response.status_code}): {body}"
            )

        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise ToolException(
                f"Aha! REST {method} {path} returned non-JSON body"
            ) from exc

    @staticmethod
    def _rest_error_excerpt(body: str) -> str:
        """Return a concise error detail without exposing an upstream HTML page."""
        text = (body or "").strip()
        if not text:
            return "empty error response"
        if re.search(r"<!doctype\s+html|<html\b", text, re.IGNORECASE):
            title = re.search(
                r"<title[^>]*>(.*?)</title>",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            text = title.group(1) if title else re.sub(r"<[^>]+>", " ", text)
            text = html.unescape(text)
            text = re.sub(r"\s+", " ", text).strip()
        return text[:500]

    def _rest_get(self, path: str, **params: Any) -> Dict[str, Any]:
        # Filter out None params so we don't send `?foo=None` upstream.
        cleaned = {k: v for k, v in params.items() if v is not None}
        return self._rest_request("GET", path, params=cleaned or None)

    def _rest_post(
        self,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._rest_request("POST", path, json=json, files=files)

    def _rest_put(self, path: str, *, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._rest_request("PUT", path, json=json)

    def _rest_delete(self, path: str) -> Dict[str, Any]:
        return self._rest_request("DELETE", path)

    def _paginate(self, path: str, **params: Any) -> Iterator[Dict[str, Any]]:
        """Walk Aha! pagination and yield each record.

        Aha! responses use::

            { "<collection>": [ ... ], "pagination": { "current_page", "total_pages", "total_records" } }

        The collection key is inferred from the response body (the first key
        that maps to a list). Endpoints without pagination return the payload
        as-is; callers can iterate a single-record result too.
        """
        page = 1
        while True:
            payload = self._rest_get(path, page=page, **params)
            collection_key = next(
                (k for k, v in payload.items() if isinstance(v, list) and k != "pagination"),
                None,
            )
            if collection_key is None:
                # Non-paginated / single-record response — yield once and stop.
                yield payload
                return

            for record in payload[collection_key]:
                yield record

            pagination = payload.get("pagination") or {}
            current = pagination.get("current_page", page)
            total = pagination.get("total_pages", current)
            if current >= total:
                return
            page = current + 1

    def _collect(
        self,
        path: str,
        *,
        max_records: int = 100,
        per_page: int = 25,
        **params: Any,
    ) -> List[Dict[str, Any]]:
        """Paginate ``path`` until ``max_records`` records are collected."""
        out: List[Dict[str, Any]] = []
        for record in self._paginate(path, per_page=per_page, **params):
            out.append(record)
            if len(out) >= max_records:
                break
        return out

    # ----- GraphQL helpers -----

    def _gql(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = {"query": query, "variables": variables or {}}
        try:
            response = self._session.post(self._graphql_url, json=body, timeout=30)
        except requests.exceptions.RequestException as exc:
            raise ToolException(f"Aha! GraphQL network error: {exc}") from exc

        if not response.ok:
            excerpt = (response.text or "")[:500]
            raise ToolException(
                f"Aha! GraphQL failed ({response.status_code}): {excerpt}"
            )

        payload = response.json()
        errors = payload.get("errors")
        if errors:
            raise ToolException(f"Aha! GraphQL errors: {errors}")
        return payload.get("data") or {}

    # ----- Output shaping -----

    @staticmethod
    def _project_record(record: Dict[str, Any], fields: Optional[List[str]]) -> Dict[str, Any]:
        if not fields or not isinstance(record, dict):
            return record
        return {k: record.get(k) for k in fields if k in record}

    @classmethod
    def _project_records(
        cls,
        records: List[Dict[str, Any]],
        fields: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        if not fields:
            return records
        return [cls._project_record(r, fields) for r in records]

    @staticmethod
    def _format_output(
        data: Any,
        output_format: Optional[str],
        *,
        empty_message: Optional[str] = None,
    ) -> Any:
        """Render ``data`` in the requested format.

        - ``json`` (default): return the Python object as-is; the LangChain
          tool layer serialises it to JSON.
        - ``csv`` / ``markdown``: render a single dict as one row or a
          list-of-dicts as multiple rows. Falls back to JSON when the shape
          does not match.
        - Empty collections return ``empty_message`` when one is supplied.
        """
        fmt = (output_format or "json").strip().lower()
        if fmt not in _OUTPUT_FORMATS:
            raise ToolException(
                f"Unsupported output_format '{output_format}'. "
                f"Use {_OUTPUT_FORMAT_VALUES}."
            )
        if data == [] and empty_message:
            return empty_message
        if fmt == "json":
            return data
        records = [data] if isinstance(data, dict) else data
        if (
            not isinstance(records, list)
            or not records
            or not all(isinstance(record, dict) for record in records)
        ):
            return data

        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ToolException(
                "Rendering CSV/markdown requires the `pandas` extra. "
                "Reinstall elitea-sdk with `pip install '.[tools]'`."
            ) from exc

        df = pd.DataFrame(records)
        if fmt == "csv":
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            return buf.getvalue()
        # markdown
        try:
            return df.to_markdown(index=False)
        except ImportError as exc:
            raise ToolException(
                "Rendering markdown requires the `tabulate` package. "
                "Install it or use `output_format='csv'`."
            ) from exc

    # ----- Reference validation -----

    @staticmethod
    def _validate_reference(value: str, pattern: re.Pattern, label: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ToolException(f"{label} reference is required")
        if not pattern.match(value):
            raise ToolException(
                f"'{value}' is not a valid Aha! {label} reference "
                f"(expected pattern: {pattern.pattern})"
            )
        return value

    # ----- REST reads -----

    def get_feature(
        self,
        reference_or_id: str,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """Get a feature by its reference number or numeric ID.

        Aha reference examples: ``DEVELOP-123``.
        """
        record = self._rest_get(f"features/{reference_or_id}").get("feature", {})
        return self._format_output(self._project_record(record, fields), output_format)

    def get_requirement(
        self,
        reference_or_id: str,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """Get a requirement by its reference number or numeric ID.

        Aha reference examples: ``ADT-123-1``.
        """
        record = self._rest_get(f"requirements/{reference_or_id}").get("requirement", {})
        return self._format_output(self._project_record(record, fields), output_format)

    def get_release(
        self,
        reference_or_id: str,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """Get a release by its reference number or numeric ID."""
        record = self._rest_get(f"releases/{reference_or_id}").get("release", {})
        return self._format_output(self._project_record(record, fields), output_format)

    def get_initiative(
        self,
        reference_or_id: str,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """Get an initiative by its reference number or numeric ID."""
        record = self._rest_get(f"initiatives/{reference_or_id}").get("initiative", {})
        return self._format_output(self._project_record(record, fields), output_format)

    def get_epic(
        self,
        reference_or_id: str,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """Get an epic by its reference number or numeric ID."""
        record = self._rest_get(f"epics/{reference_or_id}").get("epic", {})
        return self._format_output(self._project_record(record, fields), output_format)

    def get_idea(
        self,
        reference_or_id: str,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """Get an idea by its reference number or numeric ID."""
        record = self._rest_get(f"ideas/{reference_or_id}").get("idea", {})
        return self._format_output(self._project_record(record, fields), output_format)

    def get_product(
        self,
        reference_or_id: str,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """Get a product by its reference or numeric ID."""
        record = self._rest_get(f"products/{reference_or_id}").get("product", {})
        return self._format_output(self._project_record(record, fields), output_format)

    # ----- REST lists -----

    def list_products(
        self,
        updated_since: Optional[str] = None,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List Aha products, optionally filtered by updated_since (ISO-8601)."""
        records = self._collect(
            "products",
            per_page=per_page,
            max_records=max_records,
            updated_since=updated_since,
        )
        updated_detail = (
            f" updated since {updated_since.strip()!r}"
            if updated_since and updated_since.strip()
            else ""
        )
        return self._format_output(
            self._project_records(records, fields),
            output_format,
            empty_message=f"Aha! API returned no products{updated_detail}.",
        )

    def list_features(
        self,
        product_id: Optional[str] = None,
        release_id: Optional[str] = None,
        q: Optional[str] = None,
        updated_since: Optional[str] = None,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List features. Filter by product_id, release_id, free-text `q`, or updated_since.

        When ``release_id`` is supplied the nested endpoint
        ``releases/{release_id}/features`` is used; when ``product_id`` is
        supplied the endpoint is ``products/{product_id}/features``.
        Otherwise the global ``/features`` endpoint is queried.
        """
        if release_id:
            path = f"releases/{release_id}/features"
        elif product_id:
            path = f"products/{product_id}/features"
        else:
            path = "features"
        records = self._collect(
            path,
            per_page=per_page,
            max_records=max_records,
            q=q,
            updated_since=updated_since,
        )
        scope_detail = (
            f" for release {release_id.strip()!r}"
            if release_id and release_id.strip()
            else (
                f" for product {product_id.strip()!r}"
                if product_id and product_id.strip()
                else ""
            )
        )
        query_detail = f" matching query {q.strip()!r}" if q and q.strip() else ""
        updated_detail = (
            f" updated since {updated_since.strip()!r}"
            if updated_since and updated_since.strip()
            else ""
        )
        return self._format_output(
            self._project_records(records, fields),
            output_format,
            empty_message=(
                f"Aha! API returned no features{scope_detail}"
                f"{query_detail}{updated_detail}."
            ),
        )

    def list_requirements(
        self,
        feature_id: Optional[str] = None,
        q: Optional[str] = None,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List requirements owned by a feature reference/ID."""
        feature_ref = (feature_id or "").strip()
        if not feature_ref:
            raise ToolException("list_requirements: feature_id is required")
        path = f"features/{feature_ref}/requirements"
        records = self._collect(
            path,
            per_page=per_page,
            max_records=max_records,
            q=q,
        )
        query_detail = f" matching query {q.strip()!r}" if q and q.strip() else ""
        return self._format_output(
            self._project_records(records, fields),
            output_format,
            empty_message=(
                "Aha! API returned no requirements for feature "
                f"{feature_ref!r}{query_detail}."
            ),
        )

    def list_releases(
        self,
        product_id: Optional[str] = None,
        parking_lot: Optional[bool] = None,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List releases, optionally scoped to a product and/or filtered by parking_lot."""
        path = f"products/{product_id}/releases" if product_id else "releases"
        records = self._collect(
            path,
            per_page=per_page,
            max_records=max_records,
            parking_lot=parking_lot,
        )
        product_detail = (
            f" for product {product_id.strip()!r}"
            if product_id and product_id.strip()
            else ""
        )
        parking_lot_detail = (
            f" with parking_lot={str(parking_lot).lower()}"
            if parking_lot is not None
            else ""
        )
        return self._format_output(
            self._project_records(records, fields),
            output_format,
            empty_message=(
                f"Aha! API returned no releases{product_detail}"
                f"{parking_lot_detail}."
            ),
        )

    def list_initiatives(
        self,
        product_id: Optional[str] = None,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List initiatives, optionally scoped to a product."""
        path = f"products/{product_id}/initiatives" if product_id else "initiatives"
        records = self._collect(path, per_page=per_page, max_records=max_records)
        product_detail = (
            f" for product {product_id.strip()!r}"
            if product_id and product_id.strip()
            else ""
        )
        return self._format_output(
            self._project_records(records, fields),
            output_format,
            empty_message=f"Aha! API returned no initiatives{product_detail}.",
        )

    def list_epics(
        self,
        product_id: Optional[str] = None,
        release_id: Optional[str] = None,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List epics, optionally scoped to a product or release."""
        if release_id:
            path = f"releases/{release_id}/epics"
        elif product_id:
            path = f"products/{product_id}/epics"
        else:
            path = "epics"
        records = self._collect(path, per_page=per_page, max_records=max_records)
        scope_detail = (
            f" for release {release_id.strip()!r}"
            if release_id and release_id.strip()
            else (
                f" for product {product_id.strip()!r}"
                if product_id and product_id.strip()
                else ""
            )
        )
        return self._format_output(
            self._project_records(records, fields),
            output_format,
            empty_message=f"Aha! API returned no epics{scope_detail}.",
        )

    def list_ideas(
        self,
        product_id: Optional[str] = None,
        q: Optional[str] = None,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List ideas, optionally scoped to a product or filtered by free-text `q`."""
        path = f"products/{product_id}/ideas" if product_id else "ideas"
        records = self._collect(path, per_page=per_page, max_records=max_records, q=q)
        product_detail = (
            f" for product {product_id.strip()!r}"
            if product_id and product_id.strip()
            else ""
        )
        query_detail = (
            f" matching query {q.strip()!r}"
            if q and q.strip()
            else ""
        )
        return self._format_output(
            self._project_records(records, fields),
            output_format,
            empty_message=(
                f"Aha! API returned no ideas{product_detail}{query_detail}."
            ),
        )

    def search(
        self,
        q: str,
        type: Optional[str] = None,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """Full-text search across Aha records.

        Uses Aha's generic ``/api/v1/search`` endpoint. Optionally filter by
        record ``type`` (``feature``, ``requirement``, ``release``, ``idea``,
        ``epic``, etc.).
        """
        query = (q or "").strip()
        if not query:
            raise ToolException("search: query `q` is required")
        records = self._collect(
            "search",
            per_page=per_page,
            max_records=max_records,
            q=q,
            type=type,
        )
        record_type = (type or "").strip()
        record_label = f"{record_type} records" if record_type else "records"
        return self._format_output(
            self._project_records(records, fields),
            output_format,
            empty_message=(
                f"Aha! API returned no {record_label} for query {query!r}."
            ),
        )

    # ----- GraphQL reads -----

    def get_page(self, reference: str, include_parent: bool = False):
        """Fetch an Aha! page (note) by its reference number.

        Reference format: ``ABC-N-###`` (e.g. ``ABC-N-213``).
        """
        ref = self._validate_reference(reference, _PAGE_REF_RE, "page")
        data = self._gql(
            _QUERY_GET_PAGE,
            {"id": ref, "includeParent": bool(include_parent)},
        )
        return data.get("page") or {}

    def search_documents(self, query: str, searchable_type: Optional[str] = "Page"):
        """Search Aha! documents (default type: `Page`) via GraphQL.

        Returns ``[ { name, url, searchableId, searchableType }, ... ]``.
        """
        if not (query or "").strip():
            raise ToolException("search_documents: query is required")
        data = self._gql(
            _QUERY_SEARCH_DOCUMENTS,
            {"query": query, "searchableType": [searchable_type or "Page"]},
        )
        return (data.get("searchDocuments") or {}).get("nodes") or []

    def get_feature_gql(self, reference: str):
        """Fetch a feature via GraphQL — description is returned as markdown.

        Prefer this over the REST ``get_feature`` when you need the markdown
        body (REST returns HTML in ``description.body``).
        Reference format: ``DEVELOP-123``.
        """
        ref = self._validate_reference(reference, _FEATURE_REF_RE, "feature")
        data = self._gql(_QUERY_GET_FEATURE, {"id": ref})
        return data.get("feature") or {}

    def get_requirement_gql(self, reference: str):
        """Fetch a requirement via GraphQL — description is returned as markdown.

        Reference format: ``ADT-123-1``.
        """
        ref = self._validate_reference(reference, _REQUIREMENT_REF_RE, "requirement")
        data = self._gql(_QUERY_GET_REQUIREMENT, {"id": ref})
        return data.get("requirement") or {}

    # ----- Resource-type helpers -----

    @classmethod
    def _normalize_resource_type(cls, resource_type: str) -> str:
        key = (resource_type or "").strip().lower()
        return _COMMENT_RESOURCE_ALIASES.get(key, key)

    @classmethod
    def _resource_plural(cls, resource_type: str) -> str:
        key = cls._normalize_resource_type(resource_type)
        plural = _RESOURCE_PLURAL.get(key)
        if not plural:
            raise ToolException(
                f"Unsupported Aha resource type '{resource_type}'. "
                f"Accepted: {_COMMENTABLE_RESOURCE_VALUES}"
            )
        return plural

    @classmethod
    def _comment_resource(cls, resource_type: str) -> tuple[str, str]:
        canonical = cls._normalize_resource_type(resource_type)
        if canonical not in _COMMENTABLE_RESOURCE_TYPES:
            raise ToolException(
                f"Unsupported Aha comment resource type '{resource_type}'. "
                f"Accepted: {_COMMENTABLE_RESOURCE_VALUES}"
            )
        return canonical, _RESOURCE_PLURAL[canonical]

    # ----- Comments -----

    def add_comment(self, resource_type: str, resource_id: str, body: str):
        """Post a comment on an Aha record.

        Supports the record types Aha exposes comments for (features,
        requirements, ideas, releases, release phases, epics, initiatives,
        goals, pages, and to-dos). Use ``resource_type='to_do'`` for a to-do.
        """
        _, plural = self._comment_resource(resource_type)
        if not (body or "").strip():
            raise ToolException("add_comment: comment body is required")
        payload = {"comment": {"body": body}}
        response = self._rest_post(f"{plural}/{resource_id}/comments", json=payload)
        return response.get("comment") or response

    def list_comments(
        self,
        resource_type: str,
        resource_id: str,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List comments on an Aha record (paginated)."""
        canonical, plural = self._comment_resource(resource_type)
        records = self._collect(
            f"{plural}/{resource_id}/comments",
            per_page=per_page,
            max_records=max_records,
        )
        return self._format_output(
            self._project_records(records, fields),
            output_format,
            empty_message=(
                "Aha! API returned no comments for "
                f"{canonical} {resource_id.strip()!r}."
            ),
        )

    # ----- record create / update / delete -----

    # Parent scope for ``manage_record(action='create', ...)``. The path
    # segment before the plural is the parent resource; the value comes from
    # ``parent_id`` (or ``record_id`` as a legacy alias).
    _CREATE_PARENT_PATH: Dict[str, str] = {
        "feature": "releases",
        "requirement": "features",
        "idea": "products",
        "release": "products",
        "initiative": "products",
        "epic": "releases",
        "page": "products",
    }
    _SCOPED_MUTATION_PATH: Dict[str, str] = {
        "release": "products",
        "initiative": "products",
    }

    def create_record(
        self,
        record_type: str,
        parent_id: str,
        properties: Dict[str, Any],
    ):
        """Create an Aha record under its required parent scope.

        Parent scoping is release for features and epics, feature for
        requirements, and product for ideas, releases, initiatives, and pages.
        ``properties`` must contain the fields required by Aha for the selected
        record type, usually at least ``name``.
        """
        return self.manage_record(
            action="create",
            record_type=record_type,
            parent_id=parent_id,
            properties=properties,
        )

    def update_record(
        self,
        record_type: str,
        record_id: str,
        properties: Dict[str, Any],
        parent_id: Optional[str] = None,
    ):
        """Update fields on an existing Aha record.

        ``parent_id`` is the product reference/ID and is required for releases
        and initiatives because Aha scopes their mutation endpoints by product.
        It is not used for other record types.
        """
        return self.manage_record(
            action="update",
            record_type=record_type,
            record_id=record_id,
            parent_id=parent_id,
            properties=properties,
        )

    def delete_record(
        self,
        record_type: str,
        record_id: str,
        parent_id: Optional[str] = None,
    ):
        """Delete an existing Aha record.

        ``parent_id`` is the product reference/ID and is required for releases
        and initiatives because Aha scopes their mutation endpoints by product.
        It is not used for other record types. Returns a confirmation object
        after Aha accepts the deletion.
        """
        return self.manage_record(
            action="delete",
            record_type=record_type,
            record_id=record_id,
            parent_id=parent_id,
        )

    def manage_record(
        self,
        action: str,
        record_type: str,
        record_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """Legacy combined interface to create, update, or delete an Aha record.

        Prefer ``create_record``, ``update_record``, or ``delete_record`` so
        each operation can be enabled independently and only relevant inputs
        are exposed.

        - ``action='create'`` inserts under a parent scope. Parent scoping:
          feature → release, requirement → feature, idea → product, release →
          product, initiative → product, epic → release, page → product. The
          parent reference may be passed as ``parent_id`` (preferred) or as
          ``record_id`` (legacy alias).
        - ``action='update'`` and ``action='delete'`` require ``record_id``.
          Releases and initiatives also require their product reference in
          ``parent_id`` because those Aha endpoints are product-scoped.

        ``properties`` is a dict of Aha field values (e.g. ``{"name": "…",
        "description": "…"}``). See Aha REST docs for field names.
        """
        act = (action or "").strip().lower()
        rt = (record_type or "").strip().lower()
        props = dict(properties or {})

        if rt not in _MANAGEABLE_RECORD_TYPES:
            raise ToolException(
                f"manage_record does not support record_type '{record_type}'. "
                f"Accepted: {_MANAGEABLE_RECORD_VALUES}"
            )
        if act not in _MANAGE_ACTIONS:
            raise ToolException(
                f"manage_record: action must be {_MANAGE_ACTION_VALUES}"
            )

        plural = _RESOURCE_PLURAL[rt]
        singular = rt

        if act == "update":
            if not record_id:
                raise ToolException("manage_record update: record_id is required")
            path = self._mutation_path(act, rt, record_id, parent_id)
            response = self._rest_put(path, json={singular: props})
            return response.get(singular) or response

        if act == "delete":
            if not record_id:
                raise ToolException("manage_record delete: record_id is required")
            path = self._mutation_path(act, rt, record_id, parent_id)
            response = self._rest_delete(path)
            # Aha returns 204/empty on delete; surface a consistent shape.
            return {"deleted": True, "record_type": rt, "record_id": record_id, **(response or {})}

        # create
        scope = parent_id or record_id
        if not scope:
            parent_kind = self._CREATE_PARENT_PATH[rt].rstrip("s")
            raise ToolException(
                f"manage_record create {rt}: parent_id is required "
                f"({parent_kind} ref)"
            )
        parent_plural = self._CREATE_PARENT_PATH[rt]
        path = f"{parent_plural}/{scope}/{plural}"

        response = self._rest_post(path, json={singular: props})
        return response.get(singular) or response

    def _mutation_path(
        self,
        action: str,
        record_type: str,
        record_id: str,
        parent_id: Optional[str],
    ) -> str:
        parent_plural = self._SCOPED_MUTATION_PATH.get(record_type)
        plural = _RESOURCE_PLURAL[record_type]
        if not parent_plural:
            return f"{plural}/{record_id}"
        if not parent_id:
            raise ToolException(
                f"manage_record {action} {record_type}: parent_id is required "
                "(product reference/ID) because Aha scopes this endpoint by product"
            )
        return f"{parent_plural}/{parent_id}/{plural}/{record_id}"

    # ----- Record links -----

    def create_record_link(
        self,
        from_record_type: str,
        from_id: str,
        to_record_type: str,
        to_id: str,
        link_type: int,
    ):
        """Create a link between two Aha records.

        Resolves reference numbers to Aha's internal numeric IDs, then calls
        ``POST /{source_type}/{source_id}/record_links`` with Aha's documented
        ``record_type``, ``record_id``, and numeric ``link_type`` fields.
        """
        source = (from_record_type or "").strip().lower()
        target = (to_record_type or "").strip().lower()
        if source not in _RECORD_LINK_SOURCE_TYPES:
            raise ToolException(
                f"create_record_link: unsupported from_record_type "
                f"'{from_record_type}'. Accepted: {_RECORD_LINK_SOURCE_VALUES}"
            )
        if target not in _RECORD_LINK_TARGET_TYPES:
            raise ToolException(
                f"create_record_link: unsupported to_record_type "
                f"'{to_record_type}'. Accepted: {_RECORD_LINK_TARGET_VALUES}"
            )
        if isinstance(link_type, bool):
            link_code = -1
        else:
            try:
                link_code = int(link_type)
            except (TypeError, ValueError):
                link_code = -1
        if link_code not in _RECORD_LINK_TYPE_LABELS:
            raise ToolException(
                f"create_record_link: unsupported link_type '{link_type}'. "
                f"Accepted: {_RECORD_LINK_TYPE_VALUES}"
            )

        source_input = (from_id or "").strip()
        target_input = (to_id or "").strip()
        source_id = self._resolve_record_link_id(
            source,
            source_input,
            role="source",
        )
        target_id = self._resolve_record_link_id(
            target,
            target_input,
            role="target",
        )
        link: Dict[str, Any] = {
            "record_link": {
                "record_type": target,
                "record_id": int(target_id),
                "link_type": link_code,
            }
        }
        response = self._rest_post(
            f"{_RESOURCE_PLURAL[source]}/{source_id}/record_links",
            json=link,
        )
        return response.get("record_link") or response or {
            "created": True,
            "from_record_type": source,
            "from_reference_or_id": source_input,
            "from_record_id": source_id,
            "to_record_type": target,
            "to_reference_or_id": target_input,
            "to_record_id": target_id,
            "link_type": link_code,
            "link_type_name": _RECORD_LINK_TYPE_LABELS[link_code],
        }

    def _resolve_record_link_id(
        self,
        record_type: str,
        reference_or_id: str,
        *,
        role: str,
    ) -> str:
        """Resolve a record-link input to the numeric ID required by Aha."""
        if not reference_or_id:
            raise ToolException(
                f"create_record_link: {role} {record_type} reference or "
                "numeric ID is required"
            )
        if reference_or_id.isdigit():
            return reference_or_id
        if record_type in {"goal", "initiative"}:
            plural = _RESOURCE_PLURAL[record_type]
            for record in self._paginate(plural, per_page=100):
                reference_num = str(record.get("reference_num", "")).strip()
                if reference_num.casefold() != reference_or_id.casefold():
                    continue
                resolved_id = str(record.get("id", "")).strip()
                if resolved_id.isdigit():
                    return resolved_id
                break
            raise ToolException(
                f"create_record_link: Aha! returned no {record_type} with reference "
                f"{reference_or_id!r} and an internal numeric ID"
            )
        if record_type == "page":
            page_ref = self._validate_reference(
                reference_or_id,
                _PAGE_REF_RE,
                "page",
            )
            page = self._gql(
                _QUERY_GET_PAGE,
                {"id": page_ref, "includeParent": False},
            ).get("page")
            resolved_id = (
                str(page.get("id", "")).strip()
                if isinstance(page, dict)
                else ""
            )
            if resolved_id.isdigit():
                return resolved_id
            raise ToolException(
                f"create_record_link: Aha! returned no internal numeric ID for "
                f"{role} page {reference_or_id!r}"
            )
        if record_type == "release_phase":
            raise ToolException(
                f"create_record_link: {role} {record_type} requires a numeric "
                f"ID; received {reference_or_id!r}"
            )

        payload = self._rest_get(
            f"{_RESOURCE_PLURAL[record_type]}/{reference_or_id}"
        )
        record = payload.get(record_type)
        resolved_id = (
            str(record.get("id", "")).strip()
            if isinstance(record, dict)
            else ""
        )
        if not resolved_id.isdigit():
            raise ToolException(
                f"create_record_link: Aha! returned no internal numeric ID for "
                f"{role} {record_type} {reference_or_id!r}"
            )
        return resolved_id

    # ----- Copy / duplicate -----

    def copy_record(self, record_type: str, record_id: str):
        """Duplicate an Aha record.

        Only ``record_type='release'`` is supported by Aha's REST API
        (``POST /releases/{id}/duplicate``). Other types raise ToolException
        so callers can surface the limitation clearly.
        """
        rt = (record_type or "").strip().lower()
        if rt != "release":
            raise ToolException(
                "copy_record: Aha REST only supports duplicating releases. "
                "For other record types use `manage_record(action='create', ...)` "
                "with the fields you want to copy."
            )
        response = self._rest_post(f"releases/{record_id}/duplicate")
        return response.get("release") or response

    # ----- Custom fields metadata -----

    def fields_metadata(
        self,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List all custom-field definitions in the Aha account.

        Aha exposes this as an account-level collection. No record ID or
        workspace scope is required.
        """
        payload = self._rest_get("custom_field_definitions")
        records = payload.get("custom_field_definitions") or []
        return self._format_output(
            self._project_records(records, fields),
            output_format,
            empty_message="Aha! API returned no custom-field definitions.",
        )

    def field_options_metadata(
        self,
        field_id: str,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List option metadata for an Aha custom-field definition."""
        definition_id = str(field_id or "").strip()
        if not definition_id:
            raise ToolException("field_options_metadata: field_id is required")
        if not definition_id.isdigit():
            raise ToolException(
                "field_options_metadata: field_id must be the numeric custom-field "
                f"definition ID returned by fields_metadata; received {field_id!r}"
            )
        payload = self._rest_get(
            f"custom_field_definitions/{definition_id}/options"
        )
        records = (
            payload.get("options")
            or payload.get("custom_field_options")
            or []
        )
        return self._format_output(
            self._project_records(records, fields),
            output_format,
            empty_message=(
                "Aha! API returned no options for custom-field definition "
                f"'{definition_id}'."
            ),
        )

    # ----- Attachments -----

    def attach_file(
        self,
        resource_type: str,
        resource_id: str,
        filepath: str,
        filename: Optional[str] = None,
    ):
        """Upload an artifact to an Aha record description or to-do.

        ``filepath`` must be an artifact-storage path in
        ``/{bucket}/{filename}`` format. For records, the tool first resolves
        the description note ID required by Aha's attachments API.
        """
        rt = self._normalize_resource_type(resource_type)
        plural = self._resource_plural(rt)
        if not (resource_id or "").strip():
            raise ToolException("attach_file: resource_id is required")
        if not (filepath or "").strip():
            raise ToolException("attach_file: filepath is required")

        try:
            content, artifact_filename = get_file_bytes_from_artifact(
                self.elitea,
                filepath,
            )
        except Exception as exc:
            raise ToolException(
                f"attach_file: failed to retrieve artifact '{filepath}': {exc}"
            ) from exc

        if not content:
            raise ToolException(
                f"attach_file: artifact '{filepath}' was not found or is empty"
            )
        resolved_name = filename or artifact_filename
        if not resolved_name:
            raise ToolException(
                "attach_file: filename could not be resolved from the artifact"
            )

        if rt == "to_do":
            attachment_path = f"tasks/{resource_id}/attachments"
        else:
            record_payload = self._rest_get(f"{plural}/{resource_id}")
            record = record_payload.get(rt) or record_payload
            description = record.get("description") if isinstance(record, dict) else None
            note_id = description.get("id") if isinstance(description, dict) else None
            if not note_id:
                raise ToolException(
                    f"attach_file: Aha {rt} '{resource_id}' response does not "
                    "contain description.id required for attachment upload"
                )
            attachment_path = f"notes/{note_id}/attachments"

        # Aha's attachments API requires the ``attachment[data]`` form field.
        files = {"attachment[data]": (resolved_name, content)}
        # ``requests`` will set the correct multipart Content-Type header
        # automatically when ``files`` is provided; strip the JSON default.
        headers = {"Content-Type": None}
        try:
            response = self._session.post(
                f"{self._rest_url}/{attachment_path}",
                files=files,
                headers=headers,
                timeout=60,
            )
        except requests.exceptions.RequestException as exc:
            raise ToolException(f"Aha! attachment upload network error: {exc}") from exc

        if not response.ok:
            excerpt = (response.text or "")[:500]
            raise ToolException(
                f"Aha! attachment upload failed ({response.status_code}): {excerpt}"
            )
        try:
            payload = response.json() if response.content else {}
        except ValueError:
            payload = {}
        return payload.get("attachment") or payload

    # ----- Dispatchers matching Aha remote MCP tool names -----

    def find_project(
        self,
        q: Optional[str] = None,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """Find Aha products (workspaces).

        Thin wrapper around ``list_products``: matches Aha's remote-MCP tool
        name so callers can discover the correct workspace/product reference
        before running searches.
        """
        records = self._collect(
            "products",
            per_page=per_page,
            max_records=max_records,
            q=q,
        )
        return self._format_output(self._project_records(records, fields), output_format)

    def search_records(
        self,
        record_type: str,
        q: Optional[str] = None,
        feature_id: Optional[str] = None,
        product_id: Optional[str] = None,
        release_id: Optional[str] = None,
        updated_since: Optional[str] = None,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """Search Aha records of a given type.

        Dispatches to the appropriate list endpoint (``list_features``,
        ``list_requirements``, etc.) so agents can express searches uniformly
        without knowing the specific tool name.
        """
        rt = (record_type or "").strip().lower()
        common = {
            "per_page": per_page,
            "max_records": max_records,
            "output_format": output_format,
            "fields": fields,
        }
        if rt == "feature":
            return self.list_features(
                product_id=product_id,
                release_id=release_id,
                q=q,
                updated_since=updated_since,
                **common,
            )
        if rt == "requirement":
            return self.list_requirements(feature_id=feature_id, q=q, **common)
        if rt == "release":
            return self.list_releases(product_id=product_id, **common)
        if rt == "idea":
            return self.list_ideas(product_id=product_id, q=q, **common)
        if rt == "epic":
            return self.list_epics(product_id=product_id, release_id=release_id, **common)
        if rt == "initiative":
            return self.list_initiatives(product_id=product_id, **common)
        if rt == "product":
            return self.list_products(updated_since=updated_since, **common)
        raise ToolException(
            f"search_records: unsupported record_type '{record_type}'. "
            f"Accepted: {_SEARCHABLE_RECORD_VALUES}."
        )

    def read_records(
        self,
        record_type: str,
        reference_or_id: str,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """Read a single Aha record by type + reference/ID.

        Dispatches to the appropriate get endpoint. For ``record_type='page'``
        uses the GraphQL page endpoint.
        """
        rt = (record_type or "").strip().lower()
        if rt == "feature":
            return self.get_feature(reference_or_id, output_format, fields)
        if rt == "requirement":
            return self.get_requirement(reference_or_id, output_format, fields)
        if rt == "release":
            return self.get_release(reference_or_id, output_format, fields)
        if rt == "initiative":
            return self.get_initiative(reference_or_id, output_format, fields)
        if rt == "epic":
            return self.get_epic(reference_or_id, output_format, fields)
        if rt == "idea":
            return self.get_idea(reference_or_id, output_format, fields)
        if rt == "product":
            return self.get_product(reference_or_id, output_format, fields)
        if rt == "page":
            return self.get_page(reference_or_id)
        raise ToolException(
            f"read_records: unsupported record_type '{record_type}'. "
            f"Accepted: {_READABLE_RECORD_VALUES}."
        )

    # ----- Tool registry -----

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return the list of tools this wrapper exposes."""
        return [
            # REST reads
            {
                "name": "get_feature",
                "description": self.get_feature.__doc__,
                "args_schema": AhaFeatureReferenceInput,
                "ref": self.get_feature,
            },
            {
                "name": "get_requirement",
                "description": self.get_requirement.__doc__,
                "args_schema": AhaRequirementReferenceInput,
                "ref": self.get_requirement,
            },
            {
                "name": "get_release",
                "description": self.get_release.__doc__,
                "args_schema": AhaReleaseReferenceInput,
                "ref": self.get_release,
            },
            {
                "name": "get_initiative",
                "description": self.get_initiative.__doc__,
                "args_schema": AhaInitiativeReferenceInput,
                "ref": self.get_initiative,
            },
            {
                "name": "get_epic",
                "description": self.get_epic.__doc__,
                "args_schema": AhaEpicReferenceInput,
                "ref": self.get_epic,
            },
            {
                "name": "get_idea",
                "description": self.get_idea.__doc__,
                "args_schema": AhaIdeaReferenceInput,
                "ref": self.get_idea,
            },
            {
                "name": "get_product",
                "description": self.get_product.__doc__,
                "args_schema": AhaProductReferenceInput,
                "ref": self.get_product,
            },
            # REST lists
            {
                "name": "list_products",
                "description": self.list_products.__doc__,
                "args_schema": AhaListProductsInput,
                "ref": self.list_products,
            },
            {
                "name": "list_features",
                "description": self.list_features.__doc__,
                "args_schema": AhaListFeaturesInput,
                "ref": self.list_features,
            },
            {
                "name": "list_requirements",
                "description": self.list_requirements.__doc__,
                "args_schema": AhaListRequirementsInput,
                "ref": self.list_requirements,
            },
            {
                "name": "list_releases",
                "description": self.list_releases.__doc__,
                "args_schema": AhaListReleasesInput,
                "ref": self.list_releases,
            },
            {
                "name": "list_initiatives",
                "description": self.list_initiatives.__doc__,
                "args_schema": AhaListInitiativesInput,
                "ref": self.list_initiatives,
            },
            {
                "name": "list_epics",
                "description": self.list_epics.__doc__,
                "args_schema": AhaListEpicsInput,
                "ref": self.list_epics,
            },
            {
                "name": "list_ideas",
                "description": self.list_ideas.__doc__,
                "args_schema": AhaListIdeasInput,
                "ref": self.list_ideas,
            },
            {
                "name": "search",
                "description": self.search.__doc__,
                "args_schema": AhaSearchInput,
                "ref": self.search,
            },
            # GraphQL reads
            {
                "name": "get_page",
                "description": self.get_page.__doc__,
                "args_schema": AhaGetPageInput,
                "ref": self.get_page,
            },
            {
                "name": "search_documents",
                "description": self.search_documents.__doc__,
                "args_schema": AhaSearchDocumentsInput,
                "ref": self.search_documents,
            },
            {
                "name": "get_feature_gql",
                "description": self.get_feature_gql.__doc__,
                "args_schema": AhaGetFeatureGqlInput,
                "ref": self.get_feature_gql,
            },
            {
                "name": "get_requirement_gql",
                "description": self.get_requirement_gql.__doc__,
                "args_schema": AhaGetRequirementGqlInput,
                "ref": self.get_requirement_gql,
            },
            # Dispatchers (match Aha remote MCP tool names)
            {
                "name": "find_project",
                "description": self.find_project.__doc__,
                "args_schema": AhaFindProjectInput,
                "ref": self.find_project,
            },
            {
                "name": "search_records",
                "description": self.search_records.__doc__,
                "args_schema": AhaSearchRecordsInput,
                "ref": self.search_records,
            },
            {
                "name": "read_records",
                "description": self.read_records.__doc__,
                "args_schema": AhaReadRecordsInput,
                "ref": self.read_records,
            },
            # Writes
            {
                "name": "add_comment",
                "description": self.add_comment.__doc__,
                "args_schema": AhaAddCommentInput,
                "ref": self.add_comment,
            },
            {
                "name": "list_comments",
                "description": self.list_comments.__doc__,
                "args_schema": AhaListCommentsInput,
                "ref": self.list_comments,
            },
            {
                "name": "manage_record",
                "description": self.manage_record.__doc__,
                "args_schema": AhaManageRecordInput,
                "ref": self.manage_record,
            },
            {
                "name": "create_record",
                "description": self.create_record.__doc__,
                "args_schema": AhaCreateRecordInput,
                "ref": self.create_record,
            },
            {
                "name": "update_record",
                "description": self.update_record.__doc__,
                "args_schema": AhaUpdateRecordInput,
                "ref": self.update_record,
            },
            {
                "name": "delete_record",
                "description": self.delete_record.__doc__,
                "args_schema": AhaDeleteRecordInput,
                "ref": self.delete_record,
            },
            {
                "name": "create_record_link",
                "description": self.create_record_link.__doc__,
                "args_schema": AhaCreateRecordLinkInput,
                "ref": self.create_record_link,
            },
            {
                "name": "copy_record",
                "description": self.copy_record.__doc__,
                "args_schema": AhaCopyRecordInput,
                "ref": self.copy_record,
            },
            {
                "name": "fields_metadata",
                "description": self.fields_metadata.__doc__,
                "args_schema": AhaFieldsMetadataInput,
                "ref": self.fields_metadata,
            },
            {
                "name": "field_options_metadata",
                "description": self.field_options_metadata.__doc__,
                "args_schema": AhaFieldOptionsInput,
                "ref": self.field_options_metadata,
            },
            {
                "name": "attach_file",
                "description": self.attach_file.__doc__,
                "args_schema": AhaAttachFileInput,
                "ref": self.attach_file,
            },
        ]
