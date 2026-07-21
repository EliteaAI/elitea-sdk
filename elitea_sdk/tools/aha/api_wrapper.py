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

import io
import logging
import re
from typing import Any, Dict, Iterator, List, Optional

import requests
from langchain_core.tools import ToolException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, create_model, model_validator
from pydantic.fields import PrivateAttr

from ..elitea_base import BaseToolApiWrapper

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
    "epic": "epics",
    "initiative": "initiatives",
    "product": "products",
    "goal": "goals",
    "page": "pages",
    "to_do": "to_dos",
    "todo": "to_dos",
}

# Record types that ``manage_record`` can create/update/delete via REST.
# Keep the whitelist tight so the tool cannot silently hit unsupported
# endpoints. Pages are Aha's "notes" resource and expose full CRUD via REST.
_MANAGEABLE_RECORD_TYPES = {
    "feature",
    "requirement",
    "idea",
    "release",
    "initiative",
    "epic",
    "page",
}

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
        description="Response format: `json` (default), `csv`, or `markdown`.",
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

AhaReferenceInput = create_model(
    "AhaReferenceInput",
    reference_or_id=(
        str,
        Field(
            description=(
                "Aha reference number (e.g. `DEVELOP-123`) or numeric record ID."
            ),
        ),
    ),
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
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
    feature_id=(Optional[str], Field(default=None, description="Filter by feature reference/ID.")),
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

AhaAddCommentInput = create_model(
    "AhaAddCommentInput",
    resource_type=(
        str,
        Field(
            description=(
                "Aha resource type: `feature`, `requirement`, `idea`, `release`, "
                "`epic`, `initiative`, `goal`, or `to_do`."
            ),
        ),
    ),
    resource_id=(str, Field(description="Aha reference number or numeric ID of the target record.")),
    body=(str, Field(description="Comment body (HTML or plain text).")),
)

AhaListCommentsInput = create_model(
    "AhaListCommentsInput",
    resource_type=(
        str,
        Field(description="Aha resource type (see `add_comment` for the accepted values)."),
    ),
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
            description=(
                "Record type. Accepted: `feature`, `requirement`, `idea`, "
                "`release`, `initiative`, `epic`, `page`."
            )
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
        Dict[str, Any],
        Field(
            default_factory=dict,
            description=(
                "Field/value map to set on the record. See Aha REST docs for the "
                "specific fields accepted by each record type."
            ),
        ),
    ),
)

AhaCreateRecordLinkInput = create_model(
    "AhaCreateRecordLinkInput",
    from_record_type=(
        str,
        Field(description="Source record type — currently only `feature` is supported by Aha."),
    ),
    from_id=(str, Field(description="Source record reference or numeric ID.")),
    to_record_type=(
        str,
        Field(description="Target record type: e.g. `feature`, `requirement`, `idea`."),
    ),
    to_id=(str, Field(description="Target record reference or numeric ID.")),
    link_type=(
        Optional[str],
        Field(
            default=None,
            description=(
                "Optional Aha link type — see Aha's record-link vocabulary "
                "(e.g. `blocks`, `blocked_by`, `duplicate`, `related_to`)."
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
    per_page=PER_PAGE_FIELD,
    max_records=MAX_RECORDS_FIELD,
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaFieldOptionsInput = create_model(
    "AhaFieldOptionsInput",
    field_id=(str, Field(description="Custom-field ID (numeric or slug).")),
    output_format=OUTPUT_FORMAT_FIELD,
    fields=FIELDS_FIELD,
)

AhaAttachFileInput = create_model(
    "AhaAttachFileInput",
    resource_type=(
        str,
        Field(description="Aha resource type (see `add_comment` for the accepted values)."),
    ),
    resource_id=(str, Field(description="Aha reference number or numeric ID of the target record.")),
    filepath=(
        str,
        Field(
            description=(
                "Local file path to upload. Use ``artifact://<bucket>/<name>`` "
                "for artifacts stored via the SDK artifact interface."
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
            description=(
                "Record type to search: `feature`, `requirement`, `release`, "
                "`idea`, `epic`, `initiative`, or `product`."
            ),
        ),
    ),
    q=(Optional[str], Field(default=None, description="Free-text search filter.")),
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
            description=(
                "Record type: `feature`, `requirement`, `release`, `idea`, `epic`, "
                "`initiative`, `product`, or `page`."
            ),
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
            body = (response.text or "")[:500]
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
    ) -> Any:
        """Render ``data`` in the requested format.

        - ``json`` (default): return the Python object as-is; the LangChain
          tool layer serialises it to JSON.
        - ``csv`` / ``markdown``: only meaningful for a list-of-dicts. Falls
          back to JSON when the shape does not match.
        """
        fmt = (output_format or "json").strip().lower()
        if fmt == "json":
            return data
        if fmt not in {"csv", "markdown"}:
            raise ToolException(
                f"Unsupported output_format '{output_format}'. "
                "Use 'json', 'csv', or 'markdown'."
            )
        if not isinstance(data, list) or not data or not all(isinstance(r, dict) for r in data):
            return data

        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ToolException(
                "Rendering CSV/markdown requires the `pandas` extra. "
                "Reinstall elitea-sdk with `pip install '.[tools]'`."
            ) from exc

        df = pd.DataFrame(data)
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
                "Install it or use output_format='csv'."
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
        return self._format_output(self._project_records(records, fields), output_format)

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
        return self._format_output(self._project_records(records, fields), output_format)

    def list_requirements(
        self,
        feature_id: Optional[str] = None,
        q: Optional[str] = None,
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List requirements, optionally scoped to a feature by reference/ID."""
        path = f"features/{feature_id}/requirements" if feature_id else "requirements"
        records = self._collect(
            path,
            per_page=per_page,
            max_records=max_records,
            q=q,
        )
        return self._format_output(self._project_records(records, fields), output_format)

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
        return self._format_output(self._project_records(records, fields), output_format)

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
        return self._format_output(self._project_records(records, fields), output_format)

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
        return self._format_output(self._project_records(records, fields), output_format)

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
        return self._format_output(self._project_records(records, fields), output_format)

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
        if not (q or "").strip():
            raise ToolException("search: query `q` is required")
        records = self._collect(
            "search",
            per_page=per_page,
            max_records=max_records,
            q=q,
            type=type,
        )
        return self._format_output(self._project_records(records, fields), output_format)

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
    def _resource_plural(cls, resource_type: str) -> str:
        key = (resource_type or "").strip().lower()
        plural = _RESOURCE_PLURAL.get(key)
        if not plural:
            raise ToolException(
                f"Unsupported Aha resource type '{resource_type}'. "
                f"Accepted: {', '.join(sorted(set(_RESOURCE_PLURAL))) }"
            )
        return plural

    # ----- Comments -----

    def add_comment(self, resource_type: str, resource_id: str, body: str):
        """Post a comment on an Aha record.

        Supports the record types Aha exposes comments for (features,
        requirements, ideas, releases, epics, initiatives, goals, to-dos).
        """
        plural = self._resource_plural(resource_type)
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
        plural = self._resource_plural(resource_type)
        records = self._collect(
            f"{plural}/{resource_id}/comments",
            per_page=per_page,
            max_records=max_records,
        )
        return self._format_output(self._project_records(records, fields), output_format)

    # ----- manage_record: create / update dispatcher -----

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

    def manage_record(
        self,
        action: str,
        record_type: str,
        record_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """Create, update, or delete an Aha record.

        - ``action='create'`` inserts under a parent scope. Parent scoping:
          feature → release, requirement → feature, idea → product, release →
          product, initiative → product, epic → release, page → product. The
          parent reference may be passed as ``parent_id`` (preferred) or as
          ``record_id`` (legacy alias).
        - ``action='update'`` requires ``record_id`` (the record reference to
          modify).
        - ``action='delete'`` requires ``record_id``. Returns ``{"deleted":
          True, ...}`` on success — Aha's DELETE endpoints typically respond
          with an empty body.

        ``properties`` is a dict of Aha field values (e.g. ``{"name": "…",
        "description": "…"}``). See Aha REST docs for field names.
        """
        act = (action or "").strip().lower()
        rt = (record_type or "").strip().lower()
        props = dict(properties or {})

        if rt not in _MANAGEABLE_RECORD_TYPES:
            raise ToolException(
                f"manage_record does not support record_type '{record_type}'. "
                f"Accepted: {', '.join(sorted(_MANAGEABLE_RECORD_TYPES))}"
            )
        if act not in {"create", "update", "delete"}:
            raise ToolException(
                "manage_record: action must be 'create', 'update', or 'delete'"
            )

        plural = _RESOURCE_PLURAL[rt]
        singular = rt

        if act == "update":
            if not record_id:
                raise ToolException("manage_record update: record_id is required")
            response = self._rest_put(f"{plural}/{record_id}", json={singular: props})
            return response.get(singular) or response

        if act == "delete":
            if not record_id:
                raise ToolException("manage_record delete: record_id is required")
            response = self._rest_delete(f"{plural}/{record_id}")
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

    # ----- Record links -----

    def create_record_link(
        self,
        from_record_type: str,
        from_id: str,
        to_record_type: str,
        to_id: str,
        link_type: Optional[str] = None,
    ):
        """Create a link between two Aha records.

        Aha REST currently exposes record-link creation from features only
        (``POST /features/{id}/record_links``). ``to_record_type`` is passed
        through as-is so Aha can resolve it (feature, requirement, idea, …).
        """
        source = (from_record_type or "").strip().lower()
        if source != "feature":
            raise ToolException(
                "create_record_link: Aha REST only supports links originating "
                "from features (from_record_type='feature')"
            )
        target = (to_record_type or "").strip().lower()
        if target not in _RESOURCE_PLURAL:
            raise ToolException(
                f"create_record_link: unsupported to_record_type '{to_record_type}'"
            )
        link: Dict[str, Any] = {
            "record_link": {
                "linkable_type": target.capitalize(),
                "linkable_id": to_id,
            }
        }
        if link_type:
            link["record_link"]["link_type"] = link_type
        response = self._rest_post(f"features/{from_id}/record_links", json=link)
        return response.get("record_link") or response

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
        per_page: int = 25,
        max_records: int = 100,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List custom-field definitions defined in the Aha account."""
        records = self._collect(
            "custom_fields",
            per_page=per_page,
            max_records=max_records,
        )
        return self._format_output(self._project_records(records, fields), output_format)

    def field_options_metadata(
        self,
        field_id: str,
        output_format: Optional[str] = "json",
        fields: Optional[List[str]] = None,
    ):
        """List the option values defined for an Aha custom field."""
        if not (field_id or "").strip():
            raise ToolException("field_options_metadata: field_id is required")
        payload = self._rest_get(f"custom_fields/{field_id}/options")
        records = payload.get("options") or payload.get("custom_field_options") or []
        return self._format_output(self._project_records(records, fields), output_format)

    # ----- Attachments -----

    def attach_file(
        self,
        resource_type: str,
        resource_id: str,
        filepath: str,
        filename: Optional[str] = None,
    ):
        """Upload an attachment to an Aha record.

        ``filepath`` may be a local path or an ``artifact://bucket/name`` URI
        resolved via the SDK's artifact interface.
        """
        import os

        plural = self._resource_plural(resource_type)
        if not (filepath or "").strip():
            raise ToolException("attach_file: filepath is required")

        content: bytes
        resolved_name: str
        if filepath.startswith("artifact://"):
            try:
                from ...runtime.utils.artifact import read_artifact  # type: ignore
            except ImportError as exc:
                raise ToolException(
                    "attach_file: artifact:// URIs require the SDK runtime "
                    "artifact helper — provide a local filepath instead."
                ) from exc
            content = read_artifact(filepath)  # returns bytes
            resolved_name = filename or filepath.rsplit("/", 1)[-1]
        else:
            try:
                with open(filepath, "rb") as fh:
                    content = fh.read()
            except OSError as exc:
                raise ToolException(f"attach_file: cannot read '{filepath}': {exc}") from exc
            resolved_name = filename or os.path.basename(filepath)

        # Aha multipart form: uses the "attachment[file]" field.
        files = {"attachment[file]": (resolved_name, content)}
        # ``requests`` will set the correct multipart Content-Type header
        # automatically when ``files`` is provided; strip the JSON default.
        headers = {"Content-Type": None}
        try:
            response = self._session.post(
                f"{self._rest_url}/{plural}/{resource_id}/attachments",
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
            return self.list_requirements(q=q, **common)
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
            "Accepted: feature, requirement, release, idea, epic, initiative, product."
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
            "Accepted: feature, requirement, release, initiative, epic, idea, product, page."
        )

    # ----- Tool registry -----

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return the list of tools this wrapper exposes."""
        return [
            # REST reads
            {
                "name": "get_feature",
                "description": self.get_feature.__doc__,
                "args_schema": AhaReferenceInput,
                "ref": self.get_feature,
            },
            {
                "name": "get_requirement",
                "description": self.get_requirement.__doc__,
                "args_schema": AhaReferenceInput,
                "ref": self.get_requirement,
            },
            {
                "name": "get_release",
                "description": self.get_release.__doc__,
                "args_schema": AhaReferenceInput,
                "ref": self.get_release,
            },
            {
                "name": "get_initiative",
                "description": self.get_initiative.__doc__,
                "args_schema": AhaReferenceInput,
                "ref": self.get_initiative,
            },
            {
                "name": "get_epic",
                "description": self.get_epic.__doc__,
                "args_schema": AhaReferenceInput,
                "ref": self.get_epic,
            },
            {
                "name": "get_idea",
                "description": self.get_idea.__doc__,
                "args_schema": AhaReferenceInput,
                "ref": self.get_idea,
            },
            {
                "name": "get_product",
                "description": self.get_product.__doc__,
                "args_schema": AhaReferenceInput,
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
