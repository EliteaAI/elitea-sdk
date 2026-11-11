"""SharePoint document-library mode (#7352): describe indexed files with the fields of the
library lists that hold them.

A document library is a SharePoint list whose items are files; its columns (document id,
owner, approval date, category, ...) live on the list item, not in the file. This module
reads those items once per list through Microsoft Graph, joins them to the files the
toolkit indexes (by server-relative path, then by file name), and turns the item fields
into document metadata under readable snake_case names.

Safety properties, each covered by a test:
* List names from the settings never enter a request URL. Lists are resolved by comparing
  names with the site's own list catalogue; only ids returned by Graph are used in paths,
  and every path segment is percent-encoded.
* Requests go to the Microsoft Graph host only, never follow redirects, and a paging link
  is followed only when it points back at the same list's items on that host.
* Everything is bounded: lists per toolkit, page size, items per list and in total, pages,
  response bytes, fields per item, field-name and value length, request time and retries.
  A limit that is reached is a typed failure, never a silent cut, except for value length
  and fields per item, which are trimmed by design.
* 429 / 503 / 504 and connection errors are retried with backoff, honouring Retry-After.
* No field value is ever logged; logs carry counts only.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, unquote, urlsplit

import requests
from langchain_core.tools import ToolException

from .graph_wrapper import _is_graph_url as is_graph_url

logger = logging.getLogger(__name__)

GRAPH_HOST = "graph.microsoft.com"
GRAPH_BASE = f"https://{GRAPH_HOST}/v1.0"

# --------------------------------------------------------------------------- #
#  Limits (shared by the loader, the helper tool and the tests)                #
# --------------------------------------------------------------------------- #
MAX_LIBRARY_LISTS = 10
MAX_LIST_NAME_CHARS = 255
MAX_CATEGORIES = 200
MAX_METADATA_FIELD_SETTINGS = 100
PAGE_SIZE = 200
MAX_ITEMS_PER_LIST = 20_000
MAX_TOTAL_ITEMS = 50_000
# Graph may return fewer items per page than $top; the page bound only stops a list that pages
# forever, so it allows pages as small as 20 items before refusing.
MAX_PAGES_PER_LIST = MAX_ITEMS_PER_LIST // 20
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_FIELDS_PER_ITEM = 100
MAX_FIELD_NAME_CHARS = 64
MAX_VALUE_CHARS = 1_000
MAX_LIST_VALUE_ENTRIES = 50
MAX_CATALOGUE_LISTS = 2_000
REQUEST_TIMEOUT = (10, 60)          # connect, read (seconds)
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 1.0
MAX_RETRY_AFTER_SECONDS = 60.0
RETRYABLE_STATUS = frozenset({429, 503, 504})
_TRANSIENT_ERRORS = (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError)

# Status values written to every file's metadata in library mode.
STATUS_OK = "ok"
STATUS_NO_METADATA = "no metadata"
STATUS_AMBIGUOUS = "ambiguous"

# Keys the toolkit owns in document metadata; a list field never overwrites them.
RESERVED_KEYS = frozenset({
    "id", "updated_on", "created_on", "name", "path", "created", "modified", "link", "source",
    "metadata_status", "metadata_match", "library_list", "library_item_id", "web_url", "file_name",
    "chunk_id", "chunk_type", "method_name", "collection", "type", "page", "page_number",
    # keys the indexer itself reads from document metadata (grouping, deletion, content routing)
    "parent_id", "dependent_docs", "loader_content", "loader_content_type", "filename", "file_path",
    "index_meta", "content", "page_content",
})
_RESERVED_PREFIXES = ("indexer_", "loader_", "elitea_", "chunk_")

# System columns every library carries; never copied by default (they say nothing about the
# document and some, like the editor ids, identify people).
_SYSTEM_FIELDS = frozenset({
    "contenttype", "edit", "linktitle", "linktitlenomenu", "itemchildcount", "folderchildcount",
    "doc icon", "docicon", "filesizedisplay", "linkfilename", "linkfilenomenu", "linkfilenamenomenu",
    "fileleafref", "fileref", "filedirref", "complianceassetid", "id", "created", "modified",
    "attachments", "checkoutuser", "versions", "parentversionstring", "parentleafname",
    "selectfilename", "serverurl", "encodedabsurl", "basename", "fsobjtype", "uniqueid",
    "progid", "scopeid", "guid", "workflowversion", "workflowinstanceid", "instanceid", "order",
    "restricted", "originatorid", "noexecute", "contentversion", "smtotalsize", "smlastmodifieddate",
    "smtotalfilestreamsize", "smtotalfilecount", "accesspolicy", "appauthor", "appeditor",
    "author", "editor", "mediaservicemetadata", "mediaservicefastmetadata",
})

_ESCAPE_RE = re.compile(r"_x([0-9a-fA-F]{4})_")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class LibraryMetadataError(ToolException):
    """A typed, readable failure of library mode (bad setting, limit reached, Graph refusal)."""


# --------------------------------------------------------------------------- #
#  Settings                                                                    #
# --------------------------------------------------------------------------- #

def _clean_names(values: Optional[Iterable[Any]], what: str, limit: int) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    cleaned: List[str] = []
    for value in values:
        if not isinstance(value, str):
            raise LibraryMetadataError(f"Each {what} must be text.")
        name = value.strip()
        if not name:
            continue
        if len(name) > MAX_LIST_NAME_CHARS:
            raise LibraryMetadataError(f"A {what} is longer than {MAX_LIST_NAME_CHARS} characters.")
        if _CONTROL_RE.search(name):
            raise LibraryMetadataError(f"A {what} contains control characters.")
        if name not in cleaned:
            cleaned.append(name)
    if len(cleaned) > limit:
        raise LibraryMetadataError(f"At most {limit} {what}s can be set; {len(cleaned)} were given.")
    return cleaned


@dataclass(frozen=True)
class LibrarySettings:
    lists: Tuple[str, ...]
    field_map: Tuple[Tuple[str, Optional[str]], ...]   # (source field, explicit target or None)
    categories: Tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return bool(self.lists)

    @classmethod
    def from_settings(cls, library_lists=None, metadata_fields=None, categories=None) -> "LibrarySettings":
        lists = _clean_names(library_lists, "library list name", MAX_LIBRARY_LISTS)
        entries = _clean_names(metadata_fields, "metadata field", MAX_METADATA_FIELD_SETTINGS)
        field_map: List[Tuple[str, Optional[str]]] = []
        for entry in entries:
            source, sep, target = entry.partition("=")
            source = source.strip()
            if not source:
                raise LibraryMetadataError("A metadata field entry has no source field name.")
            explicit = normalize_field_name(target) if sep else None
            if sep and not explicit:
                raise LibraryMetadataError("A metadata field entry maps to an empty name.")
            field_map.append((source, explicit))
        cats = _clean_names(categories, "category", MAX_CATEGORIES)
        if cats and not lists:
            raise LibraryMetadataError("The categories filter needs at least one library list.")
        return cls(tuple(lists), tuple(field_map), tuple(cats))


# --------------------------------------------------------------------------- #
#  Field names and values                                                      #
# --------------------------------------------------------------------------- #

def normalize_field_name(name: str) -> str:
    """SharePoint internal name -> readable snake_case (``Document_x0020_Owner`` -> ``document_owner``)."""
    decoded = _ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), name or "")
    spaced = _CAMEL_RE.sub(" ", decoded)
    snake = _NON_ALNUM_RE.sub("_", spaced.lower()).strip("_")
    return snake[:MAX_FIELD_NAME_CHARS].rstrip("_")


def _is_system_field(name: str) -> bool:
    if name.startswith(("_", "@", "odata")) or name.endswith("LookupId"):
        return True
    return name.lower() in _SYSTEM_FIELDS


_CANONICAL_NAMES = frozenset({"document_id", "owner", "approver", "approval_date", "area", "country", "category"})


def _alias_for(snake: str) -> Optional[str]:
    """The readable name the issue asks for (document_id, owner, approver, approval_date, area,
    country, category) that a field name points at; used only when exactly one field matches."""
    tokens = snake.split("_")
    if snake == "document_id" or snake.endswith("_document_id") or ("document" in tokens and tokens[-1] == "id"):
        return "document_id"
    if "owner" in tokens:
        return "owner"
    if "approver" in tokens and "date" not in tokens:
        return "approver"
    if "approval" in tokens and "date" in tokens:
        return "approval_date"
    if tokens[-1] == "area":
        return "area"
    if "country" in tokens:
        return "country"
    if "category" in tokens or "categories" in tokens:
        return "category"
    return None


def _scalar(value: Any) -> Any:
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value[:MAX_VALUE_CHARS]
    if isinstance(value, dict):
        # Managed-metadata (taxonomy) and lookup values: keep the readable label only.
        for key in ("Label", "label", "LookupValue", "lookupValue"):
            label = value.get(key)
            if isinstance(label, str):
                return label[:MAX_VALUE_CHARS]
    return None


def simple_value(value: Any) -> Any:
    """A JSON-safe, bounded copy of a field value, or None when the value is not simple."""
    if isinstance(value, list):
        values = [v for v in (_scalar(entry) for entry in value[:MAX_LIST_VALUE_ENTRIES]) if v is not None]
        return values or None
    return _scalar(value)


class FieldMapper:
    """Decides, per list, which fields are copied and under which metadata key."""

    def __init__(self, settings: LibrarySettings):
        self._explicit = settings.field_map

    def plan(self, field_names: Iterable[str]) -> Dict[str, str]:
        """{source field: metadata key}.

        Without settings: every simple field under its readable name. With settings, an entry
        names a field by its SharePoint internal name, its snake_case name or the readable name
        ``list_library_categories`` shows; ``Field=name`` stores it under ``name``; ``*`` keeps
        every other field under its default name too."""
        names = sorted({n for n in field_names if isinstance(n, str) and n})
        defaults = self._default_plan(names)
        if not self._explicit:
            return self._dedupe(defaults)
        by_lower = {n.lower(): n for n in names}
        by_snake = {normalize_field_name(n): n for n in names}
        by_readable = {target: source for source, target in defaults.items()}
        plan: Dict[str, str] = {}
        explicit: Dict[str, str] = {}
        for source, target in self._explicit:
            if source == "*":
                plan.update(defaults)
                continue
            snake = normalize_field_name(source)
            actual = by_lower.get(source.lower()) or by_readable.get(snake) or by_snake.get(snake)
            if actual:
                explicit[actual] = target or defaults.get(actual) or normalize_field_name(actual)
        taken = set(explicit.values())
        plan = {src: tgt for src, tgt in plan.items() if src not in explicit and tgt not in taken}
        plan.update(explicit)
        return self._dedupe(plan)

    @staticmethod
    def _default_plan(names: List[str]) -> Dict[str, str]:
        snakes = {n: normalize_field_name(n) for n in names if not _is_system_field(n)}
        snakes = {n: s for n, s in snakes.items() if s}
        aliases: Dict[str, List[str]] = {}
        for name, snake in snakes.items():
            alias = _alias_for(snake)
            if alias:
                aliases.setdefault(alias, []).append(name)
        plan = dict(snakes)
        for alias, sources in aliases.items():
            if len(sources) == 1:          # a rename only when the rule is unambiguous
                plan[sources[0]] = alias
        return plan

    @staticmethod
    def _dedupe(plan: Dict[str, str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        used: set = set()
        # Readable names first, so the field cap never drops category, owner, ...
        for source in sorted(plan, key=lambda src: (plan[src] not in _CANONICAL_NAMES, src)):
            target = plan[source]
            if target in RESERVED_KEYS or target.startswith(_RESERVED_PREFIXES):
                target = f"list_{target}"
            candidate, n = target, 2
            while candidate in used:
                candidate, n = f"{target}_{n}", n + 1
            used.add(candidate)
            result[source] = candidate
            if len(result) >= MAX_FIELDS_PER_ITEM:
                break
        return result


# --------------------------------------------------------------------------- #
#  Graph client                                                                #
# --------------------------------------------------------------------------- #

def _segment(value: str) -> str:
    """Percent-encode one URL path segment (no '/' survives, so a value can never add a segment)."""
    return quote(str(value), safe="")


def _retry_after_seconds(resp: requests.Response, attempt: int) -> float:
    header = (resp.headers or {}).get("Retry-After")
    if header:
        try:
            return min(max(float(header), 0.0), MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
    return min(BACKOFF_BASE_SECONDS * (2 ** attempt), MAX_RETRY_AFTER_SECONDS)


class GraphListClient:
    """Read-only Graph calls for library mode: site, list catalogue, list items.

    ``get_token`` returns the bearer for each request (it may change after a refresh);
    ``refresh`` is called once on a 401 and returns whether a new token is available;
    ``on_unauthorized`` turns a final 401 into the toolkit's own authorization signal.
    """

    def __init__(self, site_url: str, get_token: Callable[[], str],
                 refresh: Optional[Callable[[], bool]] = None,
                 on_unauthorized: Optional[Callable[[requests.Response], None]] = None,
                 session: Any = None, sleep: Optional[Callable[[float], None]] = None):
        parts = urlsplit(site_url or "")
        if parts.scheme != "https" or not parts.hostname or not parts.path.strip("/"):
            raise LibraryMetadataError("Library mode needs an https SharePoint site URL with a site path.")
        self._hostname = parts.hostname
        self._site_path = [p for p in unquote(parts.path).split("/") if p]
        self._get_token = get_token
        self._refresh = refresh
        self._on_unauthorized = on_unauthorized
        self._session = session or requests
        self._sleep = sleep or time.sleep
        self._site_id: Optional[str] = None

    # -- transport ---------------------------------------------------------- #
    def _read_body(self, resp: requests.Response) -> bytes:
        declared = (resp.headers or {}).get("Content-Length")
        if declared and declared.isdigit() and int(declared) > MAX_RESPONSE_BYTES:
            raise LibraryMetadataError("A SharePoint list response is larger than the allowed size.")
        chunks, size = [], 0
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise LibraryMetadataError("A SharePoint list response is larger than the allowed size.")
            chunks.append(chunk)
        return b"".join(chunks)

    def get_json(self, url: str, params: Optional[dict] = None) -> dict:
        if not is_graph_url(url):
            raise LibraryMetadataError("Library mode only calls Microsoft Graph.")
        refreshed = False
        attempt = 0
        while True:
            try:
                resp = self._session.get(
                    url, params=params, timeout=REQUEST_TIMEOUT, allow_redirects=False, stream=True,
                    headers={"Authorization": f"Bearer {self._get_token()}", "Accept": "application/json"})
            except _TRANSIENT_ERRORS as exc:
                if attempt >= MAX_RETRIES:
                    raise LibraryMetadataError(
                        f"SharePoint did not answer after {MAX_RETRIES + 1} attempts ({type(exc).__name__}).")
                self._sleep(min(BACKOFF_BASE_SECONDS * (2 ** attempt), MAX_RETRY_AFTER_SECONDS))
                attempt += 1
                continue
            try:
                status = resp.status_code
                if status == 401 and not refreshed and self._refresh and self._refresh():
                    refreshed = True
                    continue
                if status in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                    delay = _retry_after_seconds(resp, attempt)
                    logger.info("[SP-library] Graph answered %s; retrying in %.1fs (attempt %d)",
                                status, delay, attempt + 1)
                    self._sleep(delay)
                    attempt += 1
                    continue
                self._raise_for_status(resp)
                try:
                    body = self._read_body(resp)
                except _TRANSIENT_ERRORS as exc:
                    if attempt >= MAX_RETRIES:
                        raise LibraryMetadataError(
                            f"SharePoint did not answer after {MAX_RETRIES + 1} attempts ({type(exc).__name__}).")
                    self._sleep(min(BACKOFF_BASE_SECONDS * (2 ** attempt), MAX_RETRY_AFTER_SECONDS))
                    attempt += 1
                    continue
                try:
                    return json.loads(body or b"{}")
                except ValueError:
                    raise LibraryMetadataError("SharePoint returned a response that is not JSON.")
            finally:
                resp.close()

    def _raise_for_status(self, resp: requests.Response) -> None:
        status = resp.status_code
        if 300 <= status < 400:
            raise LibraryMetadataError("SharePoint answered with a redirect; library mode does not follow redirects.")
        if status == 401:
            if self._on_unauthorized:
                self._on_unauthorized(resp)
            raise LibraryMetadataError("SharePoint refused the credential (HTTP 401). Sign in again.")
        if status == 403:
            raise LibraryMetadataError(
                "SharePoint refused access to the library list (HTTP 403). The credential needs read "
                "access to the site's lists (Graph Sites.Read.All).")
        if status == 404:
            raise LibraryMetadataError("SharePoint could not find the site or library list (HTTP 404).")
        if status >= 400:
            raise LibraryMetadataError(f"SharePoint answered HTTP {status} while reading a library list.")

    # -- resources ---------------------------------------------------------- #
    def site_id(self) -> str:
        if self._site_id is None:
            path = "/".join(_segment(p) for p in self._site_path)
            data = self.get_json(f"{GRAPH_BASE}/sites/{_segment(self._hostname)}:/{path}", {"$select": "id"})
            site_id = data.get("id")
            if not isinstance(site_id, str) or not site_id:
                raise LibraryMetadataError("SharePoint did not return the site id.")
            self._site_id = site_id
        return self._site_id

    def list_catalogue(self) -> List[dict]:
        url = f"{GRAPH_BASE}/sites/{_segment(self.site_id())}/lists"
        params: Optional[dict] = {"$select": "id,name,displayName,webUrl", "$top": PAGE_SIZE}
        lists: List[dict] = []
        prefix = urlsplit(url).path
        while url:
            data = self.get_json(url, params)
            params = None
            lists.extend(v for v in data.get("value") or [] if isinstance(v, dict))
            if len(lists) > MAX_CATALOGUE_LISTS:
                raise LibraryMetadataError(f"The site has more than {MAX_CATALOGUE_LISTS} lists.")
            url = _checked_next_link(data.get("@odata.nextLink"), prefix)
        return lists

    def resolve_lists(self, names: Sequence[str]) -> List[Tuple[str, dict]]:
        catalogue = self.list_catalogue()
        resolved, seen_ids = [], set()
        for name in names:
            wanted = name.casefold()
            matches = [entry for entry in catalogue
                       if wanted in {str(entry.get("displayName") or "").casefold(), str(entry.get("name") or "").casefold()}]
            if not matches:
                raise LibraryMetadataError(f"Library list '{name}' was not found on the site.")
            if len({m.get("id") for m in matches}) > 1:
                raise LibraryMetadataError(f"Several lists on the site are called '{name}'.")
            if matches[0].get("id") in seen_ids:
                continue
            seen_ids.add(matches[0].get("id"))
            resolved.append((name, matches[0]))
        return resolved

    def list_items(self, list_id: str, budget: int) -> List[dict]:
        if not isinstance(list_id, str) or not list_id:
            raise LibraryMetadataError("SharePoint returned a list without an id.")
        url = f"{GRAPH_BASE}/sites/{_segment(self.site_id())}/lists/{_segment(list_id)}/items"
        prefix = urlsplit(url).path
        params: Optional[dict] = {"$select": "id,webUrl", "$expand": "fields", "$top": PAGE_SIZE}
        items: List[dict] = []
        pages = 0
        limit = min(MAX_ITEMS_PER_LIST, budget)
        while url:
            pages += 1
            if pages > MAX_PAGES_PER_LIST:
                raise LibraryMetadataError(f"A library list has more than {MAX_PAGES_PER_LIST} pages of items.")
            data = self.get_json(url, params)
            params = None
            for item in data.get("value") or []:
                if isinstance(item, dict):
                    items.append(item)
            if len(items) > limit:
                if limit < MAX_ITEMS_PER_LIST:
                    raise LibraryMetadataError(f"The library lists hold more than {MAX_TOTAL_ITEMS} items in total.")
                raise LibraryMetadataError(f"A library list holds more than {MAX_ITEMS_PER_LIST} items.")
            url = _checked_next_link(data.get("@odata.nextLink"), prefix)
        return items


def _checked_next_link(link: Any, expected_path: str) -> Optional[str]:
    """A paging link is followed only when it stays on the same Graph resource."""
    if not link:
        return None
    if not isinstance(link, str) or _CONTROL_RE.search(link) or not is_graph_url(link):
        raise LibraryMetadataError("SharePoint returned a paging link to another location.")
    parts = urlsplit(link)
    if parts.username or parts.password or parts.fragment or unquote(parts.path) != unquote(expected_path):
        raise LibraryMetadataError("SharePoint returned a paging link to another location.")
    return link


# --------------------------------------------------------------------------- #
#  Join                                                                        #
# --------------------------------------------------------------------------- #

def _server_relative_path(url_or_path: Any) -> Optional[str]:
    if not isinstance(url_or_path, str) or not url_or_path:
        return None
    path = urlsplit(url_or_path).path if "://" in url_or_path else url_or_path
    path = unquote(path).strip()
    if not path.startswith("/") or path.startswith("/drives/"):
        return None
    return path.rstrip("/").casefold() or None


@dataclass
class LibraryEntry:
    list_name: str
    item_id: str
    path: Optional[str]
    name: Optional[str]
    metadata: Dict[str, Any]
    root: Optional[str] = None      # server-relative path of the list the item belongs to


@dataclass
class JoinCounts:
    by_path: int = 0
    by_name: int = 0
    no_metadata: int = 0
    ambiguous: int = 0
    category_skipped: int = 0
    list_items: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


@dataclass
class LibraryIndex:
    entries: List[LibraryEntry] = field(default_factory=list)
    by_path: Dict[str, List[LibraryEntry]] = field(default_factory=dict)
    by_name: Dict[str, List[LibraryEntry]] = field(default_factory=dict)

    def add(self, entry: LibraryEntry) -> None:
        self.entries.append(entry)
        if entry.path:
            self.by_path.setdefault(entry.path, []).append(entry)
        if entry.name:
            self.by_name.setdefault(entry.name, []).append(entry)

    def match(self, file_meta: Dict[str, Any]) -> Tuple[Optional[LibraryEntry], str]:
        """(entry, how) with how in {'path', 'name', 'ambiguous', 'none'}.

        The name fallback never lends a list item to a file that lives outside that item's own
        library: a same-named file elsewhere on the site is 'none', not a match."""
        file_path = None
        for key in ("Link", "Path", "web_url"):
            path = _server_relative_path(file_meta.get(key))
            file_path = file_path or path
            hits = self.by_path.get(path) if path else None
            if hits:
                return (hits[0], "path") if len(hits) == 1 else (None, "ambiguous")
        name = file_meta.get("Name") or file_meta.get("file_name")
        hits = self.by_name.get(name.casefold()) if isinstance(name, str) and name else None
        if hits and file_path:
            hits = [h for h in hits if not h.root or file_path.startswith(h.root + "/")]
        if not hits:
            return None, "none"
        return (hits[0], "name") if len(hits) == 1 else (None, "ambiguous")


def build_index(client: GraphListClient, settings: LibrarySettings) -> LibraryIndex:
    index = LibraryIndex()
    mapper = FieldMapper(settings)
    total = 0
    resolved = client.resolve_lists(settings.lists)
    for position, (name, catalogue_entry) in enumerate(resolved, start=1):
        items = client.list_items(catalogue_entry.get("id"), MAX_TOTAL_ITEMS - total)
        root = _server_relative_path(catalogue_entry.get("webUrl"))
        total += len(items)
        plan = mapper.plan(n for item in items for n in (item.get("fields") or {}))
        for item in items:
            fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
            metadata: Dict[str, Any] = {}
            for source, target in plan.items():
                if source in fields:
                    value = simple_value(fields[source])
                    if value is not None:
                        metadata[target] = value
            leaf = fields.get("FileLeafRef")
            path = _server_relative_path(item.get("webUrl")) or _server_relative_path(fields.get("FileRef"))
            if not isinstance(leaf, str) or not leaf:
                leaf = path.rsplit("/", 1)[-1] if path else None
            index.add(LibraryEntry(
                list_name=name, item_id=str(item.get("id") or ""), path=path,
                name=leaf.casefold() if isinstance(leaf, str) and leaf else None, metadata=metadata,
                root=root))
        logger.info("[SP-library] list %d of %d: %d items, %d fields copied",
                    position, len(resolved), len(items), len(plan))
    return index


def _categories_of(metadata: Dict[str, Any]) -> List[str]:
    value = metadata.get("category")
    values = value if isinstance(value, list) else [value]
    return [str(v).strip().casefold() for v in values if v not in (None, "")]


def apply_library_metadata(file_meta: Dict[str, Any], index: LibraryIndex, settings: LibrarySettings,
                           counts: JoinCounts) -> bool:
    """Merge list fields into ``file_meta`` in place; False when the categories filter drops the file."""
    entry, how = index.match(file_meta)
    file_meta["web_url"] = file_meta.get("Link") or file_meta.get("web_url") or ""
    file_meta["file_name"] = file_meta.get("Name") or file_meta.get("file_name") or ""
    if entry is None:
        status = STATUS_AMBIGUOUS if how == "ambiguous" else STATUS_NO_METADATA
        if settings.categories:
            counts.category_skipped += 1
            return False
        file_meta["metadata_status"] = status
        if how == "ambiguous":
            counts.ambiguous += 1
        else:
            counts.no_metadata += 1
        return True
    if settings.categories:
        wanted = {c.casefold() for c in settings.categories}
        if not wanted.intersection(_categories_of(entry.metadata)):
            counts.category_skipped += 1
            return False
    taken = {k.lower() for k in file_meta}
    for key, value in entry.metadata.items():
        file_meta[f"list_{key}" if key.lower() in taken else key] = value
    file_meta["metadata_status"] = STATUS_OK
    file_meta["metadata_match"] = how
    file_meta["library_list"] = entry.list_name
    file_meta["library_item_id"] = entry.item_id
    if how == "path":
        counts.by_path += 1
    else:
        counts.by_name += 1
    return True


def category_values(index: LibraryIndex) -> List[Dict[str, Any]]:
    """Distinct category values of the indexed lists with item counts, most used first."""
    counter: Dict[str, Tuple[str, int]] = {}
    for entry in index.entries:
        value = entry.metadata.get("category")
        for raw in value if isinstance(value, list) else [value]:
            if raw in (None, ""):
                continue
            text = str(raw).strip()
            key = text.casefold()
            label, count = counter.get(key, (text, 0))
            counter[key] = (label, count + 1)
    ordered = sorted(counter.values(), key=lambda pair: (-pair[1], pair[0].casefold()))
    return [{"value": label, "items": count} for label, count in ordered[:MAX_CATEGORIES]]


# Files listed at most when the categories filter is on (it drops files after listing).
LIBRARY_LISTING_LIMIT = MAX_TOTAL_ITEMS


def report_label(file_meta: Dict[str, Any]) -> str:
    """The file's server-relative path for the indexing report (two files may share a name)."""
    for key in ("Link", "Path"):
        value = file_meta.get(key)
        if isinstance(value, str) and value:
            path = unquote(urlsplit(value).path if "://" in value else value)
            if path.startswith("/") and not path.startswith("/drives/"):
                return path
    return str(file_meta.get("Name") or file_meta.get("Path") or "")
