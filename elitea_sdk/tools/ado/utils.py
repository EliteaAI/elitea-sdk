import contextlib
import contextvars
import difflib
import functools
from dataclasses import dataclass, fields
from typing import Annotated, Any, Optional

import requests
from azure.devops.connection import Connection
from langchain_core.tools import ToolException
from msrest.authentication import BasicAuthentication
from pydantic import BeforeValidator, Field, SecretStr, create_model


@dataclass(frozen=True)
class SearchWindow:
    truncated: bool
    paging_ceiling_reached: bool
    matches_withheld_by_permissions: bool
    next_skip: Optional[int]


@dataclass(frozen=True)
class AdoSearchPaging:
    """empty_window_stride is each index's answer to a window that permission trimming
    emptied, and the only paging behaviour that differs between the two search tools.

    Set it, and such a window still advertises a cursor, advancing by the stride rather
    than by the requested top: the rows it steps over have just been proven unreadable, and
    striding clears an unreadable stretch in tens of calls instead of hundreds. Work item
    permissions are granted per area path, so readable matches can sit below an unreadable
    stretch and are worth reaching.

    Leave it unset, and an empty window ends paging. Code read permission is granted per
    repository and code search is scoped to a single one, so a token that cannot read one
    match cannot read any of them and there is nothing below to stride towards.
    """

    default_top: int
    max_top: int
    max_skip: int
    empty_window_stride: Optional[int] = None

    def describe_window(self, skip, top, total_count, returned, info_code):
        reported_code = SEARCH_INFO_CODES.get(info_code)
        matches_withheld_by_permissions = (
            reported_code is not None and reported_code.matches_hidden_by_permissions
        )
        empty_window_step = max(top, self.empty_window_stride or top)
        cursor = skip + (top if returned else empty_window_step)
        truncated = total_count > skip + top
        paging_ceiling_reached = cursor > self.max_skip
        worth_continuing_from = returned > 0 or (
            self.empty_window_stride is not None and matches_withheld_by_permissions
        )
        return SearchWindow(
            truncated=truncated,
            paging_ceiling_reached=paging_ceiling_reached,
            matches_withheld_by_permissions=matches_withheld_by_permissions,
            next_skip=(
                cursor
                if truncated and worth_continuing_from and not paging_ceiling_reached
                else None
            ),
        )


@dataclass(frozen=True)
class SearchIndexHints:
    """Azure DevOps reuses infoCode numbers across its search indexes but not their
    meaning, so the entity-specific half of a message cannot live in the shared table.
    """

    filter_not_indexed: str


@dataclass(frozen=True)
class SearchInfoCode:
    number: int
    message: str
    matches_hidden_by_permissions: bool = False
    hint: str = ""

    def __post_init__(self):
        known_hints = {field.name for field in fields(SearchIndexHints)}
        if self.hint and self.hint not in known_hints:
            raise ValueError(
                f"Info code {self.number} names hint '{self.hint}', which is not a field of "
                f"SearchIndexHints. Known hints: {sorted(known_hints)}."
            )

    def resolve_hint(self, hints):
        if not self.hint or hints is None:
            return ""
        return getattr(hints, self.hint)


SEARCH_INFO_CODES_DOCUMENTED_BY_MICROSOFT = (
    SearchInfoCode(1, "The organization is being reindexed, so results may be incomplete."),
    SearchInfoCode(2, "Indexing has not started for this organization yet."),
    SearchInfoCode(3, "Azure DevOps rejected the query as invalid."),
    SearchInfoCode(4, "Prefix wildcard queries (a leading '*') are not supported."),
    SearchInfoCode(5, "Multi-word queries combined with code type filters are not supported."),
    SearchInfoCode(6, "The organization is being onboarded, so results may be incomplete."),
    SearchInfoCode(7, "The organization is being onboarded or reindexed, so results may be incomplete."),
    SearchInfoCode(
        8,
        "Azure DevOps chose the window size because 'top' exceeded its own maximum, so "
        "page from a smaller top rather than from this response's next_skip.",
    ),
    SearchInfoCode(9, "Branches are still being indexed, so results may be incomplete."),
    SearchInfoCode(10, "Faceting is not enabled for this organization."),
    SearchInfoCode(19, "Phrase queries are not supported together with code type filters such as 'class:' or 'def:'."),
    SearchInfoCode(20, "Wildcard queries are not supported together with code type filters such as 'class:' or 'def:'."),
)

SEARCH_INFO_CODES_OBSERVED_IN_RESPONSES = (
    SearchInfoCode(
        11,
        "Some matched results are not readable with the current token and were omitted.",
        matches_hidden_by_permissions=True,
    ),
    SearchInfoCode(15, "A filter value matched nothing in the index.", hint="filter_not_indexed"),
)

SEARCH_INFO_CODES = {
    code.number: code
    for code in SEARCH_INFO_CODES_DOCUMENTED_BY_MICROSOFT + SEARCH_INFO_CODES_OBSERVED_IN_RESPONSES
}


def create_search_client(organization_url, token):
    """A client for the almsearch host, which the azure-devops connection resolves itself."""
    secret = token.get_secret_value() if isinstance(token, SecretStr) else token
    connection = Connection(base_url=organization_url, creds=BasicAuthentication("", secret))
    return connection.clients_v7_1.get_search_client()


def describe_search_info_code(number, hints=None):
    code = SEARCH_INFO_CODES.get(number)
    if code is None:
        return f"Azure DevOps returned info code {number}."
    hint = code.resolve_hint(hints)
    return f"{code.message} {hint}" if hint else code.message


def generate_diff(base_text, target_text, file_path):
    base_lines = base_text.splitlines(keepends=True)
    target_lines = target_text.splitlines(keepends=True)
    diff = difflib.unified_diff(
        base_lines, target_lines, fromfile=f"a/{file_path}", tofile=f"b/{file_path}"
    )

    return "".join(diff)


def get_content_from_generator(content_generator):
    def safe_decode(chunk):
        try:
            return chunk.decode("utf-8")
        except UnicodeDecodeError:
            return chunk.decode("ascii", errors="backslashreplace")

    return "".join(safe_decode(chunk) for chunk in content_generator)

NO_PROJECT_MESSAGE = (
    "No Azure DevOps project specified. Pass `project` in the tool call, or set a default "
    "project in the toolkit configuration."
)

# Pipelines and LLMs send these for "not set"; they mean the configured project.
_EMPTY_PROJECT_PLACEHOLDERS = frozenset({"null", "none", "nil", "undefined", "n/a"})

# The project of the tool call in progress, as (wrapper id, project). It is per call and
# per thread, so concurrent calls on one wrapper never see each other's project, and
# nested helpers need no project argument.
_call_project: contextvars.ContextVar = contextvars.ContextVar("ado_call_project", default=None)


def normalize_project(value: Any) -> Optional[str]:
    """The project name a caller meant, or None when it meant "not set"."""
    if value is None:
        return None
    value = str(value).strip().strip("'\"").strip()
    if not value or value.lower() in _EMPTY_PROJECT_PLACEHOLDERS:
        return None
    return value


ProjectArg = Annotated[Optional[str], BeforeValidator(normalize_project)]


class AdoProjectScope:
    """Makes the configured `project` of an ADO wrapper a default each tool call may override.

    `run()` takes an optional `project` argument and makes it the project in scope for the
    call; helpers read it through `_project()`. Tools that never touch a project (the vector
    store ones) are listed in `_project_agnostic_tools` and take no argument.
    """

    _project_agnostic_tools = frozenset({
        "list_indexes", "search_index", "stepback_search_index",
        "stepback_summary_index", "remove_index",
    })

    def _resolve_project(self, project: Any = None) -> str:
        """The call's project, else the configured one, else an actionable error."""
        resolved = normalize_project(project) or normalize_project(getattr(self, "project", None))
        if resolved is None:
            raise ToolException(NO_PROJECT_MESSAGE)
        return resolved

    @contextlib.contextmanager
    def _project_scope(self, project: Any = None):
        """Make `project` the project every helper sees for the rest of this call.

        Resolution is lazy: a call whose helpers never need a project never fails for lack
        of one. A nested call that names no project keeps the outer call's.
        """
        project = normalize_project(project)
        current = _call_project.get()
        if project is None and current is not None and current[0] == id(self):
            yield
            return
        token = _call_project.set((id(self), project))
        try:
            yield
        finally:
            _call_project.reset(token)

    def _project(self) -> str:
        """The project of the tool call in progress, else the configured project."""
        current = _call_project.get()
        runtime = current[1] if current is not None and current[0] == id(self) else None
        return self._resolve_project(runtime)

    def _targets_configured_project(self) -> bool:
        # Azure DevOps matches project names case-insensitively
        configured = normalize_project(self.project)
        return configured is not None and self._project().casefold() == configured.casefold()

    def _iterate_in_project(self, project: str, make_generator):
        """Drive a generator with `project` in scope at every step, wherever it is resumed.

        Index loaders are consumed on other threads, which do not inherit the caller's
        scope, so the scope is entered around each step rather than once.
        """
        generator = None
        try:
            while True:
                with self._project_scope(project):
                    if generator is None:
                        generator = make_generator()
                    try:
                        document = next(generator)
                    except StopIteration:
                        return
                yield document
        finally:
            close = getattr(generator, "close", None)
            if close is not None:
                close()

    def _load_in_project(self, make_documents):
        """An index loader's documents, with the call's project in scope and tagged with it."""
        # Resolved now, while the call's scope is active: the loader may be first resumed
        # on a thread that never saw it
        project = self._project()

        def documents():
            for document in self._iterate_in_project(project, make_documents):
                document.metadata["project"] = project
                yield document
        return documents()

    def _in_document_project(self, document, make_generator):
        """Drive a per-document generator in the project the document was loaded from.

        The indexer processes documents on worker threads that know no call scope.
        """
        project = (document.metadata or {}).get("project")
        return self._iterate_in_project(project, make_generator) if project else make_generator()

    def _project_field(self) -> dict:
        configured = normalize_project(getattr(self, "project", None))
        # toolkit_config_schema() previews the tools from an unvalidated instance that never
        # knows the configured project, so without one the hint must not claim requiredness
        default_hint = (
            f"leave empty to use the toolkit's project '{configured}'"
            if configured else "defaults to the toolkit's project, and is required when the toolkit configures none"
        )
        return {"project": (ProjectArg, Field(
            default=None, description=f"Azure DevOps project name; {default_hint}."))}

    def _with_project(self, args_schema):
        """Extend a tool args schema with the optional per-call project, as its last field."""
        return create_model(args_schema.__name__, __base__=args_schema,
                            __module__=args_schema.__module__, **self._project_field())

    def _add_project_argument(self, tools: list) -> list:
        for tool in tools:
            if tool["name"] not in self._project_agnostic_tools \
                    and "project" not in tool["args_schema"].model_fields:
                tool["args_schema"] = self._with_project(tool["args_schema"])
        return tools

    def run(self, mode: str, *args: Any, **kwargs: Any):
        project = record_index_project(self, mode, kwargs)
        with self._project_scope(project):
            return super().run(mode, *args, **kwargs)


def record_index_project(wrapper, mode: str, kwargs: dict) -> Optional[str]:
    """Take the call's `project` out of `kwargs`, except for index_data.

    index_data keeps it, resolved to the configured project when the call names none, so
    the index records the project it was built from and a re-index reads the same one.
    """
    if mode != "index_data":
        return kwargs.pop("project", None)
    project = normalize_project(kwargs.get("project")) or normalize_project(getattr(wrapper, "project", None))
    if project is not None:
        kwargs["project"] = project
    return project


def in_caller_context(fn):
    """Wrap `fn` for a thread pool so every call runs in the context of the wrapping thread.

    Worker threads start from an empty context and would lose the call's project scope.
    """
    context = contextvars.copy_context()

    def call(*args, **kwargs):
        # One Context cannot be entered by two threads at once, so each call gets a copy
        return context.copy().run(fn, *args, **kwargs)
    return call


def with_project_argument(method):
    """Give every project-bound tool of a `get_available_tools` the per-call `project`."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return self._add_project_argument(method(self, *args, **kwargs))
    return wrapper


def check_organization_access(organization_url: str, token: str, timeout: int = 5):
    """A credentials-only connection check for toolkits that configure no project."""
    return requests.get(
        f"{organization_url}/_apis/projects?$top=1&api-version=7.0",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
