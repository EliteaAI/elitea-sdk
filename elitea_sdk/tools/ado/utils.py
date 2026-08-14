import difflib
from dataclasses import dataclass, fields
from typing import Optional

from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from pydantic import SecretStr


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