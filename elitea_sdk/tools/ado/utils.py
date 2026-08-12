import difflib
from dataclasses import dataclass

from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from pydantic import SecretStr


@dataclass(frozen=True)
class AdoSearchPaging:
    """A window emptied by permission trimming advances by empty_window_stride rather than
    by the requested top: the rows it skips have just been proven unreadable, and stepping
    over them reaches the skip ceiling in tens of calls instead of hundreds.
    """

    default_top: int
    max_top: int
    max_skip: int = 1000
    empty_window_stride: int = 50


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