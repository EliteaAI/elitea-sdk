"""Compare a source listing with the base rows of an index, without writing anything.

The functions here are pure: the toolkit lists the source and reads the index, and
this module only classifies the keys and builds the report.
"""
from dataclasses import dataclass, field
from typing import Any

REPORT_SCHEMA_VERSION = 1
MAX_LISTED_PER_GROUP = 200
# A listing can report one gap per folder or page; the report keeps the model context small.
MAX_REPORT_LINES = 20

STATUS_UP_TO_DATE = "up_to_date"
STATUS_ADD_ONLY = "add_only"
STATUS_NEEDS_CLEAN_REINDEX = "needs_clean_reindex"
# The listing cannot be trusted, so "nothing to do" and "only additions" would be guesses.
STATUS_INCOMPLETE = "incomplete"

KEY_MISMATCH_WARNING = "item ids of the listing do not match the index"
EMPTY_LISTING_WARNING = "the listing is empty but the index holds items"
UNINDEXED_NOT_RECORDED_WARNING = "re-index once to record skipped items"


@dataclass(frozen=True)
class ListedItem:
    """One item as the source or the index describes it."""
    key: str
    name: str
    path: str
    marker: str


@dataclass
class SourceListing:
    """What a toolkit returns from `_list_source_items`.

    `complete` is False when any part of the source may be missing from `items`;
    `gaps` then says why, one line each. `attest_blockers` are the reasons the next
    full index run cannot confirm the source, so it cannot remove rows either.
    `digests` holds a per-key value that a toolkit may need to rebuild the marker of
    a stored row; it lives on the listing, so no state outlives one check.
    """
    items: list[ListedItem]
    complete: bool = True
    gaps: list[str] = field(default_factory=list)
    id_family: str = ""
    attest_blockers: list[str] = field(default_factory=list)
    unchecked_note: str = ""
    unchecked_reason: str = ""
    digests: dict[str, str | None] = field(default_factory=dict)


@dataclass
class Comparison:
    added: list[ListedItem] = field(default_factory=list)
    changed: list[ListedItem] = field(default_factory=list)
    removed: list[ListedItem] = field(default_factory=list)
    unchanged: int = 0
    not_indexed: int = 0
    listing_complete: bool = True
    warnings: list[str] = field(default_factory=list)


def compare(listing: SourceListing, indexed: dict[str, ListedItem],
            unindexed_items: dict[str, str] | None = None) -> Comparison:
    """Classify every key of the listing and of the index.

    `indexed` holds the base rows only, keyed like the listing. A listing that is
    not complete never yields a removed item: a missing key is no proof of removal.
    """
    result = Comparison()
    unindexed = unindexed_items or {}
    listed = {item.key: item for item in listing.items}
    complete = listing.complete and not listing.gaps
    result.warnings.extend(listing.gaps)
    if indexed and not listed:
        # The loader refuses to remove rows on an empty listing, because a renamed or
        # removed library lists nothing; the check must not call every row removed.
        complete = False
        result.warnings.append(EMPTY_LISTING_WARNING)
    elif indexed and listed and not (set(indexed) & set(listed)):
        # A different id family (REST UniqueId against Graph driveItem id) would
        # otherwise turn every indexed row into "removed".
        complete = False
        result.warnings.append(KEY_MISMATCH_WARNING)
    result.listing_complete = complete

    for key, item in listed.items():
        stored = indexed.get(key)
        if stored is None:
            if key in unindexed and unindexed[key] == item.marker:
                result.not_indexed += 1
            else:
                result.added.append(item)
        elif stored.path != item.path:
            if complete:
                result.removed.append(stored)
                result.added.append(item)
            else:
                result.changed.append(item)
        elif stored.marker != item.marker:
            result.changed.append(item)
        else:
            result.unchanged += 1
    if complete:
        result.removed.extend(item for key, item in indexed.items() if key not in listed)
    return result


def _capped_lines(lines: list[str]) -> list[str]:
    """Distinct lines in order, cut to MAX_REPORT_LINES with a '+N more' line."""
    distinct = list(dict.fromkeys(lines))
    if len(distinct) <= MAX_REPORT_LINES:
        return distinct
    return distinct[:MAX_REPORT_LINES] + [f"+{len(distinct) - MAX_REPORT_LINES} more"]


def _status(comparison: Comparison) -> str:
    # A changed or removed item that the listing did show is a fact even when the listing is
    # partial, and the next step is the same. Only "up to date" and "add only" need a full listing.
    if comparison.changed or comparison.removed:
        return STATUS_NEEDS_CLEAN_REINDEX
    if not comparison.listing_complete:
        return STATUS_INCOMPLETE
    if comparison.added:
        return STATUS_ADD_ONLY
    return STATUS_UP_TO_DATE


def _reason(status: str, comparison: Comparison, listing: SourceListing, excluded_rows: int) -> str:
    if status == STATUS_UP_TO_DATE:
        text = "No added, changed or removed item found."
    elif status == STATUS_ADD_ONLY:
        text = "New items found. Run index_data to add them."
    elif status == STATUS_INCOMPLETE:
        text = ("The listing of the source is incomplete, so the check cannot say whether the index is up to date. "
                "See the warnings, fix the cause and run check_for_changes again.")
        if comparison.added:
            text += f" {len(comparison.added)} item(s) of the partial listing are not in the index."
    else:
        steps = []
        if comparison.removed:
            steps.append("Removed items found: run index_data with clean_index to remove them.")
        if comparison.changed:
            steps.append(
                "Changed items found: a plain index_data run re-indexes an item whose modified time moved; "
                "run index_data with clean_index when a change leaves it alone, such as an edit of a library list field.")
        text = " ".join(steps)
        if comparison.removed and listing.attest_blockers:
            text += (" Clean index will not remove the chunks of removed items, because "
                     + "; ".join(_capped_lines(listing.attest_blockers)) + ".")
    if not comparison.listing_complete and status != STATUS_INCOMPLETE:
        text += " The listing is incomplete, so removed items cannot be found."
    if excluded_rows and listing.unchecked_reason:
        text += " " + listing.unchecked_reason
    return text


def _paths(items: list[ListedItem]) -> list[str]:
    return [item.path or item.name or item.key for item in items[:MAX_LISTED_PER_GROUP]]


def build_report(index_name: str, comparison: Comparison, listing: SourceListing,
                 unindexed_recorded: bool, excluded_rows: int = 0) -> dict[str, Any]:
    """The report of the check. Counts are always full; the path lists are capped."""
    status = _status(comparison)
    warnings = list(comparison.warnings)
    if excluded_rows and listing.unchecked_note:
        warnings.append(listing.unchecked_note)
    if not unindexed_recorded:
        warnings.append(UNINDEXED_NOT_RECORDED_WARNING)
    groups = {"added": comparison.added, "changed": comparison.changed, "removed": comparison.removed}
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "index_name": index_name,
        "status": status,
        "listing_complete": comparison.listing_complete,
        "counts": {
            "added": len(comparison.added),
            "changed": len(comparison.changed),
            "removed": len(comparison.removed),
            "unchanged": comparison.unchanged,
            "not_indexed": comparison.not_indexed,
        },
        "items": {name: _paths(items) for name, items in groups.items()},
        "items_truncated": any(len(items) > MAX_LISTED_PER_GROUP for items in groups.values()),
        "warnings": _capped_lines(warnings),
        "reason": _reason(status, comparison, listing, excluded_rows),
    }
