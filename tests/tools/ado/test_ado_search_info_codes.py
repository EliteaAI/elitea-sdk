"""The paging and info-code machinery shared by both ADO search tools.

Covers what neither search test can reach on its own: that a mistyped hint is rejected
when the table is built rather than when a live response happens to carry that code,
that the table records where each code came from - Azure DevOps reuses these numbers
across indexes without preserving their meaning - and that the one paging behaviour the
two indexes disagree on is decided by the single argument that says so.
"""

from dataclasses import fields

import pytest

from elitea_sdk.tools.ado.utils import (
    SEARCH_INFO_CODES,
    SEARCH_INFO_CODES_DOCUMENTED_BY_MICROSOFT,
    SEARCH_INFO_CODES_OBSERVED_IN_RESPONSES,
    AdoSearchPaging,
    SearchIndexHints,
    SearchInfoCode,
    describe_search_info_code,
)

MATCHES_HIDDEN_INFO_CODE = next(
    code.number for code in SEARCH_INFO_CODES.values() if code.matches_hidden_by_permissions
)
ENDS_ON_UNREADABLE_WINDOW = AdoSearchPaging(default_top=5, max_top=1000, max_skip=1000)
STRIDES_PAST_UNREADABLE_WINDOW = AdoSearchPaging(
    default_top=5, max_top=50, max_skip=1000, empty_window_stride=50
)


def test_a_hint_naming_an_unknown_field_is_rejected_at_construction():
    with pytest.raises(ValueError, match="no_such_field"):
        SearchInfoCode(99, "Something happened.", hint="no_such_field")


@pytest.mark.parametrize(
    "code", [code for code in SEARCH_INFO_CODES.values() if code.hint], ids=lambda code: str(code.number)
)
def test_every_hinted_code_in_the_table_resolves_against_the_real_hints(code):
    hints = SearchIndexHints(**{field.name: "advice" for field in fields(SearchIndexHints)})

    assert code.resolve_hint(hints) == "advice"


def test_a_code_without_a_hint_resolves_to_nothing():
    code = SearchInfoCode(99, "Something happened.")

    assert code.resolve_hint(SearchIndexHints(filter_not_indexed="advice")) == ""


def test_an_index_supplying_no_hints_resolves_to_nothing():
    code = SearchInfoCode(99, "Something happened.", hint="filter_not_indexed")

    assert code.resolve_hint(None) == ""


def test_a_hint_is_appended_to_the_shared_message():
    described = describe_search_info_code(15, SearchIndexHints(filter_not_indexed="Check the branch filter."))

    assert described == "A filter value matched nothing in the index. Check the branch filter."


def test_the_same_code_reads_differently_per_index():
    repos = describe_search_info_code(15, SearchIndexHints(filter_not_indexed="Check the branch filter."))
    work_items = describe_search_info_code(15, SearchIndexHints(filter_not_indexed="Check the work item type."))

    assert repos != work_items


def test_provenance_is_recorded_and_the_two_sources_do_not_overlap():
    documented = {code.number for code in SEARCH_INFO_CODES_DOCUMENTED_BY_MICROSOFT}
    observed = {code.number for code in SEARCH_INFO_CODES_OBSERVED_IN_RESPONSES}

    assert not documented & observed
    assert documented | observed == set(SEARCH_INFO_CODES)


@pytest.mark.parametrize("number", [11, 15])
def test_undocumented_codes_stay_marked_as_observed(number):
    assert number in {code.number for code in SEARCH_INFO_CODES_OBSERVED_IN_RESPONSES}


def test_a_stride_is_the_only_thing_deciding_whether_an_unreadable_window_ends_paging():
    arguments = dict(skip=0, top=5, total_count=500, returned=0, info_code=MATCHES_HIDDEN_INFO_CODE)

    ends = ENDS_ON_UNREADABLE_WINDOW.describe_window(**arguments)
    strides = STRIDES_PAST_UNREADABLE_WINDOW.describe_window(**arguments)

    assert ends.matches_withheld_by_permissions is strides.matches_withheld_by_permissions is True
    assert ends.truncated is strides.truncated is True
    assert ends.next_skip is None
    assert strides.next_skip == STRIDES_PAST_UNREADABLE_WINDOW.empty_window_stride


@pytest.mark.parametrize("paging", [ENDS_ON_UNREADABLE_WINDOW, STRIDES_PAST_UNREADABLE_WINDOW])
def test_an_empty_window_without_permission_trimming_ends_paging_on_either_index(paging):
    window = paging.describe_window(skip=0, top=5, total_count=500, returned=0, info_code=0)

    assert window.truncated is True
    assert window.matches_withheld_by_permissions is False
    assert window.next_skip is None


@pytest.mark.parametrize("paging", [ENDS_ON_UNREADABLE_WINDOW, STRIDES_PAST_UNREADABLE_WINDOW])
def test_a_populated_window_advances_by_top_on_either_index(paging):
    window = paging.describe_window(skip=10, top=5, total_count=500, returned=5, info_code=0)

    assert window.next_skip == 15


@pytest.mark.parametrize("paging", [ENDS_ON_UNREADABLE_WINDOW, STRIDES_PAST_UNREADABLE_WINDOW])
def test_a_cursor_is_never_offered_past_the_skip_ceiling(paging):
    window = paging.describe_window(
        skip=paging.max_skip, top=5, total_count=500_000, returned=5, info_code=0
    )

    assert window.truncated is True
    assert window.paging_ceiling_reached is True
    assert window.next_skip is None
