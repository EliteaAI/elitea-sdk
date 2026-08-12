"""The shared info-code table behind both ADO search tools.

Covers the parts neither search test can reach on its own: that the per-index hint
lookup fails loudly on a bad field name, and that the table records where each code
came from - Azure DevOps reuses these numbers across indexes without preserving their
meaning, so a code observed on one index is not evidence for another.
"""

import pytest

from elitea_sdk.tools.ado.utils import (
    SEARCH_INFO_CODES,
    SEARCH_INFO_CODES_DOCUMENTED_BY_MICROSOFT,
    SEARCH_INFO_CODES_OBSERVED_IN_RESPONSES,
    SearchIndexHints,
    SearchInfoCode,
    describe_search_info_code,
)


def test_a_hint_naming_an_unknown_field_raises_rather_than_vanishing():
    code = SearchInfoCode(99, "Something happened.", hint="no_such_field")

    with pytest.raises(AttributeError):
        code.resolve_hint(SearchIndexHints(filter_not_indexed="advice"))


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
