"""Tests for expand_inclusive_upper_bound (#6533).

GitLab renders created_at/updated_at truncated to milliseconds but filters at
microsecond precision, so an upper bound copied out of a response excludes the
very record it came from. Coarser bounds are floored to the start of their unit.
"""

from datetime import datetime

import pytest

from elitea_sdk.tools.gitlab.utils import expand_inclusive_upper_bound


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-09-03", "2026-09-03T23:59:59.999999"),
        ("2026-09-03T09", "2026-09-03T09:59:59.999999"),
        ("2026-09-03T09:24", "2026-09-03T09:24:59.999999"),
        ("2026-09-03T09:24:40", "2026-09-03T09:24:40.999999"),
        ("2026-09-03T09:24:40.4", "2026-09-03T09:24:40.499999"),
        ("2026-09-03T09:24:40.498Z", "2026-09-03T09:24:40.498999Z"),
        ("2026-09-03T09:24:40.49853Z", "2026-09-03T09:24:40.498539Z"),
        ("2026-12-31T23:59:59.999Z", "2026-12-31T23:59:59.999999Z"),
    ],
)
def test_bound_is_widened_to_last_instant_of_stated_precision(value, expected):
    assert expand_inclusive_upper_bound(value) == expected


def test_ticket_timestamp_reaches_its_own_issue():
    """#19 has created_at 2026-09-03T09:24:40.498Z and was excluded by that bound."""
    assert (
        expand_inclusive_upper_bound("2026-09-03T09:24:40.498Z")
        == "2026-09-03T09:24:40.498999Z"
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-09-03T09:24:40.498Z", "2026-09-03T09:24:40.498999Z"),
        ("2026-09-03T09:24:40.4985+05:30", "2026-09-03T09:24:40.498599+05:30"),
        ("2023-11-02T17:47:50.000+01:00", "2023-11-02T17:47:50.000999+01:00"),
        ("2026-09-03T09:24:40+0100", "2026-09-03T09:24:40.999999+0100"),
        ("2026-09-03t09:24:40z", "2026-09-03t09:24:40.999999z"),
        ("2026-09-03T09:24:40", "2026-09-03T09:24:40.999999"),
    ],
)
def test_offset_is_preserved_verbatim(value, expected):
    assert expand_inclusive_upper_bound(value) == expected


def test_space_separator_is_preserved():
    assert expand_inclusive_upper_bound("2026-09-03 09:24") == "2026-09-03 09:24:59.999999"


def test_surrounding_whitespace_is_ignored():
    assert expand_inclusive_upper_bound("  2026-09-03  ") == "2026-09-03T23:59:59.999999"


@pytest.mark.parametrize(
    "value",
    [
        "2026-09-03T09:24:40.498500",
        "2026-09-03T09:24:40.498500Z",
        "2026-09-03T09:24:40.498999999Z",
    ],
)
def test_microsecond_or_finer_precision_is_left_alone(value):
    """A caller who states microseconds has expressed an exact bound already."""
    assert expand_inclusive_upper_bound(value) is value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "garbage",
        "last week",
        "today",
        "-7d",
        "1756890280",
        "2026-09-03Z",
        "2026/09/03",
        "None",
        "2026-09-03T09:24:40,498",
    ],
)
def test_non_timestamps_are_forwarded_untouched(value):
    """GitLab keeps reporting its own validation error for these."""
    assert expand_inclusive_upper_bound(value) is value


@pytest.mark.parametrize("value", [datetime(2026, 9, 3, 9, 24, 40), 1756890280, None])
def test_non_strings_keep_their_pre_fix_pass_through(value):
    """python-gitlab serialises these itself; widening must not break that path."""
    assert expand_inclusive_upper_bound(value) is value


@pytest.mark.parametrize(
    "value",
    ["2026-09-03", "2026-09-03T09:24:40.498Z", "garbage", "", "2026-13-45T99:99:99"],
)
def test_expansion_is_idempotent(value):
    once = expand_inclusive_upper_bound(value)
    assert expand_inclusive_upper_bound(once) == once


@pytest.mark.parametrize(
    "value",
    ["", "garbage", "2026-02-30", "2026-13-45T99:99:99", "2026-09-03T09:24:60Z", "-", "T"],
)
def test_never_raises(value):
    """Call sites assign params outside their try block, so this must be total."""
    assert isinstance(expand_inclusive_upper_bound(value), str)
