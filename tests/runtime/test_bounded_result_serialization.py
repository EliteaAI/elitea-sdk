"""Bounding and serialization meet at the tool-node boundary (#6532 + #6140).

`bound_and_record` caps the RESULT OBJECT, then `serialize_tool_result` renders
it. The order matters: serializing first would build a full string copy of the
payload the bound exists to avoid walking. Neither change was written with the
other in view, so this pins their interaction.
"""

import copy
import json

import pytest

from elitea_sdk.runtime.tool_result_bounds import bound_and_record, bound_tool_result
from elitea_sdk.runtime.utils.trace_limits import resolve_tool_result_limit
from elitea_sdk.tools.utils.serialization import serialize_tool_result


@pytest.fixture
def oversized_records():
    limit = resolve_tool_result_limit(None)
    return [{"id": index, "text": "x" * 200} for index in range((limit // 100) + 500)]


def test_a_bounded_result_still_serializes_as_json(oversized_records):
    bounded = bound_and_record(oversized_records, "get_issues", "github")

    rendered = serialize_tool_result(bounded)

    assert json.loads(rendered)
    assert "'id'" not in rendered


def test_the_truncation_marker_survives_serialization(oversized_records):
    bounded = bound_and_record(oversized_records, "get_issues", "github")

    assert "truncat" in serialize_tool_result(bounded).lower()


def test_serializing_does_not_undo_the_bound(oversized_records):
    limit = resolve_tool_result_limit(None)
    # Bounding edits the list in place, so the baseline has to be taken first.
    unbounded_length = len(serialize_tool_result(copy.deepcopy(oversized_records)))

    rendered = serialize_tool_result(bound_and_record(oversized_records, "get_issues", "github"))

    # JSON punctuation puts the rendered string slightly over the estimate the
    # bound works from; what matters is that it is bounded at all, not exact.
    assert len(rendered) < limit * 2
    assert len(rendered) < unbounded_length


def test_bounding_edits_the_result_in_place(oversized_records):
    """Documented, not endorsed: the caller's own list comes back truncated."""
    bound_and_record(oversized_records, "get_issues", "github")

    assert "truncat" in serialize_tool_result(oversized_records).lower()


def test_bounding_stays_idempotent_for_a_list(oversized_records):
    once, _ = bound_tool_result(oversized_records, "tool", "github")
    twice, _ = bound_tool_result(once, "tool", "github")

    assert serialize_tool_result(once) == serialize_tool_result(twice)


def test_a_small_result_is_untouched_by_either():
    records = [{"id": 1, "title": "Bug in *login*"}]

    assert json.loads(serialize_tool_result(bound_and_record(records, "t", "github"))) == records
