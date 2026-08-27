"""Pipeline HITLNode handling of the block_with_comment action (issue #5318).

The sensitive-tool guard is the primary surface for block_with_comment. The
pipeline HITLNode is a separate feature, but it must still treat a
block_with_comment resume safely: a "block" must NEVER be silently downgraded
to approve by the unknown-action fallback. It maps onto the reject route; if no
reject route is configured the node fails loud rather than approving.
"""

from unittest.mock import patch

import pytest

from elitea_sdk.runtime.tools.hitl import (
    HITLNode,
    PIPELINE_HITL_HISTORY_CONTRACT_VERSION,
    PIPELINE_HITL_INTERACTION_TYPE,
)


def _make_node(routes):
    return HITLNode(
        name="review",
        input_variables=["messages"],
        user_message={"type": "fixed", "value": "Please review."},
        routes=routes,
    )


@pytest.mark.parametrize(
    "action", ["block_with_comment", "reject_with_comment", "BLOCK_WITH_COMMENT"]
)
def test_block_with_comment_routes_like_reject(action):
    node = _make_node({"approve": "do_work", "reject": "END"})

    with patch("elitea_sdk.runtime.tools.hitl.dispatch_custom_event"), patch(
        "elitea_sdk.runtime.tools.hitl.interrupt",
        side_effect=lambda payload: {"action": action, "value": "not allowed"},
    ):
        command = node.invoke({"messages": []})

    # reject routes to END -> "__end__"; pure routing, no state mutation.
    assert command.goto == "__end__"
    assert command.update is None


def test_block_with_comment_without_reject_route_fails_loud():
    node = _make_node({"approve": "do_work"})  # no reject route configured

    with patch("elitea_sdk.runtime.tools.hitl.dispatch_custom_event"), patch(
        "elitea_sdk.runtime.tools.hitl.interrupt",
        side_effect=lambda payload: {"action": "block_with_comment", "value": "x"},
    ):
        with pytest.raises(ValueError, match="not configured"):
            node.invoke({"messages": []})


def test_pipeline_hitl_interrupt_advertises_durable_history_contract():
    node = _make_node({"approve": "do_work", "edit": "edit_work"})
    captured = {}

    def resume(payload):
        captured.update(payload)
        return {"action": "approve", "value": ""}

    with patch("elitea_sdk.runtime.tools.hitl.dispatch_custom_event"), patch(
        "elitea_sdk.runtime.tools.hitl.interrupt", side_effect=resume,
    ):
        node.invoke({"messages": []})

    assert captured["interaction_type"] == PIPELINE_HITL_INTERACTION_TYPE
    assert captured["history_contract_version"] == PIPELINE_HITL_HISTORY_CONTRACT_VERSION
