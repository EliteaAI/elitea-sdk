"""Integration tests driving the real LLMNode tool-calling loop with mid-turn injections.

These complement tests/test_injection_registry.py (which unit-tests the registry in
isolation) by exercising the actual drain point inside __perform_tool_calling: a
message pushed while the loop is running must be folded into the NEXT llm invoke,
after that iteration's ToolMessages, without orphaning any tool_call/tool_result pair.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
import pytest

from elitea_sdk.runtime import _injection_registry as reg
from elitea_sdk.runtime.tools.llm import LLMNode

THREAD_ID = 't-injection-loop'
WRAPPER_PREFIX = '[user interjected mid-task]:'


@pytest.fixture(autouse=True)
def _clean_registry():
    reg.unregister(THREAD_ID)
    yield
    reg.unregister(THREAD_ID)


def _config(thread_id=THREAD_ID):
    return {'configurable': {'thread_id': thread_id}}


def _tool(name='read_file', result='file contents'):
    return StructuredTool.from_function(
        func=lambda path='x': result,
        name=name,
        description=f'{name} tool.',
        metadata={'toolkit_type': 'fs', 'toolkit_name': 'fs', 'tool_name': name},
    )


class InjectingLLMClient:
    """Requests `tool_rounds` tool calls, pushing an injection during chosen rounds.

    Injecting from inside invoke() simulates the event-callback thread pushing while
    the loop is mid-flight — the drain must pick it up on the following iteration.
    """

    def __init__(self, tool_rounds=1, inject_on_round=None, final='done'):
        self.tool_rounds = tool_rounds
        # map of round number (1-based) -> list of (text, injection_id)
        self.inject_on_round = inject_on_round or {}
        self.final = final
        self.invoke_calls = []
        self.bound_tools = []
        self.round = 0

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages, config=None):
        self.invoke_calls.append(list(messages))
        self.round += 1
        for text, inj_id in self.inject_on_round.get(self.round, []):
            reg.push(THREAD_ID, text, injection_id=inj_id)
        if self.round <= self.tool_rounds:
            return AIMessage(
                content='',
                tool_calls=[{
                    'name': 'read_file',
                    'args': {'path': f'f{self.round}.txt'},
                    'id': f'call_{self.round}',
                }],
            )
        return AIMessage(content=self.final)


def _node(client):
    return LLMNode(
        client=client,
        available_tools=[_tool()],
        tool_names=['read_file'],
        lazy_tools_mode=False,
        input_mapping={},
        output_variables=['messages'],
    )


def _injected_in(messages):
    return [
        str(m.content) for m in messages
        if isinstance(m, HumanMessage) and str(m.content).startswith(WRAPPER_PREFIX)
    ]


def test_injection_folded_into_next_invoke_wrapped():
    reg.register(THREAD_ID)
    client = InjectingLLMClient(
        tool_rounds=1,
        inject_on_round={1: [('stop and list imports instead', 'i1')]},
    )
    result = _node(client).invoke(
        {'messages': [HumanMessage(content='Summarize these files.')]},
        config=_config(),
    )

    # Round 1 requested a tool; the injection pushed during it must appear in round 2.
    assert len(client.invoke_calls) >= 2
    assert _injected_in(client.invoke_calls[0]) == []
    assert _injected_in(client.invoke_calls[1]) == [
        f'{WRAPPER_PREFIX} stop and list imports instead'
    ]
    assert isinstance(result['messages'][-1], AIMessage)


def test_injection_appended_after_tool_results_no_orphaned_pairs():
    reg.register(THREAD_ID)
    client = InjectingLLMClient(tool_rounds=1, inject_on_round={1: [('pivot now', 'i1')]})
    _node(client).invoke(
        {'messages': [HumanMessage(content='go')]}, config=_config()
    )

    second = client.invoke_calls[1]
    # Every tool_call issued must still have its matching ToolMessage — the injected
    # HumanMessage must not sit between an AIMessage's tool_calls and their results.
    requested_ids = {
        tc['id'] for m in second if isinstance(m, AIMessage)
        for tc in (getattr(m, 'tool_calls', None) or [])
    }
    answered_ids = {m.tool_call_id for m in second if isinstance(m, ToolMessage)}
    assert requested_ids and requested_ids <= answered_ids

    # Ordering: the injection lands after the last ToolMessage.
    last_tool_idx = max(i for i, m in enumerate(second) if isinstance(m, ToolMessage))
    injected_idx = next(
        i for i, m in enumerate(second)
        if isinstance(m, HumanMessage) and str(m.content).startswith(WRAPPER_PREFIX)
    )
    assert injected_idx > last_tool_idx


def test_multiple_injections_ordered_and_deduped():
    reg.register(THREAD_ID)
    client = InjectingLLMClient(
        tool_rounds=2,
        inject_on_round={
            1: [('first', 'a'), ('second', 'b'), ('first', 'a')],  # 'a' repeated -> dedup
            2: [('third', 'c')],
        },
    )
    _node(client).invoke({'messages': [HumanMessage(content='go')]}, config=_config())

    assert _injected_in(client.invoke_calls[1]) == [
        f'{WRAPPER_PREFIX} first',
        f'{WRAPPER_PREFIX} second',
    ]
    # Round 2's injection appears in round 3, and earlier ones persist in history.
    third = _injected_in(client.invoke_calls[2])
    assert f'{WRAPPER_PREFIX} third' in third
    assert third.count(f'{WRAPPER_PREFIX} first') == 1  # never re-applied


def test_no_injection_leaves_messages_untouched():
    reg.register(THREAD_ID)
    client = InjectingLLMClient(tool_rounds=1)
    _node(client).invoke({'messages': [HumanMessage(content='go')]}, config=_config())

    for messages in client.invoke_calls:
        assert _injected_in(messages) == []


def test_missing_thread_id_in_config_is_safe_noop():
    reg.register(THREAD_ID)
    reg.push(THREAD_ID, 'never delivered', injection_id='x')
    client = InjectingLLMClient(tool_rounds=1)
    # No configurable.thread_id -> drain is skipped entirely, loop still completes.
    _node(client).invoke({'messages': [HumanMessage(content='go')]}, config={})

    for messages in client.invoke_calls:
        assert _injected_in(messages) == []
    # The injection stays queued rather than being silently consumed.
    assert reg.drain(THREAD_ID) == ['never delivered']


def test_consumed_ids_recorded_for_turn_end_report():
    """Only folded-in injections may appear in the turn-end consumed report.

    The report is what the UI uses to decide whether to roll back and re-send, so
    an id recorded here that was never actually folded in would suppress a needed
    retry.
    """
    reg.register(THREAD_ID)
    client = InjectingLLMClient(
        tool_rounds=2,
        inject_on_round={1: [('first', 'a')], 2: [('second', 'b')]},
    )
    _node(client).invoke({'messages': [HumanMessage(content='go')]}, config=_config())

    assert reg.consumed(THREAD_ID) == ['a', 'b']


def test_nothing_consumed_when_no_injection_arrives():
    reg.register(THREAD_ID)
    _node(InjectingLLMClient(tool_rounds=2)).invoke(
        {'messages': [HumanMessage(content='go')]}, config=_config()
    )
    assert reg.consumed(THREAD_ID) == []


def test_parked_fanout_injection_not_marked_consumed():
    """A queued-but-undelivered injection must not be reported as consumed."""
    reg.register(THREAD_ID)
    reg.push(THREAD_ID, 'pivot during fan-out', injection_id='a')

    node = _node(InjectingLLMClient(tool_rounds=1))
    node.child_dispatcher = object()
    parked = {}

    import asyncio
    completion = AIMessage(
        content='',
        tool_calls=[
            {'name': 'app_a', 'args': {}, 'id': 'c1'},
            {'name': 'app_b', 'args': {}, 'id': 'c2'},
        ],
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(type(node), '_collect_parallel_application_specs',
                   lambda self, *a, **k: [{'name': 'app_a'}, {'name': 'app_b'}])
        mp.setattr(type(node), '_build_parallel_dispatch_specs',
                   lambda self, *a, **k: {'c1': {'dispatch_epoch': 1}, 'c2': {}})
        asyncio.run(
            node._LLMNode__perform_tool_calling(
                completion, [HumanMessage(content='go')], node.client,
                _config(), parked_holder=parked,
            )
        )

    assert parked.get('parked') is True
    assert reg.consumed(THREAD_ID) == []


def test_steps_limit_not_mutated_by_injection_budget_bump():
    """The bump must use a local variable, never the shared Pydantic instance field."""
    reg.register(THREAD_ID)
    client = InjectingLLMClient(
        tool_rounds=3,
        inject_on_round={1: [('one', 'a')], 2: [('two', 'b')]},
    )
    node = _node(client)
    before = node.steps_limit
    node.invoke({'messages': [HumanMessage(content='go')]}, config=_config())
    assert node.steps_limit == before


def test_parked_fanout_returns_before_drain_boundary():
    """The parked sub-agent fan-out path does not drain in-batch.

    When child_dispatcher is present, a batch of 2+ Application calls parks and
    RETURNS from __perform_tool_calling before reaching the drain point, so an
    injection queued during that batch is not folded in on THIS pass. It stays
    queued, and _resume_parallel_reconcile re-enters the agent node once children
    settle — hitting the loop-top drain then. So fan-out injection is delivered
    late, not lost. Pinning the early return so that contract can't drift.
    """
    reg.register(THREAD_ID)
    reg.push(THREAD_ID, 'pivot during fan-out', injection_id='a')

    node = _node(InjectingLLMClient(tool_rounds=1))
    # Presence-sentinel only; parking short-circuits before any dispatch happens.
    node.child_dispatcher = object()
    parked = {}

    import asyncio
    completion = AIMessage(
        content='',
        tool_calls=[
            {'name': 'app_a', 'args': {}, 'id': 'c1'},
            {'name': 'app_b', 'args': {}, 'id': 'c2'},
        ],
    )
    with pytest.MonkeyPatch.context() as mp:
        # Force the all-Application batch shape that triggers parking.
        mp.setattr(type(node), '_collect_parallel_application_specs',
                   lambda self, *a, **k: [{'name': 'app_a'}, {'name': 'app_b'}])
        mp.setattr(type(node), '_build_parallel_dispatch_specs',
                   lambda self, *a, **k: {'c1': {'dispatch_epoch': 1}, 'c2': {}})
        messages, _ = asyncio.run(
            node._LLMNode__perform_tool_calling(
                completion, [HumanMessage(content='go')], node.client,
                _config(), parked_holder=parked,
            )
        )

    assert parked.get('parked') is True
    assert _injected_in(messages) == []
    # Not consumed — still pending for the next boundary/turn.
    assert reg.drain(THREAD_ID) == ['pivot during fan-out']


def test_bumped_turn_finishing_normally_gets_no_max_iterations_warning():
    """The post-loop cap check must use the bumped local, not self.steps_limit.

    Otherwise a turn that ran past the original limit thanks to an injection —
    and then finished on its own — is misreported as having hit the cap, and a
    spurious "Maximum tool execution iterations" AIMessage is appended.
    """
    reg.register(THREAD_ID)
    # 2 tool rounds with limit 2: the injection bumps to 3, so round 2's follow-up
    # invoke returns the final answer and the loop exits normally, not by cap.
    client = InjectingLLMClient(tool_rounds=2, inject_on_round={1: [('pivot', 'a')]})
    node = _node(client)
    node.steps_limit = 2
    result = node.invoke({'messages': [HumanMessage(content='go')]}, config=_config())

    contents = [str(m.content) for m in result['messages'] if isinstance(m, AIMessage)]
    assert not any('Maximum tool execution iterations' in c for c in contents)
    assert result['messages'][-1].content == client.final


def test_injection_extends_budget_near_steps_limit():
    """An injection arriving at the last iteration still gets room to be acted on."""
    reg.register(THREAD_ID)
    # Ask for more tool rounds than steps_limit allows, injecting at the boundary.
    client = InjectingLLMClient(tool_rounds=10, inject_on_round={2: [('pivot', 'a')]})
    node = _node(client)
    node.steps_limit = 2
    node.invoke({'messages': [HumanMessage(content='go')]}, config=_config())

    # Without the bump the loop would stop after 2 iterations (3 invokes at most);
    # the injection buys at least one further iteration.
    assert len(client.invoke_calls) > 3
    assert any(_injected_in(m) for m in client.invoke_calls)
