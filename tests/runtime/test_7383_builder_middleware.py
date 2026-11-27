"""#7383 S3: BuilderMiddleware turns the server's needs_input / needs_confirmation into the existing HITL cards.

needs_input       -> the Ask User clarifying_question interrupt (same payload, so the same card renders it)
needs_confirmation -> a builder_changeset interrupt (Changes card); Apply calls builder_apply, Cancel applies nothing
"""
import json

import pytest
from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.middleware import builder as builder_mw


def mcp_tool(name, responses, calls):
    def run(**kwargs):
        calls.append((name, kwargs))
        return json.dumps(responses[name](kwargs))
    schema = {'type': 'object', 'properties': {k: {} for k in ('type', 'op', 'id', 'changeset_id', 'changes')}}
    return StructuredTool.from_function(func=run, name=name, description=name, args_schema=schema,
                                        metadata={'toolkit_type': 'mcp', 'toolkit_name': 'Elitea Builder'})


@pytest.fixture
def interrupts(monkeypatch):
    record = {'payloads': [], 'answers': []}

    def fake_interrupt(payload):
        record['payloads'].append(payload)
        return record['answers'].pop(0)

    monkeypatch.setattr(builder_mw, 'interrupt', fake_interrupt)
    return record


@pytest.fixture
def middleware():
    return builder_mw.BuilderMiddleware(core_prompt='<elitea_builder>core</elitea_builder>')


def test_system_prompt_is_the_core_prompt(middleware):
    assert middleware.get_system_prompt() == '<elitea_builder>core</elitea_builder>'


def test_tools_from_other_toolkits_are_left_alone(middleware):
    other = StructuredTool.from_function(func=lambda: 'x', name='search', description='s')
    assert middleware.wrap_tool(other) is other


@pytest.mark.parametrize('name', ['post_elitea_core_builder_propose', 'builder_propose', 'elitea_builder_propose'])
def test_builder_tools_are_recognised_by_name(name):
    assert builder_mw.builder_action(name) == 'propose'


def test_ready_result_passes_through(middleware, interrupts):
    calls = []
    tool = middleware.wrap_tool(mcp_tool('builder_propose', {'builder_propose': lambda a: {
        'status': 'ready', 'changeset_id': 'c1'}}, calls))
    assert json.loads(tool.invoke({'type': 'agent', 'op': 'create'}))['status'] == 'ready'
    assert interrupts['payloads'] == []


def test_needs_input_shows_the_clarifying_card_and_returns_the_answer(middleware, interrupts):
    calls = []
    interrupts['answers'].append({'q1': 'Release Notes Bot'})
    tool = middleware.wrap_tool(mcp_tool('builder_propose', {'builder_propose': lambda a: {
        'status': 'needs_input',
        'questions': [{'header': 'Name', 'question': 'What should the agent name be?', 'options': []}]}}, calls))
    out = tool.invoke({'type': 'agent', 'op': 'create'})
    (payload,) = interrupts['payloads']
    assert payload['type'] == 'hitl'
    assert payload['guardrail_type'] == 'clarifying_question'
    assert payload['builder'] is True
    assert payload['tool_args'] == {'type': 'agent', 'op': 'create'}
    assert payload['available_actions'] == ['answer']
    assert payload['questions'][0]['id'] == 'q1'
    assert payload['questions'][0]['allow_other'] is True
    assert 'Release Notes Bot' in out
    assert 'propose again' in out


def test_needs_confirmation_apply_calls_builder_apply(middleware, interrupts):
    calls = []
    interrupts['answers'].append({'action': 'apply'})
    responses = {
        'builder_propose': lambda a: {'status': 'needs_confirmation', 'changeset_id': 'c9', 'project_id': 7,
                                      'summary': 'Delete agent "Old"', 'diff': []},
        'builder_apply': lambda a: {'status': 'applied', 'summary': '**Deleted agent "Old"**'},
    }
    apply_tool = middleware.wrap_tool(mcp_tool('builder_apply', responses, calls))
    propose = middleware.wrap_tool(mcp_tool('builder_propose', responses, calls))
    middleware.bind_tools([apply_tool, propose])
    out = json.loads(propose.invoke({'type': 'agent', 'op': 'delete', 'id': 3}))
    (payload,) = interrupts['payloads']
    assert payload['guardrail_type'] == 'builder_changeset'
    assert payload['changeset_id'] == 'c9'
    assert payload['builder_project_id'] == 7
    assert payload['builder_tool_name'] == 'builder_propose'
    assert payload['tool_args'] == {'type': 'agent', 'op': 'delete', 'id': 3}
    assert payload['available_actions'] == ['apply', 'cancel']
    assert payload['summary'] == 'Delete agent "Old"'
    assert calls[-1] == ('builder_apply', {'changeset_id': 'c9'})
    assert out['status'] == 'applied'


def test_the_card_gets_masked_args_and_the_resume_keeps_the_raw_ones(middleware, interrupts):
    """Review fix: the model's own propose arguments may carry a value; the transport never sees ``changes``."""
    calls = []
    responses = {'builder_propose': lambda a: {'status': 'needs_confirmation', 'changeset_id': 'c5', 'diff': []},
                 'builder_apply': lambda a: {'status': 'applied'}}
    middleware.bind_tools([mcp_tool('builder_apply', responses, calls)])
    interrupts['answers'].append({'action': 'cancel'})
    args = {'type': 'operation', 'op': 'call', 'id': 'post_secrets_secret',
            'changes': {'name': 'jira', 'value': 'made-up-1', 'api_key': 'made-up-2'}}
    middleware.wrap_tool(mcp_tool('builder_propose', responses, calls)).invoke(args)
    [card] = interrupts['payloads']
    assert card['tool_args'] == {'type': 'operation', 'op': 'call', 'id': 'post_secrets_secret'}
    assert 'made-up' not in json.dumps({k: v for k, v in card.items() if k != 'tool_args_raw'})
    assert card['tool_args_raw'] == args


def test_an_apply_result_that_echoes_a_secret_reaches_the_model_masked(middleware, interrupts):
    calls = []
    responses = {
        'builder_propose': lambda a: {'status': 'needs_confirmation', 'changeset_id': 'c6', 'diff': []},
        'builder_apply': lambda a: {'status': 'applied', 'result': {'data': {'api_key': 'echoed-value', 'url': 'x'}}},
    }
    middleware.bind_tools([mcp_tool('builder_apply', responses, calls)])
    interrupts['answers'].append({'action': 'apply'})
    out = middleware.wrap_tool(mcp_tool('builder_propose', responses, calls)).invoke({'type': 'agent', 'op': 'delete'})
    assert 'echoed-value' not in out and json.loads(out)['result']['data']['url'] == 'x'


def test_needs_confirmation_cancel_applies_nothing(middleware, interrupts):
    calls = []
    interrupts['answers'].append({'action': 'cancel'})
    responses = {'builder_propose': lambda a: {'status': 'needs_confirmation', 'changeset_id': 'c9', 'diff': []},
                 'builder_apply': lambda a: pytest.fail('apply must not run')}
    propose = middleware.wrap_tool(mcp_tool('builder_propose', responses, calls))
    middleware.bind_tools([middleware.wrap_tool(mcp_tool('builder_apply', responses, calls)), propose])
    out = json.loads(propose.invoke({'type': 'agent', 'op': 'delete', 'id': 3}))
    assert out['status'] == 'cancelled'
    assert [c[0] for c in calls] == ['builder_propose']


def test_non_interactive_runs_do_not_pause(interrupts):
    middleware = builder_mw.BuilderMiddleware(core_prompt='x', interactive=False)
    calls = []
    tool = middleware.wrap_tool(mcp_tool('builder_propose', {'builder_propose': lambda a: {
        'status': 'needs_confirmation', 'changeset_id': 'c1', 'diff': []}}, calls))
    out = json.loads(tool.invoke({'type': 'agent', 'op': 'delete'}))
    assert interrupts['payloads'] == []
    assert out['status'] == 'needs_confirmation'
    assert 'not available' in out['message']


def test_non_json_results_pass_through(middleware, interrupts):
    tool = middleware.wrap_tool(mcp_tool('builder_find', {'builder_find': lambda a: 'plain text'}, []))
    assert tool.invoke({'type': 'agent'}) == '"plain text"'


def test_mcp_server_style_tools_with_run_are_wrapped(middleware, interrupts):
    from langchain_core.tools import BaseTool

    class FakeMcpTool(BaseTool):
        name: str = 'post_elitea_core_builder_propose'
        description: str = 'propose'

        def _run(self, *args, **kwargs):
            return json.dumps({'status': 'needs_input', 'questions': [{'header': 'Name', 'question': 'Name?'}]})

    interrupts['answers'].append({'q1': 'Bot'})
    out = middleware.wrap_tool(FakeMcpTool()).invoke({})
    assert interrupts['payloads'][0]['guardrail_type'] == 'clarifying_question'
    assert 'Bot' in out


def test_resume_reissues_propose_and_applies_the_confirmed_changeset(middleware, monkeypatch):
    """A HITL resume re-runs the wrapped propose: the first run pauses, the second gets the decision and applies
    the changeset the platform returned (the same id, since the platform reuses an identical pending proposal)."""
    from langgraph.errors import GraphInterrupt

    calls = []
    decisions = iter([None, {'action': 'apply'}])

    def fake_interrupt(payload):
        decision = next(decisions)
        if decision is None:
            raise GraphInterrupt(())
        return decision

    monkeypatch.setattr(builder_mw, 'interrupt', fake_interrupt)
    responses = {
        'builder_propose': lambda a: {'status': 'needs_confirmation', 'changeset_id': 'c1', 'diff': []},
        'builder_apply': lambda a: {'status': 'applied', 'changeset_id': a['changeset_id']},
    }
    propose = middleware.wrap_tool(mcp_tool('builder_propose', responses, calls))
    middleware.bind_tools([middleware.wrap_tool(mcp_tool('builder_apply', responses, calls)), propose])
    with pytest.raises(GraphInterrupt):
        propose.invoke({'type': 'agent', 'op': 'delete', 'id': 3})
    out = json.loads(propose.invoke({'type': 'agent', 'op': 'delete', 'id': 3}))
    assert out == {'status': 'applied', 'changeset_id': 'c1'}
    assert [c[0] for c in calls] == ['builder_propose', 'builder_propose', 'builder_apply']


def _read_result(call_id, playbook='<playbook entity="agent">x', name='post_elitea_core_builder_read'):
    from langchain_core.messages import ToolMessage
    body = {'status': 'ok', 'state': {}}
    if playbook:
        body['playbook'] = playbook
    return ToolMessage(content=json.dumps(body), tool_call_id=call_id, name=name)


def test_a_playbook_reaches_the_model_once_per_conversation():
    """The platform attaches the playbook to every find / read; the model sees it only in the first one (review fix).
    The dedupe reads the conversation's messages, so a middleware rebuilt each turn still sends it once."""
    history = [_read_result('c1'), _read_result('c2'), _read_result('c3', playbook='<playbook entity="toolkit">y')]
    for _ in range(2):
        middleware = builder_mw.BuilderMiddleware(core_prompt='x')
        out = middleware.transform_messages_for_model(history, {})
        bodies = [json.loads(m.content) for m in out]
        assert 'playbook' in bodies[0]
        assert 'playbook' not in bodies[1]
        assert bodies[2]['playbook'] == '<playbook entity="toolkit">y'
    assert 'playbook' in json.loads(history[1].content)


def test_playbook_dedupe_leaves_other_messages_alone():
    middleware = builder_mw.BuilderMiddleware(core_prompt='x')
    other = _read_result('c2', name='search')
    assert middleware.transform_messages_for_model([_read_result('c1'), other], {}) is None


def _ref_result(call_id, pages):
    from langchain_core.messages import ToolMessage
    body = {'status': 'ok', 'topic': 'pipeline:' + ','.join(pages), 'reference_pages': pages}
    return ToolMessage(content=json.dumps(body), tool_call_id=call_id, name='post_elitea_core_builder_ref')


def test_a_reference_page_reaches_the_model_once_per_conversation():
    """#6388: a page already loaded earlier in the conversation is dropped from later results and named instead."""
    history = [
        _ref_result('r1', {'overview': 'OVERVIEW', 'node:llm': 'LLM'}),
        _ref_result('r2', {'node:llm': 'LLM', 'node:router': 'ROUTER'}),
        _ref_result('r3', {'overview': 'OVERVIEW'}),
    ]
    for _ in range(2):  # a middleware rebuilt each turn gives the same answer
        out = builder_mw.BuilderMiddleware(core_prompt='x').transform_messages_for_model(history, {})
        bodies = [json.loads(m.content) for m in out]
        assert bodies[0]['reference_pages'] == {'overview': 'OVERVIEW', 'node:llm': 'LLM'}
        assert 'already_loaded' not in bodies[0]
        assert bodies[1]['reference_pages'] == {'node:router': 'ROUTER'}
        assert bodies[1]['already_loaded'] == ['node:llm']
        assert bodies[2]['reference_pages'] == {} and bodies[2]['already_loaded'] == ['overview']
    assert json.loads(history[1].content)['reference_pages'] == {'node:llm': 'LLM', 'node:router': 'ROUTER'}


def test_pages_and_playbooks_are_deduplicated_independently():
    history = [_read_result('c1'), _ref_result('r1', {'overview': 'O'}), _read_result('c2'),
               _ref_result('r2', {'node:llm': 'L'})]
    out = builder_mw.BuilderMiddleware(core_prompt='x').transform_messages_for_model(history, {})
    bodies = [json.loads(m.content) for m in out]
    assert 'playbook' in bodies[0] and 'playbook' not in bodies[2]
    assert bodies[1]['reference_pages'] == {'overview': 'O'} and bodies[3]['reference_pages'] == {'node:llm': 'L'}
    assert 'already_loaded' not in bodies[3]


def test_new_pages_only_leave_the_history_untouched():
    history = [_ref_result('r1', {'overview': 'O'}), _ref_result('r2', {'node:llm': 'L'})]
    assert builder_mw.BuilderMiddleware(core_prompt='x').transform_messages_for_model(history, {}) is None


def test_a_wrapped_builder_tool_keeps_a_content_and_artifact_result(middleware):
    """A (content, artifact) tool result passes through unchanged when it needs no card (review fix)."""
    result = (json.dumps({'status': 'ok', 'playbook': 'p'}), {'a': 1})
    assert middleware._handle(result, 'builder_read', {}) == result


@pytest.mark.parametrize('builder, auto, interactive', [
    ({'core_prompt': 'x'}, False, True),
    ({'core_prompt': 'x', 'interactive': False}, False, False),
    ({'core_prompt': 'x'}, True, False),
])
def test_the_platform_marks_a_run_non_interactive(builder, auto, interactive):
    from elitea_sdk.runtime.clients.client import EliteAClient
    middleware = []
    EliteAClient._inject_builder(middleware, builder, auto_approve_sensitive_actions=auto)
    assert middleware[0].interactive is interactive
