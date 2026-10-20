"""#6886: InputGuardMiddleware end to end through MiddlewareManager and LLMNode.

What these pin: a block stops the model call instead of becoming an "Error: ..." reply,
a redaction reaches the model while the checkpointed state keeps the original, and a
disabled guardrail changes nothing.
"""

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from elitea_sdk.runtime.exceptions import InputGuardrailBlocked
from elitea_sdk.runtime.guardrails import input_detector, segment_hash
from elitea_sdk.runtime.middleware.base import Middleware, MiddlewareManager
from elitea_sdk.runtime.middleware.input_guard import InputGuardMiddleware
from elitea_sdk.runtime.middleware.tool_exception_handler import reraise_signal_exceptions
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.toolkits import security

VECTORS = json.loads((Path(input_detector.__file__).parent / 'input_guardrail_vectors.json').read_text())
TOKEN = ''.join(next(v for v in VECTORS['positive'] if v['rule_id'] == 'github_token')['parts'])
EMAIL = ''.join(next(v for v in VECTORS['positive'] if v['rule_id'] == 'email')['parts'])
BLOCK = {'enabled': True}
REDACT = {'enabled': True, 'categories': {'credentials': 'redact', 'personal_data': 'redact'}}


class _RecordingChat(GenericFakeChatModel):
    seen: list = []

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _node(policy, **guard_kwargs):
    llm = _RecordingChat(messages=iter([AIMessage(content='done')] * 3), seen=[])
    manager = MiddlewareManager().add(InputGuardMiddleware(policy=policy, **guard_kwargs))
    node = LLMNode(client=llm, name='agent', available_tools=[], middleware_manager=manager,
                   input_mapping={'system': {'type': 'fixed', 'value': 'You help.'},
                                  'task': {'type': 'variable', 'value': 'input'},
                                  'chat_history': {'type': 'variable', 'value': 'messages'}},
                   input_variables=['input', 'messages'], output_variables=['messages'])
    return node, llm


def _sent_text(llm):
    return '\n'.join(str(m.content) for m in llm.seen[-1])


def test_block_on_the_current_turn_stops_the_model_call():
    node, llm = _node(BLOCK)

    with pytest.raises(InputGuardrailBlocked) as excinfo:
        node.invoke({'input': f'deploy with {TOKEN}', 'messages': []}, {'configurable': {}})

    assert llm.seen == []
    assert excinfo.value.categories == ['credentials']
    assert excinfo.value.rules == ['credentials.github_token']
    assert excinfo.value.error_code == 'input_guardrail_blocked'
    assert TOKEN not in str(excinfo.value)
    assert TOKEN not in json.dumps(excinfo.value.summary)


def test_secret_moved_to_history_still_blocked():
    node, llm = _node(BLOCK)
    state = {'input': 'continue', 'messages': [HumanMessage(content=f'my token {TOKEN}'), AIMessage(content='ok')]}

    with pytest.raises(InputGuardrailBlocked) as excinfo:
        node.invoke(state, {'configurable': {}})

    assert llm.seen == []
    assert excinfo.value.segments == ['history']


def test_redact_reaches_the_model_and_leaves_state_untouched():
    node, llm = _node(REDACT)
    history = [HumanMessage(content=f'mail {EMAIL}'), AIMessage(content='noted')]
    state = {'input': f'use {TOKEN}', 'messages': list(history)}

    node.invoke(state, {'configurable': {}})

    sent = _sent_text(llm)
    assert TOKEN not in sent and EMAIL not in sent
    assert '[REDACTED:credentials]' in sent and '[REDACTED:personal_data]' in sent
    # Model view only - nothing to checkpoint, the caller's state is unchanged
    assert state['input'] == f'use {TOKEN}'
    assert state['messages'] == history


def test_redact_keeps_history_order_and_length():
    node, llm = _node(REDACT)
    history = [HumanMessage(content=f'a {EMAIL}'), AIMessage(content='b'), HumanMessage(content='c')]

    node.invoke({'input': 'next', 'messages': history}, {'configurable': {}})

    sent = [m.content for m in llm.seen[-1][1:]]  # after the system message
    assert sent == ['a contact [REDACTED:personal_data] please', 'b', 'c', 'next']


def test_content_blocks_are_redacted_and_images_recorded_not_inspected():
    guard = InputGuardMiddleware(policy=REDACT)
    image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAA'}}
    state = {'input': [{'type': 'text', 'text': f'key {TOKEN}'}, image], 'messages': []}
    events = []
    guard.callbacks['input_guardrail'] = events.append

    updates = guard.before_model(state, {})

    assert updates['input'] == [{'type': 'text', 'text': 'key token [REDACTED:credentials]'}, image]
    assert events[0]['not_inspected'] == ['image_url']


def test_tool_results_are_not_inspected():
    guard = InputGuardMiddleware(policy=BLOCK)
    state = {'input': 'go', 'messages': [ToolMessage(content=TOKEN, tool_call_id='t1')]}

    assert guard.before_model(state, {}) is None


def test_platform_inspected_segments_are_skipped():
    guard = InputGuardMiddleware(policy=BLOCK)
    text = f'use {TOKEN}'
    config = {'configurable': {'input_guardrail_inspected': [segment_hash(text)]}}

    assert guard.before_model({'input': text, 'messages': []}, config) is None


def test_subagent_task_is_reported_as_subagent_task():
    guard = InputGuardMiddleware(policy=BLOCK)
    config = {'metadata': {'parent_agent_path': [{'name': 'parent', 'call_id': 'c1'}]}}

    with pytest.raises(InputGuardrailBlocked) as excinfo:
        guard.before_model({'input': [HumanMessage(content=TOKEN)]}, config)

    assert excinfo.value.segments == ['subagent_task']


def test_subagent_block_is_a_signal_not_tool_prose():
    error = InputGuardrailBlocked(categories=['credentials'])

    with pytest.raises(InputGuardrailBlocked):
        reraise_signal_exceptions(error)


def test_exempt_project_is_not_inspected():
    guard = InputGuardMiddleware(policy={**BLOCK, 'exempt_project_ids': [7]}, project_id=7)

    assert guard.before_model({'input': TOKEN, 'messages': []}, {}) is None


def test_warn_and_record_pass_the_input_through_and_report_once():
    guard = InputGuardMiddleware(policy={'enabled': True, 'categories': {'credentials': 'warn'}})
    events = []
    guard.callbacks['input_guardrail'] = events.append
    state = {'input': 'next', 'messages': [HumanMessage(content=TOKEN)]}

    assert guard.before_model(state, {}) is None
    # A tool loop re-sends the same history on every model call
    assert guard.before_model(state, {}) is None

    assert len(events) == 1
    assert events[0]['action'] == 'warn'
    assert events[0]['audit'] == [
        {'action': 'input:credentials.github_token:warn', 'segment_kind': 'history', 'is_error': False}
    ]
    assert TOKEN not in json.dumps(events)


def test_fail_closed_blocks_on_detector_error():
    guard = InputGuardMiddleware(policy={**BLOCK, 'on_error': 'fail_closed', 'max_inspect_chars': 5})

    with pytest.raises(InputGuardrailBlocked) as excinfo:
        guard.before_model({'input': 'harmless but long', 'messages': []}, {})

    assert excinfo.value.categories == ['error']


def test_fail_open_passes_on_detector_error():
    guard = InputGuardMiddleware(policy={**BLOCK, 'max_inspect_chars': 5})
    events = []
    guard.callbacks['input_guardrail'] = events.append

    assert guard.before_model({'input': 'harmless but long', 'messages': []}, {}) is None
    assert events[0]['audit'][0]['action'] == 'input:error.timeout:fail_open'


def test_disabled_guardrail_sends_exactly_the_same_messages():
    plain_llm = _RecordingChat(messages=iter([AIMessage(content='done')]), seen=[])
    plain = LLMNode(client=plain_llm, name='agent', available_tools=[],
                    input_mapping={'system': {'type': 'fixed', 'value': 'You help.'},
                                   'task': {'type': 'variable', 'value': 'input'},
                                   'chat_history': {'type': 'variable', 'value': 'messages'}},
                    input_variables=['input', 'messages'], output_variables=['messages'])
    guarded, guarded_llm = _node({'enabled': False})
    state = {'input': f'use {TOKEN}', 'messages': [HumanMessage(content='hi'), AIMessage(content='hello')]}

    plain.invoke(dict(state), {'configurable': {}})
    guarded.invoke(dict(state), {'configurable': {}})

    assert [m.content for m in guarded_llm.seen[-1]] == [m.content for m in plain_llm.seen[-1]]


class _AfterGuard(Middleware):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_tools(self):
        return []

    def get_system_prompt(self):
        return ''

    def before_model(self, state, config):
        self.calls += 1


class _Broken(_AfterGuard):
    def before_model(self, state, config):
        raise RuntimeError('unrelated failure')


def test_manager_reraises_block_but_still_swallows_other_middleware_failures():
    after = _AfterGuard()
    blocking = MiddlewareManager().add(InputGuardMiddleware(policy=BLOCK)).add(after)

    with pytest.raises(InputGuardrailBlocked):
        blocking.run_before_model({'input': TOKEN, 'messages': []}, {})
    assert after.calls == 0

    tolerant = MiddlewareManager().add(_Broken()).add(after)
    tolerant.run_before_model({'input': 'x', 'messages': []}, {})
    assert after.calls == 1


@pytest.fixture
def _reset_security():
    security.reset_input_guard()
    yield
    security.reset_input_guard()


def test_input_guard_config_and_env_fallback(monkeypatch, _reset_security):
    assert security.has_input_guard_config() is False

    security.reset_input_guard()
    monkeypatch.setenv('ELITEA_INPUT_GUARDRAIL', json.dumps(BLOCK))
    assert security.has_input_guard_config() is True

    security.configure_input_guard({'enabled': False})
    assert security.has_input_guard_config() is False


def test_client_injects_guard_first_only_when_enabled(_reset_security):
    from elitea_sdk.runtime.clients.client import EliteAClient

    client = EliteAClient.__new__(EliteAClient)
    client.project_id = 3
    existing = [_AfterGuard()]

    client._inject_input_guard(existing, 'conv')
    assert len(existing) == 1

    security.configure_input_guard(BLOCK)
    client._inject_input_guard(existing, 'conv')
    client._inject_input_guard(existing, 'conv')

    assert isinstance(existing[0], InputGuardMiddleware)
    assert len(existing) == 2
    assert existing[0].project_id == 3
