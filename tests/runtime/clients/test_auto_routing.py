"""Native message preservation, replay and parallel task isolation without inference."""
import asyncio
import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage, SystemMessage

path = Path(__file__).resolve().parents[3]/'elitea_sdk/runtime/clients/routing.py'
spec = importlib.util.spec_from_file_location('auto_client_contract', path)
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Native:
    def __init__(self):
        self.inputs = []
        self.bound = None
    def _get_request_payload(self, messages, **kwargs):
        return {'messages': [m.model_dump(mode='json') for m in messages], **kwargs}
    def bind_tools(self, tools, **kwargs):
        self.bound = (tools, kwargs);return self
    def invoke(self, messages, config=None, **kwargs):
        self.inputs.append(messages)
        return AIMessage(content=[{'type': 'thinking', 'thinking': 'opaque fixture', 'signature': 'fixture-signature'},
                                  {'type': 'text', 'text': 'result'}],
                         tool_calls=[{'id': 'call1', 'name': 'fetch', 'args': {}}])
    async def ainvoke(self, *a, **kw):
        return self.invoke(*a, **kw)
    def stream(self, messages, config=None, **kwargs):
        self.inputs.append(messages)
        yield AIMessageChunk(content='A')
        yield AIMessageChunk(content='B', response_metadata={'model_name': 'native'})
    async def astream(self, *a, **kw):
        for item in self.stream(*a, **kw):
            yield item


def model():
    native = Native()
    owner = SimpleNamespace(base_url='https://unit.invalid', headers={}, project_id=7)
    calls = []
    def request(method, url, **kwargs):
        calls.append(kwargs['json'])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'action': 'generate', 'config': {'model_name': 'native', 'max_tokens': 8000, 'context_window': 200000}, 'pin': 'signed-fixture',
            'scope_id': kwargs['json']['scope_id'], 'state_token': 'state-fixture',
            'invocation_id': kwargs['json']['invocation_id'], 'expires_at': time.time()+3600})
    owner._request = request
    owner.get_llm = Mock(return_value=native)
    value = m.AutoChatModel(owner=owner, settings={'selection': {'mode': 'auto'}, 'routing_surface': 'agent'})
    return value, native, calls


def cfg(thread='child-one', run='run-one'):
    return {'configurable': {'thread_id': thread, 'checkpoint_ns': 'agent', 'elitea_routing_run_id': run}}


@pytest.mark.parametrize('cap', [None, -1, 2048, 32000])
def test_output_allowance_preserves_explicit_limit_and_defers_unspecified(cap):
    auto, native, requests = model()
    auto.settings['max_tokens'] = cap
    auto.invoke([HumanMessage(content='task')], cfg())
    if cap in (None, -1):
        assert 'output_cap' not in requests[0]
    else:
        assert requests[0]['output_cap'] == cap


def test_measured_output_allowance_from_gateway_reaches_native_client():
    auto, native, requests = model()
    original = auto.owner._request

    def resolve(*args, **kwargs):
        response = original(*args, **kwargs)
        binding = response.json()
        binding['config']['max_tokens'] = 32000
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: binding)

    auto.owner._request = resolve
    task = HumanMessage(content='task')
    first = auto.invoke([task], cfg())
    assert auto.owner.get_llm.call_args.args[1]['max_tokens'] == 32000
    first.response_metadata[m.PIN]['expires_at'] = 0
    auto.invoke([task, first, ToolMessage(content='value', tool_call_id='call1')], cfg())
    assert requests[-1]['prior_pin'] == 'signed-fixture'
    assert 'output_cap' not in requests[-1]
    assert auto.owner.get_llm.call_args.args[1]['max_tokens'] == 32000


@pytest.mark.parametrize('transport,fields', [
    ('chat_completions', {'reasoning': {'effort': 'high'}}),
    ('anthropic_messages', {'thinking': {'type': 'adaptive', 'display': 'summarized'},
                            'output_config': {'effort': 'high'}}),
])
def test_measured_effort_binding_survives_tool_cycle_and_pin_renewal(transport, fields):
    auto, native, requests = model()
    original = auto.owner._request

    def resolve(*args, **kwargs):
        response = original(*args, **kwargs)
        binding = response.json()
        binding['config'].update(max_tokens=32000, reasoning_effort='high',
                                 routing_transport=transport, routing_reasoning_fields=fields,
                                 routing_total_output_cap=True)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: binding)

    auto.owner._request = resolve
    task = HumanMessage(content='task')
    first = auto.invoke([task], cfg())
    binding = first.response_metadata[m.PIN]
    history = [task, first, ToolMessage(content='value', tool_call_id='call1')]
    reloaded = m.AutoChatModel(owner=auto.owner, settings=auto.settings)
    reloaded.invoke(history, cfg())
    assert len(requests) == 1
    assert auto.owner.get_llm.call_args.args[1]['routing_reasoning_fields'] == fields
    binding['expires_at'] = 0
    reloaded.invoke(history, cfg())
    assert len(requests) == 2
    assert requests[-1]['prior_pin'] == binding['pin']
    assert requests[-1]['invocation_id'] == binding['invocation_id']
    config = auto.owner.get_llm.call_args.args[1]
    assert config['routing_transport'] == transport
    assert config['reasoning_effort'] == 'high'
    assert config['routing_reasoning_fields'] == fields
    assert config['max_tokens'] == 32000


def test_native_tool_cycle_reuses_checkpoint_pin_and_exact_blocks():
    auto, native, requests = model()
    task = HumanMessage(content='Read the fixture', id='task1')
    first = auto.invoke([task], cfg())
    original = first.model_dump()
    history = [task, first, ToolMessage(content='value', tool_call_id='call1')]
    # A new wrapper simulates reconstruction from a persisted graph checkpoint.
    reloaded = m.AutoChatModel(owner=auto.owner, settings=auto.settings)
    second = reloaded.invoke(history, cfg())
    assert len(requests) == 1
    assert native.inputs[-1][1] is first
    assert first.model_dump() == original
    assert second.response_metadata[m.PIN]['pin'] == 'signed-fixture'
    assert auto.owner.get_llm.call_args.args[1]['routing_invocation_id'] == first.response_metadata[m.PIN]['invocation_id']


def test_parallel_same_saved_agent_uses_independent_invocation_ids():
    auto, native, requests = model()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: auto.invoke([HumanMessage(content='same task')], cfg(str(i))), [1, 2]))
    assert len(requests) == 2
    assert len({r.response_metadata[m.PIN]['invocation_id'] for r in results}) == 2
    assert not hasattr(auto, 'pin')


def test_completed_plain_turn_classifies_new_topic():
    auto, native, requests = model()
    first = auto.invoke([HumanMessage(content='architecture')], cfg())
    first = first.model_copy(update={'content': 'architecture result', 'tool_calls': []})
    auto.invoke([HumanMessage(content='architecture'), first, HumanMessage(content='joke')], cfg(run='run-two'))
    assert len(requests) == 2
    assert requests[-1]['prior_pin'] is None


def test_completed_native_turn_classifies_new_turn_without_old_pin():
    auto, native, requests = model()
    task = HumanMessage(content='architecture')
    first = auto.invoke([task], cfg())
    first = first.model_copy(update={'tool_calls': []})
    auto.invoke([task, first, HumanMessage(content='joke')], cfg(run='run-two'))
    assert len(requests) == 2
    assert requests[-1]['prior_pin'] is None
    assert requests[0]['invocation_id'] != requests[1]['invocation_id']


def test_expired_checkpoint_pin_reauthorized_with_same_identity():
    auto, native, requests = model();task = HumanMessage(content='task')
    first = auto.invoke([task], cfg())
    first.response_metadata[m.PIN]['expires_at'] = 0
    auto.invoke([task, first, ToolMessage(content='value', tool_call_id='call1')], cfg())
    assert len(requests) == 2
    assert requests[-1]['prior_pin'] == 'signed-fixture'
    assert requests[-1]['invocation_id'] == requests[0]['invocation_id']


def test_sync_and_async_stream_preserve_provider_chunks_and_add_pin_once():
    auto, native, requests = model()
    chunks = list(auto.stream([HumanMessage(content='task')], cfg()))
    assert ''.join(c.content for c in chunks) == 'AB'
    assert sum(m.PIN in c.response_metadata for c in chunks) == 1
    async def run():
        return [c async for c in auto.astream([HumanMessage(content='other')], cfg('other'))]
    second = asyncio.run(run())
    assert ''.join(c.content for c in second) == 'AB'
    assert second[1].response_metadata['model_name'] == 'native'


def test_classifier_projection_omits_opaque_thinking_but_generation_does_not():
    original = AIMessage(content=[{'type': 'thinking', 'thinking': 'secret reasoning', 'signature': 'sig'}, {'type': 'text', 'text': 'visible'}])
    assert m.projection([original])[0]['content'] == 'visible'
    assert original.content[0]['signature'] == 'sig'


def test_unqualified_modality_fails_and_never_silently_drops_image():
    with pytest.raises(ValueError, match='modality'):
        m.projection([HumanMessage(content=[{'type': 'image_url', 'image_url': {'url': 'data:fixture'}}])])


def test_foreign_sibling_binding_never_reused():
    auto, native, calls = model();task = HumanMessage(content='task')
    first = auto.invoke([task], cfg('parent'))
    with pytest.raises(ValueError, match='another execution scope'):
        auto.invoke([task, first, ToolMessage(content='value', tool_call_id='call1')], cfg('child'))
    assert len(calls) == 1


def test_unbound_native_history_and_mid_cycle_start_rejected():
    auto, native, calls = model();task = HumanMessage(content='task')
    old = native.invoke([task])
    with pytest.raises(ValueError, match='unfinished tool batch'):
        auto.invoke([task, old, HumanMessage(content='new task')], cfg())
    with pytest.raises(ValueError, match='user task boundary'):
        auto.invoke([task, ToolMessage(content='value', tool_call_id='call1')], cfg())
    assert calls == []


def test_native_serialized_payload_screen_includes_provider_fields():
    auto, native, calls = model()
    native._get_request_payload = lambda *a, **k: {'system': 'native wrapper '*20000}
    with pytest.raises(ValueError, match='native request exceeds'):
        auto.invoke([HumanMessage(content='Hi')], cfg())
    assert native.inputs == []


def test_completed_response_observation_is_scoped_and_provider_finish_normalized():
    auto, native, calls = model();task = HumanMessage(content='task')
    first = auto.invoke([task], cfg())
    first = first.model_copy(update={'content': 'result', 'tool_calls': [],
        'usage_metadata': {'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110,
                           'input_token_details': {'cache_read': 80}},
        'response_metadata': {**first.response_metadata, 'stop_reason': 'end_turn'}})
    auto.invoke([task, first, HumanMessage(content='new task')], cfg(run='run-two'))
    assert calls[-1]['state_token'] == 'state-fixture'
    assert calls[-1]['observation']['finish_reason'] == 'stop'
    assert calls[-1]['observation']['message_index'] == 1
    assert calls[-1]['observation']['usage']['prompt_tokens_details']['cached_tokens'] == 80


def test_synthetic_human_and_harder_live_steer_keep_whole_run_binding():
    auto, native, calls = model();task = HumanMessage(content='task')
    first = auto.invoke([task], cfg())
    first = first.model_copy(update={'content': 'initial answer', 'tool_calls': []})
    history = [task, first, HumanMessage(content='Now analyze every distributed failure case and continue coding')]
    result = auto.invoke(history, cfg())
    assert len(calls) == 1
    assert result.response_metadata[m.PIN]['invocation_id'] == first.response_metadata[m.PIN]['invocation_id']
    assert native.inputs[-1][-1] is history[-1]


def test_reserved_checkpoint_binding_survives_compaction_and_resume():
    auto, native, calls = model();task = HumanMessage(content='task')
    config = cfg();config['configurable']['elitea_routing_sink'] = {}
    first = auto.invoke([task], config)
    durable = config['configurable']['elitea_routing_sink']['binding']
    restored = cfg();restored['configurable']['elitea_routing_checkpoint'] = durable
    # Original assistant/prompt are gone, as after a before_model compaction.
    result = auto.invoke([HumanMessage(content='Compacted instructions. Continue work.')], restored)
    assert len(calls) == 1
    assert result.response_metadata[m.PIN]['pin'] == first.response_metadata[m.PIN]['pin']
    # A genuinely new external turn is a new platform run and may route again.
    restored['configurable']['elitea_routing_run_id'] = 'new-external-turn'
    auto.invoke([HumanMessage(content='A different task')], restored)
    assert len(calls) == 2


def test_oversized_live_steer_fails_without_selecting_another_model():
    auto, native, calls = model();task = HumanMessage(content='task')
    first = auto.invoke([task], cfg())
    native._get_request_payload = lambda *a, **k: {'messages': 'x'*200000}
    with pytest.raises(ValueError, match='native request exceeds'):
        auto.invoke([task, first, HumanMessage(content='large added evidence')], cfg())
    assert len(calls) == 1 and len(native.inputs) == 1


def test_multimodal_live_steer_fails_without_rerouting_or_native_dispatch():
    auto, native, calls = model();task = HumanMessage(content='task')
    first = auto.invoke([task], cfg())
    steer = HumanMessage(content=[{'type': 'image_url', 'image_url': {'url': 'data:fixture'}}])
    with pytest.raises(ValueError, match='modality'):
        auto.invoke([task, first, steer], cfg())
    assert len(calls) == 1 and len(native.inputs) == 1


def test_server_task_projection_keeps_generation_and_user_authored_xml():
    raw='<runtime_context>user-authored text</runtime_context>. Hi'
    message=HumanMessage(content='server metadata. '+raw,additional_kwargs={'elitea_routing_content':[{'type':'text','text':raw}]})
    assert m.projection([message])==[{'role':'user','content':raw}]
    assert message.content=='server metadata. '+raw
    assert m.projection([HumanMessage(content=raw)])[0]['content']==raw
    auto=m.AutoChatModel(owner=object(),settings={'routing_instructions':''})
    assert auto.with_active_instructions('known platform disclosure scaffolding').active_instructions==''
    agent=m.AutoChatModel(owner=object(),settings={})
    assert agent.with_active_instructions('Verify crash recovery').active_instructions=='Verify crash recovery'


@pytest.mark.parametrize('finish,expected', [('stop','stop'),('end_turn','stop'),('max_tokens','length'),
    ('tool_use','tool_calls'),(None,None),('unrecognized',None)])
def test_durable_response_receipt_rejoins_rebuilt_history_by_projected_position(finish,expected):
    auto,native,calls=model()
    native.invoke=lambda *a,**k: AIMessage(content='Ready to code.',response_metadata={'stop_reason':finish},
        usage_metadata={'input_tokens':100,'output_tokens':8,'total_tokens':108,'input_token_details':{'cache_read':80}})
    first_task=HumanMessage(content='full server metadata. Design',additional_kwargs={'elitea_routing_content':'Design'})
    config=cfg();config['configurable']['elitea_routing_sink']={}
    first=auto.invoke([SystemMessage(content='Authored requirements'),first_task],config)
    receipt=config['configurable']['elitea_routing_sink']['binding']['last_response']
    assert receipt['finish_reason']==expected and receipt['usage']['prompt_tokens_details']['cached_tokens']==80
    # Core/Worker rebuild visible history without provider metadata, and add a
    # constructor-owned System row. Native input retains both original sources.
    rebuilt=[SystemMessage(content='Authored requirements'),
        SystemMessage(content='attachment locations',additional_kwargs={'elitea_routing_content':''}),
        HumanMessage(content=first_task.content,additional_kwargs=first_task.additional_kwargs),
        AIMessage(content='Ready to code.'),HumanMessage(content='Go')]
    restored=cfg(run='second-run');restored['configurable']['elitea_routing_checkpoint']=first.response_metadata[m.PIN]
    auto.invoke(rebuilt,restored)
    assert calls[1]['messages'][:2]==calls[0]['messages']
    assert calls[1]['observation']['message_index']==2
    assert calls[1]['observation']['finish_reason']==expected
    assert calls[1]['generation_input_bytes']>len(str(calls[1]['messages']).encode())
    assert rebuilt[1].content=='attachment locations' and 'metadata' in rebuilt[2].content


def test_identical_replies_and_repeated_user_text_use_last_position_not_equality():
    auto,native,calls=model()
    native.invoke=lambda *a,**k: AIMessage(content='Identical reply',response_metadata={'finish_reason':'stop'})
    task=HumanMessage(content='Explain')
    first=auto.invoke([task],cfg())
    second=auto.invoke([task,first,HumanMessage(content='Explain')],cfg(run='two'))
    history=[task,AIMessage(content='Identical reply'),HumanMessage(content='Explain'),
             AIMessage(content='Identical reply'),HumanMessage(content='Go')]
    config=cfg(run='three');config['configurable']['elitea_routing_checkpoint']=second.response_metadata[m.PIN]
    auto.invoke(history,config)
    assert len(calls)==3 and calls[-1]['observation']['message_index']==3


def test_response_receipt_not_reused_across_scope_or_changed_response():
    auto,native,calls=model()
    native.invoke=lambda *a,**k: AIMessage(content='Original',response_metadata={'finish_reason':'stop'})
    first=auto.invoke([HumanMessage(content='task')],cfg())
    for thread,answer in [('foreign-sibling','Original'),('child-one','Changed visible response')]:
        config=cfg(thread,run='two');config['configurable']['elitea_routing_checkpoint']=first.response_metadata[m.PIN]
        auto.invoke([HumanMessage(content='task'),AIMessage(content=answer),HumanMessage(content='new task')],config)
        assert calls[-1]['observation'] is None


@pytest.mark.parametrize('async_mode',[False,True])
def test_stream_observation_aggregates_once_and_preserves_visible_chunks(async_mode):
    auto,native,calls=model();sent=[]
    chunks=[AIMessageChunk(content='Ready '),AIMessageChunk(content='to code.',
            response_metadata={'finish_reason':'stop','model_name':'native'},
            usage_metadata={'input_tokens':100,'output_tokens':8,'total_tokens':108})]
    def stream(*a,**k):
        sent.append(a)
        yield from chunks
    async def astream(*a,**k):
        for chunk in stream(*a,**k):yield chunk
    native.stream=stream;native.astream=astream
    config=cfg();config['configurable']['elitea_routing_sink']={}
    async def collect():return [chunk async for chunk in auto.astream([HumanMessage(content='task')],config)]
    result=asyncio.run(collect()) if async_mode else list(auto.stream([HumanMessage(content='task')],config))
    assert len(sent)==len(calls)==1 and result[:2]==chunks
    assert result[0] is chunks[0] and result[1] is chunks[1]
    receipt=config['configurable']['elitea_routing_sink']['binding']['last_response']
    assert receipt['finish_reason']=='stop' and receipt['usage']['completion_tokens']==8
    assert receipt['message_digest']==m.observation(AIMessage(content='Ready to code.'))['message_digest']
    assert len(result)==3 and result[-1].response_metadata[m.PIN]['last_response']==receipt


@pytest.mark.parametrize('async_mode',[False,True])
@pytest.mark.parametrize('failure',['cancel','exception'])
def test_partial_stream_has_no_completed_observation(async_mode,failure):
    auto,native,calls=model();sent=[]
    def stream(*a,**k):
        sent.append(a)
        yield AIMessageChunk(content='partial')
        raise RuntimeError('provider stream failed')
    async def astream(*a,**k):
        for chunk in stream(*a,**k):yield chunk
    native.stream=stream;native.astream=astream
    config=cfg();config['configurable']['elitea_routing_sink']={}
    async def run():
        iterator=auto.astream([HumanMessage(content='task')],config)
        assert (await anext(iterator)).content=='partial'
        if failure=='cancel':await iterator.aclose()
        else:
            with pytest.raises(RuntimeError,match='provider stream failed'):await anext(iterator)
    if async_mode:asyncio.run(run())
    else:
        iterator=auto.stream([HumanMessage(content='task')],config)
        assert next(iterator).content=='partial'
        if failure=='cancel':iterator.close()
        else:
            with pytest.raises(RuntimeError,match='provider stream failed'):next(iterator)
    assert len(sent)==len(calls)==1
    assert 'last_response' not in config['configurable']['elitea_routing_sink']['binding']


@pytest.mark.parametrize('async_mode',[False,True])
def test_invoke_failure_does_not_publish_response_receipt(async_mode):
    auto,native,calls=model()
    native.invoke=Mock(side_effect=RuntimeError('provider failed'))
    config=cfg();config['configurable']['elitea_routing_sink']={}
    with pytest.raises(RuntimeError,match='provider failed'):
        if async_mode:asyncio.run(auto.ainvoke([HumanMessage(content='task')],config))
        else:auto.invoke([HumanMessage(content='task')],config)
    assert 'last_response' not in config['configurable']['elitea_routing_sink']['binding']


def test_constructor_projection_preserves_substantive_system_and_authored_lookalikes():
    raw='<runtime_context>user-authored text</runtime_context>'
    messages=[SystemMessage(content='attachment paths. Authored constraint',
                           additional_kwargs={'elitea_routing_content':'Authored constraint'}),
              HumanMessage(content=raw),SystemMessage(content='Authored independent system')]
    assert m.projection(messages)==[{'role':'system','content':'Authored constraint'},
        {'role':'user','content':raw},{'role':'system','content':'Authored independent system'}]
