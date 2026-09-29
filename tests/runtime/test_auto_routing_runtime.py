"""Run actual SDK node/structured adapters with native transports mocked."""
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
from elitea_sdk.runtime.clients.routing import AutoChatModel, PIN
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime import _injection_registry as registry


def setup_auto(native):
    requests = []
    owner = SimpleNamespace(base_url='https://unit.invalid', headers={}, project_id=7)
    def request(method, url, **kwargs):
        body = kwargs['json'];requests.append(body)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'action': 'generate', 'config': {'model_name': 'fixture', 'max_tokens': 8000, 'context_window': 200000},
            'invocation_id': body['invocation_id'], 'scope_id': body['scope_id'],
            'pin': 'signed-fixture', 'state_token': 'state-fixture', 'expires_at': time.time()+3600})
    owner._request = request;owner.get_llm = Mock(return_value=native)
    auto = AutoChatModel(owner=owner, settings={'selection': {'mode': 'auto'}})
    config = {'configurable': {'thread_id': 'runtime-test', 'elitea_routing_run_id': 'server-run', 'elitea_routing_sink': {}}}
    return auto, config, requests


def test_auto_failure_reaches_worker_error_card_instead_of_becoming_answer():
    native = ChatOpenAI(model='gpt-5.4', api_key='fixture', max_tokens=8000)
    auto, config, requests = setup_auto(native)
    node = LLMNode(client=auto, name='ordinary_agent', available_tools=[])
    from queue import Empty
    with patch.object(LLMNode, '_invoke_llm_internal', side_effect=Empty()):
        with pytest.raises(Empty):
            node.invoke({'messages': [HumanMessage(content='Hi')]}, config)
    assert requests == []


def test_assistant_initialization_logging_does_not_materialize_auto():
    import ast
    from pathlib import Path
    import elitea_sdk.runtime.langchain.assistant as assistant
    tree = ast.parse(Path(assistant.__file__).read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.Try)
                 and '_get_model_default_parameters' in ast.unparse(n)
                 and 'Client created:' in ast.unparse(n))
    auto, _, requests = setup_auto(None)
    for client in [auto, SimpleNamespace(_get_model_default_parameters={'temperature': 0.7}),
                   SimpleNamespace(temperature=0.5, max_tokens=2048)]:
        logger = Mock()
        exec(compile(ast.Module(body=[block], type_ignores=[]), assistant.__file__, 'exec'),
             {'self': SimpleNamespace(client=client), 'logger': logger})
        assert logger.info.call_count == 1
    assert requests == []


def test_early_clarification_trace_survives_visible_message_projection():
    auto, config, requests = setup_auto(None)
    trace = {'classifier':{'called':True,'schema_valid':True}, 'descriptor':{'input_status':'missing'}}
    auto.owner._request = Mock(return_value=SimpleNamespace(raise_for_status=lambda:None,
        json=lambda:{'action':'clarify','text':'Please identify a current source.','trace':trace}))
    node = LLMNode(client=auto, name='ordinary_agent', available_tools=[],
                   input_mapping={'messages':{'type':'variable','value':'messages'}},
                   input_variables=['messages'], output_variables=['messages'], return_type='dict')
    result = node.invoke({'messages':[HumanMessage(content='Weather today?')]},config)
    assert result['messages'][-1].response_metadata['elitea_routing_trace'] == trace
    assert '_auto_routing' not in result
    auto.owner.get_llm.assert_not_called()


@pytest.mark.parametrize('name,count,has_pin,expected', [
    ('ask_user',1,True,True), ('ask_user',2,True,False),
    ('ask_user',1,False,False), ('delete_record',1,True,False)])
def test_auto_normalized_resume_fallback_is_only_unambiguous_ask_user(name,count,has_pin,expected):
    from elitea_sdk.runtime.langchain.langraph_agent import LangGraphAgentRunnable
    message = {'type':'ai','data':{'tool_calls':[
        {'id':str(i),'name':name,'args':{'question':'Select marker'}} for i in range(count)],
        'response_metadata':{PIN:{'pin':'signed-fixture'}} if has_pin else {}}}
    actual = LangGraphAgentRunnable._extract_original_ai_message(
        [message],name,{'question':'Select marker','id':'generated'})
    assert (actual is message) is expected


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
def test_structured_native_adapters_include_schema_and_preserve_raw_pin(provider):
    class Result(BaseModel):
        supported: bool
    if provider == 'openai':
        native = ChatOpenAI(model='gpt-5.4', api_key='fixture', max_tokens=8000)
        answer = AIMessage(content='', tool_calls=[{'id': 'structured', 'name': 'Result', 'args': {'supported': True}}])
    else:
        native = ChatAnthropic(model='claude-sonnet-4-6', api_key='fixture', max_tokens=8000, thinking={'type': 'adaptive'})
        answer = AIMessage(content='{"supported":true}')
    auto, config, requests = setup_auto(native)
    with patch.object(type(native), 'invoke', return_value=answer):
        output = auto.with_structured_output(Result, include_raw=True, method='function_calling').invoke([HumanMessage(content='Return supported true')], config)
    parsed = output['parsed']
    assert (parsed if isinstance(parsed, dict) else parsed.model_dump()) == {'supported': True}
    assert output['raw'].response_metadata[PIN]['pin'] == 'signed-fixture'
    assert requests[0]['output_schema']['required'] == ['supported']
    assert config['configurable']['elitea_routing_sink']['binding']['pin'] == 'signed-fixture'


@pytest.mark.parametrize('include_raw',[False,True])
@pytest.mark.parametrize('asynchronous',[False,True])
def test_structured_parse_failure_does_not_publish_completed_observation(include_raw,asynchronous):
    from pydantic import ValidationError
    class Result(BaseModel):
        supported: bool
    native=ChatOpenAI(model='gpt-5.4',api_key='fixture',max_tokens=8000)
    answer=AIMessage(content='',tool_calls=[{'id':'structured','name':'Result','args':{'wrong':True}}],
                     response_metadata={'finish_reason':'stop'})
    auto,config,requests=setup_auto(native)
    runnable=auto.with_structured_output(Result,include_raw=include_raw,method='function_calling')
    with patch.object(type(native),'ainvoke' if asynchronous else 'invoke',return_value=answer) as call:
        def run():
            task=[HumanMessage(content='Return supported true')]
            return asyncio.run(runnable.ainvoke(task,config)) if asynchronous else runnable.invoke(task,config)
        if include_raw:
            result=run()
            assert result['parsing_error'] is not None and result['raw'].response_metadata[PIN]['pin']=='signed-fixture'
        else:
            with pytest.raises(ValidationError):run()
    assert call.call_count==len(requests)==1
    assert 'last_response' not in config['configurable']['elitea_routing_sink']['binding']


@pytest.mark.parametrize('resumed', [False, True])
def test_actual_tool_batch_consumes_multiple_steers_without_reclassification(resumed):
    completed = []
    @tool
    def read_a() -> str:
        """Read fixture A."""
        completed.append('a');return 'A'
    @tool
    def read_b() -> str:
        """Read fixture B."""
        completed.append('b');return 'B'
    native = ChatOpenAI(model='gpt-5.4', api_key='fixture', max_tokens=8000)
    auto, config, requests = setup_auto(native)
    history = [HumanMessage(content='Read both fixtures')]
    first = AIMessage(content='', tool_calls=[{'id': 'a', 'name': 'read_a', 'args': {}}, {'id': 'b', 'name': 'read_b', 'args': {}}])
    seen = []
    def invoke(self, messages, config=None, **kwargs):
        seen.append(messages)
        if len(seen) == 1:
            return first
        assert completed == ['a', 'b']
        assert [m.type for m in messages[-4:]] == ['tool', 'tool', 'human', 'human']
        return AIMessage(content='Both fixtures checked; steering applied.')
    registry.register('runtime-test')
    try:
        with patch.object(ChatOpenAI, 'invoke', invoke):
            initial = auto.bind_tools([read_a, read_b]).invoke(history, config)
            if resumed:
                # Same logical run reconstructed from its durable SDK binding.
                binding = config['configurable']['elitea_routing_sink']['binding']
                config['configurable']['elitea_routing_checkpoint'] = binding
                config['configurable']['elitea_routing_sink'] = {}
                auto = AutoChatModel(owner=auto.owner, settings=auto.settings)
                registry.unregister('runtime-test');registry.register('runtime-test')
            assert registry.push('runtime-test', 'Also prove the failure case', injection_id='one')
            assert registry.push('runtime-test', 'Preserve the API contract', injection_id='two')
            node = LLMNode(client=auto, name='ordinary_agent', available_tools=[read_a, read_b], steps_limit=2)
            result, _ = asyncio.run(node._LLMNode__perform_tool_calling(initial, history, auto.bind_tools([read_a, read_b]), config))
        assert len(requests) == 1
        assert registry.consumed('runtime-test') == ['one', 'two']
        assert result[-1].response_metadata[PIN]['invocation_id'] == initial.response_metadata[PIN]['invocation_id']
    finally:
        registry.unregister('runtime-test')


def assert_graph_input_preserves_typed_routing_task(task):
    import yaml
    from langgraph.checkpoint.memory import MemorySaver
    from elitea_sdk.runtime.langchain.langraph_agent import create_graph
    native=ChatOpenAI(model='gpt-5.4',api_key='fixture',max_tokens=8000)
    auto,config,requests=setup_auto(native)
    schema={'state':{'input':{'type':'str'},'messages':{'type':'list','operator':'add_messages'}},
            'nodes':[{'id':'agent','type':'llm','prompt':{'template':'Platform scaffold'},
                      'input_mapping':{'system':{'type':'fixed','value':'Platform scaffold'},
                                       'task':{'type':'variable','value':'input'},
                                       'chat_history':{'type':'variable','value':'messages'}},
                      'input':['messages'],'output':['messages'],'transition':'END'}],'entry_point':'agent'}
    graph=create_graph(client=auto,yaml_schema=yaml.safe_dump(schema),tools=[],memory=MemorySaver())
    seen=[]
    def invoke(self,messages,config=None,**kwargs):
        seen.append(messages);return AIMessage(content='Hello')
    with patch.object(ChatOpenAI,'invoke',invoke):
        graph.invoke({'messages':[task]},config)
    config['configurable']['elitea_routing_run_id']='new-external-run'
    with patch.object(ChatOpenAI,'invoke',invoke):
        graph.invoke({'messages':[HumanMessage(content='Next task')]},config)
    assert requests[1]['scope_id']==requests[0]['scope_id']
    assert requests[1]['invocation_id']!=requests[0]['invocation_id']
    assert requests[1]['state_token']=='state-fixture'
    assert requests[1]['messages'][-1]['content']=='Next task'
    assert requests[0]['generation_input_bytes'] > len(str(requests[0]['messages']))
    assert requests[0]['messages'][-1]['content']=='Hi'
    assert '<runtime_context>' in seen[0][-1].content
    assert seen[0][-1].additional_kwargs['elitea_routing_content']==[{'type':'text','text':'Hi'}]


def test_actual_graph_input_flattening_preserves_typed_routing_task():
    # SDK owns consuming this wire contract; the actual Worker producer is
    # exercised separately in the explicit cross-repository integration suite.
    task = HumanMessage(
        content=[{'type': 'text', 'text': '<runtime_context>server IDs</runtime_context>'},
                 {'type': 'text', 'text': 'Hi'}],
        additional_kwargs={'elitea_routing_content': [{'type': 'text', 'text': 'Hi'}]},
    )
    assert_graph_input_preserves_typed_routing_task(task)


def test_unresolved_child_exception_has_actionable_model_contract():
    from elitea_sdk.runtime.exceptions import AutoRoutingChildModelRequired
    error = AutoRoutingChildModelRequired()
    assert isinstance(error, ValueError)
    assert error.error_code == 'auto_child_model_required'
    assert 'no effective model after default resolution' in str(error)
    assert 'default or save a valid model or Auto' in str(error)


def test_actual_graph_tool_loop_keeps_one_binding_across_resume_and_child_scopes():
    import yaml
    from copy import deepcopy
    from concurrent.futures import ThreadPoolExecutor
    from langgraph.checkpoint.memory import MemorySaver
    from elitea_sdk.runtime.langchain.langraph_agent import create_graph
    from elitea_sdk.runtime.clients.routing import routing_scope_id

    completed = []
    @tool
    def read_fixture() -> str:
        """Read the fixture."""
        completed.append(True)
        return 'fixture value'

    native = ChatOpenAI(model='gpt-5.4', api_key='fixture', max_tokens=8000)
    auto, config, requests = setup_auto(native)
    node = {'id':'agent','type':'llm','prompt':{'template':'Read and check'},
            'input_mapping':{'system':{'type':'fixed','value':'Read and check'},
                             'task':{'type':'variable','value':'input'},
                             'chat_history':{'type':'variable','value':'messages'}},
            'input':['messages'],'output':['messages'],'transition':'END'}
    schema = {'name':'react_agent','state':{'input':{'type':'str'},'messages':{'type':'list','operator':'add_messages'}},
              'nodes':[node],'entry_point':'agent'}
    graph = create_graph(client=auto,yaml_schema=yaml.safe_dump(schema),tools=[read_fixture],memory=MemorySaver())
    seen = []
    def invoke(self,messages,config=None,**kwargs):
        seen.append(config)
        if messages[-1].type == 'tool':
            return AIMessage(content='Checked')
        return AIMessage(content='',tool_calls=[{'id':'read','name':'read_fixture','args':{}}])
    with patch.object(ChatOpenAI,'invoke',invoke):
        graph.invoke({'messages':[HumanMessage(content='Read the fixture')]},config)
    assert len(requests)==1 and len(seen)==2 and completed==[True]
    assert len({routing_scope_id(c) for c in seen})==1
    assert len({c['configurable']['elitea_routing_run_id'] for c in seen})==1
    state=graph.get_state(config)
    binding=state.values['_auto_routing'][requests[0]['scope_id']]
    # Restored same-run checkpoint, even with a different internal node task id.
    resumed=deepcopy(config)
    resumed['configurable'].update(checkpoint_ns='agent:new-node-task',elitea_routing_checkpoint=binding,elitea_routing_sink={})
    with patch.object(ChatOpenAI,'invoke',return_value=AIMessage(content='Resumed')):
        auto.invoke([HumanMessage(content='Internal continuation')],resumed)
    assert len(requests)==1
    # Delegated graph entries replace inherited parent ownership and isolate siblings.
    def child(label):
        child_config=deepcopy(config)
        child_config['configurable']['thread_id']='runtime-test:ordinary-agent:'+label
        with_owner=child_config['configurable']['elitea_routing_graph_owner']
        assert with_owner['thread_id']=='runtime-test'
        return graph.invoke({'messages':[HumanMessage(content='Read the fixture')]},child_config)
    with patch.object(ChatOpenAI,'invoke',invoke), ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(child,['call-a','call-b']))
    assert len(requests)==3
    assert len({r['scope_id'] for r in requests})==3
    assert len({r['invocation_id'] for r in requests})==3
    assert all(r['state_token'] is None for r in requests[1:])
    assert config['configurable']['elitea_routing_graph_owner']['thread_id']=='runtime-test'


@pytest.mark.parametrize('fail_after_resume', [False, True])
def test_real_ask_user_interrupt_reuses_persisted_binding_after_graph_reconstruction(fail_after_resume):
    import yaml
    from langgraph.checkpoint.memory import MemorySaver
    from elitea_sdk.runtime.langchain.langraph_agent import create_graph
    from elitea_sdk.runtime.tools.ask_user import AskUserTool
    native = ChatOpenAI(model='gpt-5.4', api_key='fixture', max_tokens=8000)
    auto, config, requests = setup_auto(native)
    schema = {'name':'react_agent','state':{'input':{'type':'str'},'messages':{'type':'list','operator':'add_messages'}},
              'nodes':[{'id':'agent','type':'llm','prompt':{'template':'Ask for marker'},
                        'input_mapping':{'system':{'type':'fixed','value':'Ask for marker'},
                                         'task':{'type':'variable','value':'input'},
                                         'chat_history':{'type':'variable','value':'messages'}},
                        'input':['messages'],'output':['messages'],'transition':'END'}],'entry_point':'agent'}
    memory = MemorySaver()
    def build():
        return create_graph(client=auto.model_copy(), yaml_schema=yaml.safe_dump(schema), tools=[AskUserTool()], memory=memory)
    generated = []
    def invoke(self, messages, config=None, **kwargs):
        generated.append(messages)
        if messages[-1].type == 'tool':
            assert 'BLUE' in messages[-1].content
            if fail_after_resume:
                raise ValueError('Controlled resumed provider failure')
            return AIMessage(content='{"marker":"BLUE"}')
        return AIMessage(content='',tool_calls=[{'id':'ask-marker','name':'ask_user','args':{'questions':[
            {'header':'Marker','question':'Which marker?','options':[{'label':'RED'},{'label':'BLUE'}]}]}}])
    with patch.object(ChatOpenAI, 'invoke', invoke):
        graph = build()
        graph.invoke({'messages':[HumanMessage(content='Hi')]}, config)
        paused = graph.get_state(config)
        interrupt = paused.tasks[0].interrupts[0].value
        original = interrupt['_pending_messages'][0]['data']['response_metadata'][PIN]
        assert len(requests) == 1
        # The interrupted node never returned _auto_routing; the existing
        # original tool-call message is the durable source of the binding.
        assert not paused.values.get('_auto_routing')
        resumed = build()
        if fail_after_resume:
            with pytest.raises(ValueError, match='Controlled resumed provider failure'):
                resumed.invoke({'hitl_resume':True,'hitl_action':'answer','hitl_value':{'answers':{'q0':'BLUE'}}}, config)
        else:
            resumed.invoke({'hitl_resume':True,'hitl_action':'answer','hitl_value':{'answers':{'q0':'BLUE'}}}, config)
            final = resumed.get_state(config).values['_auto_routing'][original['scope_id']]
            assert final['invocation_id'] == original['invocation_id']
            assert final['config'] == original['config']
        assert len(requests) == 1
        assert len(generated) == 2
