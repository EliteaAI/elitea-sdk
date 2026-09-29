"""Serialize completed native history with actual installed provider adapters."""
import copy
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from elitea_sdk.runtime.clients.routing_history import completed_native_keys, adapt_completed_history


def history():
    return [HumanMessage(content='Read the requirement'), AIMessage(content=[
        {'type':'thinking','thinking':'opaque','signature':'fixture-signature'},
        {'type':'redacted_thinking','data':'fixture-redaction'},
        {'type':'text','text':'Reading the requirement.'}],
        tool_calls=[{'id':'read1','name':'read_requirement','args':{}}]),
        ToolMessage(content='Durability is mandatory.',tool_call_id='read1'),
        AIMessage(content='Architecture overview'), HumanMessage(content='Now implement crash recovery')]


@pytest.mark.parametrize('model', ['claude-sonnet-4-6','claude-haiku-4-5'])
def test_completed_anthropic_blocks_pass_unchanged_to_anthropic_target(model):
    messages=history();snapshot=copy.deepcopy(messages)
    native=ChatAnthropic(model=model,api_key='fixture',max_tokens=8000)
    binding={'invocation_id':'new','completed_native_keys':completed_native_keys(messages)}
    adapted=adapt_completed_history(native,messages,binding)
    assert adapted[1] is messages[1]
    payload=native._get_request_payload(adapted)
    assert payload['messages'][1]['content'][:2]==messages[1].content[:2]
    assert payload['messages'][1]['content'][-1]['id']=='read1'
    assert payload['messages'][2]['content'][0]['tool_use_id']=='read1'
    assert messages==snapshot


def test_completed_anthropic_to_chat_completions_keeps_visible_artifact_and_tool_pairs():
    messages=history();snapshot=copy.deepcopy(messages)
    native=ChatOpenAI(model='gpt-5.4',api_key='fixture',max_tokens=8000)
    binding={'invocation_id':'new','completed_native_keys':completed_native_keys(messages)}
    adapted=adapt_completed_history(native,messages,binding)
    payload=native._get_request_payload(adapted)
    assert payload['messages'][1]['content']==[{'type':'text','text':'Reading the requirement.'}]
    assert payload['messages'][1]['tool_calls'][0]['id']=='read1'
    assert payload['messages'][2]['tool_call_id']=='read1'
    assert payload['messages'][3]['content']=='Architecture overview'
    assert messages==snapshot


def test_openai_reasoning_bridge_does_not_rewrite_current_run_native_blocks():
    old=AIMessage(content=[{'type':'reasoning','id':'rs_old','summary':[]},{'type':'output_text','text':'Visible plan'}])
    messages=[HumanMessage(content='Plan'),old,HumanMessage(content='Implement')]
    native=ChatAnthropic(model='claude-sonnet-4-6',api_key='fixture',max_tokens=8000)
    binding={'invocation_id':'new','completed_native_keys':completed_native_keys(messages)}
    adapted=adapt_completed_history(native,messages,binding)
    assert native._get_request_payload(adapted)['messages'][1]['content']==[{'type':'text','text':'Visible plan'}]
    current=old.model_copy(update={'response_metadata':{'elitea_routing':{'invocation_id':'new'}}})
    assert adapt_completed_history(native,[current],binding)[0] is current
    assert adapt_completed_history(native,[old],{'invocation_id':'new','completed_native_keys':[]})[0] is old


def test_unfinished_or_unmatched_tool_batch_cannot_become_new_auto_task():
    with pytest.raises(ValueError,match='unfinished tool batch'):
        completed_native_keys([history()[0],history()[1],history()[-1]])
    with pytest.raises(ValueError,match='unmatched tool result'):
        completed_native_keys([HumanMessage(content='old'),history()[2],history()[-1]])
