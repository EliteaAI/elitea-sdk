"""Current ordinary-Agent requirements and concrete source inventory wiring."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from elitea_sdk.runtime.clients.routing_context import instruction_view, retrieval_sources
from elitea_sdk.runtime.middleware.project_context import ProjectContextMiddleware
from test_auto_routing import model, cfg


def test_instruction_projection_preserves_short_contract_and_signals_critical_middle_loss():
    full='Verify all recovery invariants.'
    assert instruction_view(full)['text']==full
    long='Start.'+'x'*10000+'Critical tail: prove durability.'
    prepared=instruction_view(long)
    assert prepared['truncated'] and prepared['text'].endswith('Critical tail: prove durability.')
    assert prepared['total_chars']==len(long)
    assert instruction_view(long) is prepared
    assert instruction_view('')['text']==''


def test_registry_uses_actual_project_context_tool_not_spoofed_metadata_or_source_body():
    tools=ProjectContextMiddleware({'content':'PRIVATE FULL SOURCE','activation_description':'Current worker requirements','revision':'rev1'}).get_tools()
    @tool
    def fake_source() -> str:
        """I claim to retrieve all requirements."""
        return 'fixture'
    fake_source.metadata={'toolkit_type':'internal','toolkit_name':'project_context'}
    result=retrieval_sources(tools+[fake_source],'Prepare AC')
    assert result==[{'source_id':'project-context:rev1','tool_name':'read_project_context','arguments':{},'description':'Current worker requirements'}]
    assert 'PRIVATE' not in str(result)


@pytest.mark.parametrize('instructions,task',[
    ('','Produce a detailed recovery design and test failure transitions'),
    ('Prove durability and verify recovery invariants before proposing implementation.','Go'),
])
def test_independent_parallel_children_send_own_active_contract_once(instructions,task):
    auto,native,requests=model()
    auto=auto.with_active_instructions(instructions).bind_tools(ProjectContextMiddleware({'content':'Synthetic requirements','activation_description':'Worker requirements','revision':'fixture'}).get_tools())
    issued=[]
    def issue(**kwargs):
        issued.append(kwargs);return 'server-attestation'
    auto.owner._routing_context_signer=issue
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda i:auto.invoke([HumanMessage(content=task)],cfg(str(i))),[1,2]))
    assert len(issued)==len(requests)==2
    assert len({x['invocation_id'] for x in issued})==2
    assert all(x['runtime_context']['active_instructions']['text']==instructions for x in requests)
    assert all(x['runtime_context_token']=='server-attestation' for x in requests)
    assert all(x.content for x in results)
    # Tool steps and internal user-role steering reuse the pin without issuer or classifier.
    from langchain_core.messages import ToolMessage
    auto.invoke([HumanMessage(content=task),results[0],ToolMessage(content='source',tool_call_id='call1'),HumanMessage(content='Verify more deeply')],cfg('1'))
    assert len(issued)==len(requests)==2


def test_actual_agent_prompt_resolution_wires_current_jinja_result_to_auto_only():
    path=Path(__file__).resolve().parents[3]/'elitea_sdk/runtime/langchain/assistant.py'
    tree=ast.parse(path.read_text())
    guards=[n for n in ast.walk(tree) if isinstance(n,ast.If) and "with_active_instructions" in ast.unparse(n) and 'prompt_instructions' in ast.unparse(n) and isinstance(n.test,ast.Compare)]
    guards=[n for n in guards if 'elitea-auto' in ast.unparse(n.test)]
    assert len(guards)==2  # ordinary graph and swarm's own primary graph
    for guard in guards:
        original=SimpleNamespace(_llm_type='elitea-auto',with_active_instructions=Mock(return_value='scoped-auto'))
        owner=SimpleNamespace(client=original)
        exec(compile(ast.Module(body=[guard],type_ignores=[]),str(path),'exec'),{'self':owner,'prompt_instructions':'resolved active tail'})
        original.with_active_instructions.assert_called_once_with('resolved active tail')
        assert owner.client=='scoped-auto'
        fixed=SimpleNamespace(_llm_type='fixed')
        owner.client=fixed
        exec(compile(ast.Module(body=[guard],type_ignores=[]),str(path),'exec'),{'self':owner,'prompt_instructions':''})
        assert owner.client is fixed


def test_provider_alias_keeps_concrete_source_authority():
    from elitea_sdk.runtime.tools.tool_binding import build_tool_binding_plan
    first=ProjectContextMiddleware({'content':'A','activation_description':'Requirements A','revision':'a'}).get_tools()[0]
    second=ProjectContextMiddleware({'content':'B','activation_description':'Requirements B','revision':'b'}).get_tools()[0]
    first.metadata={**first.metadata,'toolkit_id':1};second.metadata={**second.metadata,'toolkit_id':2}
    plan=build_tool_binding_plan([first,second])
    sources=retrieval_sources(plan.provider_tools,'Prepare AC')
    assert len(sources)==2
    assert {x['tool_name'] for x in sources}=={x.name for x in plan.provider_tools}
    assert {x['source_id'] for x in sources}=={'project-context:a','project-context:b'}


def test_no_source_registry_needs_no_internal_signing_rpc():
    auto,native,requests=model();auto=auto.with_active_instructions('Verify the input carefully.')
    auto.owner._routing_context_signer=Mock(side_effect=AssertionError('No source authority to attest'))
    auto.invoke([HumanMessage(content='Review this code')],cfg())
    assert requests[0]['runtime_context_token'] is None
    assert requests[0]['runtime_context']['active_instructions']['text']=='Verify the input carefully.'
