"""Tool results are bounded at the tool boundary, behind a feature flag (#6140)."""
import base64
import os
import time

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from elitea_sdk.runtime.tool_outcome import ToolResultStatus, outcome_sink
from elitea_sdk.runtime.tool_result_bounds import bound_and_record, bound_tool_result
from elitea_sdk.runtime.utils.trace_limits import (
    TOOL_RESULT_MARKER_KEY,
    TOOL_RESULT_MAX_CHARS,
    configure_tool_result_limits,
    estimate_chars,
    looks_like_encoded_blob,
    resolve_tool_result_limit,
    tool_result_bounding_enabled,
)

LIMIT = TOOL_RESULT_MAX_CHARS


@pytest.fixture(autouse=True)
def default_limits():
    """Every test starts from the shipped defaults and restores them afterwards."""
    configure_tool_result_limits(enabled=True, limit=LIMIT, per_toolkit={})
    yield
    configure_tool_result_limits(enabled=True, limit=LIMIT, per_toolkit={})


def _prose(chars):
    return ('lorem ipsum dolor sit amet ' * (chars // 27 + 1))[:chars]


# --- text -------------------------------------------------------------------

def test_oversized_text_is_cut_to_the_limit_with_a_marker():
    value, original = bound_tool_result(_prose(LIMIT * 2), 'big_reader')
    assert original == LIMIT * 2
    assert len(value) == LIMIT
    assert 'tool result truncated' in value
    assert 'big_reader' in value


def test_text_at_the_limit_is_returned_byte_identical():
    # A read tool that already capped itself at 200K must not be cut again.
    exact = _prose(LIMIT)
    value, original = bound_tool_result(exact, 'read_file')
    assert original is None
    assert value == exact


# --- structures -------------------------------------------------------------

def test_oversized_dict_keeps_its_type_and_its_small_keys():
    result = {'status': 'ok', 'rows': 42, 'table': _prose(LIMIT * 3)}
    value, original = bound_tool_result(result, 'sandbox_exec', 'sandbox')

    assert isinstance(value, dict)
    # Small sibling keys are what downstream pipeline nodes index into.
    assert value['status'] == 'ok'
    assert value['rows'] == 42
    assert estimate_chars(value) <= LIMIT
    marker = value[TOOL_RESULT_MARKER_KEY]
    assert marker['truncated'] is True
    assert marker['original_characters'] == original


def test_oversized_list_keeps_its_type_and_carries_the_marker_as_last_element():
    result = [{'chunk': _prose(LIMIT)}, {'chunk': _prose(LIMIT)}]
    value, _ = bound_tool_result(result, 'search')

    assert isinstance(value, list)
    assert value[-1][TOOL_RESULT_MARKER_KEY]['truncated'] is True
    assert estimate_chars(value) <= LIMIT


def test_small_structure_is_the_same_object_untouched():
    result = {'a': 'b', 'nested': [1, 2, 3]}
    value, original = bound_tool_result(result, 'small')
    assert original is None
    assert value is result
    assert TOOL_RESULT_MARKER_KEY not in value


# --- base64 -----------------------------------------------------------------

def test_encoded_blob_leaf_is_dropped_whole_not_cut():
    blob = base64.b64encode(os.urandom(LIMIT)).decode()
    value, _ = bound_tool_result({'image': blob, 'name': 'chart.png'}, 'plotter')

    assert 'binary content dropped' in value['image']
    # A partially-cut base64 string is a corrupt file with no error, so none survives.
    assert blob[:64] not in value['image']
    assert value['name'] == 'chart.png'


def test_long_prose_is_not_mistaken_for_an_encoded_blob():
    assert looks_like_encoded_blob(_prose(LIMIT)) is False
    assert looks_like_encoded_blob('x' * LIMIT) is False
    assert looks_like_encoded_blob(base64.b64encode(os.urandom(4096)).decode()) is True
    assert looks_like_encoded_blob('data:image/png;base64,' + 'A1b2' * 512) is True


# --- idempotence ------------------------------------------------------------

def test_bounding_twice_produces_exactly_one_marker():
    # The middleware bounds first and the tool node backstops it; both may run.
    once, _ = bound_tool_result({'table': _prose(LIMIT * 2)}, 'tool')
    twice, original = bound_tool_result(once, 'tool')
    assert original is None
    assert twice is once
    assert sum(1 for key in twice if key == TOOL_RESULT_MARKER_KEY) == 1


def test_bounding_text_twice_leaves_the_first_cut_alone():
    once, _ = bound_tool_result(_prose(LIMIT * 2), 'tool')
    twice, original = bound_tool_result(once, 'tool')
    assert original is None
    assert twice == once
    assert once.count('tool result truncated') == 1


# --- flag -------------------------------------------------------------------

def test_flag_off_passes_any_size_through_byte_identical():
    configure_tool_result_limits(enabled=False)
    huge = {'table': _prose(LIMIT * 10)}
    value, original = bound_tool_result(huge, 'tool')
    assert original is None
    assert value is huge
    assert TOOL_RESULT_MARKER_KEY not in value


def test_flag_off_still_reports_what_would_have_been_cut(caplog):
    configure_tool_result_limits(enabled=False)
    with caplog.at_level('INFO'):
        bound_tool_result({'table': _prose(LIMIT * 2)}, 'noisy_tool')
    assert any('truncation disabled' in record.getMessage()
               for record in caplog.records)


# --- per-toolkit override ---------------------------------------------------

def test_per_toolkit_override_wins_over_the_global_limit():
    configure_tool_result_limits(enabled=True, limit=LIMIT, per_toolkit={'pgvector': LIMIT * 4})
    assert resolve_tool_result_limit('pgvector') == LIMIT * 4
    assert resolve_tool_result_limit('jira') == LIMIT

    payload = {'table': _prose(LIMIT * 2)}
    value, original = bound_tool_result(payload, 'search', 'pgvector')
    assert original is None and value is payload


def test_removing_an_override_reverts_to_the_global_limit():
    configure_tool_result_limits(per_toolkit={'pgvector': LIMIT * 4})
    configure_tool_result_limits(per_toolkit={})
    assert resolve_tool_result_limit('pgvector') == LIMIT


def test_invalid_limit_values_fall_back_to_the_default():
    configure_tool_result_limits(limit='not-a-number', per_toolkit={'jira': 0, '': 500})
    assert resolve_tool_result_limit() == LIMIT
    assert resolve_tool_result_limit('jira') == LIMIT


# --- control flow -----------------------------------------------------------

def test_deferred_hitl_sentinel_passes_through_untouched():
    # Collapsing this would stringify the sentinel and silently lose the pause.
    sentinel = {'__hitl_deferred__': {'payload': _prose(LIMIT * 2)}}
    value, original = bound_tool_result(sentinel, 'child_agent')
    assert original is None
    assert value is sentinel
    assert TOOL_RESULT_MARKER_KEY not in value


def test_tool_message_and_command_are_never_inspected():
    message = ToolMessage(content=_prose(LIMIT * 2), tool_call_id='1')
    command = Command(update={'messages': [_prose(LIMIT * 2)]})
    for control in (message, command):
        value, original = bound_tool_result(control, 'tool')
        assert original is None and value is control


def test_exceptions_pass_through():
    error = ValueError(_prose(LIMIT * 2))
    value, original = bound_tool_result(error, 'tool')
    assert original is None and value is error


# --- content_and_artifact ---------------------------------------------------

def test_two_tuple_bounds_the_content_and_leaves_the_artifact_alone():
    artifact = {'raw': _prose(LIMIT * 3)}
    value, original = bound_tool_result((_prose(LIMIT * 2), artifact), 'exporter')

    content, returned_artifact = value
    assert len(content) == LIMIT
    assert original == LIMIT * 2
    # The artifact half is storage-bound raw payload, not model context.
    assert returned_artifact is artifact
    assert len(returned_artifact['raw']) == LIMIT * 3


def test_small_two_tuple_is_unchanged():
    pair = ('ok', {'raw': 'small'})
    value, original = bound_tool_result(pair, 'exporter')
    assert original is None
    assert value == pair


# --- outcome envelope -------------------------------------------------------

def test_truncation_records_a_truncated_outcome():
    with outcome_sink() as recorded:
        bound_and_record({'table': _prose(LIMIT * 2)}, 'big_tool', 'sandbox')

    assert len(recorded) == 1
    outcome = recorded[0]
    assert outcome.status is ToolResultStatus.TRUNCATED
    assert outcome.truncated is True
    assert outcome.original_size > LIMIT * 2
    assert outcome.tool_name == 'big_tool'
    assert outcome.toolkit_type == 'sandbox'


def test_no_outcome_is_recorded_when_nothing_is_cut():
    with outcome_sink() as recorded:
        bound_and_record({'a': 'b'}, 'small_tool')
    assert recorded == []


def test_bounding_failure_returns_the_original_result(monkeypatch):
    # A guard that can raise is worse than no guard: it runs on every tool call.
    monkeypatch.setattr(
        'elitea_sdk.runtime.tool_result_bounds.bound_tool_result',
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')),
    )
    payload = {'table': _prose(LIMIT * 2)}
    assert bound_and_record(payload, 'tool') is payload


# --- cost -------------------------------------------------------------------

def test_estimate_chars_aborts_early_instead_of_walking_everything():
    # The oversize check must not cost O(size) - that CPU on the event loop is the
    # very stall this issue is fixing.
    huge = {f'k{i}': 'x' * 1000 for i in range(100_000)}
    assert estimate_chars(huge, ceiling=LIMIT) > LIMIT
    assert estimate_chars({'a': 'x' * 10}, ceiling=LIMIT) < 100


def test_estimate_chars_survives_deeply_nested_results():
    deep = current = {}
    for _ in range(20_000):
        current['next'] = {}
        current = current['next']
    assert estimate_chars(deep) > 0


# --- through the middleware, the point every wrapped tool passes -------------

def _wrapped(tool):
    from elitea_sdk.runtime.middleware.strategies import LoggingStrategy
    from elitea_sdk.runtime.middleware.tool_exception_handler import (
        ToolExceptionHandlerMiddleware,
    )
    return ToolExceptionHandlerMiddleware(strategies=[LoggingStrategy()]).wrap_tool(tool)


def test_middleware_bounds_a_synchronous_tool_result():
    from langchain_core.tools import StructuredTool

    def _dump() -> str:
        return _prose(LIMIT * 2)

    tool = _wrapped(StructuredTool.from_function(func=_dump, name='dump', description='x'))
    result = tool.invoke({})
    assert len(result) == LIMIT
    assert 'tool result truncated' in result


@pytest.mark.asyncio
async def test_middleware_bounds_an_async_tool_result():
    from langchain_core.tools import StructuredTool

    async def _adump() -> str:
        return _prose(LIMIT * 2)

    tool = _wrapped(StructuredTool.from_function(coroutine=_adump, name='adump', description='x'))
    result = await tool.ainvoke({})
    assert len(result) == LIMIT


def test_middleware_leaves_a_small_result_alone():
    from langchain_core.tools import StructuredTool

    def _small() -> str:
        return 'fine'

    tool = _wrapped(StructuredTool.from_function(func=_small, name='small', description='x'))
    assert tool.invoke({}) == 'fine'


def test_truncation_does_not_flip_a_swarm_tool_message_to_error():
    """A truncated call still succeeded; marking it 'error' would fail a working tool."""
    from langchain_core.messages import AIMessage
    from langchain_core.tools import StructuredTool
    from langgraph.graph import MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    from elitea_sdk.runtime.middleware.tool_exception_handler import (
        swarm_awrap_tool_call,
        swarm_handle_tool_errors,
        swarm_wrap_tool_call,
    )

    def _dump() -> str:
        return _prose(LIMIT * 2)

    node = ToolNode(
        [_wrapped(StructuredTool.from_function(func=_dump, name='dump', description='x'))],
        handle_tool_errors=swarm_handle_tool_errors,
        wrap_tool_call=swarm_wrap_tool_call,
        awrap_tool_call=swarm_awrap_tool_call,
    )
    builder = StateGraph(MessagesState)
    builder.add_node('tools', node)
    builder.set_entry_point('tools')
    builder.set_finish_point('tools')

    call = AIMessage(content='', tool_calls=[{'name': 'dump', 'args': {}, 'id': 'c1'}])
    message = builder.compile().invoke({'messages': [call]})['messages'][-1]

    assert message.status == 'success'
    assert len(message.content) == LIMIT


def test_sandbox_pandas_result_keeps_every_structural_key():
    """The real production case: a pyodide/pandas job returning a huge CSV body.

    Pipeline nodes index into 'status' / 'error' / 'execution_info', so only the
    data body may shrink - the shape around it must survive verbatim.
    """
    rows = '\n'.join(f'{i},row-{i},value-{i * 3.14159}' for i in range(400_000))
    result = {
        'result': rows,
        'output': 'shape=(400000, 3)\n',
        'error': None,
        'status': 'success',
        'execution_info': {'duration_ms': 8123, 'packages': ['pandas']},
    }
    value, original = bound_tool_result(result, 'execute_code', 'sandbox')

    assert original > 10_000_000
    assert estimate_chars(value) <= LIMIT
    assert value['status'] == 'success'
    assert value['error'] is None
    assert value['output'] == 'shape=(400000, 3)\n'
    assert value['execution_info'] == {'duration_ms': 8123, 'packages': ['pandas']}
    assert value['result'].startswith('0,row-0,value-0.0')
    assert 'truncated' in value['result']


# --- Review round 1: non-text payloads and config-failure fallback ------------

@pytest.mark.parametrize('payload', [
    pytest.param({'status': 'ok', 'rows': list(range(100_000))}, id='long-int-list'),
    pytest.param({'status': 'ok', 'rows': [{'a': i, 'b': i * 2} for i in range(40_000)]},
                 id='array-of-records'),
    pytest.param({'status': 'ok', 'm': {str(i): i for i in range(60_000)}},
                 id='scalar-bag-dict'),
    pytest.param({'status': 'ok', 'd': {'e': {'f': [list(range(200)) for _ in range(2_000)]}}},
                 id='deeply-nested-lists'),
    pytest.param([{'a': 'z' * 100} for _ in range(5_000)], id='root-list-of-small-records'),
])
def test_non_text_payloads_actually_shrink(payload):
    """A result stamped as truncated must never still be oversized.

    Trimming only string/byte leaves left numeric lists, record arrays and scalar
    bags untouched - the payload stayed megabytes wide while carrying a marker that
    claimed it had been cut.
    """
    before = estimate_chars(payload)
    assert before > LIMIT, 'probe must start oversized'
    bounded, reported = bound_tool_result(payload, 'probe')
    assert estimate_chars(bounded) <= LIMIT
    assert reported == before


def test_bounding_a_non_text_payload_preserves_small_structural_keys():
    payload = {'status': 'ok', 'error': None, 'rows': [{'a': i, 'b': i * 2} for i in range(40_000)]}
    bounded, _ = bound_tool_result(payload, 'probe')
    assert bounded['status'] == 'ok'
    assert bounded['error'] is None
    assert isinstance(bounded['rows'], list)


def test_bounding_a_huge_non_text_payload_is_not_cpu_bound():
    """Trimming must not re-measure per candidate length: that reintroduces the
    CPU-bound serialization cost this whole feature exists to remove."""
    payload = {'status': 'ok', 'rows': list(range(2_000_000))}
    started = time.monotonic()
    bounded, _ = bound_tool_result(payload, 'probe')
    assert estimate_chars(bounded) <= LIMIT
    assert time.monotonic() - started < 30


def test_configure_rejects_a_non_mapping_without_partially_updating_state():
    """A malformed per_toolkit must not leave half the bounds applied and the old
    overrides still live."""
    configure_tool_result_limits(enabled=True, limit=123_456, per_toolkit={'pgvector': 400_000})
    with pytest.raises(TypeError):
        configure_tool_result_limits(enabled=False, limit=999, per_toolkit='garbage')
    assert tool_result_bounding_enabled() is True
    assert resolve_tool_result_limit() == 123_456
    assert resolve_tool_result_limit('pgvector') == 400_000
