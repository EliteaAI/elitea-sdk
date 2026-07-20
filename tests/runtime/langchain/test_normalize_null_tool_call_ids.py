"""Tests for normalize_null_tool_call_ids (issue #5750).

LangChain's default_tool_parser / langchain_anthropic's extract_tool_calls
set a tool_call's "id" to None (never absent) when the raw provider payload
lacks a usable id. Elitea's `tool_call.get('id', '')` pattern only guards an
ABSENT key, so the None flows into ToolMessage(tool_call_id=None), which
pydantic rejects. normalize_null_tool_call_ids repairs this before any
ToolMessage gets built.
"""
from langchain_core.messages import AIMessage

from elitea_sdk.runtime.langchain.utils import normalize_null_tool_call_ids


def test_anthropic_content_block_id_synced_with_tool_call_id():
    """List-content (Anthropic-shape) message with a null-id tool_use block
    and a matching null-id tool_calls entry: both must be patched to the
    SAME synthesized value, or langchain_anthropic's id-matching logic would
    treat them as unrelated and emit a duplicate tool_use block."""
    message = AIMessage(
        content=[{'type': 'tool_use', 'id': None, 'name': 'search', 'input': {'q': 'x'}}],
        tool_calls=[{'name': 'search', 'args': {'q': 'x'}, 'id': None, 'type': 'tool_call'}],
    )

    normalize_null_tool_call_ids(message)

    new_id = message.tool_calls[0]['id']
    assert new_id and new_id.startswith('synth_')
    assert message.content[0]['id'] == new_id


def test_positional_pairing_skips_non_tool_use_blocks():
    """A leading `thinking` block must not be mistaken for the tool_use block
    it precedes — pairing is positional among tool_use-typed blocks only,
    matching how extract_tool_calls originally built tool_calls from content."""
    message = AIMessage(
        content=[
            {'type': 'thinking', 'thinking': 'reasoning about it...'},
            {'type': 'tool_use', 'id': None, 'name': 'search', 'input': {'q': 'x'}},
        ],
        tool_calls=[{'name': 'search', 'args': {'q': 'x'}, 'id': None, 'type': 'tool_call'}],
    )

    normalize_null_tool_call_ids(message)

    new_id = message.tool_calls[0]['id']
    assert new_id and new_id.startswith('synth_')
    assert message.content[1]['id'] == new_id
    # The thinking block has no id field to begin with and must stay untouched.
    assert 'id' not in message.content[0]


def test_mixed_batch_only_synthesizes_the_null_id():
    """A batch of one real id + one null id: the real id must survive
    unchanged, only the null one gets synthesized, and the two stay distinct
    (never a shared placeholder that would collide in id-keyed dispatch)."""
    message = AIMessage(
        content='',
        tool_calls=[
            {'name': 'tool_a', 'args': {}, 'id': 'call-real-1', 'type': 'tool_call'},
            {'name': 'tool_b', 'args': {}, 'id': None, 'type': 'tool_call'},
        ],
    )

    normalize_null_tool_call_ids(message)

    assert message.tool_calls[0]['id'] == 'call-real-1'
    new_id = message.tool_calls[1]['id']
    assert new_id and new_id.startswith('synth_')
    assert new_id != 'call-real-1'


def test_plain_string_content_openai_style_no_content_block_sync_attempted():
    """OpenAI-style completions carry plain string content, not a list of
    content blocks. The helper must patch tool_calls[0]['id'] without
    attempting (or erroring on) any content-block synchronization."""
    message = AIMessage(
        content='',
        tool_calls=[{'name': 'tool_a', 'args': {}, 'id': None, 'type': 'tool_call'}],
    )

    normalize_null_tool_call_ids(message)

    new_id = message.tool_calls[0]['id']
    assert new_id and new_id.startswith('synth_')
    assert message.content == ''


def test_noop_when_no_tool_calls():
    message = AIMessage(content='just text, no tool calls')
    result = normalize_null_tool_call_ids(message)
    assert result is message
    assert result.tool_calls == []


def test_repeated_calls_synthesize_distinct_ids():
    """Two separately-repaired null-id calls in one turn must never receive
    the same synthesized id — that would collide in id-keyed dispatch maps
    and misroute independent per-call HITL decisions."""
    message = AIMessage(
        content='',
        tool_calls=[
            {'name': 'tool_a', 'args': {}, 'id': None, 'type': 'tool_call'},
            {'name': 'tool_b', 'args': {}, 'id': None, 'type': 'tool_call'},
        ],
    )

    normalize_null_tool_call_ids(message)

    ids = [tc['id'] for tc in message.tool_calls]
    assert all(i and i.startswith('synth_') for i in ids)
    assert len(set(ids)) == 2
