"""Regression tests for issue #5056.

No-LLM pipelines whose terminal node writes only to a named output variable
(e.g. ``final_summary``) never append to the ``messages`` channel. Before the
fix the runner's message-scan yielded ``None`` and the user saw the unhelpful
"Assistant run has been completed, but output is None" sentinel. The
``extract_state_fallback_output`` helper now derives a meaningful output from
the terminal state values.
"""

from elitea_sdk.runtime.langchain.constants import ELITEA_RS, PRINTER_NODE_RS
from elitea_sdk.runtime.langchain.langraph_agent import (
    collect_terminal_output_variables,
    extract_state_fallback_output,
    extract_terminal_state_output,
)

def test_returns_single_named_output_variable():
    # The #5056 repro: a non-LLM node writes only to its Output field.
    state = {
        'messages': [],
        'input': 'summarize this',
        'final_summary': 'This is the summarized result.',
    }
    assert extract_state_fallback_output(state) == 'This is the summarized result.'


def test_returns_last_non_empty_state_value():
    state = {
        'messages': [],
        'router_output': 'route-a',
        'final_summary': 'first written',
        'reverse_summary': 'last written',
    }
    assert extract_state_fallback_output(state) == 'last written'


def test_ignores_internal_state_keys():
    state = {
        'messages': [],
        'output': 'should-be-ignored',
        'input': 'should-be-ignored',
        'chat_history': ['should-be-ignored'],
        'thread_id': 'abc-123',
        'execution_finished': True,
        'context_info': {'tokens': 10},
        'state_types': {'final_summary': 'str'},
        'hitl_decisions': {},
        'hitl_interrupt': None,
        ELITEA_RS: 'elitea-internal',
        PRINTER_NODE_RS: 'printer-internal',
        'final_summary': 'the only real output',
    }
    assert extract_state_fallback_output(state) == 'the only real output'


def test_returns_none_when_no_usable_candidate():
    # Only internal keys / empty values -> sentinel behaviour preserved upstream.
    state = {
        'messages': [],
        'output': None,
        'input': 'just a prompt',
        'final_summary': '',
        'whitespace_only': '   ',
    }
    assert extract_state_fallback_output(state) is None


def test_normalizes_claude_style_block_content():
    state = {
        'messages': [],
        'result': [
            {'type': 'text', 'text': 'block one '},
            {'type': 'thinking', 'thinking': 'ignored'},
            {'type': 'text', 'text': 'block two'},
        ],
    }
    assert extract_state_fallback_output(state) == 'block one block two'


def test_coerces_non_string_values():
    state = {
        'messages': [],
        'roll': 68,
    }
    assert extract_state_fallback_output(state) == '68'


def test_handles_non_dict_input_gracefully():
    assert extract_state_fallback_output(None) is None
    assert extract_state_fallback_output('not-a-dict') is None
    assert extract_state_fallback_output([1, 2, 3]) is None


# ── collect_terminal_output_variables ───────────────────────────────────────

def test_collects_var_from_non_llm_node_routing_to_end():
    schema = {
        'nodes': [
            {'id': 'classify', 'type': 'llm', 'output': ['category'], 'transition': 'transform'},
            {'id': 'transform', 'type': 'function', 'output': ['final_summary'], 'transition': 'END'},
        ]
    }
    assert collect_terminal_output_variables(schema) == ['final_summary']


def test_ignores_llm_and_control_flow_terminal_nodes():
    schema = {
        'nodes': [
            {'id': 'agent', 'type': 'llm', 'output': ['answer'], 'transition': 'END'},
            {'id': 'route', 'type': 'router', 'output': ['route_var'], 'transition': 'END'},
        ]
    }
    assert collect_terminal_output_variables(schema) == []


def test_ignores_non_llm_node_not_routing_to_end():
    schema = {
        'nodes': [
            {'id': 'transform', 'type': 'function', 'output': ['mid'], 'transition': 'printer'},
            {'id': 'printer', 'type': 'function', 'output': ['printed'], 'transition': 'END'},
        ]
    }
    # transform routes to the printer (not END); only printer's var is terminal
    assert collect_terminal_output_variables(schema) == ['printed']


def test_collects_branch_terminals_via_decision_and_condition():
    schema = {
        'nodes': [
            {
                'id': 'a',
                'type': 'function',
                'output': ['branch_a'],
                'decision': {'nodes': ['b'], 'default_output': 'END'},
            },
            {
                'id': 'b',
                'type': 'toolkit',
                'output': ['branch_b'],
                'condition': {'conditional_outputs': [{'node': 'END'}], 'default_output': 'c'},
            },
        ]
    }
    assert collect_terminal_output_variables(schema) == ['branch_a', 'branch_b']


def test_collect_excludes_messages_output_var():
    # A terminal node declaring 'messages' as its output keeps the result in the
    # messages channel as AIMessages -> must NOT be collected so the existing
    # last-AIMessage scan handles it instead of normalizing the message list.
    schema = {
        'nodes': [
            {'id': 'n', 'type': 'function', 'output': ['messages'], 'transition': 'END'},
        ]
    }
    assert collect_terminal_output_variables(schema) == []


def test_collect_excludes_other_internal_keys():
    schema = {
        'nodes': [
            {'id': 'n', 'type': 'code', 'output': ['output', 'messages', 'real_var'], 'transition': 'END'},
        ]
    }
    assert collect_terminal_output_variables(schema) == ['real_var']


def test_collect_dedupes_and_handles_non_dict_schema():
    schema = {
        'nodes': [
            {'id': 'a', 'type': 'function', 'output': ['shared'], 'transition': 'END'},
            {'id': 'b', 'type': 'code', 'output': ['shared', 'extra'], 'transition': 'END'},
        ]
    }
    assert collect_terminal_output_variables(schema) == ['shared', 'extra']
    assert collect_terminal_output_variables(None) == []
    assert collect_terminal_output_variables('nope') == []


# ── extract_terminal_state_output ───────────────────────────────────────────

def test_terminal_output_prefers_populated_var_over_messages():
    # Classifier scenario: messages channel holds an unrelated LLM message, but
    # the terminal non-LLM node populated 'final_summary'.
    state = {
        'messages': ['irrelevant classifier message'],
        'category': 'support',
        'final_summary': 'The real terminal output',
    }
    assert extract_terminal_state_output(state, ['final_summary']) == 'The real terminal output'


def test_terminal_output_picks_last_populated_branch_var():
    # Router/decision scenario: only the executed branch's var is populated, so
    # even with several declared terminals the result is unambiguous.
    state = {'branch_a': '', 'branch_b': 'executed branch result'}
    assert extract_terminal_state_output(state, ['branch_a', 'branch_b']) == 'executed branch result'


def test_terminal_output_single_populated_var_is_unambiguous():
    state = {'branch_a': 'only this ran', 'branch_b': None, 'branch_c': '   '}
    assert extract_terminal_state_output(
        state, ['branch_a', 'branch_b', 'branch_c']
    ) == 'only this ran'


def test_terminal_output_ambiguous_multiple_populated_uses_last_declared(caplog):
    # Genuinely ambiguous: two distinct terminal vars are populated (e.g. an
    # upstream node wrote a var another branch also declares as terminal). We
    # cannot know which is the real result, so best-effort = last declared.
    import logging

    state = {'branch_a': 'first', 'branch_b': 'second'}
    with caplog.at_level(logging.WARNING):
        result = extract_terminal_state_output(state, ['branch_a', 'branch_b'])
    assert result == 'second'
    assert any('Multiple terminal output variables' in r.message for r in caplog.records)


def test_terminal_output_same_var_mutated_is_not_ambiguous():
    # A single deduped var holding its final (mutated) value -> unambiguous.
    state = {'final_summary': 'final mutated value'}
    assert extract_terminal_state_output(state, ['final_summary']) == 'final mutated value'


def test_terminal_output_returns_none_when_unpopulated():
    state = {'final_summary': '   '}
    assert extract_terminal_state_output(state, ['final_summary']) is None
    assert extract_terminal_state_output(state, ['missing_var']) is None


def test_terminal_output_returns_none_without_declared_vars():
    state = {'final_summary': 'value'}
    assert extract_terminal_state_output(state, []) is None
    assert extract_terminal_state_output(state, None) is None


def test_terminal_output_handles_non_dict_state():
    assert extract_terminal_state_output(None, ['x']) is None
    assert extract_terminal_state_output('nope', ['x']) is None
