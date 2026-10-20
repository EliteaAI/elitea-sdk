"""#6886: input guardrail detector, checked against the vectors shared with elitea_core.

The vectors file is the twin contract: pylon_main runs the same file against its own
copy of the detector, so a rule that drifts in either repo fails one of the two suites.
"""

import json
import re
import time
from pathlib import Path

import pytest

from elitea_sdk.runtime.guardrails import (
    Segment,
    audit_actions,
    effective_action,
    inspect,
    placeholder,
    policy_from_dict,
    summary,
)
from elitea_sdk.runtime.guardrails import input_detector

VECTORS = json.loads(
    (Path(input_detector.__file__).parent / 'input_guardrail_vectors.json').read_text()
)
ALL_ON = {'enabled': True, 'categories': {'credentials': 'block', 'personal_data': 'block'}}


def _text(vector):
    return ''.join(vector['parts'])


def _turn(text, kind='current_turn'):
    return [Segment(kind, 0, text)]


@pytest.mark.parametrize('vector', VECTORS['positive'], ids=lambda v: v['rule_id'])
def test_positive_vector_is_detected_by_its_rule(vector):
    result = inspect(_turn(_text(vector)), policy_from_dict(ALL_ON))

    assert [(d.category, d.rule_id) for d in result.detections] == [(vector['category'], vector['rule_id'])]
    assert result.action == 'block'


@pytest.mark.parametrize('vector', VECTORS['negative'], ids=lambda v: _text(v)[:40])
def test_negative_vector_is_not_detected(vector):
    result = inspect(_turn(_text(vector)), policy_from_dict(ALL_ON))

    assert result.detections == ()
    assert result.action == 'none'


def test_rule_set_matches_the_vectors_file():
    """A new rule must land with a vector, or the twin has nothing to check it against."""
    assert {r.rule_id for r in input_detector.RULES} == {v['rule_id'] for v in VECTORS['positive']}


def test_disabled_policy_inspects_nothing():
    text = _text(VECTORS['positive'][0])

    assert inspect(_turn(text), policy_from_dict({'enabled': False})).detections == ()
    assert inspect(_turn(text), policy_from_dict(None)).action == 'none'


def test_defaults_block_credentials_and_leave_personal_data_off():
    policy = policy_from_dict({'enabled': True})
    email = next(v for v in VECTORS['positive'] if v['rule_id'] == 'email')
    token = next(v for v in VECTORS['positive'] if v['rule_id'] == 'github_token')

    assert inspect(_turn(_text(email)), policy).action == 'none'
    assert inspect(_turn(_text(token)), policy).action == 'block'


def test_redact_replaces_only_the_value_and_keeps_the_key_name():
    vector = next(v for v in VECTORS['positive'] if v['rule_id'] == 'password_assignment')
    text = _text(vector)
    policy = policy_from_dict({'enabled': True, 'categories': {'credentials': 'redact'}})

    result = inspect(_turn(text), policy)

    redacted = result.redacted[('current_turn', 0)]
    assert redacted == f"db password={placeholder('credentials')}"
    assert vector['parts'][-1] not in redacted


def test_redacted_text_is_clean_on_a_second_pass():
    """Checkpoint C sees what B already redacted; the placeholder must not re-trigger."""
    policy = policy_from_dict({'enabled': True, 'categories': {'credentials': 'redact', 'personal_data': 'redact'}})
    text = ' and '.join(_text(v) for v in VECTORS['positive'])

    first = inspect(_turn(text), policy)
    second = inspect(_turn(first.redacted[('current_turn', 0)]), policy)

    assert len(first.detections) == len(VECTORS['positive'])
    assert second.detections == ()


def test_most_severe_action_wins_across_categories():
    policy = policy_from_dict({'enabled': True, 'categories': {'credentials': 'warn', 'personal_data': 'redact'}})
    token = _text(next(v for v in VECTORS['positive'] if v['rule_id'] == 'github_token'))
    email = _text(next(v for v in VECTORS['positive'] if v['rule_id'] == 'email'))

    result = inspect([Segment('current_turn', 0, token), Segment('history', 0, email)], policy)

    assert result.action == 'redact'
    # warn does not rewrite; only the redact-category segment changed
    assert list(result.redacted) == [('history', 0)]


def test_allow_pattern_suppresses_a_detection():
    token = _text(next(v for v in VECTORS['positive'] if v['rule_id'] == 'aws_access_key'))
    policy = policy_from_dict({'enabled': True, 'allow_patterns': ['FAKE0TEST0VECTOR$']})

    assert inspect(_turn(token), policy).detections == ()


def test_invalid_allow_pattern_is_dropped_not_raised():
    policy = policy_from_dict({'enabled': True, 'allow_patterns': ['(unclosed', 'ok']})

    assert [p.pattern for p in policy.allow_patterns] == ['ok']


def test_overlapping_rules_report_one_detection():
    """An Authorization header carrying a JWT is one value, not two findings."""
    jwt = _text(next(v for v in VECTORS['positive'] if v['rule_id'] == 'jwt')).split(' ', 1)[1]

    result = inspect(_turn(f'Authorization: Bearer {jwt}'), policy_from_dict(ALL_ON))

    assert len(result.detections) == 1


def test_size_cap_is_an_error_and_follows_on_error():
    text = 'a' * 50
    open_policy = policy_from_dict({'enabled': True, 'max_inspect_chars': 10})
    closed_policy = policy_from_dict({'enabled': True, 'max_inspect_chars': 10, 'on_error': 'fail_closed'})

    result = inspect(_turn(text), open_policy)

    assert result.error == 'oversize'
    assert effective_action(result, open_policy) == 'none'
    assert effective_action(inspect(_turn(text), closed_policy), closed_policy) == 'block'


def test_timeout_is_an_error(monkeypatch):
    ticks = iter([0.0] + [10.0] * 100)
    monkeypatch.setattr(input_detector.time, 'monotonic', lambda: next(ticks))

    result = inspect(_turn('hello'), policy_from_dict({'enabled': True}))

    assert result.error == 'timeout'


def test_internal_error_never_raises_and_carries_no_input(monkeypatch):
    secret = _text(VECTORS['positive'][0])

    def boom(*_):
        raise ValueError(f'cannot parse {secret}')

    monkeypatch.setattr(input_detector, '_find', boom)

    result = inspect(_turn(secret), policy_from_dict({'enabled': True}))

    assert result.error == 'internal'
    assert secret not in repr(result)


def test_detected_value_never_appears_in_the_summary():
    policy = policy_from_dict(ALL_ON)
    for vector in VECTORS['positive']:
        text = _text(vector)
        result = inspect(_turn(text), policy)
        payload = json.dumps(summary(result, policy)) + repr(result.detections)
        for d in result.detections:
            assert text[d.start:d.end] not in payload


def test_audit_action_format():
    token = _text(next(v for v in VECTORS['positive'] if v['rule_id'] == 'github_token'))
    policy = policy_from_dict({'enabled': True})

    entries = audit_actions(inspect(_turn(token, 'history'), policy), policy)

    assert entries == [{'action': 'input:credentials.github_token:block', 'segment_kind': 'history', 'is_error': False}]
    assert all(len(e['action']) <= 512 for e in entries)


def test_audit_action_for_an_error():
    policy = policy_from_dict({'enabled': True, 'max_inspect_chars': 1})

    entries = audit_actions(inspect(_turn('ab'), policy), policy)

    assert entries == [{'action': 'input:error.oversize:fail_open', 'segment_kind': None, 'is_error': True}]


def test_benign_prose_and_code_corpus_has_no_credential_hits():
    corpus = [
        'Please summarise the attached design doc and list the open questions.',
        'def connect(host, port=5432, password=None):\n    return psycopg.connect(host=host)',
        'export DATABASE_URL=postgres://localhost:5432/db',
        'curl -H "Content-Type: application/json" https://api.example.com/v1/items',
        'The deploy failed with exit code 137 at 2026-10-20T10:15:00Z.',
        'SELECT id, token_count FROM usage_event WHERE project_id = 1;',
        'Our ASIA-Pacific team uses tokens in the CI config; rotate them monthly.',
        'Hash: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
        'base64 of hello is aGVsbG8=',
        'Ticket #6886: input guardrail for credentials',
    ]
    policy = policy_from_dict({'enabled': True})
    hits = [text for text in corpus if inspect(_turn(text), policy).detections]

    assert hits == []


def test_100kb_input_inspects_well_under_budget():
    text = ('lorem ipsum dolor sit amet, password reset link sent. ' * 2000)[:100_000]
    policy = policy_from_dict({**ALL_ON, 'timeout_ms': 5000})

    started = time.monotonic()
    result = inspect(_turn(text), policy)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert result.error is None
    assert elapsed_ms < 50 * 4  # 50 ms target; slack for a loaded CI host


def test_placeholder_format():
    assert re.fullmatch(r'\[REDACTED:credentials\]', placeholder('credentials'))
