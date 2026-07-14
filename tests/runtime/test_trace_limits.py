import json

from elitea_sdk.runtime.utils.trace_limits import cap_trace_json, cap_trace_text


def test_cap_trace_text_marks_and_bounds_large_output():
    result = cap_trace_text('x' * 300, limit=100)

    assert len(result) == 100
    assert 'trace field truncated' in result
    assert '300 chars' in result


def test_cap_trace_text_keeps_normal_output_unchanged():
    assert cap_trace_text('normal', limit=100) == 'normal'


def test_cap_trace_json_keeps_small_structure_and_bounds_large_structure():
    assert cap_trace_json({'path': 'small'}, limit=100) == {'path': 'small'}

    result = cap_trace_json({'content': 'x' * 300}, limit=120)
    assert result['_trace_truncated'] is True
    assert result['original_characters'] > 300
    assert len(json.dumps(result, ensure_ascii=False)) <= 120
