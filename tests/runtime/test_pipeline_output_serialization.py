"""A pipeline's final output is the answer the user reads (#6532).

`normalize_message_content` feeds `result_with_state["output"]`, so a terminal node
writing a dict output variable used to surface as a Python repr — the ticket's own
symptom on a path with no LLM in it.
"""

import json

from elitea_sdk.runtime.langchain.langraph_agent import normalize_message_content


def test_dict_output_is_json_not_a_repr():
    content = {"number": 535, "title": "Bug in *login*", "state": "open", "assignee": None}

    result = normalize_message_content(content)

    assert json.loads(result) == content
    assert "'number'" not in result


def test_list_output_survives():
    content = [{"key": "PROJ-1"}, {"key": "PROJ-2"}]

    assert json.loads(normalize_message_content(content)) == content


def test_plain_text_is_untouched():
    assert normalize_message_content("Here is the answer") == "Here is the answer"


def test_content_blocks_still_join_to_text():
    blocks = [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]

    assert normalize_message_content(blocks) == "part one part two"


def test_records_borrowing_a_block_type_are_not_flattened():
    # A SharePoint page reader emits exactly this shape. The loop matched
    # type == 'text' and appended block.get('text', ''), so text_parts was [''],
    # truthy, and the answer came back empty.
    content = [
        {"type": "text", "content": "page text"},
        {"type": "image", "description": "a chart", "src": "chart.png"},
    ]

    assert json.loads(normalize_message_content(content)) == content


def test_search_hits_are_not_concatenated_into_one_string():
    content = [{"text": "match", "score": 0.9}, {"text": "second", "score": 0.5}]

    assert json.loads(normalize_message_content(content)) == content


def test_bare_text_chunks_still_join():
    assert normalize_message_content([{"text": "a"}, {"text": "b"}]) == "ab"
