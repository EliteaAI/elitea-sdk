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
