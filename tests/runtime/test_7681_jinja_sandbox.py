"""Regression tests for issue #7681 — SSTI in the remaining Jinja2 render sites.

#5429 sandboxed only Assistant._resolve_jinja2_variables. These tests cover the
other author-controlled render sites:
- EvaluateTemplate.extract (router / conditional-edge conditions) -> clear error
- StateModifierNode.invoke (state_modifier node template)          -> clear error
- Jinja2TemplatedChatMessagesTemplate.format_messages               -> raw-text fallback

Payloads are the generic dunder-traversal class only; nothing executes commands.
"""

import base64
import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from elitea_sdk.runtime.langchain.chat_message_template import Jinja2TemplatedChatMessagesTemplate
from elitea_sdk.runtime.langchain.langraph_agent import StateModifierNode
from elitea_sdk.runtime.utils.evaluate import EvaluateTemplate

TRAVERSAL = "{{ ''.__class__.__mro__ }}"
SUBCLASSES = "{{ ().__class__.__base__.__subclasses__() }}"
# Attribute name built at render time through the registered filter + `attr`, then traversed.
# (Unchained, the sandbox returns undefined -> '' and leaks nothing; the chain raises.)
B64_CLASS = base64.b64encode(b"__class__").decode()
FILTER_BUILT = "{{ ('' | attr('" + B64_CLASS + "' | base64_to_string)).__mro__ }}"


def _modifier(template, input_variables=None, output="out"):
    return StateModifierNode(
        template=template,
        input_variables=input_variables or [],
        output_variables=[output],
    )


# --- AC-1 / AC-5: router and condition templates ---------------------------------

class TestEvaluateTemplateBlocksInjection:

    @pytest.mark.parametrize("payload", [TRAVERSAL, SUBCLASSES])
    def test_payload_rejected_with_clear_error(self, payload):
        with pytest.raises(Exception) as exc:
            EvaluateTemplate(payload, {}).extract()
        message = str(exc.value)
        assert "Template rejected" in message
        assert "<class" not in message
        assert "__mro__" not in message and "__subclasses__" not in message


# --- AC-2 / AC-5: state_modifier node -------------------------------------------

class TestStateModifierBlocksInjection:

    @pytest.mark.parametrize("payload", [TRAVERSAL, SUBCLASSES, FILTER_BUILT])
    def test_payload_rejected_and_state_not_written(self, payload):
        node = _modifier(payload)
        with pytest.raises(Exception) as exc:
            node.invoke({"out": ""})
        message = str(exc.value)
        assert "Template rejected" in message
        assert "<class" not in message


class TestStateModifierUndefinedValue:

    def test_undefined_value_gives_clear_error(self):
        node = _modifier("{{ missing.value }}")
        with pytest.raises(Exception) as exc:
            node.invoke({"out": ""})
        assert "referenced a value that is not available" in str(exc.value)

    def test_unsafe_chain_collapsing_to_undefined_gives_clear_error(self):
        node = _modifier("{% for s in ()|attr('__class__')|attr('__base__')|attr('__subclasses__')() %}x{% endfor %}")
        with pytest.raises(Exception) as exc:
            node.invoke({"out": ""})
        assert "<class" not in str(exc.value)


# --- AC-3 / AC-5: SDK chat-message template --------------------------------------

class TestChatMessageTemplateBlocksInjection:

    @pytest.mark.parametrize("payload", [TRAVERSAL, SUBCLASSES])
    def test_payload_falls_back_to_raw_content(self, payload):
        template = Jinja2TemplatedChatMessagesTemplate(messages=[SystemMessage(content=payload)])
        [message] = template.format_messages()
        assert message.content == payload
        assert "<class" not in message.content


# --- AC-4: legitimate templates are unchanged ------------------------------------

class TestLegitimateTemplatesUnchanged:

    def test_condition_substitution_and_logic(self):
        query = "{% if count > 2 %}big{% else %}small{% endif %}-{{ name }}"
        assert EvaluateTemplate(query, {"count": 3, "name": "x"}).extract() == "big-x"

    def test_condition_json_loads_filter(self):
        query = "{{ (payload | json_loads(True))['k'] }}"
        assert EvaluateTemplate(query, {"payload": "{'k': 'v'}"}).extract() == "v"

    def test_condition_loop(self):
        query = "{% for i in items %}{{ i }},{% endfor %}"
        assert EvaluateTemplate(query, {"items": [1, 2]}).extract() == "1,2,"

    def test_state_modifier_filters(self):
        template = (
            "{{ (raw | from_json)['a'] }}|"
            "{{ encoded | base64_to_string }}|"
            "{{ (text | split_by_words(2)) | length }}|"
            "{{ (csv | split_by_regex(',')) | join('-') }}"
        )
        node = _modifier(template, input_variables=["raw", "encoded", "text", "csv"])
        state = {
            "raw": json.dumps({"a": 1}),
            "encoded": base64.b64encode(b"hi").decode(),
            "text": "one two three four five",
            "csv": "a,b,c",
            "out": "",
        }
        assert node.invoke(state)["out"] == "1|hi|3|a-b-c"

    def test_state_modifier_list_mutation_still_allowed(self):
        template = "{% set acc = [] %}{% for i in items %}{% set _ = acc.append(i * 2) %}{% endfor %}{{ acc | tojson }}"
        node = _modifier(template, input_variables=["items"])
        assert node.invoke({"items": [1, 2], "out": []})["out"] == [2, 4]

    def test_chat_message_substitution_and_debug_undefined(self):
        template = Jinja2TemplatedChatMessagesTemplate(
            messages=[HumanMessage(content="hi {{ name }} {{ missing }}")]
        )
        [message] = template.format_messages(name="bob")
        assert message.content == "hi bob {{ missing }}"
