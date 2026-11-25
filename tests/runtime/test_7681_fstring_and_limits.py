"""Issue #7681 — AC-6 (fstring attribute/index traversal) and AC-7 (resource exhaustion)."""

import pytest

from elitea_sdk.runtime.langchain.utils import propagate_the_input_mapping
from elitea_sdk.runtime.utils.evaluate import EvaluateTemplate
from elitea_sdk.runtime.utils.safe_templates import TemplateRejectedError, UnsafePlaceholderError, safe_format


class _Obj:
    secret = "LEAK"

    def __init__(self):
        self.items = ["first"]


# --- AC-6 ------------------------------------------------------------------------

class TestSafeFormat:

    @pytest.mark.parametrize("template", [
        "{x.__class__}",
        "{x.__init__.__globals__}",
        "{x._private}",
        "{x.items[0].__class__}",
        "{0.__class__}",
    ])
    def test_private_attribute_access_rejected(self, template):
        with pytest.raises(UnsafePlaceholderError):
            safe_format(template, _Obj(), x=_Obj())

    def test_unsafe_placeholder_is_a_keyerror(self):
        # existing missing-placeholder fallbacks (input mapping, HITL) must still apply
        assert issubclass(UnsafePlaceholderError, KeyError)

    def test_public_attribute_and_index_traversal_resolve(self):
        data = {"key": "v"}
        assert safe_format("{x.secret}|{x.items[0]}|{l[0]}|{d[key]}", x=_Obj(), l=["a"], d=data) == "LEAK|first|a|v"

    def test_plain_names_resolve(self):
        assert safe_format("hi {name}, {n}", name="bob", n=3) == "hi bob, 3"

    def test_escapes_conversion_and_spec_unchanged(self):
        assert safe_format("{{literal}} {a!r} {n:>4} {}", "p", a="x", n=7) == "{literal} 'x'    7 p"

    def test_variable_named_template_does_not_collide(self):
        assert safe_format("{template}-{v}", template="t", v=1) == "t-1"

    def test_missing_name_still_raises_keyerror(self):
        with pytest.raises(KeyError):
            safe_format("{missing}", a=1)

    def test_positional_still_raises_indexerror(self):
        with pytest.raises(IndexError):
            safe_format("{0}", a=1)


class TestInputMappingFstring:

    def test_unsafe_placeholder_in_input_mapping_not_resolved(self):
        mapping = {"out": {"type": "fstring", "value": "{x.__class__}|{y}"}}
        state = {"x": _Obj(), "y": "ok"}
        result = propagate_the_input_mapping(mapping, ["x", "y"], state)
        assert "<class" not in str(result)
        assert "ok" in result["out"]

    def test_variable_named_template_in_input_mapping(self):
        mapping = {"out": {"type": "fstring", "value": "{template}!"}}
        result = propagate_the_input_mapping(mapping, ["template"], {"template": "hi"})
        assert result["out"] == "hi!"


# --- AC-7 ------------------------------------------------------------------------

class TestResourceExhaustion:

    @pytest.mark.parametrize("payload", [
        "{{ 'a' * 10000000000 }}",
        "{{ [1] * 10000000000 }}",
        "{{ 9 ** 99999999 }}",
        "{{ '%0999999999d' % 1 }}",
        "{{ '%(a)0999999999d' % {'a': 1} }}",
        "{{ '%*d' % (999999999, 1) }}",
    ])
    def test_rejected_with_clear_error(self, payload):
        with pytest.raises(TemplateRejectedError) as exc:
            EvaluateTemplate(payload, {}).extract()
        assert "Template rejected" in str(exc.value)

    def test_normal_arithmetic_and_repetition_unchanged(self):
        assert EvaluateTemplate("{{ 2 ** 10 }}-{{ 'ab' * 3 }}-{{ 6 * 7 }}-{{ 7 % 3 }}-{{ '%05d|%%' % 42 }}", {}).extract() == "1024-ababab-42-1-00042|%"
