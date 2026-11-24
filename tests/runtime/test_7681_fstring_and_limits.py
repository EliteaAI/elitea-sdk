"""Issue #7681 — AC-6 (fstring attribute/index traversal) and AC-7 (resource exhaustion)."""

import pytest

from elitea_sdk.runtime.langchain.utils import propagate_the_input_mapping
from elitea_sdk.runtime.utils.evaluate import EvaluateTemplate
from elitea_sdk.runtime.utils.safe_templates import safe_format


class _Obj:
    secret = "LEAK"

    def __init__(self):
        self.items = ["first"]


# --- AC-6 ------------------------------------------------------------------------

class TestSafeFormat:

    @pytest.mark.parametrize("template", [
        "{x.secret}",
        "{x.__class__}",
        "{x.__init__.__globals__}",
        "{x[0]}",
        "{x.items[0]}",
    ])
    def test_traversal_left_as_written(self, template):
        out = safe_format(template, x=_Obj())
        assert out == template
        assert "LEAK" not in out and "<class" not in out

    def test_plain_names_resolve(self):
        assert safe_format("hi {name}, {n}", name="bob", n=3) == "hi bob, 3"

    def test_escapes_conversion_and_spec_unchanged(self):
        assert safe_format("{{literal}} {a!r} {n:>4}", a="x", n=7) == "{literal} 'x'    7"

    def test_missing_name_still_raises_keyerror(self):
        with pytest.raises(KeyError):
            safe_format("{missing}", a=1)

    def test_positional_still_raises_indexerror(self):
        with pytest.raises(IndexError):
            safe_format("{0}", a=1)


class TestInputMappingFstring:

    def test_traversal_in_input_mapping_not_resolved(self):
        mapping = {"out": {"type": "fstring", "value": "{x.secret}|{y}"}}
        state = {"x": _Obj(), "y": "ok"}
        result = propagate_the_input_mapping(mapping, ["x", "y"], state)
        # create_params stringifies state values, so traversal could never read objects;
        # the guarantee here is that the placeholder is not resolved and plain names still are.
        assert "LEAK" not in str(result)
        assert "ok" in result["out"]


# --- AC-7 ------------------------------------------------------------------------

class TestResourceExhaustion:

    @pytest.mark.parametrize("payload", [
        "{{ 'a' * 10000000000 }}",
        "{{ [1] * 10000000000 }}",
        "{{ 9 ** 99999999 }}",
    ])
    def test_rejected_with_clear_error(self, payload):
        with pytest.raises(Exception) as exc:
            EvaluateTemplate(payload, {}).extract()
        assert "Template rejected" in str(exc.value)

    def test_normal_arithmetic_and_repetition_unchanged(self):
        assert EvaluateTemplate("{{ 2 ** 10 }}-{{ 'ab' * 3 }}-{{ 6 * 7 }}", {}).extract() == "1024-ababab-42"
