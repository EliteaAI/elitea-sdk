"""Regression tests for code-scanning alert #19 (issue #5429) — SSTI.

_resolve_jinja2_variables previously used jinja2.Environment, which exposes
Python object internals through template syntax. Because the rendered template
string is the agent author's own instructions (self.prompt / cfg['instructions']),
a malicious author could embed a sandbox-escape payload and achieve RCE on the
shared indexer-worker runtime.

The fix swaps Environment -> SandboxedEnvironment. These tests assert:
1. The classic ().__class__.__subclasses__() escape is blocked.
2. os/system access via builtins is blocked.
3. Legitimate {{variable}} substitution still works (no regression).
"""

from unittest.mock import MagicMock

import pytest


def _get_assistant():
    """Minimal Assistant — only _resolve_jinja2_variables is exercised."""
    from elitea_sdk.runtime.langchain.assistant import Assistant

    assistant = Assistant.__new__(Assistant)
    assistant.client = MagicMock()
    assistant.prompt_variables = {}
    return assistant


class TestSandboxBlocksInjection:
    """Exploit-proof: known SSTI payloads must not execute or leak internals."""

    def test_subclasses_escape_is_neutralized(self):
        # Classic Jinja sandbox-escape primitive: walk the type hierarchy to
        # reach arbitrary classes. SandboxedEnvironment raises SecurityError on
        # access to __class__/__subclasses__, which the method catches and falls
        # back to returning the template unchanged. The key guarantee: the
        # subclass list is NEVER rendered into the output.
        assistant = _get_assistant()
        payload = "{{ ().__class__.__base__.__subclasses__() }}"
        resolved = assistant._resolve_jinja2_variables(payload)
        # No leakage of Python internals into the rendered prompt.
        assert "subclasses" not in resolved.lower() or resolved == payload
        assert "<class" not in resolved
        assert "type object" not in resolved

    def test_os_system_rce_payload_does_not_execute(self):
        # The canonical RCE gadget. If the sandbox were absent, this would import
        # os and run a command. Under SandboxedEnvironment the builtins access is
        # blocked; on the resulting SecurityError the method returns the template
        # verbatim. We assert no command output / class internals leak through.
        assistant = _get_assistant()
        payload = (
            "{% for s in ().__class__.__base__.__subclasses__() %}"
            "{% if 'warning' in s.__name__ %}"
            "{{ s()._module.__builtins__['__import__']('os').popen('id').read() }}"
            "{% endif %}{% endfor %}"
        )
        resolved = assistant._resolve_jinja2_variables(payload)
        # uid=/gid= would appear if `id` had run; they must not.
        assert "uid=" not in resolved
        assert "gid=" not in resolved

    def test_attr_access_on_builtin_blocked(self):
        # Direct attribute drill-down on a literal is an unsafe operation under
        # the sandbox; it must not render mangled-dunder internals.
        assistant = _get_assistant()
        payload = "{{ ''.__class__.__mro__ }}"
        resolved = assistant._resolve_jinja2_variables(payload)
        assert "<class" not in resolved
        assert "mro" not in resolved.lower() or resolved == payload


class TestLegitimateTemplatingStillWorks:
    """No regression: ordinary variable substitution and filters must work."""

    def test_simple_variable_substitution(self):
        assistant = _get_assistant()
        resolved = assistant._resolve_jinja2_variables(
            "hello {{name}}", extra_context={"name": "world"}
        )
        assert resolved == "hello world"

    def test_current_date_still_resolves(self):
        import re

        assistant = _get_assistant()
        resolved = assistant._resolve_jinja2_variables("date={{current_date}}")
        assert re.match(r"date=\d{4}-\d{2}-\d{2}$", resolved)

    def test_safe_filter_still_works(self):
        # upper is a safe filter and must remain available post-sandbox.
        assistant = _get_assistant()
        resolved = assistant._resolve_jinja2_variables(
            "{{ greeting | upper }}", extra_context={"greeting": "hi"}
        )
        assert resolved == "HI"

    def test_undefined_variable_left_as_token(self):
        # DebugUndefined behavior must be preserved under the sandbox.
        assistant = _get_assistant()
        resolved = assistant._resolve_jinja2_variables("{{unknown_var}}")
        assert "unknown_var" in resolved

    def test_no_template_fast_path(self):
        assistant = _get_assistant()
        text = "plain instructions, no jinja"
        assert assistant._resolve_jinja2_variables(text) == text
