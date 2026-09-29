"""The SandboxClient preamble (requests + chardet) is only injected when code can reach it.

Installing and importing requests inside Pyodide costs ~0.7s per code-node run, so a node
that never touches the client gets a light stdlib-only preamble instead. The light preamble
must still define every cheap name the full one used to leak, and the state preamble's
`alita_client = elitea_client` alias must neither force the full preamble nor NameError.
"""
from types import SimpleNamespace

import pytest

from elitea_sdk.runtime.tools.sandbox import (
    CLIENT_ALIAS_LINE,
    PyodideSandboxTool,
    _code_needs_sandbox_client,
)


def _tool(client=True):
    elitea_client = SimpleNamespace(
        base_url="http://example", project_id=2, auth_token="t",
        auth_session=None, session_cookie_name=None,
    ) if client else None
    return PyodideSandboxTool.model_construct(elitea_client=elitea_client)


@pytest.mark.parametrize("code", [
    "print(elitea_client.get_list_of_apps())",
    "print(alita_client)",
    "import requests\nrequests.get('x')",
    "r = requests.get('x')",
    "import chardet",
    "a = SandboxArtifact",
])
def test_client_names_need_the_full_preamble(code):
    assert _code_needs_sandbox_client(f"{CLIENT_ALIAS_LINE}\n{code}")


@pytest.mark.parametrize("code", [
    "import json\nprint(json.dumps({'a': 1}))",
    "print(re.sub('a', 'b', 'aa'))",
    "my_requests_count = 1",
])
def test_plain_code_does_not_need_the_client(code):
    assert not _code_needs_sandbox_client(f"{CLIENT_ALIAS_LINE}\n{code}")


def test_plain_code_gets_the_light_preamble_that_still_runs_the_alias():
    prepared = _tool()._prepare_pyodide_input(f"{CLIENT_ALIAS_LINE}\nresult = re.sub('a', 'b', 'aa')")
    #
    assert "import requests" not in prepared
    namespace = {}
    exec(prepared, namespace)  # pylint: disable=W0122
    assert namespace["result"] == "bb"
    assert namespace["alita_client"] is None
    for name in ("logging", "re", "Path", "Dict", "Optional", "Any", "Union", "quote", "urlparse", "logger"):
        assert name in namespace, name


def test_the_real_state_preamble_alone_does_not_force_the_client():
    from elitea_sdk.runtime.tools.function import FunctionTool
    #
    state_preamble = FunctionTool.model_construct()._prepare_pyodide_input({"input": "hi"})
    prepared = _tool()._prepare_pyodide_input(f"{state_preamble}\nresult = alita_state['input']")
    #
    assert "import requests" not in prepared
    namespace = {}
    exec(prepared, namespace)  # pylint: disable=W0122
    assert namespace["result"] == "hi"


def test_commented_out_client_references_do_not_count():
    assert not _code_needs_sandbox_client("# elitea_client.get_list_of_apps()\nx = 1")


def test_client_code_gets_the_full_preamble():
    prepared = _tool()._prepare_pyodide_input("print(elitea_client)")
    #
    assert "import requests" in prepared
    assert "elitea_client = SandboxClient(" in prepared


def test_no_client_means_no_preamble():
    prepared = _tool(client=False)._prepare_pyodide_input("print(1)")
    #
    assert prepared == "#elitea simplified client\nprint(1)"
