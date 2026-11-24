"""#7719: the sandbox gets a credential-free SandboxClient stub; the worker makes every platform call.

Most tests run the real stub source in CPython against the real dispatcher: `elitea_bridge` and `pyodide.ffi` are
faked with monkeypatch.setitem on sys.modules, so they are removed again after each test (a stub left in sys.modules
breaks unrelated tests). A real-Deno test at the end runs the same stub inside Pyodide.
"""
import asyncio
import json
import os
import shutil
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from elitea_sdk.runtime.clients import sandbox_client as sc
from elitea_sdk.runtime.tools import sandbox_client_bridge as bridge
from elitea_sdk.runtime.tools.sandbox import PyodideSandboxTool

TOKEN = "tok-SECRET-123456"
SECRET_VALUE = "s3cr3t-value-xyz"


class FakeArtifact:
    def __init__(self, client, bucket):
        self.client, self.bucket = client, bucket
        client.calls.append(("artifact", bucket))

    def create(self, name, data, bucket_name=None, check_if_exists=False):
        self.client.calls.append(("artifact.create", self.bucket, name, data))
        return {"filepath": f"/{self.bucket}/{name}"}

    def get(self, name, bucket_name=None, **kwargs):
        return f"content of {name}"

    def get_content_bytes(self, name, bucket_name=None):
        return b"\x00\x01binary"

    def list(self, bucket_name=None, prefix="", delimiter="/", return_as_string=True):
        return "a.txt\nb.txt" if return_as_string else {"rows": ["a.txt", "b.txt"]}

    def delete(self, name, bucket_name=None, check_exists=True):
        return {"deleted": name}

    def append(self, name, data, bucket_name=None, create_if_missing=True):
        return {"appended": name}


class FakeClient:
    """Stands in for the worker-side SandboxClient; records calls."""

    def __init__(self, project_id):
        self.project_id = project_id
        self.calls = []

    def artifact(self, bucket):
        return FakeArtifact(self, bucket)

    def unsecret(self, secret_name):
        self.calls.append(("unsecret", secret_name))
        return SECRET_VALUE if secret_name == "API_KEY" else None

    def get_private_project_secret(self, secret_name, default=sc._UNSET):
        if secret_name == "missing":
            raise sc.PrivateSecretNotFound(f"Private secret '{secret_name}' does not exist")
        if secret_name == "unshared":
            raise sc.PrivateSecretNotShared("not shared")
        return "personal-" + secret_name

    def get_app_details(self, application_id, project_id=None):
        return {"id": application_id, "project": self.project_id}

    def get_app_version_details(self, application_id, application_version_id, project_id=None):
        raise sc.ApiDetailsRequestError("Failed to fetch application version details")

    def download_artifact_s3(self, bucket, key):
        return b"\xffraw"

    def upload_artifact_s3(self, bucket, key, data, content_type=None):
        return {"size": len(data), "type": type(data).__name__}

    def get_user_data(self):
        return {"id": 3}


@pytest.fixture
def harness(monkeypatch):
    """Runs stub code in-process: elitea_bridge.call -> real dispatcher, run_sync -> asyncio.run."""
    clients = {}
    audits = []

    def factory(project_id):
        clients.setdefault(project_id, FakeClient(project_id))
        return clients[project_id]

    def make(caller="code_node", governed=False, max_result_bytes=1_000_000):
        dispatcher = bridge.ClientBridgeDispatcher(
            factory, caller=caller, project_id=3, governed=governed, max_result_bytes=max_result_bytes,
            audit=lambda **kw: audits.append(kw),
        )
        bridge_mod = types.ModuleType("elitea_bridge")
        bridge_mod.call = lambda name, args: dispatcher(name, args)
        ffi = types.ModuleType("pyodide.ffi")
        ffi.run_sync = lambda coro: asyncio.run(coro)
        pyodide = types.ModuleType("pyodide")
        pyodide.ffi = ffi
        monkeypatch.setitem(sys.modules, "elitea_bridge", bridge_mod)
        monkeypatch.setitem(sys.modules, "pyodide", pyodide)
        monkeypatch.setitem(sys.modules, "pyodide.ffi", ffi)

        def run(code):
            namespace = {}
            exec(bridge.build_client_stub("http://platform", 3) + "\n" + code, namespace)  # noqa: S102
            return namespace
        return dispatcher, run

    make.clients = clients
    make.audits = audits
    return make


# --- stub + dispatcher, end to end -------------------------------------------------------------------------------

def test_artifact_round_trip_and_bucket_object_reused(harness):
    _, run = harness()
    ns = run(
        "a = elitea_client.artifact('bkt')\n"
        "created = a.create('x.txt', 'hello')\n"
        "text = a.get('x.txt')\n"
        "listing = a.list()\n"
        "raw = a.get_content_bytes('x.txt')\n"
    )
    assert ns["created"] == {"filepath": "/bkt/x.txt"}
    assert ns["text"] == "content of x.txt"
    assert ns["listing"] == "a.txt\nb.txt"  # SandboxClient's default string shape is kept
    assert ns["raw"] == b"\x00\x01binary"
    # One worker-side artifact per bucket, so bucket_exists/create_bucket are not repeated on every call
    assert harness.clients[None].calls.count(("artifact", "bkt")) == 1


def test_bytes_both_ways(harness):
    _, run = harness()
    ns = run("up = elitea_client.upload_artifact_s3('b', 'k', b'\\x00\\x01\\x02')\n"
             "down = elitea_client.download_artifact_s3('b', 'k')\n")
    assert ns["up"] == {"size": 3, "type": "bytes"}
    assert ns["down"] == b"\xffraw"


def test_errors_keep_their_classes(harness):
    _, run = harness()
    ns = run(
        "try:\n    elitea_client.get_private_project_secret('missing')\nexcept PrivateSecretNotFound as e:\n    nf = str(e)\n"
        "try:\n    elitea_client.get_private_project_secret('unshared')\nexcept PrivateSecretError as e:\n    ns_ = type(e).__name__\n"
        "try:\n    elitea_client.get_app_version_details(1, 2)\nexcept ApiDetailsRequestError as e:\n    api = str(e)\n"
    )
    assert "does not exist" in ns["nf"]
    assert ns["ns_"] == "PrivateSecretNotShared"
    assert "Failed to fetch" in ns["api"]


def test_credentials_are_not_in_the_sandbox(harness):
    _, run = harness()
    ns = run(
        "errors = []\n"
        "for name in ('auth_token', 'headers', '_session', 'auth_session'):\n"
        "    try:\n        getattr(elitea_client, name)\n    except AttributeError as e:\n        errors.append(str(e))\n"
        "base, pid = elitea_client.base_url, elitea_client.project_id\n"
    )
    assert len(ns["errors"]) == 4 and all("#7719" in e for e in ns["errors"])
    assert (ns["base"], ns["pid"]) == ("http://platform", 3)
    assert TOKEN not in bridge.build_client_stub("http://platform", 3)


def test_user_built_client_for_another_project(harness):
    # Dev code builds SandboxClient(project_id=other, auth_token=...) to read another project's app
    _, run = harness()
    ns = run("other = SandboxClient(base_url=elitea_client.base_url, project_id=99, auth_token='ignored')\n"
             "app = other.get_app_details(5)\n"
             "same = elitea_client.get_app_details(6)\n")
    assert ns["app"] == {"id": 5, "project": 99}
    assert ns["same"] == {"id": 6, "project": None}  # None = the run's own project


def test_sync_calls_work_inside_callbacks(harness):
    _, run = harness()
    ns = run("names = sorted(['b', 'a'], key=lambda n: elitea_client.artifact('bkt').get(n))\n"
             "class W:\n    @property\n    def user(self):\n        return elitea_client.get_user_data()\n"
             "uid = W().user['id']\n")
    assert ns["names"] == ["a", "b"]
    assert ns["uid"] == 3


# --- secrets: caller split, audit, masking -----------------------------------------------------------------------

def test_code_node_reads_secrets_and_every_read_is_audited(harness):
    dispatcher, run = harness(caller="code_node")
    ns = run("key = elitea_client.unsecret('API_KEY')\n"
             "none = elitea_client.unsecret('NOPE')\n"
             "try:\n    elitea_client.get_private_project_secret('unshared')\nexcept PrivateSecretError:\n    pass\n")
    assert ns["key"] == SECRET_VALUE and ns["none"] is None
    outcomes = [(a["method"], a["secret_name"], a["outcome"]) for a in harness.audits]
    assert outcomes == [("unsecret", "API_KEY", "success"), ("unsecret", "NOPE", "not_found"),
                        ("get_private_project_secret", "unshared", "not_shared")]
    assert all(SECRET_VALUE not in json.dumps(a) for a in harness.audits)
    assert dispatcher.masker.mask(f"key={SECRET_VALUE}") == "key=***"


@pytest.mark.parametrize("caller", ["pyodide_tool", "code_mode"])
def test_model_written_code_cannot_read_secrets(harness, caller):
    _, run = harness(caller=caller)
    ns = run("try:\n    elitea_client.unsecret('API_KEY')\nexcept PermissionError as e:\n    err = str(e)\n"
             "other = elitea_client.get_user_data()\n")
    assert "not available to code written by the model" in ns["err"]
    assert ns["other"] == {"id": 3}  # the rest of the client still works
    assert harness.audits == [{"method": "unsecret", "secret_name": "API_KEY", "outcome": "denied",
                               "caller": caller, "project_id": 3, "user_id": None}]
    assert ("unsecret", "API_KEY") not in harness.clients[None].calls


def test_governed_run_refuses_every_call(harness):
    _, run = harness(governed=True)
    ns = run("try:\n    elitea_client.artifact('b')\nexcept PermissionError as e:\n    err = str(e)\n")
    assert "governed runs" in ns["err"]
    assert harness.clients == {}  # no worker-side client was even built


def test_a_failing_audit_never_fails_the_call(harness, monkeypatch):
    dispatcher, run = harness()
    dispatcher.audit = MagicMock(side_effect=RuntimeError("audit sink down"))
    assert run("v = elitea_client.unsecret('API_KEY')\n")["v"] == SECRET_VALUE


def test_masker_handles_nesting_and_ignores_short_values():
    masker = bridge.SecretMasker()
    masker.add("abc")  # too short to mask safely
    masker.add("longsecret")
    masker.add(None)
    assert masker.mask({"a": ["x longsecret y", ("abc",)], "n": 5}) == {"a": ["x *** y", ["abc"]], "n": 5}


# --- dispatcher edges ----------------------------------------------------------------------------------------------

def _call(dispatcher, name, *args, **kwargs):
    return json.loads(asyncio.run(dispatcher(name, json.dumps({"args": list(args), "kwargs": kwargs}))))


def test_unknown_and_private_names_are_refused(harness):
    dispatcher, _ = harness()
    for name in ("client._request", "client.auth_token", "os.system", "artifact.__init__"):
        reply = _call(dispatcher, name)
        assert reply["ok"] is False and reply["error_class"] == "PermissionError", name


def test_result_over_the_cap_is_an_error(harness):
    dispatcher, _ = harness(max_result_bytes=64)
    reply = _call(dispatcher, "artifact.get", "bkt", "x" * 100)
    assert reply["ok"] is False and "limit" in reply["error"]


def test_plain_dict_with_a_bytes_like_key_is_not_decoded(harness):
    _, run = harness()
    ns = run("r = elitea_client.artifact('bkt').create('x', {'__bytes__': 'abc'})\n")
    assert harness.clients[None].calls[-1] == ("artifact.create", "bkt", "x", {"__bytes__": "abc"})


def test_close_closes_worker_sessions(harness):
    dispatcher, run = harness()
    run("elitea_client.get_user_data()\n")
    client = harness.clients[None]
    client._session = MagicMock()
    dispatcher.close()
    client._session.close.assert_called_once()


def test_malformed_payload(harness):
    dispatcher, _ = harness()
    reply = json.loads(asyncio.run(dispatcher("client.get_user_data", "not json")))
    assert reply["error_class"] == "ValueError"


def test_unknown_exception_becomes_api_details_error(harness, monkeypatch):
    dispatcher, _ = harness()
    monkeypatch.setattr(FakeClient, "get_user_data", lambda self: (_ for _ in ()).throw(ConnectionError("down")))
    reply = _call(dispatcher, "client.get_user_data")
    assert reply == {"ok": False, "error_class": "ApiDetailsRequestError", "error": "down"}


def test_audit_span_shape(monkeypatch):
    spans = []
    tracer = MagicMock()
    tracer.start_span.side_effect = lambda name, attributes: spans.append((name, attributes)) or MagicMock()
    from opentelemetry import trace
    monkeypatch.setattr(trace, "get_tracer", lambda name: tracer)
    bridge.emit_secret_audit(method="unsecret", secret_name="API_KEY", outcome="denied", caller="pyodide_tool",
                             project_id=3)
    name, attrs = spans[0]
    assert name == "sandbox_secret_read:unsecret:denied"
    assert attrs["audit.observation.type"] == "tool" and attrs["audit.tool.name"] == "sandbox_secret:API_KEY"
    assert attrs["audit.is_error"] is True and attrs["project.id"] == 3


# --- sandbox tool wiring -------------------------------------------------------------------------------------------

def _tool(stateful=False, caller="pyodide_tool"):
    client = MagicMock(base_url="http://platform", project_id=3, auth_token=TOKEN, auth_session=None,
                       session_cookie_name=None, internal_run_auth=None)
    return PyodideSandboxTool.model_construct(stateful=stateful, allow_net=True, elitea_client=client,
                                              session_bytes=None, session_metadata=None, caller=caller)


def test_flag_on_uses_the_stub_without_the_token(monkeypatch):
    monkeypatch.setenv("SANDBOX_CLIENT_BRIDGE", "1")
    monkeypatch.delenv("SANDBOX_SERVICE_URL", raising=False)
    code = _tool()._prepare_pyodide_input("x = elitea_client.unsecret('A')")
    assert "elitea_bridge" in code and TOKEN not in code
    assert "import requests" not in code and "class SandboxClient" in code


@pytest.mark.parametrize("code", ["import requests\nrequests.get('https://x')",
                                  "c = globals()['elitea_' + 'client']\nprint(c)"])
def test_flag_on_never_injects_the_token(monkeypatch, code):
    # Code that does not name the client (or hides the name) gets no client at all, never the old token-carrying one
    monkeypatch.setenv("SANDBOX_CLIENT_BRIDGE", "1")
    monkeypatch.delenv("SANDBOX_SERVICE_URL", raising=False)
    tool = _tool()
    prepared = tool._prepare_pyodide_input(code)
    assert not tool._uses_client_bridge(code)
    assert TOKEN not in prepared and "class SandboxClient" not in prepared and "elitea_client = None" in prepared


@pytest.mark.parametrize("env,stateful", [({}, False), ({"SANDBOX_CLIENT_BRIDGE": "1"}, True),
                                          ({"SANDBOX_CLIENT_BRIDGE": "1", "SANDBOX_SERVICE_URL": "http://r"}, False)])
def test_old_injection_kept_when_flag_off_stateful_or_remote(monkeypatch, env, stateful):
    for key in ("SANDBOX_CLIENT_BRIDGE", "SANDBOX_SERVICE_URL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    code = _tool(stateful=stateful)._prepare_pyodide_input("x = elitea_client.unsecret('A')")
    assert "elitea_bridge" not in code and TOKEN in code


def test_bridged_run_masks_secrets_and_maps_the_result(monkeypatch):
    monkeypatch.setenv("SANDBOX_CLIENT_BRIDGE", "1")
    tool = _tool(caller="code_node")
    captured = {}

    async def fake_execute_with_bridge(code, dispatcher, **kwargs):
        captured.update(kwargs)
        dispatcher.masker.add(SECRET_VALUE)
        from elitea_sdk.runtime.langchain.sandbox_types import BridgeExecutionResult
        return BridgeExecutionResult(status="success", execution_time=0.1, result={"k": SECRET_VALUE},
                                     stdout=f"printed {SECRET_VALUE}\n")

    tool._sandbox = MagicMock(execute_with_bridge=fake_execute_with_bridge)
    limits = {"timeout_seconds": 5, "wasm_max_mem_mb": 512, "root_ca_path": "/ca.pem", "allowed_pyodide_domains": []}
    out = asyncio.run(tool._arun_bridged("code", limits, 0.0))
    # The result is pipeline state a later node may need; only printed text is masked
    assert out["result"] == {"k": SECRET_VALUE} and out["output"] == "printed ***\n"
    assert captured["root_ca_path"] == "/ca.pem" and captured["max_concurrent_calls"] == 1


def test_bridged_timeout_raises_admission_refused(monkeypatch):
    from elitea_sdk.runtime.exceptions import SandboxAdmissionRefused
    from elitea_sdk.runtime.langchain.sandbox_types import BridgeExecutionResult
    tool = _tool(caller="code_node")

    async def timed_out(code, dispatcher, **kwargs):
        return BridgeExecutionResult(status="error", execution_time=5, error="timed out", error_type="timeout")

    tool._sandbox = MagicMock(execute_with_bridge=timed_out)
    with pytest.raises(SandboxAdmissionRefused):
        asyncio.run(tool._arun_bridged("code", {"timeout_seconds": 5, "wasm_max_mem_mb": 512}, 0.0))


def test_code_nodes_are_marked_as_code_node():
    source = Path(__file__).resolve().parents[2] / "elitea_sdk" / "runtime" / "langchain" / "langraph_agent.py"
    assert 'caller="code_node"' in source.read_text()


# --- cross-project app lookups (#7719) ------------------------------------------------------------------------------

def _sandbox_client(monkeypatch):
    client = sc.SandboxClient(base_url="http://h", project_id=3, auth_token="t")
    seen = []
    response = MagicMock(ok=True)
    response.json.return_value = {}
    monkeypatch.setattr(client, "_request", lambda method, url, **kw: seen.append(url) or response)
    return client, seen


def test_get_app_details_project_id(monkeypatch):
    client, seen = _sandbox_client(monkeypatch)
    client.get_app_details(7)
    client.get_app_details(7, project_id=42)
    assert seen == ["http://h/api/v2/elitea_core/application/prompt_lib/3/7",
                    "http://h/api/v2/elitea_core/application/prompt_lib/42/7"]


def test_get_app_version_details_project_id(monkeypatch):
    client, seen = _sandbox_client(monkeypatch)
    client.get_app_version_details(7, 8)
    client.get_app_version_details(7, 8, project_id=42)
    assert seen == ["http://h/api/v2/elitea_core/version/prompt_lib/3/7/8",
                    "http://h/api/v2/elitea_core/version/prompt_lib/42/7/8"]


# --- real Pyodide ---------------------------------------------------------------------------------------------------

def _sandbox_base():
    for base in (os.environ.get("SANDBOX_BASE"), "/app"):
        if base and os.path.isfile(os.path.join(base, "pyodide", "pyodide.mjs")):
            return base
    return None


@pytest.mark.skipif(not shutil.which("deno") or _sandbox_base() is None or not os.environ.get("PYODIDE_SANDBOX_PKG"),
                    reason="needs Deno, a Pyodide distribution and PYODIDE_SANDBOX_PKG")
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_real_pyodide_stub(monkeypatch):
    from elitea_sdk.runtime.langchain.pyodide_sandbox import PyodideSandbox
    from elitea_sdk.runtime.tools.sandbox import local_sandbox_permissions
    monkeypatch.setenv("SANDBOX_BASE", _sandbox_base())
    dispatcher = bridge.ClientBridgeDispatcher(lambda pid: FakeClient(pid), caller="code_node", project_id=3,
                                               audit=lambda **kw: None)
    code = bridge.build_client_stub("http://platform", 3) + '''
import gc, os
out = {"art": elitea_client.artifact("bkt").get("x"), "raw": list(elitea_client.download_artifact_s3("b", "k")),
       "sorted": sorted(["b", "a"], key=lambda n: elitea_client.artifact("bkt").get(n))}
try:
    elitea_client.get_private_project_secret("missing")
except PrivateSecretNotFound:
    out["caught"] = True
try:
    elitea_client.auth_token
except AttributeError:
    out["no_token"] = True
out["leak"] = any("tok-SECRET" in repr(o) for o in gc.get_objects() if isinstance(o, str)) or \
    "tok-SECRET" in repr(dict(os.environ))
out
'''
    box = PyodideSandbox(stateful=False, allow_net=False, node_modules_dir="auto", **local_sandbox_permissions())
    result = asyncio.run(box.execute_with_bridge(code, dispatcher, timeout_seconds=120, max_tool_calls=50,
                                                 max_concurrent_calls=1))
    assert result.status == "success", result.error
    assert result.result == {"art": "content of x", "raw": [255, 114, 97, 119], "sorted": ["a", "b"],
                             "caught": True, "no_token": True, "leak": False}
