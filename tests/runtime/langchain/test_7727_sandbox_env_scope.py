"""#7727: a local Pyodide sandbox process receives only the allowlisted environment.

A fake `deno` on PATH writes the environment it was started with to a file, so the
real spawn paths (async execute, execute_with_bridge, sync execute) are exercised
without Deno or Pyodide.
"""
import asyncio
import json
import os
import stat
import sys
import textwrap

import pytest

from elitea_sdk.runtime.langchain.pyodide_sandbox import PyodideSandbox, SyncPyodideSandbox

SECRETS = {"POSTGRES_PASSWORD": "pg-secret", "SECRETS_MASTER_KEY": "mk-secret", "GITHUB_TOKEN": "gh-secret"}


@pytest.fixture
def fake_deno(tmp_path, monkeypatch):
    dump = tmp_path / "env.json"
    script = tmp_path / "bin" / "deno"
    script.parent.mkdir()
    # Dumps its env, then answers like main.js so execute() parses it as success.
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, os, sys
        json.dump(dict(os.environ), open({str(dump)!r}, "w"))
        sys.stdin.read() if "--bridge" not in sys.argv else sys.stdin.readline()
        if "--bridge" in sys.argv:
            print(json.dumps({{"type": "final", "success": True, "result": None, "error": None, "error_type": None}}))
        else:
            print(json.dumps({{"stdout": "", "stderr": None, "result": None, "success": True}}))
    """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("SANDBOX_BASE", str(tmp_path))
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SANDBOX_SNAPSHOT_PATH", raising=False)
    return dump


def _assert_scoped(dump):
    env = json.loads(dump.read_text())
    for key in SECRETS:
        assert key not in env, f"{key} leaked into the sandbox process"
    assert not set(SECRETS.values()) & set(env.values())
    assert env["SANDBOX_BASE"] and env["HTTPS_PROXY"] == "http://proxy.local:3128"
    assert env["DENO_DIR"] == os.path.join(env["TMPDIR"], "deno_dir")
    # Python itself may inject a few process vars (e.g. LC_CTYPE); nothing else from the worker.
    assert set(env) - {"PATH", "SANDBOX_BASE", "HTTPS_PROXY", "TMPDIR", "DENO_DIR", "LC_CTYPE"} == set()


def _sandbox(cls):
    return cls(stateful=False, allow_net=False, skip_deno_check=True, pkg_name="main.js")


def test_async_execute_passes_only_allowlisted_env(fake_deno):
    result = asyncio.run(_sandbox(PyodideSandbox).execute("print(1)", timeout_seconds=20))
    assert result.status == "success", result.stderr
    _assert_scoped(fake_deno)


def test_bridge_execute_passes_only_allowlisted_env(fake_deno):
    async def dispatcher(name, args):
        return "{}"
    asyncio.run(_sandbox(PyodideSandbox).execute_with_bridge("print(1)", dispatcher, timeout_seconds=20, max_tool_calls=5))
    _assert_scoped(fake_deno)


def test_sync_execute_passes_only_allowlisted_env(fake_deno):
    result = _sandbox(SyncPyodideSandbox).execute("print(1)", timeout_seconds=20)
    assert result.status == "success", result.stderr
    _assert_scoped(fake_deno)


def test_run_environment_omits_unset_passthrough_keys(monkeypatch, tmp_path):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy", "SANDBOX_BASE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "pg-secret")

    class _Dir:
        name = str(tmp_path)

    env = PyodideSandbox._run_environment(_Dir())
    assert set(env) <= {"PATH", "TMPDIR", "DENO_DIR"}
    assert "POSTGRES_PASSWORD" not in env
