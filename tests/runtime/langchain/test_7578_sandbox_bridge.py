"""PyodideSandbox.execute_with_bridge (#7578).

Unit tests replace Deno with a small Python script that speaks the bridge protocol, so every failure
can be forced without Deno. The integration tests at the bottom run the real `main.ts` and are skipped
where Deno or a Pyodide distribution is missing.
"""
import asyncio
import json
import re
import os
import shutil
import sys
import textwrap
import time
from pathlib import Path

import pytest

from elitea_sdk.runtime.langchain.pyodide_sandbox import PyodideSandbox

# The fake sandbox: reads the code line, then runs the scenario named in it.
FAKE_SANDBOX = textwrap.dedent(r'''
    import json, os, sys, time

    def send(message):
        sys.stdout.write(json.dumps(message) + "\n"); sys.stdout.flush()

    def call(call_id, name, args="{}"):
        send({"type": "call", "id": call_id, "name": name, "args": args})
        line = sys.stdin.readline()
        return json.loads(line) if line else None

    def final(**fields):
        send({"type": "final", "success": False, "result": None, "error": None, "error_type": None, **fields})

    scenario = json.loads(sys.stdin.readline())["code"]
    if scenario == "round_trip":
        first = call(1, "t.echo", '{"n": 1}')
        failed = call(2, "t.boom")
        send({"type": "stdout", "text": "caught " + failed["error"] + "\n"})
        final(success=True, result={"first": json.loads(first["value"]), "failed_ok": failed["ok"]})
    elif scenario == "gather":
        for i in range(3):
            send({"type": "call", "id": i, "name": "t.echo", "args": json.dumps({"i": i})})
        replies = [json.loads(sys.stdin.readline()) for _ in range(3)]
        final(success=True, result=sorted(r["id"] for r in replies))
    elif scenario == "many_calls":
        for i in range(100):
            if call(i, "t.echo") is None:
                break
        final(success=True, result="should not be reached")
    elif scenario == "slow_call":
        call(1, "t.slow")
        final(success=True)
    elif scenario == "eof":
        send({"type": "stdout", "text": "partial output\n"})
        sys.stderr.write("deno panicked\n"); sys.stderr.flush()
        sys.exit(3)
    elif scenario == "stderr_flood":
        chunk = "e" * 65536
        for _ in range(160):
            sys.stderr.write(chunk)
        sys.stderr.flush()
        final(success=True, result="done")
    elif scenario == "long_line":
        reply = call(1, "t.big")
        final(success=True, result=len(json.loads(reply["value"])))
    elif scenario == "line_over_cap":
        send({"type": "stdout", "text": "x" * 300_000})
        final(success=True)
    elif scenario == "killed_during_call":
        send({"type": "call", "id": 1, "name": "t.echo", "args": "{}"})
        os.kill(os.getpid(), 9)
    elif scenario == "fire_and_forget":
        # The program started a write with create_task and returned without awaiting it.
        send({"type": "call", "id": 1, "name": "t.slow", "args": "{}"})
        time.sleep(0.3)
        final(success=True, result="returned early")
    elif scenario == "fanout":
        for i in range(6):
            send({"type": "call", "id": i, "name": "t.counted", "args": "{}"})
        replies = [json.loads(sys.stdin.readline()) for _ in range(6)]
        final(success=True, result=len(replies))
    elif scenario == "memory_error":
        final(error="MemoryError", error_type="out_of_memory")
''')


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    script = tmp_path / "fake_sandbox.py"
    script.write_text(FAKE_SANDBOX)
    box = PyodideSandbox(pkg_name="unused.js", skip_deno_check=True)
    monkeypatch.setattr(box, "_build_command", lambda **_kwargs: [sys.executable, str(script)])
    return box


async def echo_dispatcher(name, args_json):
    if name == "t.boom":
        raise RuntimeError("tool exploded")
    if name == "t.slow":
        await asyncio.sleep(30)
    if name == "t.big":
        return json.dumps("y" * 200_000)
    return json.dumps({"tool": name, "args": json.loads(args_json)})


def run(sandbox, scenario, **kwargs):
    kwargs.setdefault("timeout_seconds", 10)
    kwargs.setdefault("max_tool_calls", 10)
    return asyncio.run(sandbox.execute_with_bridge(scenario, kwargs.pop("dispatcher", echo_dispatcher), **kwargs))


def test_round_trip_with_a_caught_tool_error(sandbox):
    seen = []
    result = run(sandbox, "round_trip", on_output=lambda kind, text: seen.append((kind, text)))
    assert result.status == "success" and result.error_type is None
    assert result.result == {"first": {"tool": "t.echo", "args": {"n": 1}}, "failed_ok": False}
    assert result.stdout == "caught tool exploded\n"
    assert seen == [("stdout", "caught tool exploded\n")]
    assert result.calls_completed == 2


def test_concurrent_calls_all_answered(sandbox):
    result = run(sandbox, "gather")
    assert result.status == "success" and result.result == [0, 1, 2]
    assert result.calls_completed == 3


def test_call_limit(sandbox):
    result = run(sandbox, "many_calls", max_tool_calls=5)
    assert result.error_type == "call_limit"
    assert result.calls_completed == 5


def test_timeout_while_a_call_is_pending_kills_the_process(sandbox):
    started = time.time()
    result = run(sandbox, "slow_call", timeout_seconds=1)
    assert result.error_type == "timeout"
    assert time.time() - started < 5


def test_eof_without_final_is_a_crash_and_keeps_partial_output(sandbox):
    result = run(sandbox, "eof")
    assert result.error_type == "sandbox_crashed"
    assert result.stdout == "partial output\n"
    assert "deno panicked" in result.error and "code 3" in result.error


def test_stderr_flood_does_not_deadlock(sandbox):
    result = run(sandbox, "stderr_flood")
    assert result.status == "success" and result.result == "done"
    assert len(result.stderr) <= 64 * 1024


def test_a_line_over_asyncio_default_limit_passes(sandbox):
    result = run(sandbox, "long_line")
    assert result.status == "success" and result.result == 200_000


def test_a_line_over_the_cap_is_result_too_large(sandbox):
    result = run(sandbox, "line_over_cap", max_line_bytes=100_000)
    assert result.error_type == "result_too_large"


def test_killed_during_a_call_is_a_crash(sandbox):
    result = run(sandbox, "killed_during_call")
    assert result.error_type == "sandbox_crashed"


def test_error_type_from_final_is_kept(sandbox):
    result = run(sandbox, "memory_error")
    assert result.status == "error" and result.error_type == "out_of_memory"


def test_cancellation_kills_the_process(sandbox):
    async def scenario():
        task = asyncio.create_task(sandbox.execute_with_bridge(
            "slow_call", echo_dispatcher, timeout_seconds=30, max_tool_calls=5))
        await asyncio.sleep(1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    started = time.time()
    asyncio.run(scenario())
    assert time.time() - started < 5


def test_calls_in_flight_at_the_end_are_reported_as_cancelled(sandbox):
    result = run(sandbox, "fire_and_forget")
    assert result.status == "success" and result.result == "returned early"
    assert result.calls_completed == 0
    assert result.calls_cancelled == 1


def test_concurrent_calls_are_capped(sandbox):
    running = 0
    peak = 0

    async def counted(name, args_json):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.2)
        running -= 1
        return "null"

    result = run(sandbox, "fanout", dispatcher=counted, max_concurrent_calls=2)
    assert result.status == "success" and result.result == 6
    assert peak == 2
    assert result.calls_completed == 6 and result.calls_cancelled == 0


def test_command_without_bridge_is_unchanged():
    box = PyodideSandbox(pkg_name="main.js", skip_deno_check=True)
    cmd = box._build_command(memory_limit_mb=128)
    assert cmd[-2:] == ["main.js", "--stdin"] and "--bridge" not in cmd
    bridged = box._build_command(memory_limit_mb=128, bridge_max_line_bytes=1000)
    assert bridged[-3:] == ["main.js", "--bridge", "--max-line-bytes=1000"]
    assert bridged[:-3] == cmd[:-2]


# Real Deno + Pyodide, same discovery as test_7185_sandbox_isolation.
MAIN_TS = Path(__file__).resolve().parents[3] / "infra" / "data" / "sandbox" / "main.ts"


def _sandbox_base():
    for base in (os.environ.get("SANDBOX_BASE"), "/app"):
        if base and os.path.isfile(os.path.join(base, "pyodide", "pyodide.mjs")):
            return base
    return None


needs_deno = pytest.mark.skipif(
    not shutil.which("deno") or _sandbox_base() is None, reason="needs Deno and a Pyodide distribution",
)

ROUND_TRIP = '''
import asyncio, json, elitea_bridge
print("start")
first = json.loads(await elitea_bridge.call("t.echo", json.dumps({"n": 1})))
try:
    await elitea_bridge.call("t.boom", "{}")
    caught = False
except Exception as exc:
    caught = "tool exploded" in str(exc)
many = await asyncio.gather(*[elitea_bridge.call("t.echo", json.dumps({"i": i})) for i in range(3)])
big = await elitea_bridge.call("t.big2mb", "{}")
{"first": first, "caught": caught, "many": len(many), "big": len(json.loads(big))}
'''


# The production copy lives in indexer_worker; point BRIDGE_MAIN_JS at it to run the same tests against both.
ENTRYPOINTS = [MAIN_TS] + ([Path(os.environ["BRIDGE_MAIN_JS"])] if os.environ.get("BRIDGE_MAIN_JS") else [])


@pytest.fixture(params=ENTRYPOINTS, ids=lambda path: path.name)
def real_sandbox(request, monkeypatch):
    from elitea_sdk.runtime.tools.sandbox import build_offline_pyodide_sandbox
    monkeypatch.setenv("PYODIDE_SANDBOX_PKG", str(request.param))
    monkeypatch.setenv("SANDBOX_BASE", _sandbox_base())
    return build_offline_pyodide_sandbox()


async def real_dispatcher(name, args_json):
    if name == "t.big2mb":
        return json.dumps("z" * 2_000_000)
    return await echo_dispatcher(name, args_json)


def real_run(box, code, **kwargs):
    kwargs.setdefault("timeout_seconds", 120)
    kwargs.setdefault("max_tool_calls", 20)
    return asyncio.run(box.execute_with_bridge(code, real_dispatcher, **kwargs))


@needs_deno
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_real_round_trip(real_sandbox):
    result = real_run(real_sandbox, ROUND_TRIP)
    assert result.status == "success", result.error
    assert result.result == {"first": {"tool": "t.echo", "args": {"n": 1}}, "caught": True, "many": 3, "big": 2_000_000}
    assert result.stdout == "start\n"
    assert result.calls_completed == 6


@needs_deno
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_real_python_exception_keeps_streamed_output(real_sandbox):
    result = real_run(real_sandbox, 'print("before")\nraise ValueError("bad")')
    assert result.error_type == "python_exception" and "ValueError" in result.error
    assert result.stdout == "before\n"


@needs_deno
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_real_oversized_final(real_sandbox):
    result = real_run(real_sandbox, '"y" * 6_000_000')
    assert result.error_type == "result_too_large"


@needs_deno
@pytest.mark.integration
@pytest.mark.timeout(300)
@pytest.mark.parametrize("memory_limit_mb", [256, 64])
def test_real_memory_error(real_sandbox, memory_limit_mb):
    code = "x = []\nwhile True:\n    x.append(bytearray(20_000_000))"
    result = real_run(real_sandbox, code, memory_limit_mb=memory_limit_mb)
    assert result.status == "error"
    # A cap below what Pyodide needs to start fails before user code; it must still be classified.
    assert result.error_type in ("out_of_memory", "sandbox_crashed")
    if memory_limit_mb == 256:
        assert result.error_type == "out_of_memory"


def _bridge_fingerprint(path):
    # String literals and constants carry the protocol (message types, keys, error types, limits);
    # comparing them survives TypeScript annotations, which a line-by-line diff would not.
    source = path.read_text()
    start = source.index("// --bridge:")
    end = source.index("function printHelp()" if path.suffix == ".js" else "async function main()")
    # Timing labels exist only in the production copy.
    block = re.sub(r"markTiming\([^)]*\);", "", source[start:end])
    literals = re.findall(r'"(?:[^"\\]|\\.)*"|`[^`]*`', block)
    constants = re.findall(r"const (BRIDGE_\w+|DEFAULT_BRIDGE_\w+) = ([^;]+);", block)
    keys = re.findall(r"\b(type|id|name|args|ok|value|error|error_type|success|result|text|code)\s*[:,}]", block)
    return sorted(literals), sorted(constants), sorted(set(keys))


@pytest.mark.skipif(not os.environ.get("BRIDGE_MAIN_JS"), reason="set BRIDGE_MAIN_JS to the indexer_worker main.js")
def test_both_entrypoints_carry_the_same_bridge_protocol():
    assert _bridge_fingerprint(MAIN_TS) == _bridge_fingerprint(Path(os.environ["BRIDGE_MAIN_JS"]))
