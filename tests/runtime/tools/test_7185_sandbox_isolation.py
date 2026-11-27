"""Two real sandbox runs on one worker: nothing the first run leaves behind reaches the second (#7185).

Each run tries to leave something behind and reports what an earlier run left. The worker URL is the same in every
run, since Deno keys a worker's cache storage by it. Runs the repository's Deno entrypoint with the Pyodide
distribution of the SDK image (/app/pyodide), so it runs in the CI image and is skipped where either is missing.
"""
import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from elitea_sdk.runtime.tools.sandbox import PyodideSandboxTool, build_offline_pyodide_sandbox

MAIN_TS = Path(__file__).resolve().parents[3] / 'infra' / 'data' / 'sandbox' / 'main.ts'


def _sandbox_base():
    for base in (os.environ.get('SANDBOX_BASE'), '/app'):
        if base and os.path.isfile(os.path.join(base, 'pyodide', 'pyodide.mjs')):
            return base
    return None


SANDBOX_BASE = _sandbox_base()
pytestmark = pytest.mark.skipif(
    not shutil.which('deno') or SANDBOX_BASE is None, reason='needs Deno and a Pyodide distribution',
)

WORKER_CACHE = r'''await run_js("""new Promise((done) => {
  const source = "caches.open('left').then((c) => c.match('http://x/a').then((r) => r ? r.text() : null)"
    + ".then((earlier) => c.put('http://x/a', new Response('from an earlier run')).then(() => postMessage(earlier))))"
    + ".catch(() => postMessage(null))";
  const worker = new Worker('data:application/javascript,' + encodeURIComponent(source), {type: 'module'});
  worker.onmessage = (event) => { done(event.data); worker.terminate(); };
  worker.onerror = (event) => { event.preventDefault(); done(null); };
})""")'''

LEAVE_AND_LOOK = r'''
import js, json
from pyodide.code import run_js
found = {}
for path in (js.Deno.env.get("SANDBOX_BASE") + "/tmp/left.txt", "/tmp/left.txt"):
    try:
        found[path] = js.Deno.readTextFileSync(path)
    except Exception:
        found[path] = None
    try:
        js.Deno.writeTextFileSync(path, "from an earlier run")
    except Exception:
        pass
try:
    found["localStorage"] = js.localStorage.getItem("left")
    js.localStorage.setItem("left", "from an earlier run")
except Exception:
    found["localStorage"] = None
found["worker cache"] = ''' + WORKER_CACHE + r'''
print(json.dumps(found))
'''


def _python_tool_sandbox():
    tool = PyodideSandboxTool(allow_net=False)
    tool._initialize_sandbox()
    return tool._sandbox


@pytest.mark.timeout(240)
@pytest.mark.parametrize('build_sandbox', [build_offline_pyodide_sandbox, _python_tool_sandbox],
                         ids=['skill-script', 'python-code'])
def test_a_later_run_finds_nothing_an_earlier_run_left_behind(monkeypatch, build_sandbox):
    monkeypatch.setenv('PYODIDE_SANDBOX_PKG', str(MAIN_TS))
    monkeypatch.setenv('SANDBOX_BASE', SANDBOX_BASE)
    monkeypatch.delenv('SANDBOX_SERVICE_URL', raising=False)

    async def two_runs():
        first = await build_sandbox().execute(LEAVE_AND_LOOK, timeout_seconds=120)
        second = await build_sandbox().execute(LEAVE_AND_LOOK, timeout_seconds=120)
        return first, second

    first, second = asyncio.run(two_runs())

    assert first.status == 'success', first.stderr
    assert second.status == 'success', second.stderr
    found = json.loads(second.stdout.strip().splitlines()[-1])
    assert found == dict.fromkeys(found, None)
    assert set(found) == {f'{SANDBOX_BASE}/tmp/left.txt', '/tmp/left.txt', 'localStorage', 'worker cache'}
