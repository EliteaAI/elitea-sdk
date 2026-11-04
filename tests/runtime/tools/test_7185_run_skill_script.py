"""A loaded skill's Python script runs in the offline sandbox and its results reach the model, the scratch and the user (#7185)."""
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from elitea_sdk.runtime.langchain.constants import SKILL_FILES_RUN_SCRIPTS_NOTE, SKILL_FILES_SCRIPTS_NOTE
from elitea_sdk.runtime.langchain.sandbox_types import CodeExecutionResult
from elitea_sdk.runtime.tool_outcome import ToolResultStatus, outcome_sink
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.tools.skill_script_runner import (
    INLINE_OUTPUT_CHARS,
    OUTPUT_PREVIEW_CHARS,
    SCRATCH_ROOT_PREFIX,
    ScratchCaps,
    ScriptRunContext,
    SkillScriptRunner,
)
from elitea_sdk.runtime.tools.skill_tools import (
    ListSkillFilesTool,
    LoadSkillTool,
    ReadSkillFileTool,
    ReadSkillOutputTool,
    RunSkillScriptTool,
    build_load_skill_tools,
    loaded_skill_names_from_messages,
)

CONVERSATION = 'conv-uuid-1'
BUCKET = 'attachments'
STEP1 = '''import json, os, sys
print("step1: progress", file=sys.stderr)
os.makedirs("out", exist_ok=True)
token = sys.argv[1] if len(sys.argv) > 1 else "none"
with open("out/step1.json", "w") as handle:
    json.dump({"token": token, "skill_dir": os.environ.get("CLAUDE_SKILL_DIR")}, handle)
print(json.dumps({"token": token, "file": "out/step1.json", "cwd": os.getcwd()}))
'''
STEP2 = '''import json, sys
path = sys.argv[sys.argv.index("--input") + 1]
with open(path) as handle:
    data = json.load(handle)
print("STEP2_RESULT=" + data["token"] * 2)
'''
HELPER = 'VALUE = 7\n'
USES_HELPER = 'import helper\nprint("helper", helper.VALUE)\n'
SHARE = '''import os
os.makedirs("output/sub", exist_ok=True)
with open("output/deck.txt", "w") as handle:
    handle.write("deck")
with open("output/sub/keep.txt", "w") as handle:
    handle.write("scratch")
print("shared")
'''
FAIL = 'import sys\nprint("partial")\nprint("ERR: broken", file=sys.stderr)\nsys.exit(3)\n'
RAISES = 'raise ValueError("boom")\n'
BIG = 'for i in range(1, 1201):\n    print(f"LINE{i:04d} " + "x" * 40)\nprint("NEEDLE=abc123")\n'
READ_INPUT = 'import sys\nprint(open("inputs/" + sys.argv[1]).read())\n'
DELETE = 'import os\nos.remove("out/step1.json")\nprint("deleted")\n'
SLOW = 'import time\ntime.sleep(5)\n'
NETWORK = 'import urllib.request\nurllib.request.urlopen("https://example.com")\n'
CLIENT = 'print(elitea_client)\n'

SKILL_FILES = {
    'SKILL.md': '# demo', 'scripts/step1.py': STEP1, 'scripts/step2.py': STEP2, 'scripts/helper.py': HELPER,
    'scripts/uses_helper.py': USES_HELPER, 'scripts/share.py': SHARE, 'scripts/fail.py': FAIL,
    'scripts/raises.py': RAISES, 'scripts/big.py': BIG, 'scripts/read_input.py': READ_INPUT,
    'scripts/delete.py': DELETE, 'scripts/slow.py': SLOW, 'scripts/network.py': NETWORK, 'scripts/client.py': CLIENT,
    'scripts/run.sh': 'echo no\n', 'reference.md': 'ref',
}


def _manifest(files):
    return [
        {'path': path, 'size': len(content), 'content_type': 'text/plain', 'is_text': True}
        for path, content in files.items()
    ]


def _skill(name='demo', files=SKILL_FILES, skill_id=5):
    return {
        'skill_id': skill_id, 'skill_version_id': 50 + skill_id, 'skill_project_id': 1, 'name': name,
        'description': f'{name} skill', 'instructions': 'Run scripts/step1.py then scripts/step2.py.',
        'files': _manifest(files),
    }


DEMO = _skill()
SHELL_ONLY = _skill('shell', {'SKILL.md': '# shell', 'scripts/run.sh': 'echo'}, skill_id=6)
PLAIN = _skill('tone', {}, skill_id=7)


class FakeClient:
    """An in-memory artifact store with the S3-style methods the runner uses, plus the skill file fetch."""

    def __init__(self, skills=(DEMO,), attachments=None):
        self.objects = {}
        self.buckets = {BUCKET}
        self.calls = []
        self._skill_scratch_cache = {}
        self._files = {skill['name']: {path: content for path, content in SKILL_FILES.items()} for skill in skills}
        self._skills = {skill['skill_id']: skill for skill in skills}
        for name, data in (attachments or {}).items():
            self.objects[(BUCKET, f"{CONVERSATION}/{name}")] = (data, '2026-11-01T00:00:00Z')

    def get_skill_file(self, skill_project_id, skill_id, version_id, path):
        self.calls.append(('get_skill_file', skill_id, path))
        return self._files[self._skills[skill_id]['name']][path].encode()

    def upload_artifact_s3(self, bucket, key, data, content_type=None):
        self.calls.append(('upload', key))
        if bucket not in self.buckets:
            return {'error': f'bucket {bucket} not found'}
        self.objects[(bucket, key)] = (data, '2026-11-02T00:00:00Z')
        return {'filepath': f'/{bucket}/{key}', 'bucket': bucket, 'filename': key, 'size': len(data)}

    def download_artifact_s3(self, bucket, key):
        self.calls.append(('download', key))
        if (bucket, key) not in self.objects:
            return {'error': 'Resource not found (404)'}
        return self.objects[(bucket, key)][0]

    def delete_artifact_s3(self, bucket, key):
        self.calls.append(('delete', key))
        self.objects.pop((bucket, key), None)
        return {'success': True}

    def list_artifacts_s3(self, bucket, prefix=None, delimiter='/'):
        prefix = prefix if not prefix or prefix.endswith('/') else f'{prefix}/'
        return {'contents': [
            {'key': key, 'size': len(data), 'lastModified': modified}
            for (object_bucket, key), (data, modified) in self.objects.items()
            if object_bucket == bucket and key.startswith(prefix or '')
        ]}

    def create_bucket(self, bucket, expiration_measure='months', expiration_value=1):
        self.calls.append(('create_bucket', bucket, expiration_measure, expiration_value))
        self.buckets.add(bucket)
        return {'name': bucket}

    def scratch_files(self):
        data = self.objects.get((BUCKET, f'{SCRATCH_ROOT_PREFIX}/{CONVERSATION}/scratch.zip'))
        if data is None:
            return {}
        with zipfile.ZipFile(io.BytesIO(data[0])) as archive:
            return {info.filename: archive.read(info) for info in archive.infolist()}


DRIVER = '''import ast, asyncio, json, sys
tree = ast.parse(sys.stdin.read())
last = ast.Expression(tree.body.pop().value)
namespace = {}
exec(compile(tree, "<program>", "exec"), namespace)
value = eval(compile(last, "<program>", "eval", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)
if asyncio.iscoroutine(value):
    value = asyncio.run(value)
sys.stdout.write(json.dumps(value))
'''


class CPythonSandbox:
    """Runs the staged program under the test interpreter so the wrapper's behaviour is proven without Deno."""

    def __init__(self, events=None):
        self.events = events if events is not None else []

    async def execute(self, program, *, timeout_seconds=None, memory_limit_mb=None, **_kwargs):
        self.events.append(('execute', timeout_seconds, memory_limit_mb))
        try:
            completed = subprocess.run(
                [sys.executable, '-c', DRIVER], input=program, capture_output=True, text=True, timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return CodeExecutionResult(status='error', execution_time=timeout_seconds, stderr='timed out', timed_out=True)
        if completed.returncode != 0:
            return CodeExecutionResult(status='error', execution_time=0.1, stderr=completed.stderr)
        return CodeExecutionResult(status='success', execution_time=0.1, result=json.loads(completed.stdout))


LIMITS = {
    'timeout_seconds': 55.0, 'wasm_max_mem_mb': 512, 'max_concurrent': 0, 'memory_pressure_pct': 0,
    'root_ca_path': None, 'allowed_pyodide_domains': [],
}


def _runner(client=None, tmp_path=None, caps=None, limits=None, sandbox=None, call_ids=('call1', 'call2', 'call3')):
    client = client or FakeClient()
    ids = iter(call_ids)
    runner = SkillScriptRunner(
        ScriptRunContext(client=client, bucket=BUCKET, conversation_key=CONVERSATION),
        file_fetcher=client.get_skill_file, sandbox_factory=lambda: sandbox or CPythonSandbox(),
        limits=limits or LIMITS, caps=caps, call_id_factory=lambda: next(ids), sandbox_base=str(tmp_path),
    )
    return runner, client


def _tools(runner, attached=(DEMO,), loaded=('demo',)):
    tools = build_load_skill_tools(list(attached), skill_file_fetcher=runner.file_fetcher, skill_script_runner=runner)
    for tool in tools:
        if hasattr(tool, 'mark_already_loaded'):
            tool.mark_already_loaded(loaded)
    return tools


def _run_tool(runner, **kwargs):
    return next(tool for tool in _tools(runner, **kwargs) if isinstance(tool, RunSkillScriptTool))


def _read_output_tool(runner):
    return next(tool for tool in _tools(runner) if isinstance(tool, ReadSkillOutputTool))


def test_a_skill_with_a_python_script_binds_the_run_and_read_output_tools(tmp_path):
    runner, _client = _runner(tmp_path=tmp_path)

    tools = build_load_skill_tools([DEMO, PLAIN], skill_script_runner=runner)

    assert [type(tool) for tool in tools] == [
        LoadSkillTool, ReadSkillFileTool, ListSkillFilesTool, RunSkillScriptTool, ReadSkillOutputTool,
    ]
    assert tools[0].scripts_runnable is True
    for tool in tools[3:]:
        assert (tool.metadata['toolkit_type'], tool.metadata['toolkit_name']) == ('internal', 'skills')


def test_without_a_runner_or_without_python_scripts_the_run_tools_stay_unbound(tmp_path):
    runner, _client = _runner(tmp_path=tmp_path)

    assert [type(tool) for tool in build_load_skill_tools([DEMO])] == [LoadSkillTool, ReadSkillFileTool, ListSkillFilesTool]
    assert [type(tool) for tool in build_load_skill_tools([SHELL_ONLY], skill_script_runner=runner)] == [
        LoadSkillTool, ReadSkillFileTool, ListSkillFilesTool,
    ]
    assert build_load_skill_tools([SHELL_ONLY], skill_script_runner=runner)[0].scripts_runnable is False


def test_the_loaded_body_tells_the_model_how_to_run_scripts_when_the_runner_is_bound(tmp_path):
    runner, _client = _runner(tmp_path=tmp_path)
    loader = build_load_skill_tools([DEMO, SHELL_ONLY], skill_script_runner=runner)[0]

    body = loader.load('demo')
    shell_body = loader.load('shell')

    assert SKILL_FILES_RUN_SCRIPTS_NOTE.format(name='demo', example='scripts/big.py') in body
    assert SKILL_FILES_SCRIPTS_NOTE not in body
    assert SKILL_FILES_SCRIPTS_NOTE in shell_body
    assert '/skills/shell/' not in shell_body


def test_the_description_states_the_sandbox_contract():
    description = RunSkillScriptTool.model_fields['description'].default

    for phrase in ('no network', 'Exit code N', 'output/', 'inputs/', '/skills/<name>/', 'Only .py files run'):
        assert phrase in description


def test_a_script_runs_with_its_arguments_and_its_output_comes_back_after_the_exit_code(tmp_path):
    runner, client = _runner(tmp_path=tmp_path)

    result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/step1.py', 'args': ['42'], 'files': []})

    head, printed, heading, progress = result.split('\n')
    assert head == 'Exit code 0'
    assert json.loads(printed) == {'token': '42', 'file': 'out/step1.json', 'cwd': str(tmp_path / 'work')}
    assert (heading, progress) == ('[stderr]', 'step1: progress')


def test_a_file_the_first_script_wrote_is_staged_back_for_the_second_call(tmp_path):
    runner, client = _runner(tmp_path=tmp_path)
    tool = _run_tool(runner)
    tool.invoke({'skill': 'demo', 'path': 'scripts/step1.py', 'args': ['42']})
    assert json.loads(client.scratch_files()['out/step1.json'])['token'] == '42'
    assert json.loads(client.scratch_files()['out/step1.json'])['skill_dir'] == str(tmp_path / 'skills' / 'demo')

    result = tool.invoke({'skill': 'demo', 'path': 'scripts/step2.py', 'args': ['--input', 'out/step1.json']})

    assert result == 'Exit code 0\nSTEP2_RESULT=4242'


def test_a_script_imports_its_sibling_module(tmp_path):
    runner, _client = _runner(tmp_path=tmp_path)

    assert _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/uses_helper.py'}) == 'Exit code 0\nhelper 7'


def test_a_file_deleted_by_a_script_leaves_the_scratch(tmp_path):
    runner, client = _runner(tmp_path=tmp_path)
    tool = _run_tool(runner)
    tool.invoke({'skill': 'demo', 'path': 'scripts/step1.py', 'args': ['1']})

    result = tool.invoke({'skill': 'demo', 'path': 'scripts/delete.py'})

    assert result == 'Exit code 0\ndeleted'
    assert client.scratch_files() == {}


def test_files_under_output_become_attachments_and_subfolders_stay_scratch(tmp_path):
    runner, client = _runner(tmp_path=tmp_path)

    result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/share.py'})

    assert result == 'Exit code 0\nshared\nFiles shared: deck.txt (4 B)'
    assert client.objects[(BUCKET, f'{CONVERSATION}/call1/deck.txt')][0] == b'deck'
    assert client.scratch_files() == {'output/sub/keep.txt': b'scratch'}


def test_a_promoted_file_is_announced_to_the_conversation_as_file_modified(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr('elitea_sdk.runtime.tools.skill_script_runner.dispatch_custom_event',
                        lambda name, payload: events.append((name, payload)))
    runner, _client = _runner(tmp_path=tmp_path)

    _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/share.py'})

    [(name, payload)] = events
    assert name == 'file_modified'
    assert payload['filepath'] == f'/{BUCKET}/{CONVERSATION}/call1/deck.txt'
    assert (payload['tool_name'], payload['toolkit'], payload['operation_type']) == ('run_skill_script', 'skills', 'create')
    assert payload['meta'] == {'bucket': BUCKET, 'file_size': 4, 'source': 'generated'}


def test_a_promoted_file_gets_a_key_no_earlier_call_shares(tmp_path):
    runner, client = _runner(tmp_path=tmp_path)
    tool = _run_tool(runner)

    tool.invoke({'skill': 'demo', 'path': 'scripts/share.py'})
    tool.invoke({'skill': 'demo', 'path': 'scripts/share.py'})

    assert {key for bucket, key in client.objects if key.endswith('deck.txt')} == {
        f'{CONVERSATION}/call1/deck.txt', f'{CONVERSATION}/call2/deck.txt',
    }


def test_a_named_attachment_is_staged_under_inputs_newest_copy_first(tmp_path):
    client = FakeClient(attachments={'notes.txt': b'uploaded'})
    client.objects[(BUCKET, f'{CONVERSATION}/call0/notes.txt')] = (b'promoted later', '2026-11-03T00:00:00Z')
    runner, _client = _runner(client=client, tmp_path=tmp_path)

    result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/read_input.py', 'args': ['notes.txt'], 'files': ['notes.txt']})

    assert result == 'Exit code 0\npromoted later'
    assert 'inputs/notes.txt' not in client.scratch_files()


def test_an_unknown_attachment_name_lists_what_is_available(tmp_path):
    client = FakeClient(attachments={'deck.pptx': b'x', 'notes.txt': b'y'})
    runner, _client = _runner(client=client, tmp_path=tmp_path)

    result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/read_input.py', 'files': ['missing.pdf']})

    assert result == 'No attachment named "missing.pdf" is in this conversation. Attachments available: deck.pptx, notes.txt.'
    assert not any(call[0] == 'upload' for call in client.calls)


def test_a_non_zero_exit_keeps_the_output_and_flags_the_result_as_an_error(tmp_path):
    runner, _client = _runner(tmp_path=tmp_path)

    with outcome_sink() as recorded:
        result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/fail.py'})

    assert result == 'Exit code 3\npartial\n[stderr]\nERR: broken'
    assert [outcome.status for outcome in recorded] == [ToolResultStatus.ERROR]
    assert recorded[0].message == 'Exit code 3'


def test_an_uncaught_exception_is_exit_code_1_with_the_traceback(tmp_path):
    runner, _client = _runner(tmp_path=tmp_path)

    result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/raises.py'})

    assert result.startswith('Exit code 1\n[stderr]\nTraceback')
    assert result.endswith('ValueError: boom')


def test_a_script_without_output_says_so(tmp_path):
    runner, _client = _runner(tmp_path=tmp_path)

    assert _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/helper.py'}) == 'Exit code 0\n(no output)'


def test_a_timeout_is_an_error_that_says_nothing_was_captured(tmp_path):
    runner, _client = _runner(tmp_path=tmp_path, limits={**LIMITS, 'timeout_seconds': 1.0})

    with outcome_sink() as recorded:
        result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/slow.py'})

    assert result == 'Exit code -1 (timed out after 1 s)\nNo output was captured; the script was stopped.'
    assert [outcome.status for outcome in recorded] == [ToolResultStatus.ERROR]


def test_output_over_the_ceiling_is_saved_and_previewed_then_readable_by_range(tmp_path):
    runner, client = _runner(tmp_path=tmp_path)

    result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/big.py'})

    lines = result.split('\n')
    assert lines[0] == 'Exit code 0'
    assert lines[1].startswith('Output too large (') and 'skill-output://call1' in lines[1]
    assert f'Preview (first {OUTPUT_PREVIEW_CHARS} characters):' in result
    assert 'NEEDLE' not in result
    assert len(result) < INLINE_OUTPUT_CHARS
    reader = _read_output_tool(runner)
    tail = reader.invoke({'ref': 'skill-output://call1', 'start_line': 1200, 'end_line': 1201})
    assert tail.startswith('Script output skill-output://call1 lines 1200-1201:\n<script_output>\n')
    assert 'NEEDLE=abc123' in tail
    assert reader.invoke({'ref': 'skill-output://nope'}).startswith('"skill-output://nope" is not an available saved script output.')
    downloads_before = sum(1 for call in client.calls if call[0] == 'download')
    assert reader.invoke({'ref': 'call1'}).startswith('"call1" is not an available saved script output.')
    assert reader.invoke({'ref': 'skill-output://../../call1'}).startswith('"skill-output://../../call1" is not an available')
    assert sum(1 for call in client.calls if call[0] == 'download') == downloads_before


def test_a_shell_script_is_refused_with_a_python_only_message(tmp_path):
    runner, client = _runner(tmp_path=tmp_path)

    result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/run.sh'})

    assert result == (
        '"scripts/run.sh" cannot run here: only Python (.py) scripts run on this platform. Read it with '
        'read_skill_file and follow it yourself.'
    )
    assert client.calls == []


def test_an_unknown_script_path_suggests_the_closest_files(tmp_path):
    runner, _client = _runner(tmp_path=tmp_path)

    result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/step3.py'})

    assert result.startswith('Skill "demo" has no file "scripts/step3.py". Did you mean: ')
    assert '"scripts/step1.py"' in result and '"scripts/step2.py"' in result


def test_a_path_outside_the_skill_is_refused(tmp_path):
    runner, _client = _runner(tmp_path=tmp_path)

    result = _run_tool(runner).invoke({'skill': 'demo', 'path': '../other/x.py'})

    assert result.startswith('Path "../other/x.py" is outside skill "demo"')


def test_running_a_script_of_a_skill_not_loaded_yet_loads_it_first(tmp_path):
    runner, _client = _runner(tmp_path=tmp_path)

    result = _run_tool(runner, loaded=()).invoke({'skill': 'demo', 'path': 'scripts/helper.py'})

    assert result.startswith('Skill "demo" is now active')
    assert result.endswith('\n\nExit code 0\n(no output)')
    assert loaded_skill_names_from_messages([
        AIMessage(content='', tool_calls=[{'id': 'c1', 'name': 'run_skill_script', 'args': {}}]),
        ToolMessage(content=result, tool_call_id='c1'),
    ]) == {'demo'}


def _seed_scratch(client, files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    client.objects[(BUCKET, f'{SCRATCH_ROOT_PREFIX}/{CONVERSATION}/scratch.zip')] = (buffer.getvalue(), '2026-11-01T00:00:00Z')


def test_a_scratch_over_its_cap_stops_the_script_before_it_starts(tmp_path):
    client = FakeClient()
    _seed_scratch(client, {'big.bin': b'x' * 200})
    runner, _client = _runner(client=client, tmp_path=tmp_path, caps=ScratchCaps(max_total_bytes=100))

    with outcome_sink() as recorded:
        result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/helper.py'})

    assert result == (
        "The script cannot start: the conversation's script storage holds 200 B in 1 files, over the limit of "
        "100 B and 200 files. A script may delete files it no longer needs."
    )
    assert [outcome.status for outcome in recorded] == [ToolResultStatus.ERROR]
    assert not any(call[0] == 'get_skill_file' for call in client.calls)


def test_files_that_would_overflow_the_scratch_are_not_kept_and_the_model_is_told(tmp_path):
    runner, client = _runner(tmp_path=tmp_path, caps=ScratchCaps(max_total_bytes=60))

    result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/step1.py', 'args': ['y' * 100]})

    assert result.split('\n')[0] == 'Exit code 0'
    assert "Files written by this call were not kept" in result
    assert client.scratch_files() == {}


def test_an_attachment_over_the_per_file_cap_is_refused(tmp_path):
    client = FakeClient(attachments={'huge.bin': b'z' * 50})
    runner, _client = _runner(client=client, tmp_path=tmp_path, caps=ScratchCaps(max_file_bytes=10))

    result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/read_input.py', 'files': ['huge.bin']})

    assert result == 'The script cannot start: attachment "huge.bin" is 50 B, over the 10 B limit for one staged file.'


def test_a_missing_attachment_bucket_is_created_before_the_first_scratch_write(tmp_path):
    client = FakeClient()
    client.buckets.clear()
    runner, _client = _runner(client=client, tmp_path=tmp_path)

    _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/step1.py', 'args': ['1']})

    assert ('create_bucket', BUCKET, 'days', 365) in client.calls
    assert 'out/step1.json' in client.scratch_files()


def test_the_sandbox_gets_the_platform_limits_and_no_network(tmp_path):
    events = []
    sandbox = CPythonSandbox(events)
    runner, _client = _runner(tmp_path=tmp_path, sandbox=sandbox, limits={**LIMITS, 'timeout_seconds': 30.0, 'wasm_max_mem_mb': 256})

    _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/helper.py'})

    assert events == [('execute', 30.0, 256)]


def test_the_offline_sandbox_denies_network_and_subprocesses(monkeypatch):
    from elitea_sdk.runtime.tools.sandbox import build_offline_pyodide_sandbox
    import elitea_sdk.runtime.langchain.pyodide_sandbox as module
    monkeypatch.setenv('PYODIDE_SANDBOX_PKG', 'main.js')
    monkeypatch.delenv('SANDBOX_SERVICE_URL', raising=False)
    monkeypatch.setattr(module.shutil, 'which', lambda name: '/usr/bin/deno')

    sandbox = build_offline_pyodide_sandbox()

    assert (sandbox.allow_net, sandbox.allow_run, sandbox.allow_ffi, sandbox.stateful) == (False, False, False, False)
    assert sandbox.allow_env == ['SANDBOX_BASE']
    assert '--allow-net' not in sandbox._build_command()


def test_the_offline_sandbox_ignores_the_remote_sandbox_service(monkeypatch):
    from elitea_sdk.runtime.tools.sandbox import build_offline_pyodide_sandbox
    import elitea_sdk.runtime.langchain.pyodide_sandbox as module
    monkeypatch.setenv('PYODIDE_SANDBOX_PKG', 'main.js')
    monkeypatch.setenv('SANDBOX_SERVICE_URL', 'https://sandbox.example.invalid/run')
    monkeypatch.setattr(module.shutil, 'which', lambda name: '/usr/bin/deno')

    sandbox = build_offline_pyodide_sandbox()

    assert isinstance(sandbox, module.PyodideSandbox)
    assert sandbox.allow_net is False


def test_a_pod_without_the_local_sandbox_reports_the_script_as_unavailable(tmp_path):
    def missing_deno():
        raise RuntimeError('Deno is required for PyodideSandbox but is not installed.')
    runner, _client = _runner(tmp_path=tmp_path)
    runner.sandbox_factory = missing_deno

    with outcome_sink() as recorded:
        result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/helper.py'})

    assert result == 'The script could not be run: the sandbox is not available right now.'
    assert [outcome.status for outcome in recorded] == [ToolResultStatus.ERROR]


def test_a_sandbox_that_never_ran_the_program_is_reported_as_unavailable(tmp_path):
    broken = MagicMock()

    async def execute(program, **kwargs):
        return CodeExecutionResult(status='error', execution_time=0.1, stderr='deno exploded')
    broken.execute = execute
    runner, _client = _runner(tmp_path=tmp_path, sandbox=broken)

    with outcome_sink() as recorded:
        result = _run_tool(runner).invoke({'skill': 'demo', 'path': 'scripts/helper.py'})

    assert result == 'The script could not be run: the sandbox is not available right now.'
    assert [outcome.status for outcome in recorded] == [ToolResultStatus.ERROR]


def _node(client, **fields):
    return LLMNode(client=MagicMock(), name='llm', available_tools=[], input_mapping={}, input_variables=[],
                   output_variables=['messages'], return_type='dict', elitea_client=client, **fields)


def test_the_llm_node_builds_a_runner_from_the_conversation_and_the_attachment_bucket():
    client = FakeClient()
    node = _node(client)

    runner = node._build_skill_script_runner({'thread_id': 'conv-7', 'attachment_bucket': 'chat-files'})
    root_runner = node._build_skill_script_runner({'thread_id': 'child-1', '__parallel_hitl_root_thread_id__': 'conv-7'})
    child_runner = node._build_skill_script_runner({
        'conversation_id': 'conv-uuid', 'thread_id': 'conv-uuid:dispatch_ab12:My Agent:call_1',
        '__parallel_hitl_root_thread_id__': 'conv-uuid:dispatch_ab12:My Agent:call_1',
    })

    assert (runner.context.bucket, runner.context.conversation_key, runner.context.client) == ('chat-files', 'conv-7', client)
    assert (root_runner.context.bucket, root_runner.context.conversation_key) == (BUCKET, 'conv-7')
    assert child_runner.context.conversation_key == 'conv-uuid'
    assert node._build_skill_script_runner({}) is None
    assert _node(None)._build_skill_script_runner({'thread_id': 'conv-7'}) is None
