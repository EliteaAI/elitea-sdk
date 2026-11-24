"""#6718: the sandbox command starts from a Pyodide snapshot only when the worker has built one.

The worker (indexer_worker, flag `sandbox_snapshot`) sets SANDBOX_SNAPSHOT_PATH after a successful
build. The SDK must then pass `--snapshot=<path>` and grant read access to that one file only;
with the variable unset, or pointing at a file that no longer exists, the command is unchanged.
"""
import pytest


@pytest.fixture
def sandbox(monkeypatch):
    from elitea_sdk.runtime.tools.sandbox import build_offline_pyodide_sandbox
    import elitea_sdk.runtime.langchain.pyodide_sandbox as module
    monkeypatch.setenv('PYODIDE_SANDBOX_PKG', 'main.js')
    monkeypatch.setenv('SANDBOX_BASE', '/sandbox')
    monkeypatch.delenv('SANDBOX_SNAPSHOT_PATH', raising=False)
    monkeypatch.setattr(module.shutil, 'which', lambda name: '/usr/bin/deno')
    return build_offline_pyodide_sandbox()


def _reads(command):
    return [flag for flag in command if flag.startswith('--allow-read')]


def test_flag_off_leaves_the_command_unchanged(sandbox):
    command = sandbox._build_command()
    assert not [flag for flag in command if flag.startswith('--snapshot')]
    assert _reads(command) == ['--allow-read=/sandbox/pyodide']


def test_a_missing_snapshot_file_leaves_the_command_unchanged(sandbox, monkeypatch, tmp_path):
    # A worker restart can remove the old file between the env being set and the next run.
    monkeypatch.setenv('SANDBOX_SNAPSHOT_PATH', str(tmp_path / 'gone.bin'))
    command = sandbox._build_command()
    assert not [flag for flag in command if flag.startswith('--snapshot')]
    assert _reads(command) == ['--allow-read=/sandbox/pyodide']


@pytest.mark.parametrize('bridge', [False, True])
def test_a_built_snapshot_is_passed_and_only_that_file_becomes_readable(sandbox, monkeypatch, tmp_path, bridge):
    snapshot = tmp_path / 'abc-1.bin'
    snapshot.write_bytes(b'x')
    monkeypatch.setenv('SANDBOX_SNAPSHOT_PATH', str(snapshot))

    command = sandbox._build_command(bridge_max_line_bytes=1024 if bridge else None)

    assert f'--snapshot={snapshot}' in command
    assert _reads(command) == [f'--allow-read=/sandbox/pyodide,{snapshot}']
    assert not [flag for flag in command if flag.startswith('--allow-write')]
    # The flag must follow the entrypoint, or Deno takes it as its own option.
    assert command.index(f'--snapshot={snapshot}') > command.index('main.js')
