"""Runs inside the Pyodide sandbox, never in the worker: unpacks the staged skill and working files, runs one skill
script with its own stdout and stderr, and reports its exit code with every working file it added, changed or
deleted, naming instead of sending the ones over the size caps (#7185). The runner submits this file's source followed by ``await run({...})``."""
import base64
import contextlib
import hashlib
import io
import os
import runpy
import sys
import traceback
import zipfile

try:
    import prepare_env
    from pyodide.code import find_imports
except ImportError:
    prepare_env = None


async def run(params: dict) -> dict:
    """The script's exit code and output, and how it changed the working directory."""
    zipfile.ZipFile(io.BytesIO(base64.b64decode(params['stage']))).extractall(params['base'])
    work, skill_root, script = params['work'], params['skill_root'], params['script']
    for folder in params['folders']:
        os.makedirs(os.path.join(work, folder), exist_ok=True)
    os.chdir(work)
    os.environ['CLAUDE_SKILL_DIR'] = os.environ['ELITEA_SKILL_DIR'] = skill_root
    sys.argv = [script, *params['argv']]
    sys.path[:0] = [os.path.dirname(script), skill_root]
    await load_bundled_packages(skill_root, script)
    exit_code, stdout, stderr = run_script(script)
    changed, too_large, deleted = collect_changes(
        work, params['manifest'], params['inputs'], params['max_file_bytes'], params['max_total_bytes'],
    )
    return {
        'exit_code': exit_code, 'stdout': stdout, 'stderr': stderr,
        'files': changed, 'too_large': too_large, 'deleted': deleted,
    }


async def load_bundled_packages(skill_root: str, script: str) -> None:
    """Load the Pyodide packages the skill's modules import, since the sandbox only scans the submitted source."""
    if prepare_env is None:
        return
    wanted = set()
    for root, _dirs, names in os.walk(skill_root):
        for name in names:
            if not name.endswith('.py'):
                continue
            with open(os.path.join(root, name), encoding='utf-8', errors='replace') as handle:
                with contextlib.suppress(SyntaxError):
                    wanted.update(find_imports(handle.read()))
    local = {
        entry[:-3] if entry.endswith('.py') else entry
        for folder in (os.path.dirname(script), skill_root)
        for entry in os.listdir(folder)
    }
    for module in sorted(wanted):
        if module.split('.', 1)[0] in local:
            continue
        with contextlib.suppress(Exception):
            await prepare_env.install_imports([module])


def run_script(script: str) -> tuple:
    """Run the script as __main__ and turn SystemExit and uncaught exceptions into an exit code, as a shell would."""
    stdout, stderr = io.StringIO(), io.StringIO()
    exit_code = 0
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            runpy.run_path(script, run_name='__main__')
        except SystemExit as exit_request:
            if isinstance(exit_request.code, int):
                exit_code = exit_request.code
            elif exit_request.code is not None:
                print(exit_request.code, file=sys.stderr)
                exit_code = 1
        except BaseException:  # pylint: disable=broad-except
            traceback.print_exc()
            exit_code = 1
    return exit_code, stdout.getvalue(), stderr.getvalue()


def collect_changes(work: str, manifest: dict, inputs_folder: str, max_file_bytes: int, max_total_bytes: int) -> tuple:
    """Working files that differ from the staged manifest, base64-encoded; the changed ones left out because they pass
    a size cap, since encoding them could exhaust the sandbox's memory; and the staged paths no longer there."""
    changed, too_large = {}, []
    total = 0
    for root, _dirs, names in os.walk(work):
        for name in names:
            full = os.path.join(root, name)
            relative = os.path.relpath(full, work).replace(os.sep, '/')
            if relative.split('/', 1)[0] == inputs_folder:
                continue
            if os.path.getsize(full) > max_file_bytes:
                too_large.append(relative)
                continue
            with open(full, 'rb') as handle:
                data = handle.read()
            if manifest.get(relative) == hashlib.sha256(data).hexdigest():
                continue
            if total + len(data) > max_total_bytes:
                too_large.append(relative)
                continue
            total += len(data)
            changed[relative] = base64.b64encode(data).decode('ascii')
    deleted = [path for path in manifest if not os.path.exists(os.path.join(work, path))]
    return changed, sorted(too_large), deleted
