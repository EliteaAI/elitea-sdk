"""Runs a skill's bundled Python script in the offline sandbox and carries its results back (#7185).

Every call rebuilds the sandbox filesystem: the skill's files at /skills/<name>/, the conversation's scratch
(one zip object per conversation in the attachment bucket) as the working directory, named attachments under
inputs/. After the run the changed scratch files go back into the zip, files written directly under output/
become conversation attachments, and output over the inline ceiling is saved next to the scratch for
read_skill_output.
"""

import asyncio
import base64
import concurrent.futures
import hashlib
import io
import json
import logging
import mimetypes
import posixpath
import time
import uuid
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain_core.callbacks.manager import dispatch_custom_event

from ..exceptions import SandboxAdmissionRefused, SkillBodyUnavailable
from ..langchain.constants import (
    RUN_SKILL_SCRIPT_FILES_SHARED,
    RUN_SKILL_SCRIPT_INPUT_NONE,
    RUN_SKILL_SCRIPT_INPUT_NOT_FOUND,
    RUN_SKILL_SCRIPT_INPUT_OVER_CAP,
    RUN_SKILL_SCRIPT_NO_OUTPUT,
    RUN_SKILL_SCRIPT_OUTPUT_SAVED,
    RUN_SKILL_SCRIPT_RESULT_PREFIX,
    RUN_SKILL_SCRIPT_SCRATCH_CAP_EXCEEDED,
    RUN_SKILL_SCRIPT_SCRATCH_OVER_CAP,
    RUN_SKILL_SCRIPT_STDERR_HEADING,
    RUN_SKILL_SCRIPT_TIMEOUT_RESULT,
    RUN_SKILL_SCRIPT_UNAVAILABLE,
    RUN_SKILL_SCRIPT_UNAVAILABLE_SANDBOX,
    RUN_SKILL_SCRIPT_UNAVAILABLE_SCRATCH,
    RUN_SKILL_SCRIPT_UNAVAILABLE_STAGING,
    SKILL_OUTPUT_URI,
    SKILL_SCRATCH_MAX_FILE_BYTES,
    SKILL_SCRATCH_MAX_FILES,
    SKILL_SCRATCH_MAX_TOTAL_BYTES,
    SKILL_SCRIPT_INLINE_OUTPUT_CHARS,
    SKILL_SCRIPT_INPUTS_FOLDER,
    SKILL_SCRIPT_OUTBOX_FOLDER,
    SKILL_SCRIPT_OUTPUT_PREVIEW_CHARS,
    SKILL_SCRIPT_SKILLS_ROOT,
    SKILL_SCRIPT_WORK_ROOT,
)
from ..tool_outcome import ToolErrorClass, ToolOutcome, ToolResultStatus, record_outcome
from .sandbox import _read_sandbox_limits_from_env, build_offline_pyodide_sandbox, enforce_sandbox_admission

logger = logging.getLogger(__name__)

SCRATCH_ROOT_PREFIX = "_skill_scratch"
SCRATCH_ZIP_NAME = "scratch.zip"
SCRATCH_OUTPUTS_FOLDER = "outputs"
STAGE_SKILLS_FOLDER = "skills"
STAGE_WORK_FOLDER = "work"
INPUT_LISTING_LIMIT = 20
RUN_SKILL_SCRIPT_TOOL_NAME = "run_skill_script"

# Runs inside the sandbox as the whole submitted program: the staged zip is unpacked at BASE, the script runs
# with its own stdout and stderr captured, and the last expression (the dict) is the sandbox's result.
SANDBOX_PROGRAM = r'''
import base64, contextlib, hashlib, io, json, os, runpy, sys, traceback, zipfile
BASE = {base!r}
STAGE = {stage!r}
SCRIPT = {script!r}
ARGV = {argv!r}
SKILL_ROOT = {skill_root!r}
WORK = {work!r}
MANIFEST = {manifest!r}
INPUTS = {inputs!r}
zipfile.ZipFile(io.BytesIO(base64.b64decode(STAGE))).extractall(BASE)
os.makedirs(os.path.join(WORK, INPUTS), exist_ok=True)
os.makedirs(os.path.join(WORK, {outbox!r}), exist_ok=True)
os.chdir(WORK)
os.environ["CLAUDE_SKILL_DIR"] = SKILL_ROOT
os.environ["ELITEA_SKILL_DIR"] = SKILL_ROOT
sys.argv = [SCRIPT] + list(ARGV)
sys.path[:0] = [os.path.dirname(SCRIPT), SKILL_ROOT]
try:
    import prepare_env
    from pyodide.code import find_imports
except ImportError:
    prepare_env = None
if prepare_env is not None:
    _wanted = set()
    for _root, _dirs, _names in os.walk(SKILL_ROOT):
        for _name in _names:
            if _name.endswith(".py"):
                try:
                    with open(os.path.join(_root, _name), encoding="utf-8", errors="replace") as _handle:
                        _wanted.update(find_imports(_handle.read()))
                except SyntaxError:
                    pass
    _local = set()
    for _folder in (os.path.dirname(SCRIPT), SKILL_ROOT):
        for _entry in os.listdir(_folder):
            _local.add(_entry[:-3] if _entry.endswith(".py") else _entry)
    for _module in sorted(_wanted):
        if _module.split(".", 1)[0] in _local:
            continue
        try:
            await prepare_env.install_imports([_module])
        except Exception:
            pass
_stdout, _stderr = io.StringIO(), io.StringIO()
_exit_code = 0
with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
    try:
        runpy.run_path(SCRIPT, run_name="__main__")
    except SystemExit as _exit:
        if _exit.code is None:
            _exit_code = 0
        elif isinstance(_exit.code, int):
            _exit_code = _exit.code
        else:
            print(_exit.code, file=sys.stderr)
            _exit_code = 1
    except BaseException:
        traceback.print_exc()
        _exit_code = 1
_files = {{}}
for _root, _dirs, _names in os.walk(WORK):
    for _name in _names:
        _full = os.path.join(_root, _name)
        _rel = os.path.relpath(_full, WORK).replace(os.sep, "/")
        if _rel.split("/", 1)[0] == INPUTS:
            continue
        with open(_full, "rb") as _handle:
            _data = _handle.read()
        if MANIFEST.get(_rel) == hashlib.sha256(_data).hexdigest():
            continue
        _files[_rel] = base64.b64encode(_data).decode("ascii")
_deleted = [_path for _path in MANIFEST if not os.path.exists(os.path.join(WORK, _path))]
{{"exit_code": _exit_code, "stdout": _stdout.getvalue(), "stderr": _stderr.getvalue(), "files": _files, "deleted": _deleted}}
'''


def format_byte_size(size: int) -> str:
    """A size the way results show it: bytes below 1 KB, then KB, then MB with one decimal."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{-(-size // 1024)} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def is_outbox_file(path: str) -> bool:
    """Whether a work-directory path is directly under output/, the only place promoted to the user."""
    folder, _separator, name = path.partition('/')
    return folder == SKILL_SCRIPT_OUTBOX_FOLDER and bool(name) and '/' not in name


def is_python_script(path: str) -> bool:
    """Whether a skill file is something the offline sandbox can run."""
    return path.lower().endswith('.py')


@dataclass
class ScratchCaps:
    max_file_bytes: int = SKILL_SCRATCH_MAX_FILE_BYTES
    max_total_bytes: int = SKILL_SCRATCH_MAX_TOTAL_BYTES
    max_files: int = SKILL_SCRATCH_MAX_FILES


class ScratchStore:
    """The conversation's script files: one zip object and the saved outputs, under a prefix of the attachment bucket."""

    def __init__(self, client, bucket: str, conversation_key: str):
        self.client = client
        self.bucket = bucket
        self.conversation_key = conversation_key
        self._cache = getattr(client, '_skill_scratch_cache', None)
        if self._cache is None:
            self._cache = {}

    @property
    def prefix(self) -> str:
        return f"{SCRATCH_ROOT_PREFIX}/{self.conversation_key}"

    @property
    def zip_key(self) -> str:
        return f"{self.prefix}/{SCRATCH_ZIP_NAME}"

    def output_key(self, call_id: str) -> str:
        return f"{self.prefix}/{SCRATCH_OUTPUTS_FOLDER}/{call_id}.txt"

    def load_files(self) -> Dict[str, bytes]:
        """Every scratch file by path; empty when the conversation has none yet."""
        data = self._download(self.zip_key)
        if data is None:
            return {}
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                return {info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()}
        except zipfile.BadZipFile:
            logger.warning("[Skills] scratch zip of %s is unreadable; starting empty", self.conversation_key)
            return {}

    def save_files(self, files: Dict[str, bytes]) -> None:
        """Replace the scratch with these files; an empty set removes the object."""
        if not files:
            self._delete(self.zip_key)
            return
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(files):
                archive.writestr(path, files[path])
        self._upload(self.zip_key, buffer.getvalue(), 'application/zip')

    def save_output(self, call_id: str, text: str) -> None:
        self._upload(self.output_key(call_id), text.encode('utf-8'), 'text/plain; charset=utf-8')

    def read_output(self, call_id: str) -> Optional[str]:
        data = self._download(self.output_key(call_id))
        return None if data is None else data.decode('utf-8', errors='replace')

    def _download(self, key: str) -> Optional[bytes]:
        if key in self._cache:
            return self._cache[key]
        data = self.client.download_artifact_s3(self.bucket, key)
        if isinstance(data, dict):
            if _is_missing(data):
                return None
            raise ScratchUnavailable(str(data.get('error')))
        self._cache[key] = data
        return data

    def _upload(self, key: str, data: bytes, content_type: str) -> None:
        result = self.client.upload_artifact_s3(self.bucket, key, data, content_type=content_type)
        if isinstance(result, dict) and result.get('error') and _is_missing(result):
            self.client.create_bucket(self.bucket, expiration_measure='days', expiration_value=365)
            result = self.client.upload_artifact_s3(self.bucket, key, data, content_type=content_type)
        if isinstance(result, dict) and result.get('error'):
            raise ScratchUnavailable(str(result['error']))
        self._cache[key] = data

    def _delete(self, key: str) -> None:
        result = self.client.delete_artifact_s3(self.bucket, key)
        if isinstance(result, dict) and result.get('error') and not _is_missing(result):
            raise ScratchUnavailable(str(result['error']))
        self._cache.pop(key, None)


class ScratchUnavailable(Exception):
    """The attachment bucket refused a scratch operation for a reason other than a missing object."""


def _is_missing(error: dict) -> bool:
    text = str(error.get('error') or '').lower()
    return 'not found' in text or 'nosuchkey' in text or 'nosuchbucket' in text or '404' in text


class AttachmentInputs:
    """Conversation attachments by name, the newest copy of a name winning, for the files a script reads."""

    def __init__(self, client, bucket: str, conversation_key: str):
        self.client = client
        self.bucket = bucket
        self.conversation_key = conversation_key
        self._rows = None

    def rows(self) -> list:
        if self._rows is None:
            listing = self.client.list_artifacts_s3(self.bucket, prefix=f"{self.conversation_key}/", delimiter=None)
            contents = listing.get('contents', []) if isinstance(listing, dict) and not listing.get('error') else []
            self._rows = [row for row in contents if row.get('key') and not row['key'].endswith('/')]
        return self._rows

    def names(self) -> List[str]:
        return sorted({posixpath.basename(row['key']) for row in self.rows()}, key=str.casefold)

    def resolve(self, name: str) -> Optional[dict]:
        """The newest attachment row whose file name matches, exactly first, then ignoring case."""
        wanted = posixpath.basename((name or '').strip())
        exact = [row for row in self.rows() if posixpath.basename(row['key']) == wanted]
        folded = exact or [row for row in self.rows() if posixpath.basename(row['key']).casefold() == wanted.casefold()]
        if not folded:
            return None
        return max(folded, key=lambda row: str(row.get('lastModified') or ''))

    def download(self, row: dict) -> bytes:
        data = self.client.download_artifact_s3(self.bucket, row['key'])
        if isinstance(data, dict):
            raise ScratchUnavailable(str(data.get('error')))
        return data


@dataclass
class ScriptRunContext:
    """What a run needs from the platform: the client acting as the run, and where the conversation's files live."""
    client: Any
    bucket: str
    conversation_key: str


@dataclass
class StagedScript:
    skill_name: str
    script_path: str
    args: List[str]
    stage_b64: str
    scratch_manifest: Dict[str, str]


class SkillScriptRunner:
    """One run_skill_script call end to end: stage, execute offline, sync the scratch, promote output/, format."""

    def __init__(
        self,
        context: ScriptRunContext,
        file_fetcher: Callable,
        sandbox_factory: Callable = build_offline_pyodide_sandbox,
        limits: Optional[dict] = None,
        caps: Optional[ScratchCaps] = None,
        call_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex[:12],
        sandbox_base: str = '/',
    ):
        self.context = context
        self.file_fetcher = file_fetcher
        self.sandbox_factory = sandbox_factory
        self.limits = limits
        self.caps = caps or ScratchCaps()
        self.call_id_factory = call_id_factory
        self.sandbox_base = sandbox_base
        self.store = ScratchStore(context.client, context.bucket, context.conversation_key)
        self.inputs = AttachmentInputs(context.client, context.bucket, context.conversation_key)

    def run(self, skill: dict, files: list, script_path: str, args: List[str], input_names: List[str]) -> str:
        """The text the model reads for one script run."""
        call_id = self.call_id_factory()
        limits = self.limits or _read_sandbox_limits_from_env()
        try:
            scratch = self.store.load_files()
        except ScratchUnavailable as exc:
            logger.warning("[Skills] scratch of %s unavailable: %s", self.context.conversation_key, exc)
            return self._fail(RUN_SKILL_SCRIPT_UNAVAILABLE.format(reason=RUN_SKILL_SCRIPT_UNAVAILABLE_SCRATCH))
        over_cap = self._describe_over_cap(scratch)
        if over_cap:
            return self._fail(over_cap)
        staged_inputs, refusal = self._stage_inputs(input_names)
        if refusal:
            return self._fail(refusal)
        try:
            staged = self._stage(skill, files, script_path, args, scratch, staged_inputs)
        except SkillBodyUnavailable as exc:
            logger.warning("[Skills] staging %r failed: %s", skill.get('name'), exc)
            return self._fail(RUN_SKILL_SCRIPT_UNAVAILABLE.format(reason=RUN_SKILL_SCRIPT_UNAVAILABLE_STAGING))
        started = time.perf_counter()
        try:
            result = self._execute(self._program(staged), limits)
        except SandboxAdmissionRefused as exc:
            logger.warning("[Skills] run_skill_script refused by the sandbox: %s", exc)
            return self._fail(RUN_SKILL_SCRIPT_UNAVAILABLE.format(reason=RUN_SKILL_SCRIPT_UNAVAILABLE_SANDBOX))
        duration = time.perf_counter() - started
        if getattr(result, 'timed_out', False):
            return self._fail(RUN_SKILL_SCRIPT_TIMEOUT_RESULT.format(timeout=_describe_seconds(limits['timeout_seconds'])))
        payload = _payload_of(result)
        if payload is None:
            logger.warning("[Skills] run_skill_script got no payload: %s", (result.stderr or '')[:500])
            return self._fail(RUN_SKILL_SCRIPT_UNAVAILABLE.format(reason=RUN_SKILL_SCRIPT_UNAVAILABLE_SANDBOX))
        logger.info(
            "[Skills] run_skill_script %s/%s exit=%s in %.2fs", skill.get('name'), script_path, payload['exit_code'], duration,
        )
        trailer = self._sync_files(call_id, scratch, payload)
        text = self._format(call_id, payload, trailer)
        if payload['exit_code'] != 0:
            record_outcome(ToolOutcome(
                status=ToolResultStatus.ERROR, message=f"Exit code {payload['exit_code']}",
                tool_name=RUN_SKILL_SCRIPT_TOOL_NAME, error_class=ToolErrorClass.TOOL_INTERNAL,
            ))
        return text

    def _fail(self, text: str) -> str:
        record_outcome(ToolOutcome(
            status=ToolResultStatus.ERROR, message=text.splitlines()[0], tool_name=RUN_SKILL_SCRIPT_TOOL_NAME,
            error_class=ToolErrorClass.INFRASTRUCTURE,
        ))
        return text

    def _describe_over_cap(self, scratch: Dict[str, bytes]) -> str:
        total = sum(len(data) for data in scratch.values())
        if total > self.caps.max_total_bytes or len(scratch) > self.caps.max_files:
            usage = f"{format_byte_size(total)} in {len(scratch)} files"
            cap = f"{format_byte_size(self.caps.max_total_bytes)} and {self.caps.max_files} files"
            return RUN_SKILL_SCRIPT_SCRATCH_OVER_CAP.format(usage=usage, cap=cap)
        return ''

    def _stage_inputs(self, names: List[str]) -> tuple:
        staged = {}
        for name in names or []:
            row = self.inputs.resolve(name)
            if row is None:
                available = ', '.join(self.inputs.names()[:INPUT_LISTING_LIMIT]) or RUN_SKILL_SCRIPT_INPUT_NONE
                return None, RUN_SKILL_SCRIPT_INPUT_NOT_FOUND.format(name=name, available=available)
            if int(row.get('size') or 0) > self.caps.max_file_bytes:
                return None, RUN_SKILL_SCRIPT_INPUT_OVER_CAP.format(
                    name=name, size=format_byte_size(int(row['size'])), cap=format_byte_size(self.caps.max_file_bytes),
                )
            try:
                staged[posixpath.basename(row['key'])] = self.inputs.download(row)
            except ScratchUnavailable:
                return None, RUN_SKILL_SCRIPT_UNAVAILABLE.format(reason=RUN_SKILL_SCRIPT_UNAVAILABLE_SCRATCH)
        return staged, ''

    def _stage(self, skill, files, script_path, args, scratch, inputs) -> StagedScript:
        name = skill.get('name')
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in files:
                data = self.file_fetcher(
                    skill.get('skill_project_id'), skill.get('skill_id'),
                    skill.get('skill_version_id', skill.get('version_id')), entry['path'],
                )
                archive.writestr(f"{STAGE_SKILLS_FOLDER}/{name}/{entry['path']}", _as_bytes(data))
            for path, data in scratch.items():
                archive.writestr(f"{STAGE_WORK_FOLDER}/{path}", data)
            for input_name, data in inputs.items():
                archive.writestr(f"{STAGE_WORK_FOLDER}/{SKILL_SCRIPT_INPUTS_FOLDER}/{input_name}", data)
        return StagedScript(
            skill_name=name, script_path=script_path, args=[str(arg) for arg in args],
            stage_b64=base64.b64encode(buffer.getvalue()).decode('ascii'),
            scratch_manifest={path: hashlib.sha256(data).hexdigest() for path, data in scratch.items()},
        )

    def _program(self, staged: StagedScript) -> str:
        base = self.sandbox_base
        skills_root = posixpath.join(base, SKILL_SCRIPT_SKILLS_ROOT.lstrip('/'))
        work = posixpath.join(base, SKILL_SCRIPT_WORK_ROOT.lstrip('/'))
        skill_root = posixpath.join(skills_root, staged.skill_name)
        return SANDBOX_PROGRAM.format(
            base=base, stage=staged.stage_b64, script=posixpath.join(skill_root, staged.script_path),
            argv=staged.args, skill_root=skill_root, work=work, manifest=staged.scratch_manifest,
            inputs=SKILL_SCRIPT_INPUTS_FOLDER, outbox=SKILL_SCRIPT_OUTBOX_FOLDER,
        )

    def _execute(self, program: str, limits: dict):
        try:
            sandbox = self.sandbox_factory()
        except RuntimeError as exc:
            raise SandboxAdmissionRefused(str(exc), "service_unavailable") from exc
        enforce_sandbox_admission(limits)
        coroutine = sandbox.execute(
            program, timeout_seconds=limits['timeout_seconds'], memory_limit_mb=limits['wasm_max_mem_mb'],
            root_ca_path=limits.get('root_ca_path'), insecure_tls_domains=limits.get('allowed_pyodide_domains'),
        )
        if not asyncio.iscoroutine(coroutine):
            return coroutine
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coroutine).result()

    def _sync_files(self, call_id: str, scratch: Dict[str, bytes], payload: dict) -> List[str]:
        """Promote output/ files, write the rest back to the scratch, and return the trailer lines."""
        trailer = []
        shared = []
        kept = dict(scratch)
        for path in payload.get('deleted') or []:
            kept.pop(path, None)
        for path, encoded in (payload.get('files') or {}).items():
            data = base64.b64decode(encoded)
            if is_outbox_file(path):
                shared.append(self._promote(call_id, path, data))
                kept.pop(path, None)
            else:
                kept[path] = data
        if shared:
            trailer.append(RUN_SKILL_SCRIPT_FILES_SHARED.format(listing=', '.join(shared)))
        total = sum(len(data) for data in kept.values())
        oversized = any(len(data) > self.caps.max_file_bytes for data in kept.values())
        if kept != scratch:
            if total > self.caps.max_total_bytes or len(kept) > self.caps.max_files or oversized:
                trailer.append(RUN_SKILL_SCRIPT_SCRATCH_CAP_EXCEEDED.format(
                    cap=f"{format_byte_size(self.caps.max_total_bytes)}, {self.caps.max_files} files, "
                        f"{format_byte_size(self.caps.max_file_bytes)} per file",
                ))
            else:
                try:
                    self.store.save_files(kept)
                except ScratchUnavailable as exc:
                    logger.warning("[Skills] scratch of %s not saved: %s", self.context.conversation_key, exc)
                    trailer.append(RUN_SKILL_SCRIPT_UNAVAILABLE.format(reason=RUN_SKILL_SCRIPT_UNAVAILABLE_SCRATCH))
        return trailer

    def _promote(self, call_id: str, path: str, data: bytes) -> str:
        """One output/ file as a conversation attachment under a key no earlier message's download shares."""
        name = posixpath.basename(path)
        key = f"{self.context.conversation_key}/{call_id}/{name}"
        upload = self.context.client.upload_artifact_s3(self.context.bucket, key, data)
        if isinstance(upload, dict) and upload.get('error'):
            logger.warning("[Skills] output %r not shared: %s", path, upload['error'])
            return f"{name} (not shared: {upload['error']})"
        name = posixpath.basename(upload['filename'])
        mime_type, _encoding = mimetypes.guess_type(name)
        try:
            dispatch_custom_event("file_modified", {
                "message": f"File '{name}' created by run_skill_script",
                "filepath": upload['filepath'],
                "tool_name": RUN_SKILL_SCRIPT_TOOL_NAME,
                "toolkit": "skills",
                "operation_type": "create",
                "media_type": "image" if mime_type and mime_type.startswith('image/') else None,
                "meta": {"bucket": self.context.bucket, "file_size": len(data), "source": "generated"},
            })
        except RuntimeError as exc:
            logger.debug("[Skills] file_modified not dispatched outside a run: %s", exc)
        return f"{name} ({format_byte_size(len(data))})"

    def _format(self, call_id: str, payload: dict, trailer: List[str]) -> str:
        body = (payload.get('stdout') or '').rstrip('\n')
        stderr = (payload.get('stderr') or '').rstrip('\n')
        if stderr:
            body = '\n'.join(part for part in (body, RUN_SKILL_SCRIPT_STDERR_HEADING, stderr) if part)
        body = body or RUN_SKILL_SCRIPT_NO_OUTPUT
        head = RUN_SKILL_SCRIPT_RESULT_PREFIX.format(exit_code=payload['exit_code'])
        if len(body) > SKILL_SCRIPT_INLINE_OUTPUT_CHARS:
            ref = SKILL_OUTPUT_URI.format(call_id=call_id)
            try:
                self.store.save_output(call_id, body)
                body = RUN_SKILL_SCRIPT_OUTPUT_SAVED.format(
                    size=format_byte_size(len(body.encode('utf-8'))), ref=ref,
                    preview_chars=SKILL_SCRIPT_OUTPUT_PREVIEW_CHARS, preview=body[:SKILL_SCRIPT_OUTPUT_PREVIEW_CHARS],
                )
            except ScratchUnavailable as exc:
                logger.warning("[Skills] output of %s not saved: %s", call_id, exc)
                body = body[:SKILL_SCRIPT_INLINE_OUTPUT_CHARS]
        return '\n'.join([head, body] + trailer)


def _describe_seconds(value) -> str:
    return str(int(value)) if float(value).is_integer() else f"{float(value):g}"


def _as_bytes(data) -> bytes:
    return data.encode('utf-8') if isinstance(data, str) else bytes(data or b'')


def _payload_of(result) -> Optional[dict]:
    """The program's result dict, or None when the sandbox never ran it to the end."""
    payload = getattr(result, 'result', None)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return None
    if not isinstance(payload, dict) or 'exit_code' not in payload:
        return None
    return payload
