"""Runs a skill's bundled Python script in the offline sandbox and carries its results back (#7185).

Every call rebuilds the sandbox filesystem: the skill's files at /skills/<name>/, the conversation's scratch
(one zip object per conversation in the platform's attachment bucket) as the working directory, named attachments
under inputs/. After the run the files the script changed are merged into the latest zip, files written directly
under output/ become conversation attachments, and output over the inline ceiling is saved next to the scratch for
read_skill_output.
"""

import asyncio
import base64
import concurrent.futures
import hashlib
import inspect
import io
import json
import logging
import mimetypes
import posixpath
import re
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.callbacks.manager import dispatch_custom_event

from ..exceptions import SandboxAdmissionRefused, SkillBodyUnavailable
from ..tool_outcome import ToolErrorClass, ToolOutcome, ToolResultStatus, record_outcome
from .sandbox import _read_sandbox_limits_from_env, build_offline_pyodide_sandbox, enforce_sandbox_admission

logger = logging.getLogger(__name__)

TOOL_NAME = "run_skill_script"
PROGRAM_SOURCE = Path(__file__).with_name('skill_script_program.py').read_text(encoding='utf-8')
INLINE_OUTPUT_CHARS = 30_000
OUTPUT_PREVIEW_CHARS = 2_000
SCRATCH_ROOT_PREFIX = "_skill_scratch"
SKILLS_ROOT = "/skills"
WORK_ROOT = "/work"
INPUTS_FOLDER = "inputs"
OUTBOX_FOLDER = "output"
INPUT_LISTING_LIMIT = 20


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
    return folder == OUTBOX_FOLDER and bool(name) and '/' not in name


def is_python_script(path: str) -> bool:
    """Whether a skill file is something the offline sandbox can run."""
    return path.lower().endswith('.py')


def is_canonical_uuid(value) -> bool:
    """Whether a value is a UUID in its canonical lowercase hyphenated form, so it is one storage path segment."""
    try:
        return str(uuid.UUID(value)) == value
    except (AttributeError, TypeError, ValueError):
        return False


@dataclass
class ScratchCaps:
    """Sized to the sandbox: about 42 MB of incompressible staged files exhausts its 512 MB of WASM memory."""
    max_file_bytes: int = 10 * 1024 * 1024
    max_total_bytes: int = 30 * 1024 * 1024
    max_files: int = 200

    def describe(self) -> str:
        return f"{format_byte_size(self.max_total_bytes)} and {self.max_files} files"


class ScriptRunRefused(Exception):
    """A step of a run stopped it; the message is what the model reads instead of a result, and the facts are set
    when the sandbox ran the script before stopping it."""

    def __init__(self, message: str, facts: Optional[dict] = None):
        super().__init__(message)
        self.facts = facts


class ScratchUnavailable(Exception):
    """The attachment bucket refused a scratch operation for a reason other than a missing object."""


def _is_missing(error: dict) -> bool:
    text = str(error.get('error') or '').lower()
    return 'not found' in text or 'nosuchkey' in text or 'nosuchbucket' in text or '404' in text


class ScratchStore:
    """The conversation's script files: one zip object and the saved outputs, under a prefix of the attachment bucket."""

    def __init__(self, client, bucket: str, conversation_uuid: str):
        self.client = client
        self.bucket = bucket
        self.prefix = f"{SCRATCH_ROOT_PREFIX}/{conversation_uuid}"

    def load_files(self) -> Dict[str, bytes]:
        """Every scratch file by path; empty when the conversation has none yet."""
        data = self._download(f"{self.prefix}/scratch.zip")
        if data is None:
            return {}
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                return {info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()}
        except zipfile.BadZipFile:
            logger.warning("[Skills] scratch zip %s is unreadable; starting empty", self.prefix)
            return {}

    def save_files(self, files: Dict[str, bytes]) -> None:
        """Replace the scratch with these files; an empty set removes the object."""
        key = f"{self.prefix}/scratch.zip"
        if not files:
            self._delete(key)
            return
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(files):
                archive.writestr(path, files[path])
        self._upload(key, buffer.getvalue(), 'application/zip')

    def save_output(self, call_id: str, text: str) -> str:
        """Keep an output too long to return, and give the skill-output:// reference read_output takes back."""
        self._upload(f"{self.prefix}/outputs/{call_id}.txt", text.encode('utf-8'), 'text/plain; charset=utf-8')
        return f"skill-output://{call_id}"

    def read_output(self, ref: str) -> Optional[str]:
        """A saved output by its reference; None when the reference is malformed or the output no longer exists."""
        match = re.fullmatch(r'skill-output://([A-Za-z0-9_-]+)', (ref or '').strip())
        if not match:
            return None
        data = self._download(f"{self.prefix}/outputs/{match.group(1)}.txt")
        return None if data is None else data.decode('utf-8', errors='replace')

    def _download(self, key: str) -> Optional[bytes]:
        data = self.client.download_artifact_s3(self.bucket, key)
        if isinstance(data, dict):
            if _is_missing(data):
                return None
            raise ScratchUnavailable(str(data.get('error')))
        return data

    def _upload(self, key: str, data: bytes, content_type: str) -> None:
        result = self.client.upload_artifact_s3(self.bucket, key, data, content_type=content_type)
        if isinstance(result, dict) and result.get('error') and _is_missing(result):
            self.client.create_bucket(self.bucket, expiration_measure='days', expiration_value=365)
            result = self.client.upload_artifact_s3(self.bucket, key, data, content_type=content_type)
        if isinstance(result, dict) and result.get('error'):
            raise ScratchUnavailable(str(result['error']))

    def _delete(self, key: str) -> None:
        result = self.client.delete_artifact_s3(self.bucket, key)
        if isinstance(result, dict) and result.get('error') and not _is_missing(result):
            raise ScratchUnavailable(str(result['error']))


class AttachmentInputs:
    """Conversation attachments by name, the newest copy of a name winning, for the files a script reads."""

    def __init__(self, client, bucket: str, conversation_uuid: str):
        self.client = client
        self.bucket = bucket
        self.conversation_uuid = conversation_uuid
        self._rows = None

    def rows(self) -> list:
        if self._rows is None:
            listing = self.client.list_artifacts_s3(self.bucket, prefix=f"{self.conversation_uuid}/", delimiter=None)
            contents = listing.get('contents', []) if isinstance(listing, dict) and not listing.get('error') else []
            self._rows = [row for row in contents if row.get('key') and not row['key'].endswith('/')]
        return self._rows

    def resolve(self, name: str) -> dict:
        """The newest attachment whose file name matches, exactly first, then ignoring case."""
        wanted = posixpath.basename((name or '').strip())
        matching = [row for row in self.rows() if posixpath.basename(row['key']) == wanted] or [
            row for row in self.rows() if posixpath.basename(row['key']).casefold() == wanted.casefold()
        ]
        if not matching:
            names = sorted({posixpath.basename(row['key']) for row in self.rows()}, key=str.casefold)
            available = ', '.join(names[:INPUT_LISTING_LIMIT]) or '(none)'
            raise ScriptRunRefused(
                f'No attachment named "{name}" is in this conversation. Attachments available: {available}.'
            )
        return max(matching, key=lambda row: str(row.get('lastModified') or ''))

    def download(self, row: dict) -> bytes:
        data = self.client.download_artifact_s3(self.bucket, row['key'])
        if isinstance(data, dict):
            raise ScratchUnavailable(str(data.get('error')))
        return data


@dataclass
class ScriptRunContext:
    """What a run needs from the platform: the client acting as the run, and where the conversation's files live.
    The conversation uuid is the one the platform verified for the run (its own when the run has no conversation)."""
    client: Any
    scratch_bucket: str
    attachment_bucket: str
    conversation_uuid: str

    def __post_init__(self):
        if not is_canonical_uuid(self.conversation_uuid):
            raise ValueError("A skill script run is keyed only by a conversation UUID.")


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
        self.store = ScratchStore(context.client, context.scratch_bucket, context.conversation_uuid)
        self.inputs = AttachmentInputs(context.client, context.attachment_bucket, context.conversation_uuid)

    def run(self, skill: dict, files: list, script_path: str, args: List[str], input_names: List[str],
            callbacks=None) -> Tuple[str, Optional[dict]]:
        """``arun`` for a caller without an event loop of its own."""
        return self._wait(self.arun(skill, files, script_path, args, input_names, callbacks))

    async def arun(self, skill: dict, files: list, script_path: str, args: List[str], input_names: List[str],
                   callbacks=None) -> Tuple[str, Optional[dict]]:
        """The text the model reads for one script run, and the run's facts when the sandbox ran the script. The
        callbacks are the tool run's own, since the run's context may no longer carry them once the sandbox answers."""
        call_id = self.call_id_factory()
        limits = self.limits or _read_sandbox_limits_from_env()
        try:
            scratch = self._load_scratch()
            program = self._program(skill, files, script_path, args, scratch, self._stage_inputs(input_names, scratch))
            payload = await self._execute(program, limits)
        except ScriptRunRefused as refusal:
            record_outcome(ToolOutcome(
                status=ToolResultStatus.ERROR, message=str(refusal).splitlines()[0], tool_name=TOOL_NAME,
                error_class=ToolErrorClass.INFRASTRUCTURE,
            ))
            return str(refusal), refusal.facts
        trailer, shared = self._sync_files(call_id, payload, callbacks)
        if payload['exit_code'] != 0:
            record_outcome(ToolOutcome(
                status=ToolResultStatus.ERROR, message=f"Exit code {payload['exit_code']}", tool_name=TOOL_NAME,
                error_class=ToolErrorClass.TOOL_INTERNAL,
            ))
        text = '\n'.join([f"Exit code {payload['exit_code']}", self._format_output(call_id, payload)] + trailer)
        return text, {
            'exit_code': payload['exit_code'],
            'timed_out': False,
            'stdout_bytes': len((payload.get('stdout') or '').encode('utf-8')),
            'stderr_bytes': len((payload.get('stderr') or '').encode('utf-8')),
            'shared': shared,
        }

    def _load_scratch(self) -> Dict[str, bytes]:
        try:
            scratch = self.store.load_files()
        except ScratchUnavailable as exc:
            logger.warning("[Skills] scratch %s unavailable: %s", self.store.prefix, exc)
            raise ScriptRunRefused(
                "The script could not be run: the conversation's script storage could not be reached."
            ) from exc
        total = sum(len(data) for data in scratch.values())
        if total > self.caps.max_total_bytes or len(scratch) > self.caps.max_files:
            raise ScriptRunRefused(
                f"The script cannot start: the conversation's script storage holds {format_byte_size(total)} in "
                f"{len(scratch)} files, over the limit of {self.caps.describe()}. "
                "A script may delete files it no longer needs."
            )
        return scratch

    def _stage_inputs(self, names: List[str], scratch: Dict[str, bytes]) -> Dict[str, bytes]:
        staged = {}
        total = sum(len(data) for data in scratch.values())
        for name in names or []:
            row = self.inputs.resolve(name)
            size = int(row.get('size') or 0)
            if size > self.caps.max_file_bytes:
                raise ScriptRunRefused(
                    f'The script cannot start: attachment "{name}" is {format_byte_size(size)}, over the '
                    f'{format_byte_size(self.caps.max_file_bytes)} limit for one staged file.'
                )
            total += size
            if total > self.caps.max_total_bytes:
                raise ScriptRunRefused(
                    f'The script cannot start: with attachment "{name}" the files staged for this run come to '
                    f'{format_byte_size(total)}, over the {format_byte_size(self.caps.max_total_bytes)} a run can stage.'
                )
            try:
                staged[posixpath.basename(row['key'])] = self.inputs.download(row)
            except ScratchUnavailable as exc:
                raise ScriptRunRefused(
                    "The script could not be run: the conversation's script storage could not be reached."
                ) from exc
        return staged

    def _program(self, skill, files, script_path, args, scratch, inputs) -> str:
        """This file's sibling program, called with everything the sandbox needs to rebuild the run's filesystem."""
        name = skill.get('name')
        buffer = io.BytesIO()
        try:
            with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for entry in files:
                    data = self.file_fetcher(
                        skill.get('skill_project_id'), skill.get('skill_id'),
                        skill.get('skill_version_id', skill.get('version_id')), entry['path'],
                    )
                    content = data.encode('utf-8') if isinstance(data, str) else bytes(data or b'')
                    archive.writestr(f"skills/{name}/{entry['path']}", content)
                for path, data in scratch.items():
                    archive.writestr(f"work/{path}", data)
                for input_name, data in inputs.items():
                    archive.writestr(f"work/{INPUTS_FOLDER}/{input_name}", data)
        except SkillBodyUnavailable as exc:
            logger.warning("[Skills] staging %r failed: %s", name, exc)
            raise ScriptRunRefused("The script could not be run: the skill's files could not be fetched right now.") from exc
        skill_root = posixpath.join(self.sandbox_base, SKILLS_ROOT.lstrip('/'), name)
        params = {
            'base': self.sandbox_base,
            'stage': base64.b64encode(buffer.getvalue()).decode('ascii'),
            'work': posixpath.join(self.sandbox_base, WORK_ROOT.lstrip('/')),
            'skill_root': skill_root,
            'script': posixpath.join(skill_root, script_path),
            'argv': [str(arg) for arg in args],
            'folders': [INPUTS_FOLDER, OUTBOX_FOLDER],
            'inputs': INPUTS_FOLDER,
            'manifest': {path: hashlib.sha256(data).hexdigest() for path, data in scratch.items()},
            'max_file_bytes': self.caps.max_file_bytes,
            'max_total_bytes': self.caps.max_total_bytes,
        }
        return f"{PROGRAM_SOURCE}\nawait run({params!r})\n"

    async def _execute(self, program: str, limits: dict) -> dict:
        """The program's result dict, or the reason the sandbox gave none."""
        unavailable = "The script could not be run: the sandbox is not available right now."
        started = time.perf_counter()
        try:
            sandbox = self.sandbox_factory()
        except RuntimeError as exc:
            logger.warning("[Skills] run_skill_script has no local sandbox: %s", exc)
            raise ScriptRunRefused(unavailable) from exc
        try:
            enforce_sandbox_admission(limits)
        except SandboxAdmissionRefused as exc:
            logger.warning("[Skills] run_skill_script refused by the sandbox: %s", exc)
            raise ScriptRunRefused(unavailable) from exc
        result = sandbox.execute(
            program, timeout_seconds=limits['timeout_seconds'], memory_limit_mb=limits['wasm_max_mem_mb'],
            root_ca_path=limits.get('root_ca_path'), insecure_tls_domains=limits.get('allowed_pyodide_domains'),
        )
        if inspect.isawaitable(result):
            result = await result
        if getattr(result, 'timed_out', False):
            timeout = limits['timeout_seconds']
            seconds = str(int(timeout)) if float(timeout).is_integer() else f"{float(timeout):g}"
            raise ScriptRunRefused(
                f"Exit code -1 (timed out after {seconds} s)\nNo output was captured; the script was stopped.",
                facts={'exit_code': -1, 'timed_out': True, 'stdout_bytes': 0, 'stderr_bytes': 0, 'shared': []},
            )
        payload = getattr(result, 'result', None)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = None
        if not isinstance(payload, dict) or 'exit_code' not in payload:
            logger.warning("[Skills] run_skill_script got no payload: %s", (result.stderr or '')[:500])
            raise ScriptRunRefused(unavailable)
        logger.info("[Skills] run_skill_script exit=%s in %.2fs", payload['exit_code'], time.perf_counter() - started)
        return payload

    @staticmethod
    def _wait(outcome):
        """A coroutine's result for a synchronous caller, inside or outside an event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(outcome)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, outcome).result()

    def _sync_files(self, call_id: str, payload: dict, callbacks) -> Tuple[List[str], List[str]]:
        """Promote output/ files and merge the rest into the latest scratch; the trailer lines and the shared names."""
        trailer, shared, labels = [], [], []
        written = {}
        for path, encoded in (payload.get('files') or {}).items():
            data = base64.b64decode(encoded)
            if not is_outbox_file(path):
                written[path] = data
                continue
            name, error = self._promote(call_id, path, data, callbacks)
            if error:
                labels.append(f"{name} (not shared: {error})")
            else:
                shared.append(name)
                labels.append(f"{name} ({format_byte_size(len(data))})")
        if labels:
            trailer.append(f"Files shared: {', '.join(labels)}")
        if payload.get('too_large'):
            trailer.append(
                f"Files not kept: {', '.join(payload['too_large'])}. A file is kept up to "
                f"{format_byte_size(self.caps.max_file_bytes)}, and one call's files up to "
                f"{format_byte_size(self.caps.max_total_bytes)}."
            )
        deleted = set(payload.get('deleted') or [])
        if not written and not deleted:
            return trailer, shared
        try:
            self._merge_into_scratch(written, deleted, trailer)
        except ScratchUnavailable as exc:
            logger.warning("[Skills] scratch %s not saved: %s", self.store.prefix, exc)
            trailer.append("Files written by this call were not kept: the conversation's script storage could not be reached.")
        return trailer, shared

    def _merge_into_scratch(self, written: Dict[str, bytes], deleted: set, trailer: List[str]) -> None:
        """Apply one call's changes to the scratch as it is now, so a call that ran alongside keeps its own files."""
        latest = self.store.load_files()
        kept = {path: data for path, data in latest.items() if path not in deleted}
        kept.update(written)
        if kept == latest:
            return
        if sum(len(data) for data in kept.values()) > self.caps.max_total_bytes or len(kept) > self.caps.max_files:
            trailer.append(
                "Files written by this call were not kept: the conversation's script storage allows "
                f"{self.caps.describe()} in total. Delete files the scripts no longer need, or write results under output/."
            )
            return
        self.store.save_files(kept)

    def _promote(self, call_id: str, path: str, data: bytes, callbacks) -> Tuple[str, Optional[str]]:
        """One output/ file as a conversation attachment under a key no earlier message's download shares; its name,
        and why it was not shared when it was not."""
        name = posixpath.basename(path)
        upload = self.context.client.upload_artifact_s3(
            self.context.attachment_bucket, f"{self.context.conversation_uuid}/{call_id}/{name}", data,
        )
        if isinstance(upload, dict) and upload.get('error'):
            logger.warning("[Skills] output %r not shared: %s", path, upload['error'])
            return name, str(upload['error'])
        name = posixpath.basename(upload['filename'])
        mime_type, _encoding = mimetypes.guess_type(name)
        try:
            dispatch_custom_event("file_modified", config={'callbacks': callbacks} if callbacks else None, data={
                "message": f"File '{name}' created by run_skill_script",
                "filepath": upload['filepath'],
                "tool_name": TOOL_NAME,
                "toolkit": "skills",
                "operation_type": "create",
                "media_type": "image" if mime_type and mime_type.startswith('image/') else None,
                "meta": {"bucket": self.context.attachment_bucket, "file_size": len(data), "source": "generated"},
            })
        except RuntimeError as exc:
            logger.debug("[Skills] file_modified not dispatched outside a run: %s", exc)
        return name, None

    def _format_output(self, call_id: str, payload: dict) -> str:
        """The script's stdout and stderr as the model reads them; saved and previewed when too long."""
        stdout = (payload.get('stdout') or '').rstrip('\n')
        stderr = (payload.get('stderr') or '').rstrip('\n')
        output = '\n'.join(part for part in (stdout, '[stderr]' if stderr else '', stderr) if part) or "(no output)"
        if len(output) <= INLINE_OUTPUT_CHARS:
            return output
        try:
            ref = self.store.save_output(call_id, output)
        except ScratchUnavailable as exc:
            logger.warning("[Skills] output of %s not saved: %s", call_id, exc)
            return output[:INLINE_OUTPUT_CHARS]
        return (
            f"Output too large ({format_byte_size(len(output.encode('utf-8')))}). Full output saved as {ref}; read it "
            f'with read_skill_output(ref="{ref}", start_line=1, end_line=200).\n'
            f"Preview (first {OUTPUT_PREVIEW_CHARS} characters):\n{output[:OUTPUT_PREVIEW_CHARS]}"
        )
