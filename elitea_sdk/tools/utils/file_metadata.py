"""File metadata utility for the ``get_file_metadata`` tool (EL-4389, PRE-1 #5432).

Produces a uniform metadata dict consumed by the artifact toolkit and used by
LLM **and non-LLM pipeline nodes** to discover how to fetch a file in bounded
chunks. The output conforms to a single Pydantic-defined contract
(``ChunkedReadResponse``) shared by BOTH:

  * ``get_file_metadata`` (inspect path), and
  * the over-limit return path of ``read_file`` / ``read_multiple_files``
    (built via :func:`build_over_limit_response`).

The two paths emit the identical shape; only the ``__result_status__``
discriminator differs. A non-LLM node distinguishes "chunking guidance" from
"file content" purely by ``obj.get("__result_status__")`` — never by
string-matching prose.

**Design principle**: there is NO separate per-type handler registry here.
Instead, this module delegates to the *loader class* already registered in
``loaders_map`` (``constants.py``). If the loader class defines a
``@classmethod get_file_metadata(cls, *, filename, file_content, file_size)``
it is called and its result merged with the generic base metadata. The merged
dict is then validated through ``ChunkedReadResponse`` before return, so any
loader that omits a required field (e.g. ``read_limits.max_output_chars``)
fails loudly. Loaders that do not define this method simply produce the base
output.

This keeps a single source of truth for extension->class mapping
(``loaders_map``) and co-locates the metadata logic with the parsing logic
inside each loader, while centralizing schema enforcement here.

Type detection priority:
  1. ``filetype.guess()`` (magic-byte) if the ``filetype`` library is
     installed AND a content sample is available.
  2. ``mimetypes.guess_type()`` (extension-based) as a fallback.
  3. ``application/octet-stream`` if nothing matches.
"""
from __future__ import annotations

import logging
import mimetypes
import os
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema contract (PRE-1 #5432)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"

#: JSON key carrying the single machine-detectable discriminator. Namespaced
#: with dunders so it never collides with a content dict's own keys.
RESULT_STATUS_KEY = "__result_status__"

#: The single universal bounded-read limit (chars of text returned to context).
#: Injected as the ``read_limits`` baseline for every file type so the
#: documented-required field is genuinely always present, even for loaders that
#: supply no ``read_limits`` of their own. Phase 4 (#5446) wires the configurable
#: per-tool value through; PRE-1 only needs a single source of truth.
DEFAULT_MAX_OUTPUT_CHARS = 200000


class ResultStatus(str, Enum):
    """The single discriminator value of a chunked-read response.

    A non-LLM pipeline node branches solely on this field:

      * ``file_metadata``     — inspect output from ``get_file_metadata``.
      * ``content_too_large`` — an over-limit read returned guidance, not
        content. Loop a bounded chunked read using ``instruction_for_readFile``.
      * ``error``             — metadata/inspection failed; see ``message``.

    Raw file content (a ``str``, or a per-type ``dict`` such as an Excel
    row-range slice) carries NO ``__result_status__`` key — that absence is
    itself how content is told apart from guidance.
    """

    FILE_METADATA = "file_metadata"
    CONTENT_TOO_LARGE = "content_too_large"
    ERROR = "error"


class InstructionForReadFile(BaseModel):
    """Exact ``read_file`` params needed to fetch the next chunk.

    ``extra="allow"`` so a loader may add type-specific instruction sub-keys.
    """

    model_config = ConfigDict(extra="allow")

    first_class_params: Dict[str, str] = Field(default_factory=dict)
    extra_params: Dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class ReadLimits(BaseModel):
    """Per-type read caps and estimates.

    Only two keys are universal and REQUIRED:

      * ``max_output_chars`` — THE single bounded-read limit (chars of text
        returned to context). One limit, no second number.
      * ``full_read_allowed`` — whether an unbounded full read is permitted.

    Everything else is optional and MUST be type-prefixed so no other loader
    assumes the key exists (e.g. ``excel_full_read_max_bytes``,
    ``estimated_output_chars``). ``extra="allow"`` carries those through.
    """

    model_config = ConfigDict(extra="allow")

    max_output_chars: int
    full_read_allowed: bool


class OverLimitContext(BaseModel):
    """Why an over-limit response was returned (present only on over-limit)."""

    model_config = ConfigDict(extra="allow")

    limit_chars: int
    actual_chars: int
    requested: Optional[str] = None


class ChunkedReadResponse(BaseModel):
    """The unified chunked-read contract shared by both code paths.

    ``extra="allow"`` lets type-specific fields ride through unchanged:
    ``sheets`` (Excel), ``image_count``/``image_names`` (Docx), and the flat
    ``total_<unit>`` totals (``total_rows`` / ``total_lines`` / ``total_pages``
    / ``total_sheets``) named by ``unit``.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # The single discriminator, exposed in JSON as ``__result_status__``.
    result_status: ResultStatus = Field(
        default=ResultStatus.FILE_METADATA, alias=RESULT_STATUS_KEY
    )
    schema_version: str = SCHEMA_VERSION

    filename: Optional[str] = None
    type: Optional[str] = None
    extension: Optional[str] = None
    filesize: Optional[int] = None

    #: Authoritative natural unit for this file type: "rows" | "lines" |
    #: "pages" | "sheets" | None. Names which flat ``total_<unit>`` is the
    #: total of record.
    unit: Optional[str] = None

    read_limits: Optional[ReadLimits] = None
    instruction_for_readFile: InstructionForReadFile = Field(
        default_factory=InstructionForReadFile
    )

    # Present only when result_status == content_too_large.
    context: Optional[OverLimitContext] = None

    # Present only when result_status == error.
    message: Optional[str] = None

    @model_validator(mode="after")
    def _read_limits_required_for_guidance(self) -> "ChunkedReadResponse":
        """``read_limits`` is mandatory on guidance responses.

        A ``file_metadata`` / ``content_too_large`` object without
        ``read_limits.max_output_chars`` gives the looping node no limit to
        chunk against. Only ``error`` responses (which carry no guidance) may
        omit it. The central builders always inject the universal baseline, so
        this only bites a caller that hand-builds a guidance object incorrectly.
        """
        if self.result_status is not ResultStatus.ERROR and self.read_limits is None:
            raise ValueError(
                "read_limits is required for "
                f"__result_status__={self.result_status.value!r}"
            )
        return self


def _dump(model: ChunkedReadResponse) -> Dict[str, Any]:
    """Serialize a response model to a plain dict using the JSON aliases.

    ``mode="json"`` so ``__result_status__`` is the plain string value
    (e.g. ``"file_metadata"``), not a ``ResultStatus`` enum object — a non-LLM
    node keys on the string without importing the enum.
    """
    return model.model_dump(mode="json", by_alias=True, exclude_none=True)


def validate_chunked_read_response(obj: Dict[str, Any]) -> ChunkedReadResponse:
    """Validate a dict against the contract, raising on non-conformance.

    Downstream consumers and tests use this as the single enforcement point.
    """
    return ChunkedReadResponse.model_validate(obj)


def build_error_response(
    message: str,
    *,
    filename: Optional[str] = None,
    extension: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    """Build an ``error`` chunked-read response through the validated model.

    The single error-emission entry point: every caller routes here so error
    responses carry ``schema_version`` and the discriminator and stay within the
    one PRE-1 contract (no hand-built dicts that silently drift). ``extra``
    carries optional context such as ``filepath`` or ``bucket``.
    """
    model = ChunkedReadResponse(
        result_status=ResultStatus.ERROR,
        filename=filename,
        extension=extension,
        message=message,
    )
    payload = _dump(model)
    # extra=allow on the model lets these ride through; merge after dump so they
    # appear in output without needing declared fields.
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    return payload


def _count_lines(file_content) -> int:
    """Count lines in *file_content* (bytes, bytearray, or str) without decoding.

    Scans for newline bytes/chars directly — no decode, no extra copy.
    The +1 handles files that don't end with a trailing newline.
    """
    if not file_content:
        return 0
    nl = b'\n' if isinstance(file_content, (bytes, bytearray)) else '\n'
    return file_content.count(nl) + (1 if not file_content.endswith(nl) else 0)


def build_line_range_metadata(file_content, *, file_type_note: str = "file") -> dict:
    """Return a loader-conformant dict for line-oriented text files (PRE-3 #5434).

    Shared by EliteATextLoader, EliteACodeLoader, EliteAMarkdownLoader,
    EliteACSVLoader, and EliteAJSONLoader so the logic lives in one place. Pass
    *file_type_note* for the human-readable ``notes`` string (e.g. "text file",
    "source file", "Markdown file", "JSON file").

    Single-line honesty (#5436): line slicing reads the file, then splits on
    newlines — a file with no usable line breaks is one "line", so a line range
    returns the *whole* file regardless of start_line/end_line. When such a file
    also exceeds the output cap there is no way to read it in bounded chunks, so
    we refuse the full read (``full_read_allowed=False``) and DON'T advertise
    start_line/end_line as if they worked. (The 150 MB artifact upload ceiling
    bounds the worst case, so this is a refuse-and-explain, not a memory guard.)
    """
    total_lines = _count_lines(file_content)
    content_size = len(file_content) if file_content else 0

    # A single (or zero) line file that is also over the output cap cannot be
    # chunked by line — be honest rather than offer params that do nothing.
    if total_lines <= 1 and content_size > DEFAULT_MAX_OUTPUT_CHARS:
        return {
            "unit": "lines",
            "total_lines": total_lines,
            "read_limits": {"full_read_allowed": False},
            "instruction_for_readFile": {
                "first_class_params": {},
                "notes": (
                    f"This {file_type_note} has no usable line breaks "
                    f"({content_size} characters on a single line) and exceeds "
                    f"the {DEFAULT_MAX_OUTPUT_CHARS}-character read limit. Line "
                    f"slicing would return the whole file, so a bounded read is "
                    f"not possible — the full read is refused."
                ),
            },
        }

    range_hint = f"Valid range 1..{total_lines}. " if total_lines else ""
    return {
        "unit": "lines",
        "total_lines": total_lines,
        "instruction_for_readFile": {
            "first_class_params": {
                "start_line": (
                    f"integer (1-indexed, inclusive) — first line to read. "
                    f"{range_hint}Omit to read from the beginning."
                ),
                "end_line": (
                    f"integer (1-indexed, inclusive) — last line to read. "
                    f"{range_hint}Omit to read to the end."
                ),
            },
            "notes": (
                f"Use start_line/end_line together to read a bounded slice "
                f"of a large {file_type_note} and keep tokens bounded."
            ),
        },
    }


def _detect_type(filename: str, file_content: Optional[bytes]):
    """Return (mime, extension) for the given file."""
    extension = os.path.splitext(filename or "")[-1].lower()

    # Magic-byte detection first if content available
    if file_content:
        try:
            import filetype  # type: ignore
            kind = filetype.guess(file_content)
            if kind is not None:
                if not extension and kind.extension:
                    extension = "." + kind.extension
                return kind.mime, extension
        except Exception:  # pylint: disable=broad-except
            logger.debug("filetype.guess failed; falling back to mimetypes",
                         exc_info=True)

    mime, _ = mimetypes.guess_type(filename or "")
    if mime is None:
        mime = "application/octet-stream"
    return mime, extension


def get_file_metadata(
    filename: str,
    file_content: Optional[bytes] = None,
    file_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a schema-conformant metadata dict for the given file.

    Delegates to the loader class's ``get_file_metadata`` classmethod when one
    exists (using ``loaders_map`` as the single extension->class registry),
    merges its output over the generic base, then validates the result through
    :class:`ChunkedReadResponse`. The returned dict carries
    ``__result_status__ == "file_metadata"``.
    """
    mime, extension = _detect_type(filename, file_content)

    base: Dict[str, Any] = {
        RESULT_STATUS_KEY: ResultStatus.FILE_METADATA.value,
        "schema_version": SCHEMA_VERSION,
        "filename": filename,
        "type": mime,
        "extension": extension,
        "filesize": file_size if file_size is not None else (
            len(file_content) if file_content is not None else None
        ),
        "unit": None,
        # Universal read_limits baseline so the documented-required field is
        # ALWAYS present, even for loaders that supply no read_limits of their
        # own (e.g. Docx). Loaders refine it via key-wise merge below.
        "read_limits": {
            "max_output_chars": DEFAULT_MAX_OUTPUT_CHARS,
            "full_read_allowed": True,
        },
        "instruction_for_readFile": {
            "first_class_params": {},
            "extra_params": {},
            "notes": "",
        },
    }

    # Look up loader class from loaders_map (single source of truth).
    try:
        from elitea_sdk.runtime.langchain.document_loaders.constants import loaders_map
        loader_entry = loaders_map.get(extension)
    except Exception:  # pylint: disable=broad-except
        loader_entry = None

    extra: Dict[str, Any] = {}
    if loader_entry:
        loader_cls = loader_entry.get("class")
        if loader_cls and hasattr(loader_cls, "get_file_metadata"):
            try:
                extra = loader_cls.get_file_metadata(
                    filename=filename,
                    file_content=file_content,
                    file_size=file_size,
                ) or {}
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Loader.get_file_metadata failed for %s (%s): %s",
                               filename, extension, e)
                extra = {}

    # Shallow-merge: loader overrides base on conflicting keys. The nested
    # instruction and read_limits blocks are merged key-wise so loaders can
    # supply only the sub-keys they care about while the universal baseline
    # (max_output_chars + full_read_allowed) always survives.
    instr = dict(base["instruction_for_readFile"])
    loader_instr = extra.pop("instruction_for_readFile", None)
    if isinstance(loader_instr, dict):
        instr.update(loader_instr)

    read_limits = dict(base["read_limits"])
    loader_limits = extra.pop("read_limits", None)
    if isinstance(loader_limits, dict):
        read_limits.update(loader_limits)

    base.update(extra)
    base["instruction_for_readFile"] = instr
    base["read_limits"] = read_limits

    try:
        return _dump(validate_chunked_read_response(base))
    except ValidationError as e:
        # A loader produced a non-conformant shape. Don't crash the read;
        # surface as an error response (tests assert conformance separately).
        logger.warning("get_file_metadata for %s (%s) failed schema validation: %s",
                       filename, extension, e)
        return build_error_response(
            f"Metadata failed schema validation: {e}",
            filename=filename, extension=extension,
        )


#: Appended unconditionally to every over-limit response's notes so the
#: caller is never tempted to guess a start/end value.
GET_FILE_METADATA_DIRECTIVE = (
    "For the exact count (lines/rows/pages/sheets) and full structural "
    "details (sheet names, attachment list, etc.), call get_file_metadata on "
    "this same file before choosing a range. Do not guess a start/end value."
)


def guard_text_read(
    content: str,
    filename: str,
    *,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    requested: Optional[str] = None,
) -> Any:
    """Return *content* unchanged if within the cap, else over-limit guidance.

    For VCS-style readers (GitHub/GitLab/Bitbucket/ADO/LocalGit) where a read
    always yields a fully-downloaded ``str`` — no dict/Excel-style results, so
    ``total_lines`` is computed directly from the content already in hand, no
    re-fetch or extra I/O.
    """
    actual_chars = len(content)
    if actual_chars <= max_output_chars:
        return content

    metadata = get_file_metadata(filename, file_content=None)
    if metadata.get(RESULT_STATUS_KEY) == ResultStatus.ERROR.value:
        return metadata

    if metadata.get("unit") in (None, "lines"):
        total_lines = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
        metadata["total_lines"] = total_lines
        metadata["unit"] = "lines"

    return build_over_limit_response(
        metadata, actual_chars=actual_chars, limit_chars=max_output_chars, requested=requested,
    )


def build_over_limit_response(
    metadata: Dict[str, Any],
    *,
    actual_chars: int,
    limit_chars: int,
    requested: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the over-limit guidance response from a file's metadata.

    Single entry point for the over-limit return path of ``read_file`` /
    ``read_multiple_files`` (Phase 4 / #5446). Reuses the file's own
    ``get_file_metadata`` output so the over-limit shape is byte-for-byte the
    inspect shape, differing only in ``__result_status__`` and the added
    ``context``. This guarantees a non-LLM node can detect over-limit and loop
    a bounded chunked read using the same ``instruction_for_readFile``.

    ``metadata`` is the dict returned by :func:`get_file_metadata` (or any
    contract-conformant dict). It must carry ``read_limits`` (guaranteed for
    anything produced by :func:`get_file_metadata`); the model validator rejects
    a guidance object without it.
    """
    model = validate_chunked_read_response(metadata)
    model.result_status = ResultStatus.CONTENT_TOO_LARGE
    # By definition a full read is not allowed on the over-limit path — the read
    # just exceeded the limit. Pin it so the looping node never re-attempts one.
    if model.read_limits is not None:
        model.read_limits.full_read_allowed = False
    model.context = OverLimitContext(
        limit_chars=limit_chars,
        actual_chars=actual_chars,
        requested=requested,
    )
    existing_notes = model.instruction_for_readFile.notes or ""
    model.instruction_for_readFile.notes = (
        f"{existing_notes} {GET_FILE_METADATA_DIRECTIVE}".strip()
    )
    return _dump(model)
