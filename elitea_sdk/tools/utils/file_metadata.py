"""File metadata utility for the ``get_file_metadata`` tool (EL-4389).

Produces a uniform metadata dict consumed by the artifact toolkit and used by
the LLM to discover what extra parameters a subsequent ``read_file`` call may
accept for a given file type.

**Design principle**: there is NO separate per-type handler registry here.
Instead, this module delegates to the *loader class* already registered in
``loaders_map`` (``constants.py``). If the loader class defines a
``@classmethod get_file_metadata(cls, *, filename, file_content, file_size)``
it is called and its result merged with the generic base metadata. Loaders
that do not define this method simply produce the base output (mime + size +
empty ``instruction_for_readFile``).

This keeps a single source of truth for extension→class mapping (``loaders_map``)
and co-locates the metadata logic with the parsing logic inside each loader.

Output shape (always):

    {
        "filename": "<name>",
        "type": "<mime/type-or-application/octet-stream>",
        "filesize": <int|None>,
        "extension": "<dot-extension|empty>",
        "instruction_for_readFile": {
            "extra_params": {<name>: <human-readable description>, ...},
            "notes": "<optional free text>"
        },
        # ... plus any loader-specific fields (e.g. "sheets" for Excel)
    }

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
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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
    """Return a metadata dict for the given file.

    Delegates to the loader class's ``get_file_metadata`` classmethod when one
    exists, using ``loaders_map`` as the single extension→class registry.
    """
    mime, extension = _detect_type(filename, file_content)

    base: Dict[str, Any] = {
        "filename": filename,
        "type": mime,
        "extension": extension,
        "filesize": file_size if file_size is not None else (
            len(file_content) if file_content is not None else None
        ),
        "instruction_for_readFile": {"extra_params": {}, "notes": ""},
    }

    # Look up loader class from loaders_map (single source of truth).
    try:
        from elitea_sdk.runtime.langchain.document_loaders.constants import loaders_map
        loader_entry = loaders_map.get(extension)
    except Exception:  # pylint: disable=broad-except
        loader_entry = None

    if not loader_entry:
        return base

    loader_cls = loader_entry.get("class")
    if not loader_cls or not hasattr(loader_cls, "get_file_metadata"):
        return base

    try:
        extra = loader_cls.get_file_metadata(
            filename=filename,
            file_content=file_content,
            file_size=file_size,
        ) or {}
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Loader.get_file_metadata failed for %s (%s): %s",
                       filename, extension, e)
        return base

    # Shallow-merge: loader overrides base on conflicting keys.
    base.update(extra)
    # Defensive: ensure instruction structure stays well-formed.
    instr = base.get("instruction_for_readFile") or {}
    if "extra_params" not in instr:
        instr["extra_params"] = {}
    if "notes" not in instr:
        instr["notes"] = ""
    base["instruction_for_readFile"] = instr
    return base
