"""Over-limit guidance shared by SharePoint's file-reading tools.

Lives outside the wrappers because both the facade (``read_document``) and the
Graph backend (``read_file_from_sharing_link``) must build it, and only the
backend knows a shared file's real name — importing it back from the facade
would be circular.
"""

from typing import Any, Dict, Optional

from ..utils.file_metadata import (
    DEFAULT_MAX_OUTPUT_CHARS, build_over_limit_response,
)
from ...runtime.langchain.document_loaders.EliteAExcelLoader import (
    build_excel_metadata_from_estimate,
)


def build_excel_over_limit_response(
    estimate,
    *,
    filename: str,
    requested: str,
    tool_name: str = "read_document",
    sheet_name: Optional[str] = None,
    supports_sheet_name: bool = True,
) -> Dict[str, Any]:
    """Build content_too_large guidance for an ExcelReadLimitExceeded catch.

    Unlike the artifact toolkit's read_file, SharePoint's readers expose no
    row-range params, so sheet_name is the only narrowing lever — and only
    ``read_document`` has even that. Pass ``supports_sheet_name=False`` for a
    tool without it so the guidance refuses plainly instead of advertising a
    parameter the caller cannot pass. If a sheet_name was already supplied and
    it is still over limit, there is nothing further to suggest either.
    """
    sheet_names = [s.get("name", "") for s in estimate.sheets]
    actual_chars = (estimate.estimated_output_chars or estimate.sampled_chars
                    or (DEFAULT_MAX_OUTPUT_CHARS + 1))

    # Reuse the shared builder for the full diagnostic metadata (row/image/byte
    # limits etc.), then replace only instruction_for_readFile to reflect this
    # tool's narrower surface.
    metadata: Dict[str, Any] = build_excel_metadata_from_estimate(estimate)
    metadata["filename"] = filename

    if supports_sheet_name and sheet_name is None and sheet_names:
        metadata["instruction_for_readFile"] = {
            "first_class_params": {
                "sheet_name": (
                    "string — name of a single sheet to read instead of the "
                    "whole workbook. Available sheets: " + ", ".join(sheet_names)
                ),
            },
            "notes": (
                f"This workbook exceeds the {DEFAULT_MAX_OUTPUT_CHARS}-character "
                f"read limit. Retry {tool_name} with sheet_name set to one of "
                "the sheets listed above to read a smaller subset."
            ),
        }
    else:
        metadata["instruction_for_readFile"] = {
            "first_class_params": {},
            "notes": (
                (f"Sheet '{sheet_name}' " if sheet_name else "This workbook ")
                + f"still exceeds the {DEFAULT_MAX_OUTPUT_CHARS}-character read "
                f"limit even at the narrowest scope {tool_name} supports. "
                "Reading it in full is refused; no smaller read is available "
                "through this tool."
            ),
        }

    return build_over_limit_response(
        metadata, actual_chars=actual_chars, limit_chars=DEFAULT_MAX_OUTPUT_CHARS,
        requested=requested, include_metadata_directive=False,
    )
