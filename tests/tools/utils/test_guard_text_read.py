"""Tests for guard_text_read (Phase 5, #5447) — the shared VCS-style read cap.

Covers:
  * Content within the cap is returned unchanged (str passthrough).
  * Over-cap content returns the PRE-1 (#5432) structured guidance object.
  * total_lines/unit are computed from the content already in hand (no re-fetch).
  * total_lines reflects the true full-file content, not a pre-sliced excerpt
    (bug #1: a caller that already sliced content before calling guard_text_read
    must pass full_content, or the reported total silently describes the slice).
  * The guidance note is self-contained (a valid start_line/end_line range) and
    never references get_file_metadata (bug #2: VCS-style readers register no
    such tool, so pointing an LLM at it is a dead end).
"""

from elitea_sdk.tools.utils.file_metadata import (
    GET_FILE_METADATA_DIRECTIVE,
    RESULT_STATUS_KEY,
    ResultStatus,
    guard_text_read,
)

MAX = 100  # small cap so tests do not build huge strings


def test_small_content_returns_unchanged():
    body = "y" * (MAX - 10)
    assert guard_text_read(body, "small.py", max_output_chars=MAX) == body


def test_over_limit_returns_guidance_object():
    body = "x" * (MAX + 50)
    result = guard_text_read(body, "big.py", max_output_chars=MAX)

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["context"]["actual_chars"] == MAX + 50
    assert result["context"]["limit_chars"] == MAX


def test_over_limit_note_is_self_contained_not_a_dead_tool_reference():
    # No get_file_metadata tool exists for VCS-style readers (bug #2) — the
    # note must never send the LLM chasing a tool that isn't registered.
    body = "\n".join(f"line {i}" for i in range(1, 30))
    result = guard_text_read(body, "big.py", max_output_chars=MAX)

    notes = result["instruction_for_readFile"]["notes"]
    assert GET_FILE_METADATA_DIRECTIVE not in notes
    assert "start_line" in result["instruction_for_readFile"]["first_class_params"]
    assert "end_line" in result["instruction_for_readFile"]["first_class_params"]


def test_over_limit_computes_total_lines_from_content_in_hand():
    body = "\n".join(f"line {i}" for i in range(1, 30))  # 29 lines, no trailing \n
    result = guard_text_read(body, "big.py", max_output_chars=MAX)

    assert result["unit"] == "lines"
    assert result["total_lines"] == 29


def test_over_limit_uses_full_content_for_total_lines_not_the_slice():
    # bug #1: a caller may pass an already-sliced `content` (what's actually
    # being measured/returned) alongside the true pre-slice `full_content`.
    # total_lines must reflect the whole file, not the slice.
    full = "\n".join(f"line {i}" for i in range(1, 5000))  # 4999 lines
    sliced = "\n".join(f"line {i}" for i in range(1, 3))  # tiny slice, still over MAX via padding
    padded_slice = sliced + ("z" * (MAX + 10))

    result = guard_text_read(padded_slice, "big.py", max_output_chars=MAX, full_content=full)

    assert result["total_lines"] == 4999


def test_over_limit_defaults_full_content_to_content_when_omitted():
    body = "\n".join(f"line {i}" for i in range(1, 30))
    result = guard_text_read(body, "big.py", max_output_chars=MAX)

    assert result["total_lines"] == 29


def test_over_limit_carries_requested_context():
    body = "z" * (MAX + 1)
    result = guard_text_read(body, "big.py", max_output_chars=MAX, requested="start_line=1, end_line=None")

    assert result["context"]["requested"] == "start_line=1, end_line=None"


def test_exactly_at_limit_returns_unchanged():
    body = "a" * MAX
    assert guard_text_read(body, "exact.py", max_output_chars=MAX) == body
