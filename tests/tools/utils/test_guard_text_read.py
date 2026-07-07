"""Tests for guard_text_read (Phase 5, #5447) — the shared VCS-style read cap.

Covers:
  * Content within the cap is returned unchanged (str passthrough).
  * Over-cap content returns the PRE-1 (#5432) structured guidance object.
  * total_lines/unit are computed from the content already in hand (no re-fetch).
  * The GET_FILE_METADATA_DIRECTIVE always lands in the guidance notes.
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
    assert GET_FILE_METADATA_DIRECTIVE in result["instruction_for_readFile"]["notes"]


def test_over_limit_computes_total_lines_from_content_in_hand():
    body = "\n".join(f"line {i}" for i in range(1, 30))  # 29 lines, no trailing \n
    result = guard_text_read(body, "big.py", max_output_chars=MAX)

    assert result["unit"] == "lines"
    assert result["total_lines"] == 29


def test_over_limit_carries_requested_context():
    body = "z" * (MAX + 1)
    result = guard_text_read(body, "big.py", max_output_chars=MAX, requested="start_line=1, end_line=None")

    assert result["context"]["requested"] == "start_line=1, end_line=None"


def test_exactly_at_limit_returns_unchanged():
    body = "a" * MAX
    assert guard_text_read(body, "exact.py", max_output_chars=MAX) == body
