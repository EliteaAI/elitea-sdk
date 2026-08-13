"""
End-to-end regression test for issue #6185: read_file on a large JSON/text
artifact whose first non-ASCII byte falls after chardet's 10 KB sample
window returned content_too_large with total_lines=0, and start_line/
end_line chunking was silently ignored.

Exercises the real Artifact client (elitea_sdk/runtime/clients/artifact.py)
with only the underlying S3 client mocked, so both fixes are covered
together:
  1. decode_text() no longer raises on a late (past-sample-window)
     non-ASCII byte -> Artifact.get() returns a str, not a loader-fallback
     dict, so start_line/end_line slicing actually applies.
  2. When a file *is* still over the cap, the over-limit guidance reports
     the real total_lines instead of 0.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.runtime.clients.artifact import Artifact
from elitea_sdk.runtime.tools.artifact import ArtifactWrapper
from elitea_sdk.tools.utils.file_metadata import (
    RESULT_STATUS_KEY,
    ResultStatus,
    get_file_metadata,
)

MAX = 200_000


def _build_late_nonascii_file(num_fields: int = 300) -> bytes:
    # Valid, pretty-printed JSON (matches the real .json artifact from #6185) -
    # first ~10 KB (well within chardet's sample window) is pure ASCII, and a
    # non-ASCII char only appears far past that window, in a later field's
    # value. This is the exact shape that made chardet confidently
    # mis-detect 'ascii', and it's important the content is *valid* JSON so
    # a pre-fix run exercises the real EliteAJSONLoader dict-return fallback
    # rather than a JSONDecodeError.
    fields = {}
    for i in range(num_fields):
        if i == 250:
            fields[f"field_{i}"] = "price – discount"
        else:
            fields[f"field_{i}"] = "x" * 100
    text = json.dumps(fields, indent=2, ensure_ascii=False)
    assert len(text.encode("utf-8")[:10 * 1024].decode("ascii", errors="strict")) > 0
    return text.encode("utf-8")


def make_real_artifact_wrapper(file_bytes: bytes) -> ArtifactWrapper:
    """Wire the real Artifact client (only the S3 client mocked) into a wrapper."""
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = True
    mock_client.download_artifact_s3.return_value = file_bytes
    artifact = Artifact(client=mock_client, bucket_name="test-bucket")

    wrapper = ArtifactWrapper.model_construct(
        bucket="test-bucket",
        max_single_read_size=MAX,
        artifact=artifact,
    )
    wrapper.llm = None
    return wrapper


def test_late_nonascii_file_is_read_as_text_not_loader_fallback_dict():
    file_bytes = _build_late_nonascii_file(num_fields=300)
    wrapper = make_real_artifact_wrapper(file_bytes)

    content = wrapper.read_file(filename="work_item.json")

    assert isinstance(content, str)
    assert content == file_bytes.decode("utf-8")


def test_late_nonascii_file_chunked_read_returns_requested_lines():
    file_bytes = _build_late_nonascii_file(num_fields=300)
    wrapper = make_real_artifact_wrapper(file_bytes)

    chunk = wrapper.read_file(filename="work_item.json", start_line=1, end_line=10)

    assert isinstance(chunk, str)
    expected_lines = file_bytes.decode("utf-8").split("\n")[0:10]
    assert chunk.rstrip("\n").split("\n") == expected_lines


def test_over_limit_guidance_reports_real_total_lines_not_zero():
    # File exceeds the cap even before slicing, and has a late non-ASCII
    # byte - this is the exact over-limit shape from the #6185 report,
    # where total_lines came back 0 because file_content=None was passed
    # to get_file_metadata on the fallback path.
    small_max = 1_000
    lines = ["line " + str(i) + ": " + "y" * 50 for i in range(100)]
    lines[80] = "line 80: price – discount"
    text = "\n".join(lines)
    file_bytes = text.encode("utf-8")
    assert len(text) > small_max

    wrapper = make_real_artifact_wrapper(file_bytes)
    wrapper.max_single_read_size = small_max

    result = wrapper.read_file(filename="work_item.json")

    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["total_lines"] == 100
    assert result["unit"] == "lines"


def test_over_limit_guidance_with_start_line_still_reports_full_file_total_lines():
    # Even when the caller already sliced with start_line/end_line, the
    # over-limit total_lines must reflect the WHOLE file, not just the
    # (still-too-large) slice that was returned.
    small_max = 1_000
    lines = ["line " + str(i) + ": " + "y" * 50 for i in range(200)]
    lines[150] = "line 150: price – discount"
    text = "\n".join(lines)
    file_bytes = text.encode("utf-8")

    wrapper = make_real_artifact_wrapper(file_bytes)
    wrapper.max_single_read_size = small_max

    result = wrapper.read_file(filename="work_item.json", start_line=1, end_line=200)

    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["total_lines"] == 200


def test_over_limit_response_does_not_recount_lines_already_supplied_by_loader():
    # Perf: EliteAJSONLoader.get_file_metadata already counts total_lines from
    # this same full_content via build_line_range_metadata. _over_limit_response
    # must not redo that O(n) scan itself and overwrite the loader's value -
    # only fill it in when the loader metadata has no total_lines key at all
    # (e.g. unrecognized extensions).
    small_max = 1_000
    lines = ["line " + str(i) + ": " + "y" * 50 for i in range(100)]
    text = "\n".join(lines)
    file_bytes = text.encode("utf-8")

    wrapper = make_real_artifact_wrapper(file_bytes)
    wrapper.max_single_read_size = small_max

    real_get_metadata = get_file_metadata

    def spy_get_metadata(*args, **kwargs):
        metadata = real_get_metadata(*args, **kwargs)
        if metadata.get("total_lines") is not None:
            # Sentinel: deliberately wrong so a passing test proves the value
            # was taken verbatim from the loader, not recomputed from scratch.
            metadata["total_lines"] = 999999
        return metadata

    with patch("elitea_sdk.runtime.tools.artifact.get_file_metadata_dict", spy_get_metadata):
        result = wrapper.read_file(filename="work_item.json")

    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["total_lines"] == 999999
