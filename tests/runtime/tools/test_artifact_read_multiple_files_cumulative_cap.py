"""Tests for artifact read_multiple_files cumulative batch cap (#5780).

Before this fix, ArtifactWrapper.read_multiple_files had its own bespoke loop
with only a per-file cap (via read_file) and no cumulative budget across the
batch, unlike the shared capped_read_multiple_files helper used by the
GitHub/GitLab/ADO/Bitbucket/LocalGit toolkits. These tests confirm artifact
now shares that same cumulative-budget behavior.
"""

from unittest.mock import MagicMock

from elitea_sdk.runtime.tools.artifact import ArtifactWrapper
from elitea_sdk.tools.utils.file_metadata import DEFAULT_MAX_OUTPUT_CHARS

MAX = 100  # small cap so tests do not build huge strings


def make_wrapper(bodies: dict) -> ArtifactWrapper:
    wrapper = ArtifactWrapper.model_construct(
        bucket="test-bucket", max_single_read_size=MAX, artifact=MagicMock(),
    )
    wrapper.llm = None
    wrapper.artifact.get = MagicMock(side_effect=lambda artifact_name, **_: bodies[artifact_name])
    return wrapper


def test_small_batch_all_returned_normally():
    wrapper = make_wrapper({"a.txt": "hello a", "b.txt": "hello b"})

    result = wrapper.read_multiple_files(file_paths=["a.txt", "b.txt"])

    assert result == {"a.txt": "hello a", "b.txt": "hello b"}


def test_batch_exceeding_cumulative_cap_skips_later_files():
    # The cumulative budget is the shared DEFAULT_MAX_OUTPUT_CHARS constant,
    # not the per-instance max_single_read_size, so the per-file cap here is
    # set to match it. First file alone consumes the whole cumulative budget
    # (right at the per-file cap, so it's still returned as plain content);
    # the second file must be skipped without being fetched at all.
    wrapper = ArtifactWrapper.model_construct(
        bucket="test-bucket",
        max_single_read_size=DEFAULT_MAX_OUTPUT_CHARS,
        artifact=MagicMock(),
    )
    wrapper.llm = None
    big_body = "x" * DEFAULT_MAX_OUTPUT_CHARS
    bodies = {"big.txt": big_body, "small.txt": "hello"}
    wrapper.artifact.get = MagicMock(side_effect=lambda artifact_name, **_: bodies[artifact_name])

    result = wrapper.read_multiple_files(file_paths=["big.txt", "small.txt"])

    assert result["big.txt"] == big_body
    assert "Skipped" in result["small.txt"]
    assert "cumulative" in result["small.txt"]
    wrapper.artifact.get.assert_called_once()


def test_per_file_over_cap_result_is_measured_by_its_actual_returned_size():
    # A file that individually exceeds the per-file cap returns a small
    # guidance dict, not its (uncapped) content — the cumulative budget must
    # be charged for that dict's actual serialized size, not the size of the
    # oversized file it refused to return, so later small files in the same
    # batch are still read normally rather than wrongly skipped.
    first_body = "x" * (MAX + 500)
    wrapper = make_wrapper({"first.txt": first_body, "second.txt": "hello"})

    result = wrapper.read_multiple_files(file_paths=["first.txt", "second.txt"])

    assert isinstance(result["first.txt"], dict)
    assert result["second.txt"] == "hello"
