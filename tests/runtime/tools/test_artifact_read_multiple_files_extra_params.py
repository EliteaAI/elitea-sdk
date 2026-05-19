"""Test that read_multiple_files forwards extra_params to each read_file call (EL-4629)."""

from unittest.mock import MagicMock, patch, call

import pytest


def test_read_multiple_files_forwards_extra_params():
    """extra_params should be passed to every read_file invocation."""
    from elitea_sdk.runtime.tools.artifact import ArtifactWrapper

    wrapper = MagicMock(spec=ArtifactWrapper)
    # Use the real method but on the mock instance
    wrapper.read_multiple_files = ArtifactWrapper.read_multiple_files.__get__(wrapper)
    wrapper.read_file = MagicMock(return_value="content")

    extra = '{"extract_images": true}'
    result = wrapper.read_multiple_files(
        file_paths=["/bucket/a.docx", "/bucket/b.docx"],
        extra_params=extra,
    )

    assert wrapper.read_file.call_count == 2
    for c in wrapper.read_file.call_args_list:
        assert c.kwargs.get('extra_params') == extra


def test_read_multiple_files_without_extra_params():
    """Without extra_params, read_file should receive extra_params=None."""
    from elitea_sdk.runtime.tools.artifact import ArtifactWrapper

    wrapper = MagicMock(spec=ArtifactWrapper)
    wrapper.read_multiple_files = ArtifactWrapper.read_multiple_files.__get__(wrapper)
    wrapper.read_file = MagicMock(return_value="content")

    wrapper.read_multiple_files(file_paths=["file.txt"])

    wrapper.read_file.assert_called_once()
    assert wrapper.read_file.call_args.kwargs.get('extra_params') is None
