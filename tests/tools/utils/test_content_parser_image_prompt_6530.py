"""Integration test for the image-only prompt extra_params key (#6530).

Verifies that read_file's extra_params={"prompt": "..."} reaches
EliteAImageLoader.prompt for image files, leaves the generic
image_processing_prompt default unchanged when omitted, and is a silent
no-op (no error, no leakage) when passed for a non-image file.
"""

from unittest.mock import patch

from elitea_sdk.runtime.langchain.document_loaders.EliteAImageLoader import EliteAImageLoader
from elitea_sdk.tools.utils.content_parser import image_processing_prompt, parse_file_content


def test_prompt_reaches_image_loader_prompt():
    captured = {}

    def fake_get_content(self):
        captured["prompt"] = self.prompt
        return "described"

    with patch.object(EliteAImageLoader, "get_content", fake_get_content):
        result = parse_file_content(
            file_name="shot.png",
            file_content=b"fake-bytes",
            is_capture_image=True,
            extra_params={"prompt": "Describe only the red circle"},
        )

    assert result == "described"
    assert captured["prompt"] == "Describe only the red circle"


def test_default_prompt_used_when_prompt_absent():
    captured = {}

    def fake_get_content(self):
        captured["prompt"] = self.prompt
        return "described"

    with patch.object(EliteAImageLoader, "get_content", fake_get_content):
        parse_file_content(
            file_name="shot.png",
            file_content=b"fake-bytes",
            is_capture_image=True,
        )

    assert captured["prompt"] == image_processing_prompt


def test_prompt_is_noop_for_non_image_file():
    # A .txt file goes through EliteATextLoader, which has no `prompt` concept
    # at all — the key must be silently ignored, not raise or leak.
    result = parse_file_content(
        file_name="notes.txt",
        file_content=b"hello world",
        extra_params={"prompt": "Describe only the red circle"},
    )
    assert "hello world" in result
