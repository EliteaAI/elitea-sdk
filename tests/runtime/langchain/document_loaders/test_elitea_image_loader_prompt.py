"""Regression tests for EliteAImageLoader custom prompt plumbing.

Reference: EliteaAI/elitea_issues#6530 — read_file's extra_params can now pass
a targeted extra_params prompt for images, matching the loader's own `prompt` kwarg.
These tests avoid LLM/OCR dependencies by checking constructor state directly.
"""

from elitea_sdk.runtime.langchain.constants import DEFAULT_MULTIMODAL_PROMPT
from elitea_sdk.runtime.langchain.document_loaders.EliteAImageLoader import EliteAImageLoader


def test_loader_uses_custom_prompt_when_provided():
    loader = EliteAImageLoader(
        file_path="/tmp/fake_image.png",
        prompt="Describe only the chart's axis labels",
    )
    assert loader.prompt == "Describe only the chart's axis labels"


def test_loader_falls_back_to_default_multimodal_prompt():
    loader = EliteAImageLoader(file_path="/tmp/fake_image.png")
    assert loader.prompt == DEFAULT_MULTIMODAL_PROMPT
