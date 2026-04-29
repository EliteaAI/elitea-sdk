"""
Pytest tests for EliteAImageLoader.

Test cases are auto-discovered from:
  tests/runtime/langchain/document_loaders/test_data/EliteAImageLoader/input/*.json

Each (input_name, config_index) pair becomes a standalone test.
Tags declared in the input JSON are applied as pytest marks for -m filtering.

Run:
  pytest tests/runtime/langchain/document_loaders/test_elitea_image_loader.py -v
  pytest tests/runtime/langchain/document_loaders/test_elitea_image_loader.py -v -k "image_simple"
  pytest -m "loader_image" -v
  pytest -m "loader_image and feature_llm" -v

Note: This loader requires LLM support. Set DEFAULT_LLM_MODEL_FOR_CODE_ANALYSIS env var
      to enable LLM-based image analysis. Without it, tests will use OCR only.
"""

from pathlib import Path
from typing import Any, Dict

import pytest
from loader_helpers import collect_loader_test_params, run_loader_assert
from loader_test_runner import _get_llm_for_tests

_LOADER_NAME = "EliteAImageLoader"

# (input_name, config_index) pairs to skip.
# Reason: LLM-generated content is chunked via markdown_chunker(max_tokens=512);
# non-deterministic LLM response length produces variable chunk counts vs baseline.
# Root cause: '_max_tokens' key had underscore prefix → never passed to loader.
# Fix in progress: see several_in_one_png.json (changing _max_tokens → max_tokens)
# and baseline regeneration required.
_SKIP = {
    ("several_in_one_png", 1),  # variable chunk count due to _max_tokens prefix bug
}


@pytest.fixture(scope="module")
def llm_instance():
    """Create LLM instance once per module for multimodal image analysis."""
    return _get_llm_for_tests()


@pytest.mark.parametrize(
    "input_name, config_index, config, file_path, baseline_path",
    collect_loader_test_params(_LOADER_NAME),
)
def test_loader(
    tmp_path: Path,
    input_name: str,
    config_index: int,
    config: Dict[str, Any],
    file_path: Path,
    baseline_path: Path,
    llm_instance,
) -> None:
    if (input_name, config_index) in _SKIP:
        pytest.skip(f"{input_name} config{config_index}: known failure — pending fix")
    if config.get("_use_llm") and llm_instance is None:
        pytest.skip(f"{input_name} config{config_index}: requires LLM (set ELITEA_* env vars)")
    run_loader_assert(
        _LOADER_NAME, tmp_path, input_name, config_index, config, file_path, baseline_path, llm=llm_instance
    )
