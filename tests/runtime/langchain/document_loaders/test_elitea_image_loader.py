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

------------------------------------------------------------------------------
NOTE ON THE `_max_tokens` HISTORY — this is a TEST bug, not an SDK bug.
------------------------------------------------------------------------------
Every input JSON under test_data/EliteAImageLoader/input/ used to declare
`"_max_tokens": <N>` (leading underscore). That was authored incorrectly.

The test harness in loader_test_runner.py deliberately strips keys starting
with `_` before forwarding config to the loader constructor:

    kwargs = {k: v for k, v in config.items() if not k.startswith('_')}

That rule is CORRECT and MUST stay — the underscore-prefixed namespace is
reserved for test-runner metadata that must never leak into the loader:
    _name             — display label for the parametrized case
    _use_llm          — whether the harness should inject an LLM instance
    _prompt_default   — whether to inject the built-in image_processing_prompt

`max_tokens` is different: it is a real loader constructor kwarg
(EliteAImageLoader.__init__ reads it via `kwargs.get('max_tokens', 512)`).
It was written with the wrong prefix and got silently dropped, so the loader
always used its default of 512 tokens regardless of what the JSON said.

Why this only surfaced recently:
  - The loader-side bug where AlitaImageLoader ignored `max_tokens` entirely
    (issue #4259) was fixed in R-2.0.2 — the loader now honors max_tokens and
    chunks via markdown_chunker.
  - As soon as chunking became functional, the test-side typo started
    mattering: LLM output for larger images exceeded 2 × 512 tokens and split
    into 3 chunks against 2-chunk baselines.

The fix is entirely in the tests:
  1. Rename `_max_tokens` → `max_tokens` in every input JSON (done in this
     commit) so the value actually reaches the loader.
  2. Regenerate baselines for the two currently-skipped configs, which were
     captured when the effective value was 512 and will no longer match with
     the corrected 2048/8192 setting. This is a follow-up because baseline
     regeneration needs LLM credentials and produces LLM-derived content.

Nothing in elitea_sdk/ needs to change. The SDK contract (accept `max_tokens`
without an underscore) is consistent with every other loader kwarg.
"""

from pathlib import Path
from typing import Any, Dict

import pytest
from loader_helpers import collect_loader_test_params, run_loader_assert
from loader_test_runner import _get_llm_for_tests

_LOADER_NAME = "EliteAImageLoader"

# (input_name, config_index) pairs to skip until their baselines are regenerated
# against the now-correct max_tokens value.
#
# Both entries load the LLM into the pipeline (_use_llm: true). Their JSON
# configs used to have `_max_tokens` (stripped) so the loader chunked at the
# 512-token default and produced N documents; the committed baselines captured
# that N. After renaming `_max_tokens` → `max_tokens` in this same commit, the
# loader now sees the intended token budget (2048 for elitea_screenshot_jpeg,
# 8192 for several_in_one_png) and the chunk count against the same LLM output
# will not match the old baseline until it is regenerated.
_SKIP = {
    ("several_in_one_png", 1),  # baseline captured at effective max_tokens=512; regenerate at 8192
    ("elitea_screenshot_jpeg", 1),  # baseline captured at effective max_tokens=512; regenerate at 2048
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
