"""
Unit tests for Phase 3 (#5445): skip_size_check deprecated-and-neutered on the
artifact read tools.

Coverage:
  SSC_GUARD    — size guard always applies to LLM/pipeline read_file calls
  SSC_DEPR     — skip_size_check=True is inert but logs a deprecation warning
  SSC_BYPASS   — internal _bypass_size_limit=True still returns full content
                 (grep/edit regression guard)
  SSC_MULTI    — read_multiple_files enforces per file, no bypass forwarding
  SSC_SCHEMA   — skip_size_check absent from both LLM args_schemas

All pure unit tests: no network, no S3. The artifact client is mocked and the
wrapper is built with model_construct to skip the elitea-client validator.
"""

import logging
from unittest.mock import MagicMock

import pytest

from elitea_sdk.runtime.tools.artifact import (
    ArtifactWrapper,
    SKIP_SIZE_CHECK_DEPRECATION_MSG,
)

MAX = 100  # small cap so tests do not build huge strings


def make_wrapper(content: str) -> ArtifactWrapper:
    """Build an ArtifactWrapper whose artifact.get() returns `content`."""
    wrapper = ArtifactWrapper.model_construct(
        bucket="test-bucket",
        max_single_read_size=MAX,
        artifact=MagicMock(),
    )
    wrapper.artifact.get = MagicMock(return_value=content)
    wrapper.llm = None
    return wrapper


# ---------------------------------------------------------------------------
# SSC_GUARD — guard always applies for LLM/pipeline calls
# ---------------------------------------------------------------------------

def test_large_content_is_guarded_without_any_flag():
    wrapper = make_wrapper("x" * (MAX + 50))
    result = wrapper.read_file(filename="big.txt")
    assert "exceeds size limit" in result
    assert len(result) < MAX + 50  # truncation message, not the payload


def test_small_content_returns_full():
    body = "y" * (MAX - 10)
    wrapper = make_wrapper(body)
    assert wrapper.read_file(filename="small.txt") == body


# ---------------------------------------------------------------------------
# SSC_DEPR — skip_size_check is inert but logs when True
# ---------------------------------------------------------------------------

def test_skip_size_check_true_still_guards_and_warns(caplog):
    wrapper = make_wrapper("z" * (MAX + 50))
    with caplog.at_level(logging.WARNING):
        result = wrapper.read_file(filename="big.txt", skip_size_check=True)
    assert "exceeds size limit" in result  # inert: guard still applied
    assert SKIP_SIZE_CHECK_DEPRECATION_MSG in caplog.text


def test_skip_size_check_false_guards_without_warning(caplog):
    wrapper = make_wrapper("z" * (MAX + 50))
    with caplog.at_level(logging.WARNING):
        result = wrapper.read_file(filename="big.txt", skip_size_check=False)
    assert "exceeds size limit" in result
    assert SKIP_SIZE_CHECK_DEPRECATION_MSG not in caplog.text


def test_default_call_emits_no_deprecation(caplog):
    wrapper = make_wrapper("ok")
    with caplog.at_level(logging.WARNING):
        wrapper.read_file(filename="small.txt")
    assert SKIP_SIZE_CHECK_DEPRECATION_MSG not in caplog.text


# ---------------------------------------------------------------------------
# SSC_BYPASS — internal callers keep full content (grep/edit regression guard)
# ---------------------------------------------------------------------------

def test_bypass_returns_full_content_for_large_file():
    body = "a" * (MAX + 500)
    wrapper = make_wrapper(body)
    # read_file with the private bypass (used by _read_file → grep/edit)
    assert wrapper.read_file(filename="big.txt", _bypass_size_limit=True) == body


def test_internal_read_file_returns_full_content():
    body = "b" * (MAX + 500)
    wrapper = make_wrapper(body)
    assert wrapper._read_file("big.txt") == body


# ---------------------------------------------------------------------------
# SSC_MULTI — read_multiple_files enforces per file, no bypass forwarding
# ---------------------------------------------------------------------------

def test_read_multiple_files_guards_each_file():
    wrapper = make_wrapper("c" * (MAX + 50))
    results = wrapper.read_multiple_files(file_paths=["big.txt"])
    assert "exceeds size limit" in results["big.txt"]


def test_read_multiple_files_true_is_inert_and_warns(caplog):
    wrapper = make_wrapper("c" * (MAX + 50))
    with caplog.at_level(logging.WARNING):
        results = wrapper.read_multiple_files(
            file_paths=["big.txt"], skip_size_check=True
        )
    assert "exceeds size limit" in results["big.txt"]
    assert SKIP_SIZE_CHECK_DEPRECATION_MSG in caplog.text


# ---------------------------------------------------------------------------
# SSC_SCHEMA — skip_size_check absent from both LLM args_schemas
# ---------------------------------------------------------------------------

def test_read_multiple_files_schema_has_no_skip_size_check():
    wrapper = make_wrapper("ok")
    schema = wrapper._get_file_operation_schemas()["read_multiple_files"]
    assert "skip_size_check" not in schema.model_fields


def test_read_file_schema_has_no_skip_size_check():
    wrapper = make_wrapper("ok")
    tools = wrapper.get_available_tools()
    read_file_tool = next(t for t in tools if t["name"] == "read_file")
    assert "skip_size_check" not in read_file_tool["args_schema"].model_fields
