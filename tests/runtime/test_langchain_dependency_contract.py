"""Import contracts for non-public LangChain APIs used by the SDK."""


def test_non_public_framework_symbols_used_by_sdk_remain_available():
    """Fail early when an upstream release removes an SDK runtime dependency."""

    from langchain.agents.middleware.summarization import (
        _DEFAULT_MESSAGES_TO_KEEP,
    )
    from langchain_community.document_loaders.directory import _is_visible
    from langchain_core.runnables.config import var_child_runnable_config
    from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
    from langgraph.managed.base import is_managed_value

    assert isinstance(_DEFAULT_MESSAGES_TO_KEEP, int)
    assert callable(_is_visible)
    assert hasattr(var_child_runnable_config, "get")
    assert CONFIG_KEY_SCRATCHPAD == "__pregel_scratchpad"
    assert callable(is_managed_value)


def _runtime_pins():
    """The pins the SDK ships with live under optional-dependencies.runtime, not
    project.dependencies — reading the wrong table silently pins nothing."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    with open(root / "pyproject.toml", "rb") as handle:
        data = tomllib.load(handle)

    pins = {}
    for spec in data["project"]["optional-dependencies"]["runtime"]:
        if "==" in spec:
            name, version = spec.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def test_installed_versions_match_the_shipped_pins():
    """Asserted, not skipped: a stale venv makes every ToolMessage.status assertion
    in the suite meaningless while still reporting green."""
    from importlib.metadata import version

    pins = _runtime_pins()
    for package in ("langgraph", "langgraph-prebuilt", "langchain-core"):
        assert package in pins, f"{package} lost its pin in pyproject.toml"
        assert version(package) == pins[package], (
            f"{package} installed {version(package)}, pinned {pins[package]} — "
            "reinstall with .venv/bin/pip install -e '.[all]'"
        )


def test_tool_node_default_error_handler_still_reraises():
    """Why the swarm ToolNodes must pass handle_tool_errors explicitly (#6172): the
    upstream default lets a raising tool escape the subgraph and kill the graph.
    If a future release starts swallowing instead, our design needs re-review."""
    import pytest
    from pydantic import BaseModel, ValidationError
    from langgraph.prebuilt.tool_node import (
        ToolInvocationError,
        _default_handle_tool_errors,
    )

    with pytest.raises(ValueError):
        _default_handle_tool_errors(ValueError("boom"))

    class _Args(BaseModel):
        value: int

    try:
        _Args(value="not-an-int")
    except ValidationError as source:
        invocation_error = ToolInvocationError("t", source, {"value": "not-an-int"})

    assert _default_handle_tool_errors(invocation_error) == invocation_error.message
