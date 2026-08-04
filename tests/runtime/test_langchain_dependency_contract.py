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
