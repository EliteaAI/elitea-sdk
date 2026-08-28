"""Shared tool copy-and-patch helper for wrapping middleware."""

import types

from langchain_core.tools import BaseTool


def rebind_invoke_after_copy(copied: BaseTool) -> None:
    """``_patch_tool_invoke`` binds ``invoke`` with ``types.MethodType``; ``model_copy()``
    keeps that bound method in ``__dict__`` still bound to the *original*, so without
    re-binding, the copy's ``invoke`` routes execution back past every patch on the copy."""
    if 'invoke' in copied.__dict__:
        stale = copied.__dict__['invoke']
        if callable(stale) and hasattr(stale, '__func__'):
            object.__setattr__(copied, 'invoke', types.MethodType(stale.__func__, copied))
