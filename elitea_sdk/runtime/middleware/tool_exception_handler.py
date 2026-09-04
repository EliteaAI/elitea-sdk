"""
Tool Exception Handler Middleware

Provides intelligent error handling for tool execution using pluggable strategies:
- Transform errors into human-readable messages (LLM-powered)
- Circuit breaker pattern for repeatedly failing tools
- Error tracking and monitoring
- Composable strategy pipeline

Example:
    from elitea_sdk.runtime.middleware import (
        ToolExceptionHandlerMiddleware,
        TransformErrorStrategy,
        CircuitBreakerStrategy,
        LoggingStrategy
    )

    middleware = ToolExceptionHandlerMiddleware(
        strategies=[
            TransformErrorStrategy(llm=my_llm),
            CircuitBreakerStrategy(threshold=3),
            LoggingStrategy()
        ]
    )
"""

import asyncio
import inspect
import logging
from functools import wraps
from typing import List, Optional, Dict, Any, Callable, Tuple

from elitea_sdk.runtime.exceptions import budget_exceeded_from, SandboxAdmissionRefused
from elitea_sdk.runtime.tool_outcome import (
    ToolOutcome,
    ToolResultStatus,
    classify_tool_error,
    outcome_sink,
    record_outcome,
    retriable_for,
)
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.errors import GraphBubbleUp

from ..langchain.utils import log_tool_result
from ..tool_result_bounds import bound_and_record, toolkit_type_of
from .base import Middleware
from .strategies import (
    ExceptionHandlerStrategy,
    ExceptionContext,
    TransformErrorStrategy,
    CircuitBreakerStrategy,
    LoggingStrategy
)
from .tool_patching import rebind_invoke_after_copy
from ..tools.application import Application

logger = logging.getLogger(__name__)


def _str_or_none(value: Any) -> Optional[str]:
    """Tool metadata is free-form, and the envelope must never fail validation on it."""
    return value if isinstance(value, str) else None


def reraise_signal_exceptions(error: Exception) -> None:
    """Anything re-raised here is a signal for the caller, not a tool failure.

    Module-level so the swarm ToolNode handler below shares one list with the
    middleware; two copies would drift and silently start swallowing a signal.
    """
    # MCP authorization is a cross-cutting auth concern; GraphBubbleUp carries
    # interrupt() and other graph-level signals that must reach the graph.
    if isinstance(error, (McpAuthorizationRequired, GraphBubbleUp)):
        raise error

    # "This execution path does not exist" - the agent loop branches on it to fall
    # back from ainvoke to invoke, so turning it into prose would break that fallback.
    if isinstance(error, NotImplementedError):
        raise error

    # Budget rejections bypass the strategies: TransformErrorStrategy would
    # spend another LLM call rewriting a policy error into tool output
    budget_error = budget_exceeded_from(error)
    if budget_error is not None:
        raise budget_error from error


def swarm_handle_tool_errors(error: Exception) -> str:
    """``handle_tool_errors`` for the swarm ToolNodes (#6172).

    ToolNode's own default re-raises anything but a ToolInvocationError, which takes
    the whole swarm graph down; plain ``True`` would swallow the signals above, since
    ToolNode only protects GraphBubbleUp before consulting this callable. The
    ``Exception`` annotation is what langgraph's _infer_handled_types reads.

    SandboxAdmissionRefused intentionally falls through to classification below rather
    than being re-raised here: the middleware bypasses it (pipeline path), but on the
    agent path the turn should degrade gracefully, not abort.
    """
    reraise_signal_exceptions(error)
    error_class = classify_tool_error(error)
    outcome = ToolOutcome(
        status=ToolResultStatus.ERROR,
        message=f"Error executing tool: {error}",
        error_class=error_class,
        retriable=retriable_for(error_class),
        exception_type=type(error).__name__,
        retry_after=getattr(error, 'retry_after', None),
    )
    record_outcome(outcome)
    return outcome.message


def _stamp_error_status(result, recorded):
    """Mark a ToolMessage as failed when the middleware recorded a non-success outcome.

    The middleware returns its error prose instead of raising, so ToolNode builds a
    success message over it (#6477). The signal is the recorded envelope, never the text.
    """
    # Only ERROR flips the message: TRUNCATED is a recorded outcome on a call that
    # still succeeded, and marking it 'error' would fail a working tool (#6140).
    if not any(o.status is ToolResultStatus.ERROR for o in recorded):
        return result
    if isinstance(result, list):
        return [_stamp_error_status(item, recorded) for item in result]
    # A status ToolNode already set (its own error path) is more specific - leave it.
    if isinstance(result, ToolMessage) and result.status == 'success':
        return result.model_copy(update={'status': 'error'})
    # Command: handoff control flow, never a middleware-wrapped tool.
    return result


def swarm_wrap_tool_call(request, execute):
    """``wrap_tool_call`` for the swarm ToolNodes (#6477)."""
    with outcome_sink() as recorded:
        result = execute(request)
    return _stamp_error_status(result, recorded)


async def swarm_awrap_tool_call(request, execute):
    """``awrap_tool_call``: without it ToolNode would run the sync wrapper on the loop."""
    with outcome_sink() as recorded:
        result = await execute(request)
    return _stamp_error_status(result, recorded)


class ToolExceptionHandlerMiddleware(Middleware):
    """
    Wraps agent tools with intelligent exception handling using pluggable strategies.

    Uses a strategy pattern to allow flexible error handling configurations.
    Each strategy processes exceptions in sequence, allowing composition of
    behaviors like logging, circuit breaking, and error transformation.

    Example:
        ```python
        middleware = ToolExceptionHandlerMiddleware(
            strategies=[
                TransformErrorStrategy(llm=my_llm),
                CircuitBreakerStrategy(threshold=3),
                LoggingStrategy()
            ]
        )

        assistant = client.application(
            application_id='app_123',
            middleware=[middleware]
        )
        ```
    """

    # Set on every wrapped tool so coverage of new construction paths is assertable
    WRAPPED_MARKER = '_elitea_error_wrapped'

    def __init__(
        self,
        strategies: List[ExceptionHandlerStrategy],
        conversation_id: Optional[str] = None,
        callbacks: Optional[Dict[str, Callable]] = None,
        excluded_tools: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize Tool Exception Handler Middleware.

        Args:
            strategies: List of exception handler strategies to apply (in sequence)
            conversation_id: Conversation ID for state tracking
            callbacks: Optional dict of callback functions for events
            excluded_tools: List of tool names to not wrap with error handling
        """
        super().__init__(conversation_id, callbacks, **kwargs)

        if not strategies:
            raise ValueError("At least one strategy is required.")

        self.strategies = strategies
        self.excluded_tools = set(excluded_tools or [])

        # Wrapped tools cache to avoid double-wrapping (keyed by object id)
        self._wrapped_tools_cache: Dict[int, BaseTool] = {}

        logger.debug(
            f"ToolExceptionHandlerMiddleware initialized with {len(strategies)} strategies: "
            f"{[s.__class__.__name__ for s in strategies]}"
        )

    def get_tools(self) -> List[BaseTool]:
        """
        This middleware doesn't add new tools.
        Tools are wrapped via the wrap_tool() method called by the agent.
        """
        return []

    def get_system_prompt(self) -> str:
        """Add instructions for handling tool errors."""
        return """### Tool Error Handling

When a tool fails with an error:
* Read the error message carefully - it contains guidance on what went wrong
* All the issues are mostly related to 3rd party APIs used by the tools (corresponding exceptions will be raised)
* If the error suggests a fix (e.g., missing or invalid parameter), reply with suggested fix
* If no alternative exists, inform the user about the issue and ask for help from support team (https://elitea.ai/docs/support/contact-support/)
"""

    def wrap_tool(self, tool: BaseTool) -> BaseTool:
        """
        Wrap a tool with exception handling logic.

        Sync and async entry points are both covered, and the tool keeps its own class so
        nothing carried on the instance (patched invoke, response_format) is lost.

        Args:
            tool: Original tool to wrap

        Returns:
            Wrapped tool with error handling, or original if excluded/already wrapped
        """
        # Don't wrap Application tools - they have their own invocation logic
        # and wrapping causes state variables to be lost due to args_schema filtering
        if isinstance(tool, Application):
            logger.debug(f"Tool '{tool.name}' is an Application, skipping error handling wrapper")
            return tool

        # Don't wrap if tool is in exclusion list
        if tool.name in self.excluded_tools:
            logger.debug(f"Tool '{tool.name}' is excluded from error handling")
            return tool

        # Check if already wrapped (avoid double-wrapping)
        # Use object identity (id) as cache key, not tool.name, because different toolkits
        # can have tools with the same name (e.g., index_data in both GitHub and Confluence).
        # Name-based caching would return the same wrapped object for both, breaking
        # the dedup logic in Assistant.__init__ which relies on distinct objects.
        cache_key = id(tool)
        if cache_key in self._wrapped_tools_cache:
            logger.debug(f"Tool '{tool.name}' already wrapped, returning cached version")
            return self._wrapped_tools_cache[cache_key]

        if getattr(tool, self.WRAPPED_MARKER, False):
            return tool

        sync_target = self._sync_target(tool)
        async_target = self._async_target(tool)
        if sync_target is None and async_target is None:
            # Loud on purpose: this tool's failures reach the LLM as raw provider errors
            logger.error(
                "Tool '%s' (%s) exposes no wrappable callable - its errors bypass the "
                "tool error contract", tool.name, type(tool).__name__,
            )
            return tool

        try:
            wrapped_tool = self._patched_copy(tool, sync_target, async_target)
        except Exception as e:
            logger.error(f"Failed to wrap tool '{tool.name}': {e}", exc_info=True)
            return tool  # Return original tool if wrapping fails

        self._wrapped_tools_cache[cache_key] = wrapped_tool
        logger.debug(
            "Wrapped tool '%s' with error handling (sync=%s, async=%s)",
            tool.name,
            sync_target[0] if sync_target else None,
            async_target[0] if async_target else None,
        )
        return wrapped_tool

    def _patched_copy(
        self,
        tool: BaseTool,
        sync_target: Optional[Tuple[str, Callable]],
        async_target: Optional[Tuple[str, Callable]],
    ) -> BaseTool:
        """Copy the tool and replace only its execution callables, as the sensitive-action
        guard does - a rebuilt StructuredTool silently drops coroutine and patched invoke."""
        copied = tool.model_copy()
        rebind_invoke_after_copy(copied)

        if sync_target is not None:
            attr, original = sync_target
            setattr(copied, attr, self._sync_wrapper(tool, original))
        if async_target is not None:
            attr, original = async_target
            setattr(copied, attr, self._async_wrapper(tool, original))

        # Routes pydantic ValidationError, raised during BaseTool.run() input parsing
        # before the wrapped callable is ever reached, through the same strategies.
        copied.handle_validation_error = self._validation_error_router(tool)

        # Downstream code (e.g. swarm agent detection) inspects the unwrapped tool
        object.__setattr__(copied, '_original_tool', tool)
        object.__setattr__(copied, self.WRAPPED_MARKER, True)
        return copied

    def _bound(self, tool: BaseTool, result):
        """Bound an oversized result here, the one point every wrapped tool passes."""
        return bound_and_record(result, getattr(tool, 'name', None), toolkit_type_of(tool))

    def _sync_wrapper(self, tool: BaseTool, original: Callable) -> Callable:
        forward_run_manager = self._accepts_run_manager(original)

        @wraps(original)
        def error_handled_func(*args, run_manager=None, **kwargs):
            if forward_run_manager:
                kwargs['run_manager'] = run_manager
            try:
                result = original(*args, **kwargs)
            except Exception as e:
                return self._shape_error_output(tool, self._handle_tool_exception(tool, e, args, kwargs))
            self._notify_success(tool.name)
            return self._bound(tool, result)

        return error_handled_func

    def _async_wrapper(self, tool: BaseTool, original: Callable) -> Callable:
        forward_run_manager = self._accepts_run_manager(original)

        @wraps(original)
        async def error_handled_coroutine(*args, run_manager=None, **kwargs):
            if forward_run_manager:
                kwargs['run_manager'] = run_manager
            try:
                result = await original(*args, **kwargs)
            except Exception as e:
                # Signals are pure control flow - re-raise on the loop, no thread hop.
                self._reraise_signals(e)
                # The strategies do a blocking LLM completion; on the loop it would stall
                # every sibling of a parallel tool batch for the length of a completion.
                message = await asyncio.to_thread(self._run_error_pipeline, tool, e, args, kwargs)
                return self._shape_error_output(tool, message)
            self._notify_success(tool.name)
            return self._bound(tool, result)

        return error_handled_coroutine

    @staticmethod
    def _shape_error_output(tool: BaseTool, message: str):
        """A content_and_artifact tool must return a two-tuple or BaseTool.run raises over
        it; the artifact is None because the failed call produced no raw output."""
        if getattr(tool, 'response_format', 'content') == 'content_and_artifact':
            return message, None
        return message

    def _validation_error_router(self, tool: BaseTool) -> Callable:
        def route_validation_error(e):
            context = ExceptionContext(tool=tool, error=e, args=(), kwargs={})
            self._classify_into(context)
            context = self._run_strategies(context)
            return self._finalize_outcome(context).message

        return route_validation_error

    def _handle_tool_exception(self, tool: BaseTool, error: Exception, args, kwargs) -> str:
        """Shared by both wrappers so the sync and async paths shape errors identically."""
        self._reraise_signals(error)
        return self._run_error_pipeline(tool, error, args, kwargs)

    @staticmethod
    def _reraise_signals(error: Exception) -> None:
        # Pipeline-path only: shaping this into prose would put it back into an output
        # variable, which is the bug this bypass exists to prevent.
        if isinstance(error, SandboxAdmissionRefused):
            raise error
        reraise_signal_exceptions(error)

    def _run_error_pipeline(self, tool: BaseTool, error: Exception, args, kwargs) -> str:
        """Blocking: TransformErrorStrategy spends an LLM completion in here."""
        context = ExceptionContext(tool=tool, error=error, args=args, kwargs=kwargs)
        self._classify_into(context)
        # A strategy may raise ToolException (e.g. circuit breaker) - let it through as-is
        context = self._run_strategies(context)
        return self._finalize_outcome(context).message

    def _notify_success(self, tool_name: str) -> None:
        for strategy in self.strategies:
            try:
                strategy.on_success(tool_name)
            except Exception as e:
                logger.error(f"Strategy {strategy.__class__.__name__} on_success failed: {e}")

    def _run_strategies(self, context: ExceptionContext) -> ExceptionContext:
        """A strategy may return a replacement context instead of mutating this one; carry
        the metadata over so neither later strategies nor the envelope lose the facts."""
        for strategy in self.strategies:
            result = strategy.handle_exception(context)
            if result is not context:
                # The replacement's own keys win; it may have classified more precisely.
                result.metadata = {**context.metadata, **result.metadata}
            context = result
        return context

    def _classify_into(self, context: ExceptionContext) -> None:
        """Classify before the strategies run, since a strategy may gate on error_class."""
        error_class = classify_tool_error(context.error)
        context.metadata['error_class'] = error_class
        context.metadata['retriable'] = retriable_for(error_class)
        context.metadata['exception_type'] = context.error_type
        context.metadata['retry_after'] = getattr(context.error, 'retry_after', None)

    def _finalize_outcome(self, context: ExceptionContext) -> ToolOutcome:
        """Assemble the envelope once the message is final. message is a pass-through."""
        tool_metadata = getattr(context.tool, 'metadata', None) or {}
        outcome = ToolOutcome(
            status=ToolResultStatus.ERROR,
            message=context.error_message or context.error_str,
            tool_name=_str_or_none(context.tool_name),
            error_class=context.metadata.get('error_class'),
            retriable=bool(context.metadata.get('retriable', False)),
            exception_type=_str_or_none(context.metadata.get('exception_type')),
            toolkit_type=_str_or_none(tool_metadata.get('toolkit_type')),
            retry_after=context.metadata.get('retry_after'),
        )
        log_tool_result(
            logger, type(self).__name__, outcome.tool_name,
            tool_metadata.get('toolkit_id'),
            # message is excluded so the classified fields survive the preview cap;
            # the prose is already logged as the node's response.
            outcome.model_dump(mode='json', exclude={'message'}),
            label='error outcome',
        )
        # Hand the envelope to a caller that installed a sink (a pipeline node); the
        # agent loop installs none, so this is a no-op there.
        record_outcome(outcome)
        return outcome

    @staticmethod
    def _accepts_run_manager(func: Callable) -> bool:
        try:
            return 'run_manager' in inspect.signature(func).parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _sync_target(tool: BaseTool) -> Optional[Tuple[str, Callable]]:
        """Which attribute holds the sync implementation, and the callable behind it."""
        if isinstance(tool, StructuredTool) and type(tool)._run is StructuredTool._run:
            return ('func', tool.func) if callable(tool.func) else None
        # BaseTool._run is an abstract stub that raises; patching it would fabricate a
        # sync path an async-only tool never had.
        if type(tool)._run is not BaseTool._run and callable(getattr(tool, '_run', None)):
            return '_run', tool._run
        if callable(getattr(tool, 'func', None)):
            return 'func', tool.func
        return None

    @staticmethod
    def _async_target(tool: BaseTool) -> Optional[Tuple[str, Callable]]:
        """Same for the async implementation. None also covers the tools whose inherited
        ``_arun`` just runs the sync path in an executor - patching sync already covers it."""
        if isinstance(tool, StructuredTool) and type(tool)._arun is StructuredTool._arun:
            coroutine = getattr(tool, 'coroutine', None)
            return ('coroutine', coroutine) if callable(coroutine) else None
        if type(tool)._arun is not BaseTool._arun and callable(getattr(tool, '_arun', None)):
            return '_arun', tool._arun
        coroutine = getattr(tool, 'coroutine', None)
        if callable(coroutine):
            return 'coroutine', coroutine
        return None

    def on_conversation_start(self, conversation_id: str) -> Optional[str]:
        """Reset strategy state on conversation start."""
        super().on_conversation_start(conversation_id)

        # Reset all strategies
        for strategy in self.strategies:
            try:
                strategy.reset()
            except Exception as e:
                logger.error(
                    f"Strategy {strategy.__class__.__name__} reset failed: {e}",
                    exc_info=True
                )

        # Clear wrapped tools cache
        self._wrapped_tools_cache.clear()

        logger.info(
            f"Reset error handling state for conversation {conversation_id}, "
            f"cleared {len(self.strategies)} strategies"
        )
        return None

    def on_conversation_end(self, conversation_id: str) -> None:
        """Log error statistics on conversation end."""
        super().on_conversation_end(conversation_id)

        # Try to get error summary from LoggingStrategy if present
        for strategy in self.strategies:
            if isinstance(strategy, LoggingStrategy):
                error_summary = strategy.get_error_summary()
                if error_summary:
                    logger.info(
                        f"Tool error summary for conversation {conversation_id}: "
                        f"{error_summary}"
                    )

                    # Fire summary callback
                    self._fire_callback('conversation_error_summary', {
                        'conversation_id': conversation_id,
                        'error_counts': error_summary,
                    })
                break

