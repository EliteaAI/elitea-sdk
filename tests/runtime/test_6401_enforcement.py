"""#6401: a detected provider/MCP failure becomes an actual failure.

#6168 detected `status: Error/Failed` and MCP `isError`, logged a shadow line, and let
the call continue as a success. This suite covers the enforcement half on the SDK side:

* the new top classifier rung, which reads the provider's own declared `error_category`
  off the raised exception instead of guessing from prose;
* the proof that a provider failure and an ordinary raising toolkit tool reach
  `_finalize_outcome` through the same `except` branch, so #6171's
  `last_tool_outcome.status` finally reads `error` for a failing provider tool;
* the two cost levers that this enforcement makes necessary (POLICY enrichment skip,
  per-conversation enrichment dedup).

The three MCP raise sites themselves are covered in `test_6168_failure_signal_shadow.py`,
where the shadow-log assertions already live.
"""
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from elitea_sdk.runtime.middleware.strategies import (
    ExceptionContext,
    LoggingStrategy,
    TransformErrorStrategy,
)
from elitea_sdk.runtime.middleware.tool_exception_handler import (
    ToolExceptionHandlerMiddleware,
)
from elitea_sdk.runtime.tool_outcome import (
    PROVIDER_CATEGORY_ATTR,
    ToolErrorClass,
    ToolResultStatus,
    _PROVIDER_CATEGORY_CLASSES,
    classify_tool_error,
    outcome_sink,
)
from elitea_sdk.runtime.tools.mcp_remote_tool import McpRemoteTool
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired


def _provider_exception(message, category, error_type=None):
    """What provider_worker._provider_failure_exception produces: a plain ToolException
    with the provider's machine-readable signal attached to the instance, no subclass."""
    exc = ToolException(message)
    setattr(exc, PROVIDER_CATEGORY_ATTR, category)
    exc.provider_error_type = error_type
    return exc


class _Args(BaseModel):
    query: str = Field(description="anything")


def _tool_raising(exc, name="generate_image"):
    def _func(query: str) -> str:
        raise exc

    return StructuredTool.from_function(
        func=_func, name=name, description="test tool", args_schema=_Args,
    )


class _StubLLM:
    """TransformErrorStrategy only calls .invoke(messages).content."""

    def __init__(self, content="STUB REWRITE"):
        self._content = content

    def invoke(self, messages, *args, **kwargs):
        return type("_Resp", (), {"content": self._content})()


def _hermetic(fn):
    """get_toolkit_faq does a real 5s-timeout urlopen and the code loaders walk the
    filesystem; all three only feed the prompt, which the stub LLM ignores."""
    for target in (
        "elitea_sdk.runtime.middleware.faq_fetcher.get_toolkit_faq",
        "elitea_sdk.runtime.utils.tool_code_loader.load_tool_code",
        "elitea_sdk.runtime.utils.tool_code_extractor.extract_tool_code",
    ):
        fn = patch(target, return_value=None)(fn)
    return fn


class TestProviderCategoryRung:
    """The provider states its own category, so it beats every heuristic below it."""

    @pytest.mark.parametrize("category,expected", sorted(_PROVIDER_CATEGORY_CLASSES.items()))
    def test_every_category_maps_off_the_exception_attribute(self, category, expected):
        exc = _provider_exception("generate_image failed: boom", category)

        assert classify_tool_error(exc) is expected

    def test_all_four_classes_are_reachable_through_the_rung(self):
        """Guards against a table that silently loses a whole class."""
        reached = {
            classify_tool_error(_provider_exception("x", category))
            for category in _PROVIDER_CATEGORY_CLASSES
        }

        assert reached == set(ToolErrorClass)

    def test_unmapped_category_falls_through_to_the_ladder(self):
        """AC 2: an unknown category must not short-circuit into a guess."""
        exc = _provider_exception("something odd", "brand_new_category_nobody_mapped")

        assert classify_tool_error(exc) is None

    def test_ladder_below_is_unchanged(self):
        assert classify_tool_error(ValueError("bad argument")) is ToolErrorClass.INPUT
        assert classify_tool_error(TimeoutError("slow")) is ToolErrorClass.INFRASTRUCTURE
        assert classify_tool_error(ToolException("nothing recognisable")) is None

    def test_non_string_category_is_ignored(self):
        exc = ToolException("bad argument")
        setattr(exc, PROVIDER_CATEGORY_ATTR, {"not": "a string"})

        assert classify_tool_error(exc) is None

    def test_rung_beats_the_prose_fallback(self):
        """The prose says "timed out", the provider says the input was wrong. The
        declared field wins — this is the whole point of putting the rung on top."""
        exc = _provider_exception("generate_image failed: request timed out", "invalid_input")

        assert classify_tool_error(exc) is ToolErrorClass.INPUT

    def test_rung_beats_a_status_attribute(self):
        exc = _provider_exception("generate_image failed", "invalid_input")
        exc.status_code = 503

        assert classify_tool_error(exc) is ToolErrorClass.INPUT

    def test_rung_applies_across_the_cause_chain(self):
        """A toolkit that re-wraps the provider exception must not lose the category."""
        inner = _provider_exception("provider said no", "rate_limit")
        outer = ToolException("wrapper prose with no signal in it")
        outer.__cause__ = inner

        assert classify_tool_error(outer) is ToolErrorClass.INFRASTRUCTURE


class TestSamePathAsAnyOtherToolFailure:
    """AC 8: no parallel shaping layer. A provider failure and an ordinary toolkit
    failure differ only in what the classifier can read off them."""

    def _run(self, exc):
        middleware = ToolExceptionHandlerMiddleware(
            strategies=[TransformErrorStrategy(llm=None), LoggingStrategy()]
        )
        wrapped = middleware.wrap_tool(_tool_raising(exc))
        with outcome_sink() as recorded:
            message = wrapped.run({"query": "x"})
        return message, recorded

    def test_provider_failure_records_an_error_outcome(self):
        exc = _provider_exception("generate_image failed: quota", "rate_limit", "RateLimit")

        message, recorded = self._run(exc)

        assert len(recorded) == 1
        outcome = recorded[0]
        assert outcome.status is ToolResultStatus.ERROR
        assert outcome.error_class is ToolErrorClass.INFRASTRUCTURE
        assert outcome.retriable is True
        assert outcome.tool_name == "generate_image"
        assert outcome.exception_type == "ToolException"
        assert message == outcome.message

    def test_ordinary_toolkit_failure_records_the_same_shape(self):
        message, recorded = self._run(ValueError("issue_number must be numeric"))

        assert len(recorded) == 1
        outcome = recorded[0]
        assert outcome.status is ToolResultStatus.ERROR
        assert outcome.error_class is ToolErrorClass.INPUT
        assert outcome.retriable is False
        assert message == outcome.message

    def test_provider_failure_is_no_longer_reported_as_success(self):
        """The #6171 regression this issue exists to fix: before enforcement the sink
        stayed empty and the state projection fabricated status=success."""
        exc = _provider_exception("generate_image failed: bad size", "invalid_input")

        _, recorded = self._run(exc)

        assert [o.status for o in recorded] == [ToolResultStatus.ERROR]

    def test_unmapped_category_still_records_an_error_outcome(self):
        """An unclassified failure is still a failure — only error_class is None."""
        exc = _provider_exception("generate_image failed", "not_in_the_table")

        _, recorded = self._run(exc)

        assert recorded[0].status is ToolResultStatus.ERROR
        assert recorded[0].error_class is None

    def test_error_outcome_is_logged_with_the_derived_class(self, caplog):
        exc = _provider_exception("generate_image failed: oom", "out_of_memory")

        with caplog.at_level(logging.INFO):
            self._run(exc)

        lines = [r.getMessage() for r in caplog.records if "error outcome" in r.getMessage()]
        assert lines
        assert "infrastructure" in lines[-1]


class TestEnrichmentSkippedForPolicy:
    """POLICY joins INFRASTRUCTURE on the template path: rewriting a refusal adds
    nothing the template does not already say, and enforcement multiplies the volume."""

    @_hermetic
    def test_policy_error_stays_on_the_template_path(self, _faq, _load, _extract):
        stub = _StubLLM()
        stub.invoke = MagicMock(wraps=stub.invoke)
        middleware = ToolExceptionHandlerMiddleware(
            strategies=[TransformErrorStrategy(llm=stub), LoggingStrategy()]
        )
        exc = _provider_exception("generate_image failed: bad key", "authentication_error")
        wrapped = middleware.wrap_tool(_tool_raising(exc))

        result = wrapped.run({"query": "x"})

        stub.invoke.assert_not_called()
        assert result == (
            "Tool 'generate_image' failed.\n"
            "\n"
            "Error: generate_image failed: bad key\n"
            "\n"
            "Please check the input parameters and try again, "
            "or use an alternative approach."
        )

    @_hermetic
    def test_input_class_provider_failure_still_enriches(self, _faq, _load, _extract):
        stub = _StubLLM()
        stub.invoke = MagicMock(wraps=stub.invoke)
        middleware = ToolExceptionHandlerMiddleware(
            strategies=[TransformErrorStrategy(llm=stub), LoggingStrategy()]
        )
        exc = _provider_exception("generate_image failed: bad size", "invalid_input")
        wrapped = middleware.wrap_tool(_tool_raising(exc))

        result = wrapped.run({"query": "x"})

        stub.invoke.assert_called_once()
        assert result.startswith("STUB REWRITE")


class TestEnrichmentDedup:
    """A retry loop hits the same failure repeatedly; each attempt used to pay another
    completion. The cache is per-conversation and bounded."""

    def _context(self, message="issue_number must be numeric", tool_name="update_issue"):
        tool = _tool_raising(ValueError(message), name=tool_name)
        return ExceptionContext(
            tool=tool,
            error=ValueError(message),
            args=(),
            kwargs={},
            metadata={"error_class": ToolErrorClass.INPUT},
        )

    @_hermetic
    def test_identical_failures_pay_one_completion(self, _faq, _load, _extract):
        stub = _StubLLM()
        stub.invoke = MagicMock(wraps=stub.invoke)
        strategy = TransformErrorStrategy(llm=stub)

        first = strategy.handle_exception(self._context()).error_message
        second = strategy.handle_exception(self._context()).error_message

        assert stub.invoke.call_count == 1
        assert first == second
        assert first.startswith("STUB REWRITE")

    @_hermetic
    def test_key_ignores_whitespace_and_case(self, _faq, _load, _extract):
        stub = _StubLLM()
        stub.invoke = MagicMock(wraps=stub.invoke)
        strategy = TransformErrorStrategy(llm=stub)

        strategy.handle_exception(self._context("Issue_number must be numeric"))
        strategy.handle_exception(self._context("issue_number   must\nbe numeric"))

        assert stub.invoke.call_count == 1

    @_hermetic
    def test_different_failures_are_not_conflated(self, _faq, _load, _extract):
        stub = _StubLLM()
        stub.invoke = MagicMock(wraps=stub.invoke)
        strategy = TransformErrorStrategy(llm=stub)

        strategy.handle_exception(self._context("issue_number must be numeric"))
        strategy.handle_exception(self._context("title is too long"))
        strategy.handle_exception(self._context("issue_number must be numeric", tool_name="other_tool"))

        assert stub.invoke.call_count == 3

    @_hermetic
    def test_reset_clears_the_cache(self, _faq, _load, _extract):
        stub = _StubLLM()
        stub.invoke = MagicMock(wraps=stub.invoke)
        strategy = TransformErrorStrategy(llm=stub)

        strategy.handle_exception(self._context())
        strategy.reset()
        strategy.handle_exception(self._context())

        assert stub.invoke.call_count == 2

    @_hermetic
    def test_cache_is_bounded(self, _faq, _load, _extract):
        from elitea_sdk.runtime.middleware.strategies import _ENRICHMENT_CACHE_MAX_ENTRIES

        strategy = TransformErrorStrategy(llm=_StubLLM())
        for i in range(_ENRICHMENT_CACHE_MAX_ENTRIES + 10):
            strategy.handle_exception(self._context(f"failure number {i}"))

        assert len(strategy._enrichment_cache) == _ENRICHMENT_CACHE_MAX_ENTRIES

    @_hermetic
    def test_conversation_start_resets_through_the_middleware(self, _faq, _load, _extract):
        stub = _StubLLM()
        stub.invoke = MagicMock(wraps=stub.invoke)
        strategy = TransformErrorStrategy(llm=stub)
        middleware = ToolExceptionHandlerMiddleware(strategies=[strategy])
        wrapped = middleware.wrap_tool(_tool_raising(ValueError("bad input"), name="update_issue"))

        wrapped.run({"query": "x"})
        middleware.on_conversation_start("conv-2")
        middleware.wrap_tool(_tool_raising(ValueError("bad input"), name="update_issue")).run({"query": "x"})

        assert stub.invoke.call_count == 2


class TestSignalsStillBypassEnforcement:
    """AC 5: removing the swallow in McpRemoteTool._run must not start shaping signals."""

    def _tool(self):
        return McpRemoteTool(
            name="do_thing",
            description="test tool",
            client=MagicMock(),
            server="unused",
            server_url="https://example.invalid/mcp",
            session_id="sess-1",
            original_tool_name="do_thing",
            metadata={"toolkit_name": "github", "toolkit_type": "mcp"},
        )

    def test_authorization_required_propagates_raw_from_run(self):
        signal = McpAuthorizationRequired("login first", server_url="https://example.invalid/mcp")

        with patch.object(McpRemoteTool, "_run_in_new_loop", side_effect=signal):
            with pytest.raises(McpAuthorizationRequired) as excinfo:
                self._tool()._run()

        assert excinfo.value is signal

    def test_authorization_required_is_not_shadow_logged(self, caplog):
        """It is a handshake, not a tool failure — it must not inflate the counters."""
        signal = McpAuthorizationRequired("login first", server_url="https://example.invalid/mcp")

        with patch.object(McpRemoteTool, "_run_in_new_loop", side_effect=signal):
            with caplog.at_level(logging.WARNING):
                with pytest.raises(McpAuthorizationRequired):
                    self._tool()._run()

        assert not any("TOOL_FAILURE_SHADOW" in r.getMessage() for r in caplog.records)

    def test_middleware_reraises_the_signal_instead_of_shaping_it(self):
        signal = McpAuthorizationRequired("login first", server_url="https://example.invalid/mcp")
        middleware = ToolExceptionHandlerMiddleware(
            strategies=[TransformErrorStrategy(llm=None), LoggingStrategy()]
        )
        wrapped = middleware.wrap_tool(_tool_raising(signal))

        with pytest.raises(McpAuthorizationRequired):
            wrapped.run({"query": "x"})


class TestShadowLogEmittedOncePerFailure:
    """The isError raise travels back out through _run's catch-all. That handler also
    shadow-logs, so without a marker one MCP failure is counted twice — once as
    `mcp_is_error/remote` and once, wrongly, as `mcp_exception_swallowed`."""

    def _tool(self):
        return McpRemoteTool(
            name="do_thing",
            description="test tool",
            client=MagicMock(),
            server="unused",
            server_url="https://example.invalid/mcp",
            session_id="sess-1",
            original_tool_name="do_thing",
            metadata={"toolkit_name": "github", "toolkit_type": "mcp"},
        )

    def _shadow_payloads(self, records):
        return [
            json.loads(r.getMessage()[len("TOOL_FAILURE_SHADOW "):])
            for r in records if "TOOL_FAILURE_SHADOW" in r.getMessage()
        ]

    def test_is_error_failure_logs_exactly_one_line_through_run(self, caplog):
        tool = self._tool()
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.call_tool = AsyncMock(
            return_value={"isError": True, "content": [{"type": "text", "text": "boom"}]}
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("elitea_sdk.runtime.tools.mcp_remote_tool.McpClient", return_value=mock_client):
            with caplog.at_level(logging.WARNING):
                with pytest.raises(ToolException, match="boom"):
                    tool._run(query="x")

        payloads = self._shadow_payloads(caplog.records)
        assert len(payloads) == 1
        assert payloads[0]["detected_by"] == "mcp_is_error/remote"

    def test_transport_failure_still_logs_the_swallow_site(self, caplog):
        """An unmarked exception has not been logged anywhere else, so it must be logged here."""
        tool = self._tool()

        with patch.object(McpRemoteTool, "_run_in_new_loop", side_effect=RuntimeError("connection refused")):
            with caplog.at_level(logging.WARNING):
                with pytest.raises(RuntimeError):
                    tool._run()

        payloads = self._shadow_payloads(caplog.records)
        assert len(payloads) == 1
        assert payloads[0]["detected_by"] == "mcp_exception_swallowed"
