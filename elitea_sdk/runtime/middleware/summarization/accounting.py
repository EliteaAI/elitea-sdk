"""
Token accounting for the context manager.

One counting path for the summarization trigger, the cutoff-fitting loop and
the number reported to the user, with an explicit record of where that number
came from (``token_source``).

Sources, best first:

1. ``provider``   - tokens reported by the provider in ``AIMessage.usage_metadata``,
                    normalised to the same arithmetic the usage analytics uses
2. ``tokenizer``  - an exact counter injected by the caller (e.g. a callable
                    wrapping LiteLLM's ``/utils/token_counter``)
3. ``approximate`` - character-heuristic counting (``_count_tokens_image_aware``)

With no tokenizer injected and no provider metadata available the accountant
returns exactly what the plain approximate counter returns.
"""

import json
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

logger = logging.getLogger(__name__)

TOKEN_SOURCE_PROVIDER = 'provider'
TOKEN_SOURCE_TOKENIZER = 'tokenizer'
TOKEN_SOURCE_APPROXIMATE = 'approximate'

# Characters per token by LangSmith provider id. Anthropic tokenizes denser than
# the 4.0 default; these mirror langchain's own _get_approximate_token_counter.
_CHARS_PER_TOKEN_BY_PROVIDER = {
    'anthropic': 3.3,
    'anthropic-chat': 3.3,
}


@dataclass
class ContextMeasurement:
    """A single accounting of what occupies the model's context window."""

    total: int = 0
    prompt_tokens: int = 0
    history_tokens: int = 0
    system_tokens: int = 0
    tool_schema_tokens: int = 0
    tool_message_tokens: int = 0
    user_facing_tokens: int = 0
    overhead_tokens: int = 0
    token_source: str = TOKEN_SOURCE_APPROXIMATE
    provider_input_tokens: Optional[int] = None
    provider_output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_context_info(self) -> Dict[str, Any]:
        """The token-related half of ``last_context_info``."""
        return {
            'token_count': self.total,
            'prompt_tokens': self.prompt_tokens,
            'token_count_user_facing': self.user_facing_tokens,
            'system_tokens': self.system_tokens,
            'tool_schema_tokens': self.tool_schema_tokens,
            'tool_message_tokens': self.tool_message_tokens,
            'overhead_tokens': self.overhead_tokens,
            'provider_input_tokens': self.provider_input_tokens,
            'provider_output_tokens': self.provider_output_tokens,
            'cache_read_tokens': self.cache_read_tokens,
            'cache_creation_tokens': self.cache_creation_tokens,
            'token_source': self.token_source,
        }


def empty_token_info() -> Dict[str, Any]:
    """Zeroed token fields, so downstream consumers never see a missing key."""
    return ContextMeasurement().as_context_info()


def read_provider_usage(messages) -> Optional[Dict[str, int]]:
    """Provider-reported usage from the most recent AIMessage, if any.

    Returns ``input_tokens`` / ``output_tokens`` plus the cache counters when
    the provider reports them. ``input_tokens`` describes the turn that produced
    this message, not the messages currently in hand.
    """
    for message in reversed(list(messages or [])):
        if not isinstance(message, AIMessage):
            continue
        usage = getattr(message, 'usage_metadata', None)
        if not usage:
            continue
        try:
            reading = {
                'input_tokens': int(usage.get('input_tokens') or 0),
                'output_tokens': int(usage.get('output_tokens') or 0),
            }
        except (TypeError, ValueError):
            return None
        details = usage.get('input_token_details') or {}
        if isinstance(details, dict):
            for src, dest in (('cache_read', 'cache_read_tokens'),
                              ('cache_creation', 'cache_creation_tokens')):
                if details.get(src) is not None:
                    try:
                        reading[dest] = int(details[src])
                    except (TypeError, ValueError):
                        pass
        return reading
    return None


def provider_prompt_tokens(usage: Dict[str, int]) -> int:
    """Whole-prompt tokens from a provider usage reading.

    ``input_tokens`` is inclusive of the cached blocks on some dialects and
    exclusive of them on others: Anthropic's wire format reports the uncached
    part only and ``langchain_anthropic`` adds the cached tokens back, while a
    proxy that re-shapes the same response as OpenAI passes the uncached number
    through with the cached ones in ``prompt_tokens_details``. Both conventions
    have to land on the same figure the usage analytics reports, which always
    prices the cached blocks as their own line items on top of the uncached
    input, so the convention is detected rather than assumed: cached tokens are
    added only when ``input_tokens`` is too small to already contain them.
    """
    input_tokens = int(usage.get('input_tokens') or 0)
    cached = int(usage.get('cache_read_tokens') or 0) + int(usage.get('cache_creation_tokens') or 0)
    if cached and input_tokens < cached:
        return input_tokens + cached
    return input_tokens


def resolve_chars_per_token(model) -> Optional[float]:
    """Characters-per-token for the model's provider, or None if unknown."""
    if model is None:
        return None
    try:
        params = model._get_ls_params()  # pylint: disable=protected-access
    except Exception:  # noqa: BLE001 - provider introspection is best-effort
        return None
    provider = (params or {}).get('ls_provider')
    if not provider:
        return None
    return _CHARS_PER_TOKEN_BY_PROVIDER.get(str(provider).lower())


def serialize_tool_schemas(tools) -> str:
    """The tool schemas as the provider receives them, as one text blob.

    Falls back to ``name: description`` for anything that cannot be converted,
    so an unusual tool still contributes its (smaller) share instead of zero.
    """
    if not tools:
        return ''
    parts = []
    for tool in tools:
        try:
            from langchain_core.utils.function_calling import convert_to_openai_tool
            parts.append(json.dumps(convert_to_openai_tool(tool), sort_keys=True))
        except Exception:  # noqa: BLE001 - accounting must never break a turn
            name = getattr(tool, 'name', '') or ''
            description = getattr(tool, 'description', '') or ''
            parts.append(f"{name}: {description}")
    return '\n'.join(parts)


class ContextAccountant:
    """Counts context tokens through the best source available.

    Args:
        token_counter: the approximate counter, used as the floor source.
        model: chat model, used only to resolve a provider-specific
            characters-per-token ratio when ``provider_aware`` is on.
        tokenizer: optional exact counter ``(messages) -> int``. Never called
            unless supplied; there is no network call on the hot path.
        provider_aware: when True the approximate counter is re-parameterised
            per provider. Off by default so counts stay identical to the
            pre-accountant behaviour.
    """

    def __init__(
        self,
        token_counter: Callable,
        *,
        model=None,
        tokenizer: Optional[Callable] = None,
        provider_aware: bool = False,
    ):
        self._base_counter = token_counter
        self._tokenizer = tokenizer
        self._model = model
        self._provider_aware = provider_aware
        self._approximate_counter = self._build_approximate_counter()

    def _build_approximate_counter(self) -> Callable:
        if not self._provider_aware:
            return self._base_counter
        chars_per_token = resolve_chars_per_token(self._model)
        if chars_per_token is None:
            return self._base_counter
        try:
            return partial(self._base_counter, chars_per_token=chars_per_token)
        except TypeError:
            return self._base_counter

    @property
    def approximate_counter(self) -> Callable:
        """The counter used for the approximate source."""
        return self._approximate_counter

    def set_tokenizer(self, tokenizer: Optional[Callable]) -> None:
        """Install (or clear) the exact counter."""
        self._tokenizer = tokenizer

    def count(self, messages) -> int:
        """Token count for ``messages`` from the best local source."""
        return self._count_with_source(messages)[0]

    def count_content(self, content) -> int:
        """Tokens for a raw content value (str or provider content-block list)."""
        if not content:
            return 0
        try:
            return self._count_with_source([SystemMessage(content=content)])[0]
        except Exception as exc:  # noqa: BLE001 - accounting must never break a turn
            logger.warning(f"Failed to count content tokens: {exc}")
            return 0

    def count_tool_schemas(self, tools) -> int:
        """Tokens spent on the tool definitions sent with every request."""
        return self.count_content(serialize_tool_schemas(tools))

    def _count_with_source(self, messages) -> tuple:
        if not messages:
            return 0, TOKEN_SOURCE_APPROXIMATE
        if self._tokenizer is not None:
            try:
                return int(self._tokenizer(messages)), TOKEN_SOURCE_TOKENIZER
            except Exception as exc:  # noqa: BLE001 - never fail a turn on counting
                logger.warning(f"Exact token counter failed, falling back to estimate: {exc}")
        return int(self._approximate_counter(messages)), TOKEN_SOURCE_APPROXIMATE

    def measure(
        self,
        messages: Optional[List[BaseMessage]] = None,
        *,
        user_facing_messages: Optional[List[BaseMessage]] = None,
        extra_messages: Optional[List[BaseMessage]] = None,
        system_tokens: int = 0,
        tool_schema_tokens: int = 0,
        overhead_tokens: int = 0,
        provider_usage: Optional[Dict[str, int]] = None,
        prefer_provider: bool = True,
    ) -> ContextMeasurement:
        """Measure what occupies the context window.

        ``messages`` is everything that will be sent; ``user_facing_messages``
        is the subset a person sees (tool traffic removed). ``extra_messages``
        covers content not yet in the message list, such as the pending user
        input. ``system_tokens`` / ``tool_schema_tokens`` / ``overhead_tokens``
        account for the prompt prefix the middleware cannot see on its own and
        default to zero.

        With ``prefer_provider`` the total is taken from the provider's own
        ``usage_metadata`` when present, which is what the usage analytics
        reports. Pass ``prefer_provider=False`` when measuring a request that
        has not been sent yet - the newest usage then belongs to a previous call.
        """
        messages = list(messages or [])
        extra = list(extra_messages or [])

        history_tokens, source = self._count_with_source(messages + extra)

        if user_facing_messages is None:
            user_facing_tokens = history_tokens
        else:
            user_facing_tokens = self._count_with_source(list(user_facing_messages) + extra)[0]

        prompt_tokens = history_tokens + system_tokens + tool_schema_tokens + overhead_tokens
        measurement = ContextMeasurement(
            total=prompt_tokens,
            prompt_tokens=prompt_tokens,
            history_tokens=history_tokens,
            system_tokens=system_tokens,
            tool_schema_tokens=tool_schema_tokens,
            tool_message_tokens=max(0, history_tokens - user_facing_tokens),
            user_facing_tokens=user_facing_tokens,
            overhead_tokens=overhead_tokens,
            token_source=source,
        )

        usage = provider_usage if provider_usage is not None else read_provider_usage(messages)
        if usage:
            measurement.provider_input_tokens = usage.get('input_tokens')
            measurement.provider_output_tokens = usage.get('output_tokens')
            measurement.cache_read_tokens = usage.get('cache_read_tokens')
            measurement.cache_creation_tokens = usage.get('cache_creation_tokens')

            if prefer_provider:
                self._apply_provider_total(measurement, usage)

        return measurement

    @staticmethod
    def _apply_provider_total(measurement: ContextMeasurement, usage: Dict[str, int]) -> None:
        """Replace the estimated total with what the provider actually billed.

        The normalised prompt total covers the whole request - system content,
        tool schemas, history and cached blocks alike - so together with
        ``output_tokens`` it is the same figure the usage analytics reports.
        The estimated components stay as they are and whatever the prompt total
        has beyond them lands in ``overhead_tokens``, so the components still
        add up to ``prompt_tokens`` and the answer on top of that to ``total``.
        """
        prompt_tokens = provider_prompt_tokens(usage)
        if prompt_tokens <= 0:
            return

        measurement.prompt_tokens = prompt_tokens
        measurement.total = prompt_tokens + (measurement.provider_output_tokens or 0)
        measurement.token_source = TOKEN_SOURCE_PROVIDER
        measurement.overhead_tokens = max(
            0,
            prompt_tokens
            - measurement.history_tokens
            - measurement.system_tokens
            - measurement.tool_schema_tokens,
        )
