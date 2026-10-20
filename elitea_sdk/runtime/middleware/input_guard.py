"""Input guardrail middleware (#6886).

Checkpoint C of the input guardrail: inspects what is about to reach the model -
the current turn (or a sub-agent's task) and the user turns in history - and
records, warns, redacts or blocks per the platform policy.

Checkpoints A and B run on the platform before the SDK is reached; their verdict
arrives as segment hashes in ``configurable['input_guardrail_inspected']`` so a
value the platform already handled is not reported twice.

Redaction is model-view only: the current turn is replaced through the 'input'
update and history through 'transient_messages', neither of which is written to the
checkpoint. Stored copies are the platform's job (D5).
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from langchain_core.messages import BaseMessage, HumanMessage

from ..exceptions import InputGuardrailBlocked
from ..guardrails.input_detector import (
    Policy,
    Segment,
    effective_action,
    inspect,
    policy_from_dict,
    segment_hash,
    summary,
)
from .base import Middleware

logger = logging.getLogger(__name__)

EVENT_NAME = 'input_guardrail'
INSPECTED_CONFIG_KEY = 'input_guardrail_inspected'
# Platform-generated context blocks are not user input
_PLATFORM_BLOCK_PREFIX = '<runtime_context'


class _Location:
    """Where a segment's text came from, so a redaction can be written back."""

    __slots__ = ('target', 'message_index', 'block_index')

    def __init__(self, target: str, message_index: Optional[int], block_index: Optional[int]):
        self.target = target
        self.message_index = message_index
        self.block_index = block_index


class InputGuardMiddleware(Middleware):
    """Inspect model input against the input guardrail policy before each model call."""

    def __init__(
        self,
        policy: Optional[Dict[str, Any]] = None,
        project_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        callbacks: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ):
        super().__init__(conversation_id, callbacks, **kwargs)
        self.policy: Policy = policy_from_dict(policy)
        self.project_id = project_id
        # Hashes already reported by this instance. History is re-sent on every
        # model call of a tool loop; without this a record/warn hit repeats per call.
        self._reported: Set[str] = set()

    def get_tools(self):
        return []

    def get_system_prompt(self) -> str:
        return ''

    @property
    def active(self) -> bool:
        if not self.policy.active:
            return False
        try:
            return int(self.project_id) not in self.policy.exempt_project_ids
        except (TypeError, ValueError):
            return True

    def before_model(self, state: dict, config: dict) -> Optional[dict]:
        if not self.active or not isinstance(state, dict):
            return None

        configurable = (config or {}).get('configurable') or {}
        skip = set(configurable.get(INSPECTED_CONFIG_KEY) or [])
        is_subagent = bool(((config or {}).get('metadata') or {}).get('parent_agent_path'))

        segments, locations, not_inspected = self._collect(state, skip, is_subagent)
        if not segments:
            return None

        result = inspect(segments, self.policy, not_inspected)
        action = effective_action(result, self.policy)
        if result.detections or result.error:
            hit = {(d.segment_kind, d.segment_index) for d in result.detections}
            self._emit(
                summary(result, self.policy),
                [segment_hash(s.text) for s in segments if (s.kind, s.index) in hit],
                config,
            )

        if action == 'block':
            raise InputGuardrailBlocked(
                categories=result.categories or ['error'],
                rules=result.rules,
                segments=sorted({d.segment_kind for d in result.detections}),
                summary=summary(result, self.policy),
            )
        if not result.redacted:
            return None
        return self._apply_redactions(state, segments, locations, result.redacted)

    # --- collection -------------------------------------------------------------

    def _collect(
        self, state: dict, skip: Set[str], is_subagent: bool,
    ) -> Tuple[List[Segment], List[_Location], List[str]]:
        segments: List[Segment] = []
        locations: List[_Location] = []
        not_inspected: List[str] = []
        counters: Dict[str, int] = {}

        def add(kind: str, text: str, location: _Location) -> None:
            if not text or text.lstrip().startswith(_PLATFORM_BLOCK_PREFIX) or segment_hash(text) in skip:
                return
            index = counters.get(kind, 0)
            counters[kind] = index + 1
            segments.append(Segment(kind, index, text))
            locations.append(location)

        turn_kind = 'subagent_task' if is_subagent else 'current_turn'
        current = state.get('input')
        if isinstance(current, str):
            add(turn_kind, current, _Location('input', None, None))
        elif isinstance(current, list):
            for i, item in enumerate(current):
                if isinstance(item, BaseMessage):
                    if isinstance(item, HumanMessage):
                        self._collect_content(item.content, turn_kind, add, not_inspected, 'input', i)
                else:
                    self._collect_block(item, turn_kind, add, not_inspected, _Location('input', i, None))

        for i, message in enumerate(state.get('messages') or []):
            # User turns only: tool results are outside input-guardrail scope
            if isinstance(message, HumanMessage):
                self._collect_content(message.content, 'history', add, not_inspected, 'messages', i)
        return segments, locations, not_inspected

    def _collect_content(self, content, kind, add, not_inspected, target, message_index) -> None:
        if isinstance(content, str):
            add(kind, content, _Location(target, message_index, None))
            return
        for b, block in enumerate(content or []):
            self._collect_block(block, kind, add, not_inspected, _Location(target, message_index, b))

    @staticmethod
    def _collect_block(block, kind, add, not_inspected, location: _Location) -> None:
        if isinstance(block, str):
            add(kind, block, location)
        elif isinstance(block, dict) and block.get('type') == 'text':
            add(kind, block.get('text') or '', location)
        elif isinstance(block, dict):
            not_inspected.append(str(block.get('type') or 'unknown'))

    # --- redaction --------------------------------------------------------------

    def _apply_redactions(self, state, segments, locations, redacted) -> dict:
        updates: Dict[str, Any] = {}
        current = state.get('input')
        messages = list(state.get('messages') or [])
        input_changed = messages_changed = False
        for segment, location in zip(segments, locations):
            new_text = redacted.get((segment.kind, segment.index))
            if new_text is None:
                continue
            if location.target == 'input':
                current = self._replace_in_input(current, location, new_text)
                input_changed = True
            else:
                message = messages[location.message_index]
                messages[location.message_index] = message.model_copy(
                    update={'content': _replace_content(message.content, location.block_index, new_text)}
                )
                messages_changed = True
        if input_changed:
            updates['input'] = current
        if messages_changed:
            updates['transient_messages'] = messages
        return updates

    @staticmethod
    def _replace_in_input(current, location: _Location, new_text: str):
        # message_index is the position in an 'input' list; None means 'input' is a str
        if location.message_index is None:
            return new_text
        current = list(current)
        item = current[location.message_index]
        if isinstance(item, BaseMessage):
            current[location.message_index] = item.model_copy(
                update={'content': _replace_content(item.content, location.block_index, new_text)}
            )
        else:
            current[location.message_index] = _replace_block(item, new_text)
        return current

    # --- reporting --------------------------------------------------------------

    def _emit(self, payload: Dict[str, Any], hashes: List[str], config: dict) -> None:
        if payload['action'] != 'block' and not payload['error'] and all(h in self._reported for h in hashes):
            return
        self._reported.update(hashes)
        logger.info(
            "[INPUT_GUARDRAIL] action=%s rules=%s segments=%s error=%s elapsed_ms=%s",
            payload['action'], payload['rules'], payload['segments'], payload['error'], payload['elapsed_ms'],
        )
        self._fire_callback(EVENT_NAME, payload)
        if not config:
            return
        try:
            from langchain_core.callbacks import dispatch_custom_event
            dispatch_custom_event(name=EVENT_NAME, data=payload, config=config)
        except Exception as e:
            # Outside a runnable context there is no parent run to attach to
            logger.debug(f"[INPUT_GUARDRAIL] Failed to dispatch event: {e}")


def _replace_block(block, new_text: str):
    if isinstance(block, dict):
        return {**block, 'text': new_text}
    return new_text


def _replace_content(content, block_index: Optional[int], new_text: str):
    if block_index is None or isinstance(content, str):
        return new_text
    content = list(content)
    content[block_index] = _replace_block(content[block_index], new_text)
    return content
