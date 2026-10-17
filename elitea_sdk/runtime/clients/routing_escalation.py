"""Escalation signals the SDK forwards to the Auto resolve endpoint.

The SDK only reports what it observed. It never picks, ranks or names a model:
the Gateway decides whether and where to escalate. Both fields are optional and
carry flags and codes only, never request or response text.

Platform contract (``run_config['configurable']``):

    elitea_routing_escalation = {"event": "regenerate" | "upgrade_control",
                                 "from_message_id": "<id of the Auto answer>" | None}

The platform sets it for a run that regenerates an Auto answer or that was
started by the "upgrade this answer" control. Any other shape is ignored.
"""
import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

ESCALATION_KEY = 'elitea_routing_escalation'
ESCALATION_EVENTS = frozenset({'regenerate', 'upgrade_control'})
MAX_MESSAGE_ID_LENGTH = 256

# Frozen rules (changing them changes the escalation label source; update tests).
RULE_WINDOW = 300  # characters of the answer inspected
INABILITY_PHRASES = ("i cannot", "i can't", "i don't have access", "unable to")
REFUSAL_PHRASES = ("i can't help with", "i cannot help with", "i can't assist with", "i cannot assist with",
                   "i can't comply", "i cannot comply", "i must decline", "i won't be able to help")
FINISH_REASONS = frozenset({'stop', 'tool_calls', 'length', 'content_filter', 'refusal', 'pause_turn'})
FINISH_ALIASES = {'end_turn': 'stop', 'tool_use': 'tool_calls', 'max_tokens': 'length'}


def finish_reason_of(message):
    """Provider finish reason normalised to the Gateway vocabulary, or None."""
    metadata = message.response_metadata
    finish = metadata.get('finish_reason') or metadata.get('stop_reason')
    finish = FINISH_ALIASES.get(finish, finish)
    return finish if finish in FINISH_REASONS else None


def answer_text(message):
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(block if isinstance(block, str) else block.get('text', '') or ''
                         for block in content
                         if isinstance(block, str) or (isinstance(block, dict) and block.get('type') in {'text', 'output_text'}))
    return ''


def _window(message):
    return answer_text(message)[:RULE_WINDOW].lower().replace('’', "'")


def is_inability(message):
    text = _window(message)
    return any(phrase in text for phrase in INABILITY_PHRASES)


def is_refusal(message, finish_reason):
    if finish_reason == 'refusal' or message.additional_kwargs.get('refusal'):
        return True
    text = _window(message)
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def escalation_signal(configurable):
    """``{"event", "from_message_id"}`` from the run config, or None when absent or malformed."""
    value = (configurable or {}).get(ESCALATION_KEY)
    if not isinstance(value, dict) or value.get('event') not in ESCALATION_EVENTS:
        return None
    from_id = value.get('from_message_id')
    from_id = from_id if isinstance(from_id, str) and 0 < len(from_id) <= MAX_MESSAGE_ID_LENGTH else None
    return {'event': value['event'], 'from_message_id': from_id}


def response_signal(message, tool_messages=(), prior_observation=None):
    """Flags for one answer. ``tool_messages`` is that turn's tool cycle, oldest first."""
    finish = finish_reason_of(message) or (prior_observation or {}).get('finish_reason')
    tools = list(tool_messages)
    return {'finish_reason': finish, 'refusal': bool(is_refusal(message, finish)),
            'tool_failure': bool(tools and tools[-1].status == 'error'),
            'inability': is_inability(message)}


def last_response_signal(messages, latest_index, prior_observation=None):
    """State of the last assistant answer before the current user task, or None.

    ``prior_observation`` supplies the finish reason when history was rebuilt
    without provider metadata (it is the receipt the SDK already echoes).
    """
    index = next((i for i in range(latest_index - 1, -1, -1) if isinstance(messages[i], AIMessage)), None)
    if index is None:
        return None
    turn_start = next((i for i in range(index - 1, -1, -1) if isinstance(messages[i], HumanMessage)), -1)
    tools = [m for m in messages[turn_start + 1:latest_index] if isinstance(m, ToolMessage)]
    return response_signal(messages[index], tools, prior_observation)


def is_final_answer(message):
    """A guardable answer: no tool call requested (tool-cycle turns are never guarded)."""
    if getattr(message, 'tool_calls', None) or finish_reason_of(message) == 'tool_calls':
        return False
    content = message.content
    return not (isinstance(content, list) and any(isinstance(b, dict) and b.get('type') == 'tool_use' for b in content))


# The Gateway accepts the guard's request context as a dict or string of at most 64,000 bytes.
GUARD_LATEST_CHARS = 5_000
GUARD_RECENT_TURNS = 4
GUARD_RECENT_CHARS = 900
GUARD_CONTEXT_BYTES = 60_000


def guard_request_context(rows):
    """Bounded context for the guard from resolve-style projection rows (role/content dicts).

    ``latest`` is the newest user text, ``recent`` the last user/assistant turns
    before it, ``turns_omitted`` the older user/assistant turns left out. Size is
    measured on the ASCII-escaped JSON (the larger encoding) and kept under
    GUARD_CONTEXT_BYTES: ``recent`` is dropped oldest-first, then ``latest`` shrinks.
    """
    latest_index = next((i for i in range(len(rows) - 1, -1, -1) if rows[i]['role'] == 'user'), None)
    if latest_index is None:
        return {'latest': '', 'recent': [], 'turns_omitted': 0}
    text = lambda row: row['content'] if isinstance(row.get('content'), str) else ''
    earlier = [row for row in rows[:latest_index] if row['role'] in {'user', 'assistant'} and text(row)]
    recent = [{'role': row['role'], 'text': text(row)[:GUARD_RECENT_CHARS]} for row in earlier[-GUARD_RECENT_TURNS:]]
    context = {'latest': text(rows[latest_index])[:GUARD_LATEST_CHARS], 'recent': recent,
               'turns_omitted': len(earlier) - len(recent)}
    while len(json.dumps(context).encode()) >= GUARD_CONTEXT_BYTES:
        if context['recent']:
            context['recent'] = context['recent'][1:]
            context['turns_omitted'] += 1
        else:
            context['latest'] = context['latest'][:len(context['latest']) // 2]
    return context
