"""Target-aware adaptation of completed history; active native turns stay intact."""
import copy
import hashlib
import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableBinding

NATIVE_BLOCKS = {'thinking', 'redacted_thinking', 'reasoning', 'reasoning_content'}


def message_key(message):
    return hashlib.sha256(json.dumps({'content': message.content,
        'tool_calls': getattr(message, 'tool_calls', []),
        'additional_kwargs': message.additional_kwargs}, sort_keys=True,
        ensure_ascii=False, default=str).encode()).hexdigest()


def completed_native_keys(messages):
    """Authorize adaptation only before a new task with complete tool pairs.

    These content keys are SDK checkpoint metadata, not a provider/request
    authority. Keeping keys rather than indexes survives history compaction.
    """
    boundary = max(i for i, message in enumerate(messages) if isinstance(message, HumanMessage))
    pending = set()
    keys = []
    for message in messages[:boundary]:
        if isinstance(message, HumanMessage) and pending:
            raise ValueError('Auto cannot switch across an unfinished tool batch; resume its original run')
        if isinstance(message, AIMessage):
            calls = {call['id'] for call in message.tool_calls}
            if isinstance(message.content, list):
                calls.update(block['id'] for block in message.content if isinstance(block, dict)
                             and block.get('type') == 'tool_use' and block.get('id'))
                if any(isinstance(block, dict) and block.get('type') in NATIVE_BLOCKS for block in message.content):
                    keys.append(message_key(message))
            if pending.intersection(calls):
                raise ValueError('Duplicate pending tool call in Auto history')
            pending.update(calls)
        elif isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                raise ValueError('Auto history contains an unmatched tool result')
            pending.remove(message.tool_call_id)
    if pending:
        raise ValueError('Auto cannot switch across an unfinished tool batch; resume its original run')
    return keys


def adapt_completed_history(native, messages, binding):
    """Retain Anthropic native blocks on Messages; bridge foreign private blocks.

    The installed ChatOpenAI serializer already drops Anthropic thinking but
    does not drop redacted_thinking. Native Anthropic accepts its own prior
    thinking/redacted blocks across completed turns, but not OpenAI reasoning
    items. Only known completed messages receive this provider-specific copy.
    Original checkpoint objects and every current-run native message survive.
    """
    target = native.bound if isinstance(native, RunnableBinding) else native
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    if isinstance(target, ChatAnthropic):
        unsupported = {'reasoning', 'reasoning_content'}
    elif isinstance(target, ChatOpenAI):
        unsupported = {'thinking', 'redacted_thinking', 'reasoning_content'}
        if not target.use_responses_api:
            unsupported.add('reasoning')
    else:
        return messages  # Fixture/native adapter without these wire contracts.
    completed = set(binding.get('completed_native_keys') or [])
    result = []
    for message in messages:
        own = message.response_metadata.get('elitea_routing') if isinstance(message, AIMessage) else None
        current = own and own.get('invocation_id') == binding['invocation_id']
        if (isinstance(message, AIMessage) and not current and isinstance(message.content, list)
                and message_key(message) in completed):
            blocks = [copy.deepcopy(block) for block in message.content
                      if not (isinstance(block, dict) and block.get('type') in unsupported)]
            if isinstance(target, ChatAnthropic):
                blocks = [({'type': 'text', 'text': block.get('text', '')}
                           if isinstance(block, dict) and block.get('type') == 'output_text' else block)
                          for block in blocks]
            if blocks != message.content:
                message = message.model_copy(update={'content': blocks or ''})
        result.append(message)
    return result
