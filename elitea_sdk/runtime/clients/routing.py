"""Deferred Auto binding. Native clients retain message and streaming ownership.

No mutable conversation pin lives on this object: pins travel with assistant
messages in the existing graph checkpoint, isolating parallel invocations.
"""
import asyncio
import copy
import hashlib
import json
import time
from typing import Any
from pydantic import ConfigDict, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda, RunnableBinding
from langchain_core.utils.function_calling import convert_to_openai_tool

PIN = 'elitea_routing'


def projection(messages):
    result = []
    for message in messages:
        role = 'user' if isinstance(message, HumanMessage) else 'tool' if isinstance(message, ToolMessage) else 'system' if isinstance(message, SystemMessage) else 'assistant'
        content = (message.additional_kwargs.get("elitea_routing_content", message.content)
                   if isinstance(message, (HumanMessage, SystemMessage)) else message.content)
        if isinstance(message, SystemMessage) and 'elitea_routing_content' in message.additional_kwargs and not content:
            continue  # Constructor-owned scaffolding; native generation retains it.
        if isinstance(content, list):
            text = []
            for block in content:
                if isinstance(block, str):
                    text.append(block)
                elif isinstance(block, dict) and block.get('type') in {'text', 'output_text'}:
                    text.append(block.get('text', ''))
                elif isinstance(block, dict) and block.get('type') in {'thinking', 'redacted_thinking', 'tool_use', 'reasoning', 'reasoning_content'}:
                    continue  # Opaque provider content remains on the generation path.
                else:
                    raise ValueError('Auto does not yet qualify this input modality')
            content = '\n'.join(text)
        row = {'role': role, 'content': content or ''}
        if isinstance(message, ToolMessage):
            row['tool_call_id'] = message.tool_call_id
        calls = getattr(message, 'tool_calls', None)
        if calls:
            row['tool_calls'] = [{'id': c['id'], 'type': 'function', 'function': {'name': c['name'], 'arguments': c['args']}} for c in calls]
        result.append(row)
    return result


def capture_routing_task(message, config):
    """Keep only this graph invocation's typed task across input flattening."""
    values = config.setdefault('configurable', {})
    values.pop('elitea_routing_task_projection', None)
    if isinstance(message, HumanMessage) and 'elitea_routing_content' in message.additional_kwargs:
        values['elitea_routing_task_projection'] = {
            'thread_id': values.get('thread_id'), 'run_id': values.get('elitea_routing_run_id'),
            'content': copy.deepcopy(message.additional_kwargs['elitea_routing_content'])}


def routing_task_kwargs(config):
    values = (config or {}).get('configurable') or {}
    projection = values.get('elitea_routing_task_projection') or {}
    if (projection and projection.get('thread_id') == values.get('thread_id')
            and projection.get('run_id') == values.get('elitea_routing_run_id')):
        return {'elitea_routing_content': copy.deepcopy(projection['content'])}
    return {}


def routing_scope_id(config):
    values = (config or {}).get('configurable') or {}
    if not values.get('thread_id'):
        raise ValueError('Auto requires the execution thread identity')
    owner = values.get('elitea_routing_graph_owner') or {}
    # LangGraph adds a per-task namespace inside nodes. The graph entry owns
    # the durable scope; a delegated graph replaces it with its own identity.
    namespace = (owner.get('checkpoint_ns', '') if owner.get('thread_id') == values['thread_id']
                 else values.get('checkpoint_ns', ''))
    return hashlib.sha256(json.dumps({'thread_id': values['thread_id'],
        'checkpoint_ns': namespace}, sort_keys=True).encode()).hexdigest()


def observation(message):
    row = projection([message])[0]
    usage = message.usage_metadata or {}
    details = usage.get('input_token_details') or {}
    finish = message.response_metadata.get('finish_reason') or message.response_metadata.get('stop_reason')
    finish = {'end_turn': 'stop', 'tool_use': 'tool_calls', 'max_tokens': 'length'}.get(finish, finish)
    finish = finish if finish in {'stop', 'tool_calls', 'length', 'content_filter', 'refusal', 'pause_turn'} else None
    def counter(value):
        return value if type(value) is int and 0 <= value <= 1_000_000_000 else 0
    model = message.response_metadata.get('model_name') or message.response_metadata.get('model')
    return {'message_digest': hashlib.sha256(json.dumps({k: row.get(k) for k in
        ('role', 'content', 'tool_calls', 'tool_call_id')}, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest(), 'finish_reason': finish,
        'returned_model': model[:256] if isinstance(model, str) else None,
        'usage': {'prompt_tokens': counter(usage.get('input_tokens')), 'completion_tokens': counter(usage.get('output_tokens')),
                  'prompt_tokens_details': {'cached_tokens': counter(details.get('cache_read')),
                                            'cache_creation_tokens': counter(details.get('cache_creation'))}}}


class AutoChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    owner: Any = Field(exclude=True)
    settings: dict = Field(exclude=True)
    routing_tools: list = Field(default_factory=list, exclude=True)
    tool_kwargs: dict = Field(default_factory=dict, exclude=True)
    active_instructions: str = Field(default="", exclude=True)

    @property
    def _llm_type(self):
        return 'elitea-auto'

    def with_active_instructions(self, text):
        return self.model_copy(update={"active_instructions": self.settings.get("routing_instructions", text or "")})

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={'routing_tools': list(tools), 'tool_kwargs': dict(kwargs)})

    def _materialize(self, messages, run_config=None, output_schema=None):
        configurable = (run_config or {}).get('configurable') or {}
        thread_id = configurable.get('thread_id')
        if not thread_id:
            raise ValueError('Auto requires the execution thread identity')
        latest_index = next((i for i in range(len(messages)-1, -1, -1) if isinstance(messages[i], HumanMessage)), None)
        if latest_index is None:
            raise ValueError('Auto requires a task source')
        latest = messages[latest_index]
        scope_id = routing_scope_id(run_config)
        run_id = configurable.get('elitea_routing_run_id')
        if not run_id:
            raise ValueError('Auto requires the platform run identity')
        invocation_id = hashlib.sha256(json.dumps({'scope_id': scope_id, 'run_id': run_id}, sort_keys=True).encode()).hexdigest()
        previous_index = next((i for i in range(len(messages)-1, -1, -1)
                               if isinstance(messages[i], AIMessage) and messages[i].response_metadata.get(PIN)), None)
        previous = messages[previous_index] if previous_index is not None else None
        previous_binding = previous.response_metadata[PIN] if previous else None
        if previous_binding and previous_binding.get('scope_id') != scope_id:
            raise ValueError('Routing state belongs to another execution scope')
        sink = configurable.get('elitea_routing_sink')
        checkpoint = (sink or {}).get('binding') or configurable.get('elitea_routing_checkpoint')
        if checkpoint and checkpoint.get('scope_id') == scope_id:
            previous_binding = checkpoint
        binding = copy.deepcopy(previous_binding) if previous_binding and previous_binding.get('invocation_id') == invocation_id else None
        from elitea_sdk.runtime.clients.routing_history import completed_native_keys
        completed = binding.get('completed_native_keys', []) if binding else completed_native_keys(messages)
        after_user = messages[latest_index+1:]
        if binding is None and any(isinstance(message, (AIMessage, ToolMessage)) for message in after_user):
            raise ValueError('Auto must begin at a user task boundary, not inside an unbound tool cycle')
        if binding is None or binding.get('invocation_id') != invocation_id or binding.get('expires_at', 0) <= time.time()+60:
            prior_observation = None
            receipt = (previous_binding or {}).get('last_response') or {}
            if (receipt.get('scope_id') == scope_id
                    and receipt.get('invocation_id') == (previous_binding or {}).get('invocation_id')):
                # Core rebuilds completed visible history without provider
                # metadata. Match the actual prior response digest; the Gateway
                # additionally checks its signed input prefix and message order.
                for index in range(len(messages)-1, -1, -1):
                    if isinstance(messages[index], AIMessage) and observation(messages[index])['message_digest'] == receipt.get('message_digest'):
                        prior_observation = {**receipt, 'message_index': len(projection(messages[:index]))}
                        break
            if prior_observation is None and previous is not None:
                prior_observation = {**observation(previous), 'message_index': len(projection(messages[:previous_index]))}
            tools = [convert_to_openai_tool(tool) for tool in self.routing_tools]
            cap = self.settings.get('max_tokens')
            # Unspecified output belongs to the selected measured contract.
            # Sending a synthetic 8k limit would exclude 32k-only presets.
            output_limit = {} if cap in (None, -1) else {'output_cap': cap}
            from elitea_sdk.runtime.clients.routing_context import instruction_view, retrieval_sources
            context = {'active_instructions': dict(instruction_view(self.active_instructions)),
                       'retrieval_options': retrieval_sources(self.routing_tools, str(latest.content))}
            issuer = getattr(self.owner, '_routing_context_signer', None)
            if not issuer:
                context['retrieval_options'] = []
            context_token = issuer(context=context, tools=tools, scope_id=scope_id, invocation_id=invocation_id) if issuer and binding is None and context['retrieval_options'] else None
            response = self.owner._request('post', f'{self.owner.base_url}/llm/v1/auto-routing/resolve',
                headers={**self.owner.headers, 'X-Project-Id': str(self.owner.project_id)},
                json={'selection': self.settings['selection'], 'surface': self.settings.get('routing_surface', 'agent'),
                      'messages': projection(messages), 'tools': tools, **output_limit,
                      'generation_input_bytes': len(json.dumps(
                          {'messages': [message.model_dump(mode='json') for message in messages],
                           'tools': tools, 'output_schema': output_schema}, ensure_ascii=False).encode()) + 64*len(messages),
                      'runtime_context': context, 'runtime_context_token': context_token,
                      'output_schema': output_schema or configurable.get('elitea_routing_output_schema'),
                      'invocation_id': invocation_id, 'scope_id': scope_id,
                      'state_token': previous_binding.get('state_token') if previous_binding else None,
                      'observation': prior_observation,
                      'prior_pin': binding.get('pin') if binding else None}, timeout=(5, 90))
            response.raise_for_status()
            binding = response.json()
            binding['completed_native_keys'] = completed
        if binding.get('action') != 'clarify' and sink is not None:
            sink['binding'] = copy.deepcopy(binding)
        if binding.get('action') == 'clarify':
            return None, binding
        config = {**binding['config'], 'streaming': self.settings.get('streaming', True),
                  'routing_pin': binding['pin'], 'routing_invocation_id': binding['invocation_id']}
        native = self.owner.get_llm(config['model_name'], config)
        if self.routing_tools:
            native = native.bind_tools(self.routing_tools, **self.tool_kwargs)
        return native, binding

    @staticmethod
    def _native_messages(native, messages, binding):
        from elitea_sdk.runtime.clients.routing_history import adapt_completed_history
        messages = adapt_completed_history(native, messages, binding)
        target = native.bound if isinstance(native, RunnableBinding) else native
        if 'langchain_anthropic' not in type(target).__module__:
            return messages
        from ..langchain.assistant import _make_anthropic_system_content
        return [message.model_copy(update={'content': _make_anthropic_system_content(message.content, target)})
                if isinstance(message, SystemMessage) and isinstance(message.content, str) else message
                for message in messages]

    @staticmethod
    def _admit_native(native, messages, binding, stop=None, **kwargs):
        # A same-run steer cannot expand the frozen text-only qualification.
        # Validate content without classifying again or altering the pin.
        for message in messages:
            if isinstance(message, HumanMessage) and isinstance(message.content, list):
                for block in message.content:
                    if not isinstance(block, str) and not (isinstance(block, dict) and block.get('type') in {'text', 'output_text'}):
                        raise ValueError('Auto does not yet qualify this input modality')
        # A conservative screening bound over the actual LangChain provider
        # payload, including native blocks and bound tools. This is not exact
        # tokenization. The provider remains the final context-limit authority.
        target = native.bound if isinstance(native, RunnableBinding) else native
        bound_kwargs = native.kwargs if isinstance(native, RunnableBinding) else {}
        wire_kwargs = {**bound_kwargs, **kwargs}
        wire_kwargs.pop('ls_structured_output_format', None)  # LangChain tracing-only field.
        payload = target._get_request_payload(messages, stop=stop, **wire_kwargs)
        if isinstance(payload.get('response_format'), type):
            # Same expansion used by the installed OpenAI parse transport.
            from openai.lib._parsing import type_to_response_format_param
            payload['response_format'] = type_to_response_format_param(payload['response_format'])
        size = len(json.dumps(payload, ensure_ascii=False).encode()) + 64*len(messages)
        if size + binding['config']['max_tokens'] > binding['config']['context_window']:
            raise ValueError('Auto native request exceeds the conservative context allowance')

    @staticmethod
    def _binding_metadata(binding):
        return {'elitea_routing_trace': binding.get('trace', {})} if binding.get('action') == 'clarify' else {PIN: copy.deepcopy(binding)}

    @staticmethod
    def _stamp(message, binding):
        metadata = {**message.response_metadata, **AutoChatModel._binding_metadata(binding)}
        return message.model_copy(update={'response_metadata': metadata})

    @staticmethod
    def _remember_response(message, binding, config):
        if message is None or binding.get('action') == 'clarify':
            return
        # Advisory receipt only, never billing or an admission credential.
        binding['last_response'] = {**observation(message), 'scope_id': binding['scope_id'],
                                    'invocation_id': binding['invocation_id'], 'completed_at': time.time()}
        sink = ((config or {}).get('configurable') or {}).get('elitea_routing_sink')
        if sink is not None:
            sink['binding'] = copy.deepcopy(binding)

    def invoke(self, input, config=None, *, stop=None, **kwargs):
        messages = self._convert_input(input).to_messages()
        native, binding = self._materialize(messages, config)
        if native is not None:
            messages = self._native_messages(native, messages, binding)
            self._admit_native(native, messages, binding, stop, **kwargs)
        response = AIMessage(content=binding['text']) if native is None else native.invoke(messages, config, stop=stop, **kwargs)
        self._remember_response(response, binding, config)
        return self._stamp(response, binding)

    async def ainvoke(self, input, config=None, *, stop=None, **kwargs):
        messages = self._convert_input(input).to_messages()
        native, binding = await asyncio.to_thread(self._materialize, messages, config)
        if native is not None:
            messages = self._native_messages(native, messages, binding)
            self._admit_native(native, messages, binding, stop, **kwargs)
        response = AIMessage(content=binding['text']) if native is None else await native.ainvoke(messages, config, stop=stop, **kwargs)
        self._remember_response(response, binding, config)
        return self._stamp(response, binding)

    def stream(self, input, config=None, *, stop=None, **kwargs):
        messages = self._convert_input(input).to_messages()
        native, binding = self._materialize(messages, config)
        if native is None:
            yield AIMessageChunk(content=binding['text'])
        else:
            messages = self._native_messages(native, messages, binding)
            self._admit_native(native, messages, binding, stop, **kwargs)
            complete = None
            for chunk in native.stream(messages, config, stop=stop, **kwargs):
                complete = chunk if complete is None else complete + chunk
                yield chunk
            self._remember_response(complete, binding, config)
        yield AIMessageChunk(content='', response_metadata=self._binding_metadata(binding))

    async def astream(self, input, config=None, *, stop=None, **kwargs):
        messages = self._convert_input(input).to_messages()
        native, binding = await asyncio.to_thread(self._materialize, messages, config)
        if native is None:
            yield AIMessageChunk(content=binding['text'])
        else:
            messages = self._native_messages(native, messages, binding)
            self._admit_native(native, messages, binding, stop, **kwargs)
            complete = None
            async for chunk in native.astream(messages, config, stop=stop, **kwargs):
                complete = chunk if complete is None else complete + chunk
                yield chunk
            self._remember_response(complete, binding, config)
        yield AIMessageChunk(content='', response_metadata=self._binding_metadata(binding))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # BaseChatModel.generate compatibility; ordinary invoke/stream delegate
        # directly to native clients to avoid duplicate provider-usage callbacks.
        return ChatResult(generations=[ChatGeneration(message=self.invoke(messages, stop=stop, **kwargs))])

    @staticmethod
    def _structured_native(native, schema, kwargs):
        from ..tools.llm import LLMNode
        from ..langchain.utils import make_anthropic_compatible_schema
        options = dict(kwargs)
        if LLMNode._is_anthropic_client(native):
            if isinstance(schema, type):
                schema = make_anthropic_compatible_schema(schema)
            if options.get('method', 'function_calling') == 'function_calling' and LLMNode._is_anthropic_thinking_client(native):
                options['method'] = 'json_schema'
        return native.with_structured_output(schema, **options)

    def with_structured_output(self, schema, **kwargs):
        output_schema = schema.model_json_schema() if isinstance(schema, type) else schema
        include_raw = kwargs.get('include_raw', False)
        options = {**kwargs, 'include_raw': True}

        def prepare(value, config):
            messages = self._convert_input(value).to_messages()
            native, binding = self._materialize(messages, config, output_schema=output_schema)
            if native is None:
                raise ValueError(binding['text'])
            messages = self._native_messages(native, messages, binding)
            runnable = self._structured_native(native, schema, options)
            # Current pinned LangChain adapters expose the provider binding as
            # the raw arm, before their parser. Count the transformed schema too.
            provider = runnable.steps[0].steps__['raw']
            self._admit_native(provider, messages, binding)
            return runnable, messages, binding

        def finish(result, binding, config):
            if result.get('parsing_error') is not None and not include_raw:
                raise result['parsing_error']
            if result.get('raw') is not None:
                if result.get('parsing_error') is None:
                    self._remember_response(result['raw'], binding, config)
                result = {**result, 'raw': self._stamp(result['raw'], binding)}
            if include_raw:
                return result
            return result['parsed']

        def invoke(value, config=None):
            runnable, messages, binding = prepare(value, config)
            return finish(runnable.invoke(messages, config), binding, config)

        async def ainvoke(value, config=None):
            runnable, messages, binding = await asyncio.to_thread(prepare, value, config)
            return finish(await runnable.ainvoke(messages, config), binding, config)

        return RunnableLambda(invoke, ainvoke)
