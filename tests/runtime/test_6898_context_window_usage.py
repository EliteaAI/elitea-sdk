"""#6898: summarization usage is reported once, and the request's real window reaches the context floor."""

from typing import Any, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.runtime.context_floor import resolve_floor_tokens
from elitea_sdk.runtime.middleware.summarization import SummarizationMiddleware


class _UsageChat(GenericFakeChatModel):
    """Fake chat model whose replies carry provider usage."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = result.generations[0].message
        message.usage_metadata = {'input_tokens': 120, 'output_tokens': 30, 'total_tokens': 150,
                                  'input_token_details': {'cache_read': 20}}
        message.response_metadata = {'model_name': 'low-tier-model'}
        return result


class _Recorder(BaseCallbackHandler):
    def __init__(self):
        self.starts: List[Any] = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.starts.append(kwargs.get('run_id'))


def _middleware():
    model = _UsageChat(messages=iter([AIMessage(content='summary one'), AIMessage(content='summary two')]))
    return SummarizationMiddleware(model=model, trigger=('tokens', 1000), keep=('messages', 2))


def test_summary_call_inside_a_graph_skips_run_callbacks_and_is_reported_once():
    middleware = _middleware()
    recorder = _Recorder()

    class State(TypedDict):
        text: str

    def node(state):
        return {'text': middleware._create_summary([HumanMessage(content='hello'), AIMessage(content='hi')])}

    graph = StateGraph(State)
    graph.add_node('summarize', node)
    graph.add_edge(START, 'summarize')
    graph.add_edge('summarize', END)
    graph.compile().invoke({'text': ''}, config={'callbacks': [recorder]})

    assert recorder.starts == []
    middleware.last_context_info = {'message_count': 2}
    assert middleware.last_context_info['auxiliary_usage'] == [{
        'model_name': 'low-tier-model', 'input_tokens': 120, 'output_tokens': 30,
        'cache_read_tokens': 20, 'cache_creation_tokens': 0,
    }]


def test_no_auxiliary_usage_without_summary_calls():
    middleware = _middleware()
    middleware.last_context_info = {'message_count': 1}
    assert 'auxiliary_usage' not in middleware.last_context_info


def test_a_plain_invoke_inside_a_runnable_still_reaches_run_callbacks():
    recorder = _Recorder()
    model = _UsageChat(messages=iter([AIMessage(content='x')]))
    RunnableLambda(lambda value: model.invoke(value)).invoke('hi', config={'callbacks': [recorder]})
    assert len(recorder.starts) == 1


def test_context_window_is_stamped_on_a_copy():
    llm = _UsageChat(messages=iter([]))
    stamped = EliteAClient._with_context_window(llm, {'effective_window': 272000})
    assert stamped is not llm
    assert resolve_floor_tokens(stamped) == 272000
    assert getattr(llm, '_elitea_context_window', None) is None


def test_bound_llm_is_copied_and_keeps_its_binding():
    bound = _UsageChat(messages=iter([])).bind(stop=['END'])
    stamped = EliteAClient._with_context_window(bound, {'effective_window': 128000})
    assert stamped is not bound
    assert stamped.kwargs == bound.kwargs
    assert resolve_floor_tokens(stamped) == 128000
    assert getattr(bound.bound, '_elitea_context_window', None) is None


def test_without_a_window_the_llm_is_returned_unchanged():
    llm = _UsageChat(messages=iter([]))
    assert EliteAClient._with_context_window(llm, {}) is llm
    assert EliteAClient._with_context_window(llm, None) is llm
