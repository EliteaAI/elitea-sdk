"""A run that finishes without an answer returns an empty output, not a sentence.

The runner used to return "Assistant run has been completed, but output is None.
Adding last message if any: ..." here. Chat showed it as the assistant's reply and
agent evaluation scored it as a real answer. Sonnet's ``AIMessage(content=[])``
reply with no tool calls is the common trigger.
"""
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import Assistant


class _DummyRuntime:
    def get_mcp_toolkits(self):
        return []


class _FakeChat(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _invoke(reply: AIMessage) -> dict:
    assistant = Assistant(
        elitea=_DummyRuntime(),
        data={"instructions": "Answer the question.", "tools": [], "meta": {}},
        client=_FakeChat(messages=cycle([reply])),
        tools=[],
        memory=MemorySaver(),
        app_type="predict",
    )
    return assistant.runnable().invoke(
        {"messages": [HumanMessage(content="list the files")]},
        config={"configurable": {"thread_id": "finished-without-answer"}},
    )


def test_empty_reply_gives_empty_output():
    result = _invoke(AIMessage(content=[]))
    assert result["execution_finished"] is True
    assert result["output"] == ''


def test_text_reply_is_the_output():
    result = _invoke(AIMessage(content="file_a.txt"))
    assert result["output"] == "file_a.txt"
