import base64

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Serialized with langgraph-checkpoint==2.1.2. Keeping this fixture independent
# of the installed serializer verifies that persisted checkpoints remain readable
# after upgrading to langgraph-checkpoint==4.1.1.
_CHECKPOINT_2_1_2_FIXTURE = (
    "gqhtZXNzYWdlc5LH+QWUumxhbmdjaGFpbl9jb3JlLm1lc3NhZ2VzLmFpqUFJTWVzc2FnZYmn"
    "Y29udGVudJGCpHR5cGWkdGV4dKR0ZXh0p3dvcmtpbmexYWRkaXRpb25hbF9rd2FyZ3OAsXJl"
    "c3BvbnNlX21ldGFkYXRhgKR0eXBlomFppG5hbWXAomlkwKp0b29sX2NhbGxzkYSkbmFtZaZz"
    "ZWFyY2ikYXJnc4GhcaZlbGl0ZWGiaWSmY2FsbC0xpHR5cGWpdG9vbF9jYWxssmludmFsaWRf"
    "dG9vbF9jYWxsc5CudXNhZ2VfbWV0YWRhdGHAs21vZGVsX3ZhbGlkYXRlX2pzb27HtAWUvGxh"
    "bmdjaGFpbl9jb3JlLm1lc3NhZ2VzLnRvb2yrVG9vbE1lc3NhZ2WJp2NvbnRlbnSlZm91bmSx"
    "YWRkaXRpb25hbF9rd2FyZ3OAsXJlc3BvbnNlX21ldGFkYXRhgKR0eXBlpHRvb2ykbmFtZcCiaW"
    "TArHRvb2xfY2FsbF9pZKZjYWxsLTGoYXJ0aWZhY3TApnN0YXR1c6dzdWNjZXNzs21vZGVsX3"
    "ZhbGlkYXRlX2pzb26obWV0YWRhdGGBpnNvdXJjZbpsYW5nZ3JhcGgtY2hlY2twb2ludC0yLj"
    "EuMg=="
)


def test_checkpoint_2_1_2_payload_is_readable() -> None:
    serializer = JsonPlusSerializer()

    restored = serializer.loads_typed(
        ("msgpack", base64.b64decode(_CHECKPOINT_2_1_2_FIXTURE))
    )

    ai_message, tool_message = restored["messages"]
    assert isinstance(ai_message, AIMessage)
    assert ai_message.content == [{"type": "text", "text": "working"}]
    assert ai_message.tool_calls == [
        {
            "name": "search",
            "args": {"q": "elitea"},
            "id": "call-1",
            "type": "tool_call",
        }
    ]

    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "found"
    assert tool_message.tool_call_id == "call-1"
    assert tool_message.status == "success"
    assert restored["metadata"] == {"source": "langgraph-checkpoint-2.1.2"}
