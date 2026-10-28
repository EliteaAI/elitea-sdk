"""Chat clients must fail on a stalled stream instead of hanging forever.

A fake upstream sends SSE headers and one chunk, then keeps the socket open
silent for STALL seconds. With an idle (per-read) timeout shorter than STALL
the client must raise; a stream with steady chunks must not be cut off.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from elitea_sdk.runtime.clients.client import LLM_STREAM_IDLE_TIMEOUT

STALL = 6
IDLE = 2


def _sse(h, event, data):
    h.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
    h.wfile.flush()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    gap = STALL
    chunks = 1

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        anthropic = self.path.endswith("/messages")
        try:
            if anthropic:
                _sse(self, "message_start", {"type": "message_start", "message": {
                    "id": "m", "type": "message", "role": "assistant", "model": "x",
                    "content": [], "usage": {"input_tokens": 1, "output_tokens": 1}}})
                _sse(self, "content_block_start", {"type": "content_block_start", "index": 0,
                                                   "content_block": {"type": "text", "text": ""}})
            for i in range(self.chunks):
                if anthropic:
                    _sse(self, "content_block_delta", {"type": "content_block_delta", "index": 0,
                                                       "delta": {"type": "text_delta", "text": "a"}})
                else:
                    self.wfile.write(("data: " + json.dumps({"id": "c", "object": "chat.completion.chunk",
                        "created": 1, "model": "x", "choices": [{"index": 0, "delta": {"content": "a"},
                        "finish_reason": None}]}) + "\n\n").encode())
                    self.wfile.flush()
                time.sleep(self.gap)
            if anthropic:
                _sse(self, "message_stop", {"type": "message_stop"})
            else:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _clients(base, timeout):
    return [
        ChatAnthropic(base_url=base, model="x", api_key="k", streaming=True, max_tokens=50,
                      max_retries=3, timeout=timeout),
        ChatOpenAI(base_url=f"{base}/v1", model="x", api_key="k", streaming=True, max_retries=3,
                   timeout=timeout),
    ]


def test_default_idle_timeout_is_300():
    assert LLM_STREAM_IDLE_TIMEOUT == 300


@pytest.mark.parametrize("idx", [0, 1], ids=["anthropic", "openai"])
def test_stalled_stream_fails_at_idle_timeout(server, idx):
    srv, base = server
    _Handler.gap, _Handler.chunks = STALL, 1
    llm = _clients(base, IDLE)[idx]
    start = time.monotonic()
    with pytest.raises(Exception) as exc:
        list(llm.stream("hi"))
    elapsed = time.monotonic() - start
    assert elapsed < STALL, f"waited the full stall ({elapsed:.1f}s)"
    assert "timeout" in type(exc.value).__name__.lower() or "timeout" in str(exc.value).lower() \
        or "timeout" in repr(exc.value.__cause__).lower()


@pytest.mark.parametrize("idx", [0, 1], ids=["anthropic", "openai"])
def test_long_active_stream_not_cut(server, idx):
    srv, base = server
    _Handler.gap, _Handler.chunks = 1, 5  # 5s total > IDLE, gaps < IDLE
    llm = _clients(base, IDLE)[idx]
    out = "".join(c.content if isinstance(c.content, str) else "" for c in llm.stream("hi"))
    assert out.startswith("a")


# --- readable error text ---------------------------------------------------
from elitea_sdk.runtime.tools.llm import _llm_idle_timeout
from elitea_sdk.runtime.utils.failure_signals import describe_llm_error


@pytest.mark.parametrize("idx", [0, 1], ids=["anthropic", "openai"])
def test_stall_error_is_readable_and_names_the_limit(server, idx):
    # The raw exception text is just "timed out"; the user must see what stalled and for how long.
    _Handler.gap, _Handler.chunks = STALL, 1
    llm = _clients(server[1], IDLE)[idx]
    with pytest.raises(Exception) as exc:
        llm.invoke("hi")
    assert str(exc.value) == "timed out"
    assert _llm_idle_timeout(llm) == IDLE
    msg = describe_llm_error(exc.value, _llm_idle_timeout(llm))
    assert msg.startswith(f"LLM stream stalled: no data received for {IDLE}s")


def test_wrapped_timeout_detected_via_cause():
    import httpx
    try:
        try:
            raise httpx.ReadTimeout("timed out")
        except httpx.ReadTimeout as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert describe_llm_error(outer).startswith("LLM stream stalled: no data received.")


def test_other_errors_pass_through_unchanged():
    assert describe_llm_error(ValueError("bad request"), 300) == "bad request"


def test_proxy_cut_stream_is_readable():
    # Reproduced via /llm/v1: the proxy's idle limit closes the socket and httpx raises this.
    import httpx
    err = httpx.RemoteProtocolError("peer closed connection without sending complete message body")
    assert describe_llm_error(err).startswith("LLM stream interrupted:")
