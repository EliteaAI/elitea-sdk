import httpx

from elitea_sdk.runtime.utils import mcp_adapter, mcp_transport_negotiation


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler):
        self._handler = handler
        self.requests = []
        self.size_trips = []
        self.ssl_verify_flags = []

    @property
    def calls(self):
        return [(request.method, str(request.url)) for request in self.requests]

    async def handle_async_request(self, request):
        self.requests.append(request)
        return await self._handler(request)


def patch_probe_transport(monkeypatch, handler):
    transport = RecordingTransport(handler)

    def build_factory(ssl_verify, limit, trip):
        transport.size_trips.append(trip)
        transport.ssl_verify_flags.append(ssl_verify)

        def factory(headers=None, timeout=None, auth=None):
            return httpx.AsyncClient(transport=transport, headers=headers, timeout=timeout, follow_redirects=True)
        return factory

    monkeypatch.setattr(mcp_transport_negotiation, "build_httpx_client_factory", build_factory)
    monkeypatch.setattr(mcp_adapter, "build_httpx_client_factory", build_factory)
    return transport


def respond(status, headers=None, body=b""):
    async def handler(request):
        return httpx.Response(status, headers=headers or {}, content=body)
    return handler
