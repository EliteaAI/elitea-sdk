"""Tests for the Microsoft Graph toolkit: scopes, request rules, host guard, requests, batch, artifacts."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.configurations.ms_graph import MsGraphConfiguration, normalize_ms_graph_scopes
from elitea_sdk.tools.ms_graph import MsGraphToolkit
from elitea_sdk.tools.ms_graph.api_wrapper import MsGraphApiWrapper
from elitea_sdk.tools.ms_graph.rules import RequestRules, RuleViolation, normalize_path, split_graph_url

G = "https://graph.microsoft.com/v1.0"
AUTHORITY = "https://login.microsoftonline.com/tenant-1"


def _resp(status=200, body=None, headers=None, content=None, url=None):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    if content is None:
        content = json.dumps(body).encode() if body is not None else b""
    hdrs = {"Content-Type": "application/json"} if body is not None else {}
    hdrs.update(headers or {})
    resp.headers = _Headers(hdrs)
    resp.iter_content.return_value = [content] if content else []
    resp.json.return_value = body
    resp.text = content.decode(errors="replace")
    resp.url = url
    return resp


class _Headers(dict):
    """Case-insensitive lookup like requests' headers."""

    def __init__(self, data):
        super().__init__({k.lower(): v for k, v in data.items()})

    def get(self, key, default=None):
        return super().get(key.lower(), default)

    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def __contains__(self, key):
        return super().__contains__(key.lower())


class FakeGraph:
    """Routes requests.request(method, url, ...) to canned responses and records calls."""

    def __init__(self):
        self.routes = {}
        self.calls = []

    def on(self, method, url, *responses):
        self.routes[(method, url)] = list(responses)
        return self

    def __call__(self, method, url, headers=None, params=None, data=None, **kwargs):
        self.calls.append({"method": method, "url": url, "params": params, "data": data, "headers": headers})
        queue = self.routes.get((method, url))
        if not queue:
            raise AssertionError(f"Unexpected request {method} {url}")
        return queue.pop(0) if len(queue) > 1 else queue[0]


@pytest.fixture
def graph():
    fake = FakeGraph()
    with patch("elitea_sdk.tools.ms_graph.graph_client.requests.request", side_effect=fake):
        yield fake


def _wrapper(**kwargs):
    defaults = dict(token="stub-token", client_id="cid", oauth_discovery_endpoint=AUTHORITY,
                    scopes=["Mail.Read"], configuration_uuid="cfg-1")
    return MsGraphApiWrapper(**{**defaults, **kwargs})


# ---------------------------------------------------------------- scopes

class TestScopes:
    def test_free_text_scopes_are_kept_prefix_stripped_and_deduplicated(self):
        assert normalize_ms_graph_scopes(
            "https://graph.microsoft.com/Mail.Read, Files.ReadWrite.All offline_access mail.read openid"
        ) == ["Mail.Read", "Files.ReadWrite.All"]

    def test_empty_means_user_read(self):
        assert normalize_ms_graph_scopes(None) == ["User.Read"]
        assert normalize_ms_graph_scopes([]) == ["User.Read"]
        assert normalize_ms_graph_scopes("offline_access") == ["User.Read"]

    def test_graph_default_is_allowed_alone(self):
        assert normalize_ms_graph_scopes(".default") == ["https://graph.microsoft.com/.default"]
        with pytest.raises(ValueError, match="cannot be combined"):
            normalize_ms_graph_scopes(["https://graph.microsoft.com/.default", "Mail.Read"])

    @pytest.mark.parametrize("scope", [
        "https://contoso.sharepoint.com/AllSites.Read",
        "api://my-api/access",
        "Mail",
        "Mail.Read;drop",
    ])
    def test_non_graph_scopes_are_rejected(self, scope):
        with pytest.raises(ValueError, match="not a Microsoft Graph permission"):
            normalize_ms_graph_scopes(scope)

    def test_credential_needs_app_or_client(self):
        with pytest.raises(ValueError, match="Select a Default app"):
            MsGraphConfiguration(scopes=["Mail.Read"])
        assert MsGraphConfiguration(oauth_app_id="app-1").scopes == ["User.Read"]

    def test_credential_rejects_non_entra_authority(self):
        with pytest.raises(ValueError, match="login.microsoftonline.com"):
            MsGraphConfiguration(client_id="c", client_secret="s", oauth_discovery_endpoint="https://evil.example")


# ---------------------------------------------------------------- rules

class TestRules:
    def test_default_allows_get_only(self):
        rules = RequestRules()
        rules.check("GET", "/me/messages")
        with pytest.raises(RuleViolation, match="POST is not allowed"):
            rules.check("POST", "/me/sendMail")

    def test_prefix_matches_whole_segments_and_star_one_segment(self):
        rules = RequestRules(["GET"], allowed_paths=["/me/messages", "/users/*/events"])
        rules.check("GET", "/me/messages")
        rules.check("GET", "/me/messages/AAMk=")
        rules.check("GET", "/users/abc/events/1")
        for path in ("/me/messagesX", "/me/events", "/users/abc/messages", "/users"):
            with pytest.raises(RuleViolation, match="not in the toolkit's allowed paths"):
                rules.check("GET", path)

    def test_deny_wins_over_allow(self):
        rules = RequestRules(["GET", "POST"], allowed_paths=["/me"], denied_paths=["/me/sendMail"])
        rules.check("POST", "/me/messages")
        with pytest.raises(RuleViolation, match="denied"):
            rules.check("POST", "/me/sendmail")

    @pytest.mark.parametrize("path", [
        "/v1.0/me/sendMail", "/beta/me/sendMail", "https://graph.microsoft.com/v1.0/me/sendMail",
        "/me//sendMail", "/ME/SENDMAIL/", "/me/%73endMail", "/me/sendMail?x=1",
    ])
    def test_path_tricks_do_not_bypass_deny(self, path):
        rules = RequestRules(["POST"], denied_paths=["/me/sendMail"])
        with pytest.raises(RuleViolation):
            rules.check("POST", path)

    @pytest.mark.parametrize("path", ["/me/messages/../sendMail", "/me/%2e%2e/users", "/me/%252e%252e/x"])
    def test_dot_segments_and_double_encoding_are_refused(self, path):
        with pytest.raises(ToolException):
            normalize_path(path)

    @pytest.mark.parametrize("url", [
        "https://evil.example/v1.0/me", "http://graph.microsoft.com/v1.0/me",
        "https://graph.microsoft.com.evil.example/v1.0/me", "https://user@graph.microsoft.com/v1.0/me",
        "//evil.example/me",
    ])
    def test_only_graph_urls(self, url):
        with pytest.raises(ToolException, match="Only https://graph.microsoft.com"):
            split_graph_url(url)

    def test_describe_lists_rules(self):
        text = RequestRules(["GET", "PATCH"], ["/me"], ["/me/drive"]).describe()
        assert "GET, PATCH" in text and "/me/drive" in text


# ---------------------------------------------------------------- graph_request

class TestGraphRequest:
    def test_get_sends_token_query_and_returns_json(self, graph):
        graph.on("GET", f"{G}/me/messages", _resp(200, {"value": [{"id": "1"}]}, {"request-id": "r1"}))
        result = _wrapper().graph_request("/me/messages", query={"$top": 1})
        assert result == {"status": 200, "headers": {"content-type": "application/json", "request-id": "r1"},
                          "body": {"value": [{"id": "1"}]}}
        call = graph.calls[0]
        assert call["headers"]["Authorization"] == "Bearer stub-token"
        assert call["params"] == {"$top": 1}

    def test_beta_and_absolute_url_versions(self, graph):
        graph.on("GET", "https://graph.microsoft.com/beta/me", _resp(200, {"id": "me"}))
        assert _wrapper().graph_request("/me", api_version="beta")["body"] == {"id": "me"}
        assert _wrapper().graph_request("https://graph.microsoft.com/beta/me")["body"] == {"id": "me"}

    def test_post_json_body_when_allowed(self, graph):
        graph.on("POST", f"{G}/me/sendMail", _resp(202))
        result = _wrapper(allowed_methods=["GET", "POST"]).graph_request(
            "/me/sendMail", method="POST", body={"message": {"subject": "hi"}})
        assert result["status"] == 202 and "body" in result and result["body"] is None
        call = graph.calls[0]
        assert json.loads(call["data"]) == {"message": {"subject": "hi"}}
        assert call["headers"]["Content-Type"] == "application/json"

    def test_rules_are_checked_before_any_request(self, graph):
        with pytest.raises(RuleViolation, match="DELETE is not allowed"):
            _wrapper().graph_request("/me/messages/1", method="DELETE")
        assert graph.calls == []

    def test_batch_path_is_refused_in_graph_request(self, graph):
        with pytest.raises(ToolException, match="graph_batch"):
            _wrapper(allowed_methods=["POST"]).graph_request("/$batch", method="POST", body={"requests": []})

    def test_caller_cannot_override_authorization(self, graph):
        graph.on("GET", f"{G}/me", _resp(200, {"id": "me"}))
        _wrapper().graph_request("/me", headers={"Authorization": "Bearer other", "ConsistencyLevel": "eventual"})
        headers = graph.calls[0]["headers"]
        assert headers["Authorization"] == "Bearer stub-token"
        assert headers["ConsistencyLevel"] == "eventual"

    def test_follow_next_link_merges_pages(self, graph):
        graph.on("GET", f"{G}/me/messages",
                 _resp(200, {"value": [{"id": "1"}], "@odata.nextLink": f"{G}/me/messages?$skip=1"}))
        graph.on("GET", f"{G}/me/messages?$skip=1", _resp(200, {"value": [{"id": "2"}]}))
        result = _wrapper().graph_request("/me/messages", follow_next_link=3)
        assert result["body"]["value"] == [{"id": "1"}, {"id": "2"}]
        assert "next_link" not in result

    def test_next_link_on_another_host_is_refused(self, graph):
        graph.on("GET", f"{G}/me/messages",
                 _resp(200, {"value": [], "@odata.nextLink": "https://evil.example/steal"}))
        with pytest.raises(ToolException, match="Only https://graph.microsoft.com"):
            _wrapper().graph_request("/me/messages", follow_next_link=1)
        assert len(graph.calls) == 1

    def test_large_json_is_truncated(self, graph):
        graph.on("GET", f"{G}/me/messages", _resp(200, {"value": ["x" * 5000]}))
        result = _wrapper(max_response_chars=1000).graph_request("/me/messages")
        assert result["truncated"] is True and len(result["body"]) == 1000

    def test_403_adds_permission_hint(self, graph):
        graph.on("GET", f"{G}/me/messages",
                 _resp(403, {"error": {"code": "ErrorAccessDenied", "message": "Access is denied"}}))
        with pytest.raises(ToolException, match="ErrorAccessDenied.*add it to the credential scopes"):
            _wrapper().graph_request("/me/messages")

    def test_401_refreshes_once_then_retries(self, graph):
        graph.on("GET", f"{G}/me", _resp(401, {"error": {"message": "expired"}}), _resp(200, {"id": "me"}))
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "new-token"}
        with patch("elitea_sdk.tools.ms_graph.graph_client.requests.post", return_value=token_resp) as post:
            result = _wrapper(refresh_token="rt").graph_request("/me")
        assert result["body"] == {"id": "me"}
        assert graph.calls[1]["headers"]["Authorization"] == "Bearer new-token"
        assert post.call_args.kwargs["data"]["scope"] == "offline_access Mail.Read"
        assert post.call_args.args[0] == f"{AUTHORITY}/oauth2/v2.0/token"

    def test_401_without_refresh_asks_to_sign_in(self, graph):
        graph.on("GET", f"{G}/me", _resp(401, {"error": {"message": "expired"}}))
        with pytest.raises(Exception) as exc:
            _wrapper(oauth_app_id="app-1").graph_request("/me")
        assert type(exc.value).__name__ == "McpAuthorizationRequired"
        provided = exc.value.resource_metadata["provided_settings"]
        assert provided["oauth_app_id"] == "app-1" and provided["credential_type"] == "ms_graph"
        assert "mcp_client_secret" not in provided

    def test_throttling_retries_with_retry_after(self, graph):
        graph.on("GET", f"{G}/me", _resp(429, {"error": {}}, {"Retry-After": "1"}), _resp(200, {"id": "me"}))
        with patch("elitea_sdk.tools.ms_graph.graph_client.time.sleep") as sleep:
            assert _wrapper().graph_request("/me")["body"] == {"id": "me"}
        sleep.assert_called_once_with(1.0)

    def test_redirect_to_storage_drops_token(self, graph):
        storage = "https://contoso-my.sharepoint.com/download.aspx?tempauth=x"
        graph.on("GET", f"{G}/me/drive/items/1/content", _resp(302, headers={"Location": storage}))
        graph.on("GET", storage, _resp(200, content=b"%PDF-1.7", headers={"Content-Type": "application/pdf"}))
        elitea = MagicMock()
        elitea.artifact.return_value.client.upload_artifact_s3.return_value = {
            "filepath": "/ms-graph/f.pdf", "filename": "f.pdf"}
        result = _wrapper(elitea=elitea).graph_request("/me/drive/items/1/content")
        assert "Authorization" not in graph.calls[1]["headers"]
        assert result["artifact"]["filepath"] == "/ms-graph/f.pdf"
        assert result["artifact"]["size"] == 8

    def test_binary_without_artifact_storage_is_an_error(self, graph):
        graph.on("GET", f"{G}/me/photo/$value", _resp(200, content=b"\x89PNG", headers={"Content-Type": "image/png"}))
        with pytest.raises(ToolException, match="binary"):
            _wrapper().graph_request("/me/photo/$value")

    def test_download_name_from_drive_path(self, graph):
        graph.on("GET", f"{G}/me/drive/root:/docs/report.pdf:/content",
                 _resp(200, content=b"%PDF", headers={"Content-Type": "application/pdf"}))
        elitea = MagicMock()
        upload = elitea.artifact.return_value.client.upload_artifact_s3
        upload.return_value = {}
        _wrapper(elitea=elitea).graph_request("/me/drive/root:/docs/report.pdf:/content",
                                              download_bucket="reports", download_file_name="r.pdf")
        assert upload.call_args.args[:2] == ("reports", "r.pdf")
        _wrapper(elitea=elitea).graph_request("/me/drive/root:/docs/report.pdf:/content")
        assert upload.call_args.args[0] == "ms-graph" and upload.call_args.args[1].endswith("-report.pdf")

    def test_upload_from_artifact(self, graph):
        graph.on("PUT", f"{G}/me/drive/root:/a.txt:/content", _resp(201, {"id": "item"}))
        with patch("elitea_sdk.tools.utils.get_file_bytes_from_artifact", return_value=(b"hello", "a.txt")):
            result = _wrapper(allowed_methods=["PUT"], elitea=MagicMock()).graph_request(
                "/me/drive/root:/a.txt:/content", method="PUT", upload_from_artifact="/bucket/a.txt",
                upload_content_type="text/plain")
        assert result["body"] == {"id": "item"}
        call = graph.calls[0]
        assert call["data"] == b"hello" and call["headers"]["Content-Type"] == "text/plain"

    def test_upload_over_limit_is_refused(self, graph):
        with patch("elitea_sdk.tools.utils.get_file_bytes_from_artifact", return_value=(b"x" * (1024 * 1024 + 1), "a")):
            with pytest.raises(ToolException, match="upload limit"):
                _wrapper(allowed_methods=["PUT"], max_upload_mb=1, elitea=MagicMock()).graph_request(
                    "/me/drive/root:/a:/content", method="PUT", upload_from_artifact="/b/a")
        assert graph.calls == []

    def test_download_over_limit_is_refused(self, graph):
        graph.on("GET", f"{G}/me/drive/items/1/content",
                 _resp(200, content=b"x", headers={"Content-Type": "application/pdf", "Content-Length": str(2 * 1024 * 1024)}))
        with pytest.raises(ToolException, match="limit"):
            _wrapper(max_download_mb=1, elitea=MagicMock()).graph_request("/me/drive/items/1/content")


# ---------------------------------------------------------------- graph_batch

class TestGraphBatch:
    def test_batch_of_gets_works_with_get_only(self, graph):
        graph.on("POST", f"{G}/$batch", _resp(200, {"responses": [
            {"id": "1", "status": 200, "body": {"id": "u1"}},
            {"id": "2", "status": 404, "body": {"error": {"code": "NotFound"}}},
        ]}))
        result = _wrapper().graph_batch([{"id": "1", "url": "/users/u1"}, {"id": "2", "url": "users/u2"}])
        assert result["responses"]["1"] == {"status": 200, "body": {"id": "u1"}}
        sent = json.loads(graph.calls[0]["data"])["requests"]
        assert [r["url"] for r in sent] == ["/users/u1", "/users/u2"]

    def test_each_sub_request_is_checked(self, graph):
        with pytest.raises(RuleViolation, match="DELETE"):
            _wrapper().graph_batch([{"id": "1", "url": "/me"}, {"id": "2", "method": "DELETE", "url": "/me/messages/1"}])
        with pytest.raises(RuleViolation, match="denied"):
            _wrapper(denied_paths=["/users"]).graph_batch([{"id": "1", "url": "/users/u1?$select=id"}])
        assert graph.calls == []

    def test_batch_refuses_absolute_urls_and_duplicate_ids(self, graph):
        with pytest.raises(ToolException, match="relative"):
            _wrapper().graph_batch([{"id": "1", "url": "https://graph.microsoft.com/v1.0/me"}])
        with pytest.raises(ToolException, match="unique"):
            _wrapper().graph_batch([{"id": "1", "url": "/me"}, {"id": "1", "url": "/me"}])


# ---------------------------------------------------------------- toolkit

class TestToolkit:
    CONFIG = {"client_id": "cid", "client_secret": "sec", "oauth_discovery_endpoint": AUTHORITY,
              "scopes": ["Mail.Read"], "configuration_uuid": "cfg-1"}

    def test_tools_and_rules_in_description(self):
        toolkit = MsGraphToolkit.get_toolkit(
            ms_graph_configuration=self.CONFIG, tokens={f"cfg-1:{AUTHORITY}": {"access_token": "t"}},
            allowed_methods=["GET", "PATCH"], denied_paths=["/me/drive"])
        tools = {t.name: t for t in toolkit.get_tools()}
        assert set(tools) == {"graph_request", "graph_batch"}
        assert "Allowed methods: GET, PATCH" in tools["graph_request"].description
        assert "/me/drive" in tools["graph_batch"].description

    def test_missing_token_asks_to_sign_in_without_admin_secret(self):
        with patch("elitea_sdk.runtime.utils.mcp_oauth.fetch_oauth_authorization_server_metadata", return_value={}):
            with pytest.raises(Exception) as exc:
                MsGraphToolkit.get_toolkit(ms_graph_configuration={**self.CONFIG, "oauth_app_id": "app-1"}, tokens={})
        assert type(exc.value).__name__ == "McpAuthorizationRequired"
        provided = exc.value.resource_metadata["provided_settings"]
        assert provided["scopes"] == ["offline_access", "Mail.Read"]
        assert "mcp_client_secret" not in provided

    def test_config_schema(self):
        props = MsGraphToolkit.toolkit_config_schema().model_json_schema()["properties"]
        assert props["allowed_methods"]["default"] == ["GET"]
        assert props["allowed_paths"]["default"] == [] and props["denied_paths"]["default"] == []
