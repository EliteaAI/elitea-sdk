"""Tests for the Teams toolkit: reading chats, channel listing, search, sending and registration."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.teams import graph_wrapper as gw
from elitea_sdk.tools.teams.api_wrapper import TeamsApiWrapper
from elitea_sdk.tools.teams.graph_wrapper import TeamsGraphWrapper

G = "https://graph.microsoft.com/v1.0"
FAKE_TOKEN = "stub-token"
ME = "00000000-0000-0000-0000-00000000000a"
ALICE = "00000000-0000-0000-0000-0000000000a1"
BOB = "00000000-0000-0000-0000-0000000000b0"
TEAM = "11111111-1111-1111-1111-111111111111"
CHANNEL = "19:general@thread.tacv2"
CHAT = "19:chat-1@thread.v2"
NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)


def _resp(status=200, json_body=None, headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.json.return_value = json_body if json_body is not None else {}
    resp.text = str(json_body)
    resp.url = "https://graph.microsoft.com/stub"
    resp.headers = headers or {}
    return resp


class FakeGraph:
    """Routes requests.request(method, url, ...) to canned responses and records calls."""

    def __init__(self):
        self.routes = {}
        self.calls = []

    def on(self, method, url, *responses):
        self.routes[(method, url)] = list(responses)
        return self

    def __call__(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params, "json": json, "headers": headers})
        queue = self.routes.get((method, url))
        if not queue:
            raise AssertionError(f"Unexpected request {method} {url}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        return item

    def find(self, method, url):
        return [c for c in self.calls if c["method"] == method and c["url"] == url]


def _user(uid, email, name):
    return {"id": uid, "mail": email, "userPrincipalName": email, "displayName": name}


def _msg(msg_id, created, sender_id=ALICE, name="Alice", text="hello", msg_type="message",
         deleted=None, reply_to=None, modified=None):
    return {
        "id": msg_id,
        "replyToId": reply_to,
        "messageType": msg_type,
        "createdDateTime": created,
        "lastModifiedDateTime": modified or created,
        "deletedDateTime": deleted,
        "from": {"user": {"id": sender_id, "displayName": name}} if sender_id else None,
        "body": {"contentType": "html", "content": f"<p>{text}</p>"},
    }


@pytest.fixture
def graph():
    fake = FakeGraph()
    fake.on("GET", f"{G}/me", _resp(json_body=_user(ME, "me@example.com", "Me")))
    fake.on("GET", f"{G}/users/alice@example.com", _resp(json_body=_user(ALICE, "alice@example.com", "Alice")))
    fake.on("GET", f"{G}/users/bob@example.com", _resp(json_body=_user(BOB, "bob@example.com", "Bob")))
    with patch("requests.request", side_effect=fake), patch.object(gw, "_now", return_value=NOW), \
            patch.object(gw.time, "sleep") as sleep:
        fake.sleep = sleep
        yield fake


@pytest.fixture
def wrapper():
    return TeamsGraphWrapper(token=FAKE_TOKEN, scopes=["Chat.Read"])


class TestHttp:
    def test_refreshes_token_once_on_401(self, graph):
        w = TeamsGraphWrapper(token=FAKE_TOKEN, scopes=["Chat.Read"], refresh_token="stub-refresh",
                              oauth_token_endpoint="https://login.example.com/token")
        graph.on("GET", f"{G}/me/joinedTeams", _resp(401, {"error": {"message": "expired"}}),
                 _resp(json_body={"value": []}))
        with patch("requests.post", return_value=_resp(json_body={"access_token": "stub-new"})) as post:
            assert w.list_teams() == []
        assert post.call_args.kwargs["data"]["grant_type"] == "refresh_token"
        assert graph.calls[-1]["headers"]["Authorization"] == "Bearer stub-new"

    def test_401_without_oauth_raises_tool_exception(self, graph, wrapper):
        graph.on("GET", f"{G}/me/joinedTeams", _resp(401, {"error": {"message": "expired"}}))
        with pytest.raises(ToolException, match="invalid or expired"):
            wrapper.list_teams()

    def test_retries_throttled_request_honouring_retry_after(self, graph, wrapper):
        graph.on("GET", f"{G}/me/joinedTeams", _resp(429, {}, {"Retry-After": "7"}),
                 _resp(json_body={"value": [{"id": TEAM, "displayName": "Eng"}]}))
        assert wrapper.list_teams()[0]["id"] == TEAM
        graph.sleep.assert_called_once_with(7.0)

    def test_forbidden_error_mentions_scopes(self, graph, wrapper):
        graph.on("GET", f"{G}/me/joinedTeams", _resp(403, {"error": {"message": "Missing role"}}))
        with pytest.raises(ToolException, match="Missing role.*scopes"):
            wrapper.list_teams()


class TestResolution:
    def test_user_falls_back_to_mail_filter(self, graph, wrapper):
        graph.on("GET", f"{G}/users/carol@example.com", _resp(404, {"error": {"message": "not found"}}))
        graph.on("GET", f"{G}/users", _resp(json_body={"value": [_user("c1", "carol@example.com", "Carol")]}))
        assert wrapper._resolve_user("carol@example.com")["id"] == "c1"
        assert graph.find("GET", f"{G}/users")[0]["params"]["$filter"] == "mail eq 'carol@example.com'"
        wrapper._resolve_user("CAROL@example.com")
        assert len(graph.find("GET", f"{G}/users")) == 1  # cached

    def test_team_and_channel_by_name(self, graph, wrapper):
        graph.on("GET", f"{G}/me/joinedTeams", _resp(json_body={"value": [{"id": TEAM, "displayName": "Eng"}]}))
        graph.on("GET", f"{G}/teams/{TEAM}/channels",
                 _resp(json_body={"value": [{"id": CHANNEL, "displayName": "General"}]}))
        assert wrapper._resolve_channel(wrapper._resolve_team("eng"), "general") == CHANNEL

    def test_ambiguous_team_name_lists_ids(self, graph, wrapper):
        graph.on("GET", f"{G}/me/joinedTeams", _resp(json_body={"value": [
            {"id": "t1", "displayName": "Eng"}, {"id": "t2", "displayName": "eng"}]}))
        with pytest.raises(ToolException, match="ambiguous.*t1, t2"):
            wrapper._resolve_team("Eng")

    def test_chat_by_topic_and_by_email(self, graph, wrapper):
        chats = {"value": [
            {"id": CHAT, "chatType": "group", "topic": "Release crew",
             "members": [{"userId": ME}, {"userId": ALICE}, {"userId": BOB}]},
            {"id": "19:1on1", "chatType": "oneOnOne", "topic": None,
             "members": [{"userId": ME}, {"userId": ALICE}]},
        ]}
        graph.on("GET", f"{G}/me/chats", _resp(json_body=chats))
        assert wrapper._resolve_chat("release crew") == CHAT
        assert wrapper._resolve_chat("alice@example.com") == "19:1on1"
        assert wrapper._resolve_chat(CHAT) == CHAT


    def test_unknown_chat_suggests_near_matches_only(self, graph, wrapper):
        topics = [f"Unrelated meeting {i}" for i in range(40)] + ["Release crew", "8:orgid:fe7044f0"]
        graph.on("GET", f"{G}/me/chats", _resp(json_body={"value": [
            {"id": f"19:c{i}@thread.v2", "chatType": "group", "topic": t} for i, t in enumerate(topics)]}))
        with pytest.raises(ToolException) as exc:
            wrapper._resolve_chat("Relase crew")
        text = str(exc.value)
        assert text.startswith("Chat 'Relase crew' not found. Did you mean: Release crew?")
        assert "list_chats" in text and "email" in text
        assert "Unrelated meeting" not in text and "8:orgid" not in text

    def test_unknown_chat_without_near_match_gives_next_step(self, graph, wrapper):
        graph.on("GET", f"{G}/me/chats", _resp(json_body={"value": [
            {"id": CHAT, "chatType": "group", "topic": "Release crew"}]}))
        with pytest.raises(ToolException) as exc:
            wrapper._resolve_chat("asdasdas")
        assert str(exc.value) == (
            "Chat 'asdasdas' not found. Use list_chats to find it, or pass the chat ID; "
            "for a 1:1 chat pass the person's email."
        )

    def test_unknown_team_and_channel_point_to_their_list_tools(self, graph, wrapper):
        graph.on("GET", f"{G}/me/joinedTeams", _resp(json_body={"value": [{"id": TEAM, "displayName": "Eng"}]}))
        graph.on("GET", f"{G}/teams/{TEAM}/channels",
                 _resp(json_body={"value": [{"id": CHANNEL, "displayName": "General"}]}))
        with pytest.raises(ToolException, match="list_teams"):
            wrapper._resolve_team("nonexistent-team-xyz")
        with pytest.raises(ToolException, match="Did you mean: General\\?.*list_channels"):
            wrapper._resolve_channel(TEAM, "Genral")

class TestFindChatMessages:
    def _setup(self, graph, messages, read_at=None):
        info = {"id": CHAT, "chatType": "group", "topic": "Release crew"}
        if read_at:
            info["viewpoint"] = {"lastMessageReadDateTime": read_at}
        graph.on("GET", f"{G}/chats/{CHAT}", _resp(json_body=info))
        graph.on("GET", f"{G}/chats/{CHAT}/messages", _resp(json_body={"value": messages}))

    def test_since_filters_server_side_and_flags_new(self, graph, wrapper):
        self._setup(graph, [
            _msg("3", "2026-09-24T11:00:00.000Z", BOB, "Bob"),
            _msg("2", "2026-09-24T10:00:00.000Z"),
            _msg("1", "2026-09-24T08:00:00.000Z", modified="2026-09-24T10:30:00.000Z"),
        ])
        result = wrapper.find_chat_messages(CHAT, since="2026-09-24T09:00:00Z")
        params = graph.find("GET", f"{G}/chats/{CHAT}/messages")[0]["params"]
        assert params["$orderby"] == "lastModifiedDateTime desc"
        assert params["$filter"] == "lastModifiedDateTime gt 2026-09-24T08:59:59.000Z"
        # message 1 was edited after since but created before it
        assert [m["id"] for m in result["messages"]] == ["2", "3"]
        assert all(m["is_new"] for m in result["messages"])
        assert result["watermark"] == "2026-09-24T11:00:00.000Z"
        assert result["latest_message_id"] == "3"

    def test_senders_by_email_match_user_id(self, graph, wrapper):
        self._setup(graph, [
            _msg("2", "2026-09-24T10:00:00.000Z", BOB, "Bob"),
            _msg("1", "2026-09-24T09:30:00.000Z", ALICE, "Alice"),
        ])
        result = wrapper.find_chat_messages(CHAT, senders=["alice@example.com"], since="2026-09-24T09:00:00Z")
        assert [m["id"] for m in result["messages"]] == ["1"]
        assert result["messages"][0]["matched_by"] == ["from:alice@example.com"]
        assert result["messages"][0]["from"]["email"] == "alice@example.com"
        # watermark still advances past non-matching messages
        assert result["latest_message_id"] == "2"

    def test_skips_system_deleted_and_own_messages(self, graph, wrapper):
        self._setup(graph, [
            _msg("4", "2026-09-24T11:00:00.000Z", ME, "Me"),
            _msg("3", "2026-09-24T10:40:00.000Z", None, msg_type="systemEventMessage"),
            _msg("2", "2026-09-24T10:20:00.000Z", deleted="2026-09-24T10:21:00.000Z"),
            _msg("1", "2026-09-24T10:00:00.000Z", text="Deploy <b>done</b>"),
        ])
        result = wrapper.find_chat_messages(CHAT, since="2026-09-24T09:00:00Z")
        assert [m["id"] for m in result["messages"]] == ["1"]
        assert result["messages"][0]["text"] == "Deploy done"
        own = wrapper.find_chat_messages(CHAT, since="2026-09-24T09:00:00Z", include_own=True)
        assert [m["id"] for m in own["messages"]] == ["1", "4"]
        assert own["messages"][1]["from_me"] is True

    def test_after_message_id_excludes_anchor_keeps_same_timestamp(self, graph, wrapper):
        self._setup(graph, [
            _msg("3", "2026-09-24T10:00:00.000Z"),
            _msg("2", "2026-09-24T09:00:00.000Z"),
            _msg("1", "2026-09-24T09:00:00.000Z"),
        ])
        graph.on("GET", f"{G}/chats/{CHAT}/messages/1", _resp(json_body=_msg("1", "2026-09-24T09:00:00.000Z")))
        result = wrapper.find_chat_messages(CHAT, after_message_id="1")
        assert [m["id"] for m in result["messages"]] == ["2", "3"]

    def test_without_bound_new_means_unread(self, graph, wrapper):
        self._setup(graph, [
            _msg("2", "2026-09-24T10:00:00.000Z"),
            _msg("1", "2026-09-24T08:00:00.000Z"),
        ], read_at="2026-09-24T09:00:00.000Z")
        result = wrapper.find_chat_messages(CHAT)
        assert result["new_means"] == "unread"
        assert [m["id"] for m in result["messages"]] == ["2"]
        everything = wrapper.find_chat_messages(CHAT, only_new=False)
        assert [(m["id"], m["is_new"]) for m in everything["messages"]] == [("1", False), ("2", True)]

    def test_limit_returns_oldest_and_cursor_at_last_returned(self, graph, wrapper):
        self._setup(graph, [_msg(str(i), f"2026-09-24T10:0{i}:00.000Z") for i in range(5, 0, -1)])
        result = wrapper.find_chat_messages(CHAT, since="2026-09-24T09:00:00Z", limit=2)
        assert [m["id"] for m in result["messages"]] == ["1", "2"]
        assert result["truncated"] is True
        assert result["latest_message_id"] == "2"
        assert result["watermark"] == "2026-09-24T10:02:00.000Z"

    def test_text_contains(self, graph, wrapper):
        self._setup(graph, [_msg("2", "2026-09-24T10:00:00.000Z", text="Release is GREEN"),
                            _msg("1", "2026-09-24T09:30:00.000Z", text="lunch?")])
        result = wrapper.find_chat_messages(CHAT, since="2026-09-24T09:00:00Z", text_contains="green")
        assert [m["id"] for m in result["messages"]] == ["2"]


class TestCompactOutput:
    def _members(self, n):
        people = [{"userId": ME, "displayName": "Me", "email": "me@example.com"}]
        people += [{"userId": f"u{i}", "displayName": f"User {i}", "email": f"u{i}@example.com"} for i in range(n)]
        return people

    def _chats(self, graph, n=30):
        graph.on("GET", f"{G}/me/chats", _resp(json_body={"value": [{
            "id": CHAT, "chatType": "meeting", "topic": "All hands", "webUrl": "https://teams.example/chat",
            "members": self._members(n),
            "lastMessagePreview": {"id": "m1", "createdDateTime": "2026-09-24T10:00:00Z",
                                   "from": {"user": {"displayName": "User 1"}},
                                   "body": {"contentType": "text", "content": "x" * 500}},
            "viewpoint": {"lastMessageReadDateTime": "2026-09-24T09:00:00Z"},
        }]}))

    def test_list_chats_requests_only_as_many_chats_as_limit(self, graph, wrapper):
        self._chats(graph)
        wrapper.list_chats(limit=5)
        assert graph.find("GET", f"{G}/me/chats")[0]["params"]["$top"] == 5

    def test_list_chats_caps_page_size_and_pages_fully_when_filtering(self, graph, wrapper):
        self._chats(graph)
        wrapper.list_chats(limit=500)
        assert graph.find("GET", f"{G}/me/chats")[0]["params"]["$top"] == 50
        graph.calls.clear()
        self._chats(graph)
        wrapper.list_chats(limit=5, topic_contains="hands")
        assert graph.find("GET", f"{G}/me/chats")[0]["params"]["$top"] == 50

    def test_list_chats_default_is_compact(self, graph, wrapper):
        self._chats(graph)
        chat = wrapper.list_chats()["chats"][0]
        assert chat["member_count"] == 31
        assert [m["displayName"] for m in chat["members"]] == [f"User {i}" for i in range(5)]
        assert chat["members_truncated"] is True
        assert all(set(m) == {"displayName", "email"} for m in chat["members"])
        assert "webUrl" not in chat and "lastReadDateTime" not in chat
        assert len(chat["lastMessage"]["text"]) == 300
        assert "id" not in chat["lastMessage"]
        assert chat["has_unread"] is True

    def test_list_chats_members_all_and_none(self, graph, wrapper):
        self._chats(graph)
        everyone = wrapper.list_chats(members="all")["chats"][0]
        assert len(everyone["members"]) == 30 and "members_truncated" not in everyone  # self excluded
        no_members = wrapper.list_chats(members="none")["chats"][0]
        assert "members" not in no_members and no_members["member_count"] == 31

    def test_list_chats_opt_in_ids_links_and_sizes(self, graph, wrapper):
        self._chats(graph)
        chat = wrapper.list_chats(max_members=2, include_ids=True, include_links=True,
                                  last_message_chars=10)["chats"][0]
        assert chat["members"][0] == {"displayName": "User 0", "email": "u0@example.com", "userId": "u0"}
        assert len(chat["members"]) == 2
        assert chat["webUrl"] == "https://teams.example/chat"
        assert chat["lastMessage"]["id"] == "m1" and len(chat["lastMessage"]["text"]) == 10

    def test_list_chats_member_filter_still_sees_hidden_members(self, graph, wrapper):
        self._chats(graph)
        graph.on("GET", f"{G}/users/u25@example.com", _resp(json_body=_user("u25", "u25@example.com", "User 25")))
        found = wrapper.list_chats(member="u25@example.com")
        assert found["count"] == 1  # u25 is not in the shown members, but the filter sees everyone

    def test_messages_cap_text_and_drop_noise(self, graph, wrapper):
        msg = _msg("1", "2026-09-24T10:00:00.000Z", text="y" * 1500)
        msg.update({"webUrl": "https://teams.example/m", "chatId": CHAT, "importance": "normal",
                    "attachments": [{"name": "a.pdf", "contentType": "reference", "contentUrl": "https://x/a.pdf"}]})
        graph.on("GET", f"{G}/chats/{CHAT}", _resp(json_body={"id": CHAT, "chatType": "group"}))
        graph.on("GET", f"{G}/chats/{CHAT}/messages", _resp(json_body={"value": [msg]}))
        item = wrapper.find_chat_messages(CHAT, since="2026-09-24T09:00:00Z")["messages"][0]
        assert len(item["text"]) == 1000 and item["text_truncated"] is True
        assert not {"webUrl", "chatId", "lastModifiedDateTime", "importance"} & set(item)
        assert item["attachments"] == [{"name": "a.pdf", "contentType": "reference"}]
        full = wrapper.find_chat_messages(CHAT, since="2026-09-24T09:00:00Z", text_max_chars=0, include_links=True)
        item = full["messages"][0]
        assert len(item["text"]) == 1500 and "text_truncated" not in item
        assert item["webUrl"] == "https://teams.example/m"
        assert item["attachments"][0]["contentUrl"] == "https://x/a.pdf"

    def test_text_filter_matches_beyond_the_output_cap(self, graph, wrapper):
        msg = _msg("1", "2026-09-24T10:00:00.000Z", text="a" * 2000 + " needle")
        graph.on("GET", f"{G}/chats/{CHAT}", _resp(json_body={"id": CHAT, "chatType": "group"}))
        graph.on("GET", f"{G}/chats/{CHAT}/messages", _resp(json_body={"value": [msg]}))
        result = wrapper.find_chat_messages(CHAT, since="2026-09-24T09:00:00Z", text_contains="needle", text_max_chars=50)
        assert result["count"] == 1 and len(result["messages"][0]["text"]) == 50

    def test_edited_message_keeps_last_modified(self, graph, wrapper):
        msg = _msg("1", "2026-09-24T10:00:00.000Z", modified="2026-09-24T10:30:00.000Z")
        graph.on("GET", f"{G}/chats/{CHAT}", _resp(json_body={"id": CHAT, "chatType": "group"}))
        graph.on("GET", f"{G}/chats/{CHAT}/messages", _resp(json_body={"value": [msg]}))
        item = wrapper.find_chat_messages(CHAT, since="2026-09-24T09:00:00Z")["messages"][0]
        assert item["lastModifiedDateTime"] == "2026-09-24T10:30:00.000Z"

    def test_search_hit_link_is_opt_in(self):
        hit = {"summary": "s", "resource": {"id": "1", "chatId": CHAT, "webLink": "https://teams.example/l"}}
        assert "webLink" not in TeamsGraphWrapper._format_search_hit(hit)
        assert TeamsGraphWrapper._format_search_hit(hit, include_links=True)["webLink"] == "https://teams.example/l"

    def test_list_channels_filter_limit_and_links(self, graph, wrapper):
        graph.on("GET", f"{G}/teams/{TEAM}/channels", _resp(json_body={"value": [
            {"id": f"19:c{i}@thread.tacv2", "displayName": f"Dev {i}" if i < 4 else f"Ops {i}",
             "membershipType": "standard", "webUrl": f"https://teams.example/c{i}"} for i in range(6)]}))
        result = wrapper.list_channels(TEAM, name_contains="dev", limit=3)
        assert result["count"] == 3 and result["truncated"] is True and result["total_matching"] == 4
        assert all("webUrl" not in c for c in result["channels"])
        assert wrapper.list_channels(TEAM, include_links=True)["channels"][0]["webUrl"]


class TestSearch:
    def test_builds_kql_and_filters_by_exact_time(self, graph, wrapper):
        hits = [
            {"summary": "deploy <c0>done</c0>", "resource": {
                "id": "m2", "createdDateTime": "2026-09-24T10:00:00Z", "chatId": CHAT,
                "from": {"emailAddress": {"name": "Alice", "address": "alice@example.com"}}}},
            {"summary": "older", "resource": {"id": "m1", "createdDateTime": "2026-09-24T08:00:00Z",
                                               "channelIdentity": {"teamId": TEAM, "channelId": CHANNEL}}},
        ]
        graph.on("POST", f"{G}/search/query", _resp(json_body={"value": [{"hitsContainers": [
            {"hits": hits, "moreResultsAvailable": False}]}]}))
        result = wrapper.search_teams_messages(query="deploy", senders=["alice@example.com", "Bob"],
                                               since="2026-09-24T09:00:00Z")
        request = graph.find("POST", f"{G}/search/query")[0]["json"]["requests"][0]
        assert request["entityTypes"] == ["chatMessage"]
        assert request["query"]["queryString"] == (
            'deploy AND (from:"alice@example.com" OR from:"Bob") AND sent>=2026-09-24')
        assert [m["id"] for m in result["messages"]] == ["m2"]
        assert result["messages"][0]["chatId"] == CHAT
        assert result["messages"][0]["summary"] == "deploy done"

    def test_requires_some_criteria(self, graph, wrapper):
        with pytest.raises(ToolException):
            wrapper.search_teams_messages()


class TestSendChatMessage:
    def test_one_recipient_uses_one_on_one_chat(self, graph, wrapper):
        graph.on("POST", f"{G}/chats", _resp(201, {"id": "19:1on1"}))
        graph.on("POST", f"{G}/chats/19:1on1/messages",
                 _resp(201, {"id": "m1", "createdDateTime": "2026-09-24T12:00:00Z"}))
        result = wrapper.send_chat_message("Hi", recipients=["alice@example.com"])
        chat_body = graph.find("POST", f"{G}/chats")[0]["json"]
        assert chat_body["chatType"] == "oneOnOne"
        assert [m["user@odata.bind"] for m in chat_body["members"]] == [
            f"{G}/users('{ME}')", f"{G}/users('{ALICE}')"]
        assert graph.find("POST", f"{G}/chats/19:1on1/messages")[0]["json"] == {
            "body": {"contentType": "text", "content": "Hi"}}
        assert result == {"status": "sent", "message_id": "m1", "chat_id": "19:1on1", "chat_created": False,
                          "createdDateTime": "2026-09-24T12:00:00Z"}

    def test_group_reuses_chat_with_same_members(self, graph, wrapper):
        graph.on("GET", f"{G}/me/chats", _resp(json_body={"value": [
            {"id": "19:other", "chatType": "group", "members": [{"userId": ME}, {"userId": ALICE}]},
            {"id": CHAT, "chatType": "group", "members": [{"userId": ME}, {"userId": ALICE}, {"userId": BOB}]},
        ]}))
        graph.on("POST", f"{G}/chats/{CHAT}/messages", _resp(201, {"id": "m1"}))
        result = wrapper.send_chat_message("Hi", recipients=["alice@example.com", "bob@example.com"])
        assert result["chat_id"] == CHAT and result["chat_created"] is False
        assert not graph.find("POST", f"{G}/chats")

    def test_group_is_created_with_topic(self, graph, wrapper):
        graph.on("GET", f"{G}/me/chats", _resp(json_body={"value": []}))
        graph.on("POST", f"{G}/chats", _resp(201, {"id": "19:new"}))
        graph.on("POST", f"{G}/chats/19:new/messages", _resp(201, {"id": "m1"}))
        result = wrapper.send_chat_message("Hi", recipients=["alice@example.com", "bob@example.com", "me@example.com"],
                                           topic="Release crew")
        body = graph.find("POST", f"{G}/chats")[0]["json"]
        assert body["chatType"] == "group" and body["topic"] == "Release crew"
        assert len(body["members"]) == 3
        assert result["chat_created"] is True

    def test_mentions_are_resolved_and_prepended(self, graph, wrapper):
        graph.on("POST", f"{G}/chats/{CHAT}/messages", _resp(201, {"id": "m1"}))
        wrapper.send_chat_message("Please check <this>", chat=CHAT, mentions=["alice@example.com"])
        body = graph.find("POST", f"{G}/chats/{CHAT}/messages")[0]["json"]
        assert body["body"] == {"contentType": "html", "content": '<at id="0">Alice</at> Please check &lt;this&gt;'}
        assert body["mentions"][0]["mentioned"]["user"]["id"] == ALICE

    def test_inline_mention_is_placed_in_the_text(self, graph, wrapper):
        graph.on("POST", f"{G}/chats/{CHAT}/messages", _resp(201, {"id": "m1"}))
        wrapper.send_chat_message("Thanks @[alice@example.com], see <below>\nbye", chat=CHAT)
        body = graph.find("POST", f"{G}/chats/{CHAT}/messages")[0]["json"]
        assert body["body"] == {"contentType": "html",
                                "content": 'Thanks <at id="0">Alice</at>, see &lt;below&gt;<br>bye'}
        assert len(body["mentions"]) == 1 and body["mentions"][0]["mentioned"]["user"]["id"] == ALICE

    def test_inline_and_listed_mentions_do_not_duplicate(self, graph, wrapper):
        graph.on("POST", f"{G}/chats/{CHAT}/messages", _resp(201, {"id": "m1"}))
        wrapper.send_chat_message("Hi @[alice@example.com]", chat=CHAT, mentions=["alice@example.com"])
        body = graph.find("POST", f"{G}/chats/{CHAT}/messages")[0]["json"]
        assert body["body"]["content"] == 'Hi <at id="0">Alice</at>'
        assert len(body["mentions"]) == 1

    def test_inline_mention_in_html_message(self, graph, wrapper):
        graph.on("POST", f"{G}/chats/{CHAT}/messages", _resp(201, {"id": "m1"}))
        wrapper.send_chat_message("<b>Hi</b> @[alice@example.com]!", chat=CHAT, html=True)
        body = graph.find("POST", f"{G}/chats/{CHAT}/messages")[0]["json"]
        assert body["body"]["content"] == '<b>Hi</b> <at id="0">Alice</at>!'

    def test_requires_chat_or_recipients(self, graph, wrapper):
        with pytest.raises(ToolException, match="chat .* or recipients"):
            wrapper.send_chat_message("Hi")


class TestSendChannelMessage:
    URL = f"{G}/teams/{TEAM}/channels/{CHANNEL}/messages"

    def test_new_post_with_subject(self, graph, wrapper):
        graph.on("POST", self.URL, _resp(201, {"id": "p1", "webUrl": "https://teams.example/p1"}))
        result = wrapper.send_channel_message(TEAM, CHANNEL, "Build is green", subject="CI", importance="high")
        body = graph.find("POST", self.URL)[0]["json"]
        assert body == {"body": {"contentType": "text", "content": "Build is green"},
                        "subject": "CI", "importance": "high"}
        assert result["message_id"] == "p1" and result["thread_id"] == "p1"
        assert result["web_url"] == "https://teams.example/p1"

    def test_reply_goes_to_replies_endpoint(self, graph, wrapper):
        graph.on("POST", f"{self.URL}/p1/replies", _resp(201, {"id": "r1"}))
        result = wrapper.send_channel_message(TEAM, CHANNEL, "Thanks", subject="ignored", reply_to_message_id="p1")
        assert "subject" not in graph.find("POST", f"{self.URL}/p1/replies")[0]["json"]
        assert result["message_id"] == "r1" and result["thread_id"] == "p1"


class TestToolkit:
    def test_tools_and_schema(self):
        from elitea_sdk.tools.teams import TeamsToolkit

        names = [t["name"] for t in TeamsApiWrapper.model_construct().get_available_tools()]
        assert names == ["list_teams", "list_channels", "list_chats", "find_chat_messages",
                         "search_teams_messages", "send_chat_message", "send_channel_message"]
        schema = TeamsToolkit.toolkit_config_schema().model_json_schema()
        assert "teams_configuration" in schema["properties"]
        assert schema["metadata"]["label"] == "Teams"

    def test_get_tools_uses_token_by_config_uuid(self):
        from elitea_sdk.tools.teams import get_tools

        endpoint = "https://login.microsoftonline.com/stub-tenant"
        tools = get_tools({"settings": {
            "selected_tools": ["send_chat_message"],
            "teams_configuration": {"client_id": "stub-client", "oauth_discovery_endpoint": endpoint,
                                    "configuration_uuid": "cfg-1"},
            "tokens": {f"cfg-1:{endpoint}": {"access_token": FAKE_TOKEN, "refresh_token": "stub-refresh"}},
        }})
        assert [t.name for t in tools] == ["send_chat_message"]
        backend = tools[0].api_wrapper._backend
        assert backend._token == FAKE_TOKEN
        assert backend._oauth_token_endpoint == f"{endpoint}/oauth2/v2.0/token"

    def test_missing_token_raises_authorization_required(self):
        from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired
        from elitea_sdk.tools.teams import get_tools

        with patch("elitea_sdk.runtime.utils.mcp_oauth.fetch_oauth_authorization_server_metadata",
                   return_value=None):
            with pytest.raises(McpAuthorizationRequired) as exc:
                get_tools({"settings": {"teams_configuration": {
                    "client_id": "stub-client", "client_secret": "stub-secret",
                    "oauth_discovery_endpoint": "https://login.microsoftonline.com/stub-tenant",
                    "scopes": ["Chat.Read"]}}})
        meta = exc.value.resource_metadata
        assert meta["resource_name"] == "Teams"
        assert meta["provided_settings"]["mcp_client_id"] == "stub-client"
        assert meta["provided_settings"]["scopes"] == ["offline_access", "Chat.Read"]
        assert meta["provided_settings"]["mcp_client_secret"] != "stub-secret"

    def test_registered_in_sdk(self):
        from elitea_sdk.configurations import AVAILABLE_CONFIGURATIONS
        from elitea_sdk.tools import AVAILABLE_TOOLS

        assert "teams" in AVAILABLE_CONFIGURATIONS
        assert "teams" in AVAILABLE_TOOLS


class TestErrorContract:
    def test_404_names_resource_and_is_flagged_not_found(self, graph, wrapper):
        graph.on("GET", f"{G}/me/joinedTeams", _resp(404, {"error": {"message": "Team gone."}}))
        with pytest.raises(ToolException) as exc:
            wrapper.list_teams()
        text = str(exc.value)
        assert "HTTP 404: Team gone (resource: /stub)." in text
        assert ".." not in text
        assert "do not retry with the same ID" in text
        from elitea_sdk.runtime.tool_outcome import ToolErrorClass, classify_tool_error
        assert classify_tool_error(exc.value) is ToolErrorClass.INPUT

    def test_unexpected_exception_becomes_tool_exception(self, graph, wrapper):
        import requests
        graph.on("GET", f"{G}/me/joinedTeams", _resp())
        with patch("requests.request", side_effect=requests.ConnectionError("Connection refused")):
            with pytest.raises(ToolException, match="Failed to list teams: Connection refused") as exc:
                wrapper.list_teams()
        from elitea_sdk.runtime.tool_outcome import ToolErrorClass, classify_tool_error
        assert classify_tool_error(exc.value) is ToolErrorClass.INFRASTRUCTURE

    def test_existing_tool_exception_is_not_double_wrapped(self, graph, wrapper):
        graph.on("GET", f"{G}/me/joinedTeams", _resp(401, {"error": {"message": "expired"}}))
        with pytest.raises(ToolException) as exc:
            wrapper.list_teams()
        assert not str(exc.value).startswith("Failed to")
