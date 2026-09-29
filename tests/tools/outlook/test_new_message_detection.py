"""Tests for Outlook new-message detection, thread tracking and sent-message IDs."""

from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.tools import ToolException

from elitea_sdk.tools.outlook.api_wrapper import OutlookApiWrapper
from elitea_sdk.tools.outlook.graph_wrapper import OutlookGraphWrapper

GRAPH = "https://graph.microsoft.com/v1.0/me"
FAKE_TOKEN = "stub-token"


def _resp(status=200, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.json.return_value = json_body if json_body is not None else {}
    resp.text = text
    resp.url = "https://graph.microsoft.com/stub"
    if resp.ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}", response=resp)
    return resp


def _msg(msg_id, received, sender="alice@example.com", to=("me@example.com",), cc=(),
         is_read=False, conversation="conv-1", folder="inbox-id", is_draft=False):
    return {
        "id": msg_id,
        "conversationId": conversation,
        "parentFolderId": folder,
        "subject": f"Subject {msg_id}",
        "from": {"emailAddress": {"address": sender}},
        "toRecipients": [{"emailAddress": {"address": a}} for a in to],
        "ccRecipients": [{"emailAddress": {"address": a}} for a in cc],
        "receivedDateTime": received,
        "sentDateTime": received,
        "isRead": is_read,
        "isDraft": is_draft,
    }


@pytest.fixture
def wrapper():
    return OutlookGraphWrapper(token=FAKE_TOKEN, scopes=["Mail.ReadWrite"])


class TestHeaders:
    def test_immutable_id_is_requested(self, wrapper):
        assert wrapper._auth_headers()["Prefer"] == 'IdType="ImmutableId"'

    def test_extra_preference_is_appended(self, wrapper):
        prefer = wrapper._auth_headers(prefer='outlook.body-content-type="text"')["Prefer"]
        assert prefer == 'IdType="ImmutableId", outlook.body-content-type="text"'


class TestListMessages:
    def test_unread_filter_puts_orderby_property_first(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": []})) as get:
            wrapper.list_messages(unread_only=True)
        params = get.call_args.kwargs["params"]
        assert params["$filter"].startswith("receivedDateTime ge ")
        assert params["$filter"].endswith("and isRead eq false")
        assert params["$orderby"] == "receivedDateTime desc"

    def test_follows_next_link_up_to_limit(self, wrapper):
        page1 = {"value": [_msg(f"m{i}", "2026-09-23T10:00:00Z") for i in range(100)],
                 "@odata.nextLink": "https://graph.microsoft.com/next"}
        page2 = {"value": [_msg(f"n{i}", "2026-09-22T10:00:00Z") for i in range(100)]}
        with patch("requests.get", side_effect=[_resp(json_body=page1), _resp(json_body=page2)]) as get:
            result = wrapper.list_messages(limit=150)
        assert len(result) == 150
        assert get.call_args_list[1].args[0] == "https://graph.microsoft.com/next"

    def test_all_folders_uses_mailbox_messages_endpoint(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": []})) as get:
            wrapper.search_messages(query="from:bob@example.com", folder="all")
        assert get.call_args.args[0] == f"{GRAPH}/messages"
        params = get.call_args.kwargs["params"]
        assert params["$search"] == '"from:bob@example.com"'
        assert "$orderby" not in params


class TestFindNewMessages:
    def test_since_filters_server_side_oldest_first(self, wrapper):
        page = {"value": [_msg("m1", "2026-09-23T10:00:05Z"), _msg("m2", "2026-09-23T11:00:00Z")]}
        with patch("requests.get", return_value=_resp(json_body=page)) as get:
            result = wrapper.find_new_messages(since="2026-09-23T10:00:00Z")
        params = get.call_args.kwargs["params"]
        assert params["$filter"] == "receivedDateTime gt 2026-09-23T10:00:00Z"
        assert params["$orderby"] == "receivedDateTime asc"
        assert result["count"] == 2
        assert all(m["is_new"] for m in result["messages"])
        assert result["watermark"] == "2026-09-23T11:00:00Z"
        assert result["latest_message_id"] == "m2"

    def test_senders_filtered_server_side(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": []})) as get:
            wrapper.find_new_messages(senders=["Bob@Example.com", "o'neil@example.com"])
        flt = get.call_args.kwargs["params"]["$filter"]
        assert flt.startswith("receivedDateTime ge ")
        assert "from/emailAddress/address eq 'bob@example.com'" in flt
        assert "from/emailAddress/address eq 'o''neil@example.com'" in flt

    def test_distribution_list_matched_client_side(self, wrapper):
        page = {"value": [
            _msg("m1", "2026-09-23T10:00:00Z", to=("team-dl@example.com",)),
            _msg("m2", "2026-09-23T10:01:00Z", to=("other@example.com",)),
            _msg("m3", "2026-09-23T10:02:00Z", cc=("team-dl@example.com",)),
            _msg("m4", "2026-09-23T10:03:00Z", sender="boss@example.com"),
        ]}
        with patch("requests.get", return_value=_resp(json_body=page)) as get:
            result = wrapper.find_new_messages(
                recipients=["team-dl@example.com"], senders=["boss@example.com"], only_new=False)
        assert "from/emailAddress" not in get.call_args.kwargs["params"].get("$filter", "")
        by_id = {m["id"]: m["matched_by"] for m in result["messages"]}
        assert by_id == {
            "m1": ["to:team-dl@example.com"],
            "m3": ["cc:team-dl@example.com"],
            "m4": ["from:boss@example.com"],
        }

    def test_without_bound_new_means_unread(self, wrapper):
        page = {"value": [_msg("m1", "2026-09-23T10:00:00Z", is_read=True),
                          _msg("m2", "2026-09-23T09:00:00Z", is_read=False)]}
        with patch("requests.get", return_value=_resp(json_body=page)):
            result = wrapper.find_new_messages(only_new=False)
        assert {m["id"]: m["is_new"] for m in result["messages"]} == {"m1": False, "m2": True}
        assert result["watermark"] == "2026-09-23T10:00:00Z"

    def test_after_message_id_uses_anchor_timestamp_and_excludes_anchor(self, wrapper):
        anchor = {"id": "anchor", "conversationId": "conv-1", "receivedDateTime": "2026-09-23T10:00:00Z"}
        page = {"value": [_msg("anchor", "2026-09-23T10:00:00Z"),
                          _msg("tie", "2026-09-23T10:00:00Z"),
                          _msg("later", "2026-09-23T12:00:00Z")]}
        with patch("requests.get", side_effect=[_resp(json_body=anchor), _resp(json_body=page)]) as get:
            result = wrapper.find_new_messages(after_message_id="anchor")
        assert get.call_args_list[0].args[0] == f"{GRAPH}/messages/anchor"
        assert get.call_args.kwargs["params"]["$filter"] == "receivedDateTime ge 2026-09-23T10:00:00Z"
        assert [m["id"] for m in result["messages"]] == ["tie", "later"]

    def test_later_of_since_and_anchor_wins(self, wrapper):
        anchor = {"id": "anchor", "receivedDateTime": "2026-09-20T10:00:00Z"}
        with patch("requests.get", side_effect=[_resp(json_body=anchor), _resp(json_body={"value": []})]) as get:
            result = wrapper.find_new_messages(after_message_id="anchor", since="2026-09-23T00:00:00Z")
        assert get.call_args.kwargs["params"]["$filter"] == "receivedDateTime gt 2026-09-23T00:00:00Z"
        assert result["watermark"] == "2026-09-23T00:00:00Z"

    def test_missing_anchor_gives_clear_error(self, wrapper):
        with patch("requests.get", return_value=_resp(404, text="ErrorItemNotFound")):
            with pytest.raises(ToolException, match="not found.*since"):
                wrapper.find_new_messages(after_message_id="gone")

    def test_inefficient_filter_falls_back_to_client_sort(self, wrapper):
        rejected = _resp(400, text='{"error":{"code":"InefficientFilter"}}')
        page = {"value": [_msg("b", "2026-09-23T12:00:00Z"), _msg("a", "2026-09-23T11:00:00Z")]}
        with patch("requests.get", side_effect=[rejected, _resp(json_body=page)]) as get:
            result = wrapper.find_new_messages(since="2026-09-23T10:00:00Z", senders=["alice@example.com"])
        assert "$orderby" not in get.call_args.kwargs["params"]
        assert [m["id"] for m in result["messages"]] == ["a", "b"]

    def test_truncated_when_more_available(self, wrapper):
        page = {"value": [_msg(f"m{i}", f"2026-09-23T10:00:{i:02d}Z") for i in range(5)]}
        with patch("requests.get", return_value=_resp(json_body=page)):
            result = wrapper.find_new_messages(since="2026-09-23T09:00:00Z", limit=3)
        assert result["count"] == 3
        assert result["truncated"] is True
        assert result["watermark"] == "2026-09-23T10:00:02Z"

    def test_invalid_since(self, wrapper):
        with pytest.raises(ToolException, match="Invalid date"):
            wrapper.find_new_messages(since="yesterday")


class TestCheckNewMessages:
    def test_unread_check_uses_folder_counter(self, wrapper):
        folder = {"id": "inbox-id", "displayName": "Inbox", "unreadItemCount": 42, "totalItemCount": 100}
        page = {"value": [_msg("m1", "2026-09-23T10:00:00Z")]}
        with patch("requests.get", side_effect=[_resp(json_body=folder), _resp(json_body=page)]) as get:
            result = wrapper.check_new_messages()
        assert get.call_args.kwargs["params"]["$top"] == 5
        assert result["has_new"] is True
        assert result["new_count"] == 42
        assert result["unread_count"] == 42
        assert result["latest_message_id"] == "m1"

    def test_since_counts_new_including_read(self, wrapper):
        folder = {"id": "inbox-id", "displayName": "Inbox", "unreadItemCount": 7}
        page = {"value": [_msg("m1", "2026-09-23T10:00:00Z", is_read=True),
                          _msg("m2", "2026-09-23T11:00:00Z")]}
        with patch("requests.get", side_effect=[_resp(json_body=folder), _resp(json_body=page)]):
            result = wrapper.check_new_messages(since="2026-09-23T09:00:00Z")
        assert result["new_count"] == 2
        assert result["unread_count"] == 1
        assert result["folder_unread_count"] == 7
        assert result["watermark"] == "2026-09-23T11:00:00Z"

    def test_nothing_new_keeps_watermark(self, wrapper):
        folder = {"id": "inbox-id", "displayName": "Inbox", "unreadItemCount": 0}
        with patch("requests.get", side_effect=[_resp(json_body=folder), _resp(json_body={"value": []})]):
            result = wrapper.check_new_messages(since="2026-09-23T09:00:00Z")
        assert result["has_new"] is False
        assert result["watermark"] == "2026-09-23T09:00:00Z"


class TestGetThreadMessages:
    def test_thread_sorted_and_new_replies_counted(self, wrapper):
        ref = {"id": "sent-1", "conversationId": "conv-1", "receivedDateTime": "2026-09-23T10:00:00Z"}
        thread = {"value": [
            _msg("reply-2", "2026-09-23T13:00:00Z", sender="bob@example.com"),
            _msg("sent-1", "2026-09-23T10:00:00Z", sender="me@example.com", folder="sent-id", is_read=True),
            _msg("reply-1", "2026-09-23T12:00:00Z", sender="alice@example.com"),
            _msg("my-followup", "2026-09-23T12:30:00Z", sender="me@example.com", folder="sent-id", is_read=True),
        ]}
        sent_folder = {"id": "sent-id"}
        with patch("requests.get", side_effect=[
            _resp(json_body=ref), _resp(json_body=thread), _resp(json_body=sent_folder),
        ]) as get:
            result = wrapper.get_thread_messages(after_message_id="sent-1")
        thread_call = get.call_args_list[1]
        assert thread_call.args[0] == f"{GRAPH}/messages"
        assert thread_call.kwargs["params"]["$filter"] == "conversationId eq 'conv-1'"
        assert "$orderby" not in thread_call.kwargs["params"]
        assert [m["id"] for m in result["messages"]] == ["sent-1", "reply-1", "my-followup", "reply-2"]
        assert result["new_count"] == 3
        assert result["new_replies_count"] == 2
        assert result["latest_message_id"] == "reply-2"
        from_me = {m["id"]: m["from_me"] for m in result["messages"]}
        assert from_me["sent-1"] and from_me["my-followup"] and not from_me["reply-1"]

    def test_only_new_by_conversation_id(self, wrapper):
        thread = {"value": [_msg("a", "2026-09-23T10:00:00Z"), _msg("b", "2026-09-23T12:00:00Z")]}
        with patch("requests.get", side_effect=[_resp(json_body=thread), _resp(json_body={"id": "sent-id"})]):
            result = wrapper.get_thread_messages(conversation_id="conv-1", since="2026-09-23T11:00:00Z",
                                                 only_new=True)
        assert [m["id"] for m in result["messages"]] == ["b"]
        assert result["total_in_thread"] == 2

    def test_requires_identifier(self, wrapper):
        with pytest.raises(ToolException, match="Provide"):
            wrapper.get_thread_messages()


class TestSentMessageIds:
    def test_send_mail_returns_draft_ids(self, wrapper):
        draft = {"id": "imm-1", "conversationId": "conv-9", "internetMessageId": "<x@example.com>",
                 "subject": "Hi", "toRecipients": [{"emailAddress": {"address": "bob@example.com"}}]}
        with patch("requests.post", side_effect=[_resp(201, json_body=draft), _resp(202)]) as post:
            result = wrapper.send_mail(to=["bob@example.com"], subject="Hi", body="Hello")
        assert post.call_args_list[0].args[0] == f"{GRAPH}/messages"
        assert post.call_args_list[1].args[0] == f"{GRAPH}/messages/imm-1/send"
        assert result == {
            "status": "sent", "message_id": "imm-1", "conversation_id": "conv-9",
            "internet_message_id": "<x@example.com>", "subject": "Hi", "to": ["bob@example.com"],
        }

    def test_send_mail_falls_back_to_sendmail_and_looks_up_sent_items(self, wrapper):
        sent = {"value": [{"id": "sent-7", "conversationId": "conv-7", "subject": "Hi",
                           "sentDateTime": "2999-01-01T00:00:00Z", "toRecipients": []}]}
        with patch("requests.post", side_effect=[_resp(403, text="AccessDenied"), _resp(202)]) as post, \
                patch("requests.get", return_value=_resp(json_body=sent)), \
                patch("elitea_sdk.tools.outlook.graph_wrapper.time.sleep"):
            result = wrapper.send_mail(to=["bob@example.com"], subject="Hi", body="Hello")
        assert post.call_args_list[1].args[0] == f"{GRAPH}/sendMail"
        assert result["message_id"] == "sent-7"
        assert result["conversation_id"] == "conv-7"

    def test_fallback_without_sent_copy_does_not_fail(self, wrapper):
        with patch("requests.post", side_effect=[_resp(403), _resp(202)]), \
                patch("requests.get", return_value=_resp(json_body={"value": []})), \
                patch("elitea_sdk.tools.outlook.graph_wrapper.time.sleep"):
            result = wrapper.send_mail(to=["bob@example.com"], subject="Hi", body="Hello")
        assert result["status"] == "sent"
        assert result["message_id"] is None
        assert "Do not resend" in result["note"]

    def test_failed_send_deletes_draft(self, wrapper):
        draft = {"id": "imm-1", "conversationId": "conv-9"}
        with patch("requests.post", side_effect=[_resp(201, json_body=draft), _resp(400, text="bad")]), \
                patch("requests.delete", return_value=_resp(204)) as delete:
            with pytest.raises(ToolException):
                wrapper.send_mail(to=["bob@example.com"], subject="Hi", body="Hello")
        assert delete.call_args.args[0] == f"{GRAPH}/messages/imm-1"

    def test_reply_returns_ids(self, wrapper):
        draft = {"id": "reply-imm", "conversationId": "conv-1", "subject": "RE: Hi", "toRecipients": []}
        with patch("requests.post", side_effect=[_resp(201, json_body=draft), _resp(202)]) as post:
            result = wrapper.reply_to_message(message_id="orig", body="Thanks", reply_all=True)
        assert post.call_args_list[0].args[0] == f"{GRAPH}/messages/orig/createReplyAll"
        assert post.call_args_list[0].kwargs["json"] == {"comment": "Thanks"}
        assert result["message_id"] == "reply-imm"
        assert result["conversation_id"] == "conv-1"
        assert result["in_reply_to"] == "orig"


class TestToolRegistration:
    def test_new_tools_are_exposed(self):
        names = [t["name"] for t in OutlookApiWrapper.model_construct().get_available_tools()]
        for name in ("check_new_messages", "find_new_messages", "get_thread_messages"):
            assert name in names

    def test_run_dispatches_to_backend(self):
        api = OutlookApiWrapper(token=FAKE_TOKEN, scopes=["Mail.Read"])
        with patch.object(OutlookGraphWrapper, "find_new_messages", return_value={"count": 0}) as find:
            assert api.run("find_new_messages", senders=["a@example.com"]) == {"count": 0}
        assert find.call_args.kwargs["senders"] == ["a@example.com"]
