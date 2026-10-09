"""Tests for the compact-by-default Outlook output (recipient caps, body caps, opt-in ids)."""

from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException

from .test_new_message_detection import GRAPH, _msg, _resp, wrapper  # noqa: F401


def _mass_mail(msg_id="m1", n_to=40, n_cc=12, preview="p" * 255):
    msg = _msg(msg_id, "2026-09-23T10:00:00Z",
               to=[f"to{i}@example.com" for i in range(n_to)],
               cc=[f"cc{i}@example.com" for i in range(n_cc)])
    msg["bodyPreview"] = preview
    msg["hasAttachments"] = False
    return msg


class TestListMessagesCompact:
    def test_recipients_capped_and_counted(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": [_mass_mail()]})):
            (item,) = wrapper.list_messages()
        assert len(item["to"]) == 5 and item["to_more"] == 35
        assert len(item["cc"]) == 5 and item["cc_more"] == 7
        assert "conversationId" not in item

    def test_small_recipient_lists_have_no_more_counter(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": [_mass_mail(n_to=2, n_cc=0)]})):
            (item,) = wrapper.list_messages()
        assert len(item["to"]) == 2 and item["cc"] == []
        assert "to_more" not in item and "cc_more" not in item

    def test_preview_capped_by_default_and_configurable(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": [_mass_mail()]})):
            assert len(wrapper.list_messages()[0]["bodyPreview"]) == 200
            assert len(wrapper.list_messages(preview_chars=50)[0]["bodyPreview"]) == 50

    def test_include_preview_false_does_not_request_it(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": [_msg("m1", "2026-09-23T10:00:00Z")]})) as get:
            (item,) = wrapper.list_messages(include_preview=False)
        assert "bodyPreview" not in get.call_args.kwargs["params"]["$select"]
        assert "bodyPreview" not in item

    def test_preview_is_requested_by_default(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": []})) as get:
            wrapper.list_messages()
        assert "bodyPreview" in get.call_args.kwargs["params"]["$select"]

    def test_max_recipients_and_include_ids(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": [_mass_mail()]})):
            (item,) = wrapper.list_messages(max_recipients=20, include_ids=True)
        assert len(item["to"]) == 20 and item["to_more"] == 20
        assert len(item["cc"]) == 12 and "cc_more" not in item
        assert item["conversationId"] == "conv-1"

    def test_search_path_is_compact_too(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": [_mass_mail()]})) as get:
            (item,) = wrapper.search_messages(query="budget")
        assert "bodyPreview" in get.call_args.kwargs["params"]["$select"]
        assert item["to_more"] == 35


class TestFindAndCheckCompact:
    def test_find_new_messages_is_compact(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": [_mass_mail()]})):
            result = wrapper.find_new_messages(since="2026-09-23T09:00:00Z")
        item = result["messages"][0]
        assert item["to_more"] == 35 and len(item["bodyPreview"]) == 200
        assert "conversationId" not in item and "parentFolderId" not in item
        assert item["is_new"] is True

    def test_find_new_messages_options(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body={"value": [_mass_mail()]})):
            result = wrapper.find_new_messages(since="2026-09-23T09:00:00Z", preview_chars=30,
                                               max_recipients=1, include_ids=True)
        item = result["messages"][0]
        assert len(item["bodyPreview"]) == 30 and len(item["to"]) == 1 and item["to_more"] == 39
        assert item["conversationId"] == "conv-1"

    def test_check_new_messages_latest_is_brief(self, wrapper):
        folder = {"id": "inbox-id", "displayName": "Inbox", "unreadItemCount": 1}
        with patch("requests.get", side_effect=[_resp(json_body=folder), _resp(json_body={"value": [_mass_mail()]})]):
            result = wrapper.check_new_messages()
        (latest,) = result["latest"]
        assert set(latest) == {"id", "subject", "from", "receivedDateTime", "isRead"}
        assert result["latest_message_id"] == "m1"

    def test_check_new_messages_latest_limit(self, wrapper):
        folder = {"id": "inbox-id", "displayName": "Inbox", "unreadItemCount": 3}
        page = {"value": [_msg(f"m{i}", f"2026-09-23T1{i}:00:00Z") for i in range(3)]}
        with patch("requests.get", side_effect=[_resp(json_body=folder), _resp(json_body=page)]):
            result = wrapper.check_new_messages(since="2026-09-23T09:00:00Z", latest_limit=2)
        assert [m["id"] for m in result["latest"]] == ["m2", "m1"]
        assert result["new_count"] == 3


class TestGetMessageCompact:
    def _data(self, content, content_type="text"):
        data = _mass_mail()
        data["body"] = {"contentType": content_type, "content": content}
        data["importance"] = "normal"
        return data

    def test_text_body_requested_via_prefer_header(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body=self._data("hello"))) as get:
            result = wrapper.get_message("m1")
        assert 'outlook.body-content-type="text"' in get.call_args.kwargs["headers"]["Prefer"]
        assert result["body"] == "hello" and result["bodyContentType"] == "text"
        assert "body_truncated" not in result

    def test_html_format_skips_text_preference(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body=self._data("<b>hi</b>", "html"))) as get:
            result = wrapper.get_message("m1", body_format="html")
        assert "body-content-type" not in get.call_args.kwargs["headers"]["Prefer"]
        assert result["bodyContentType"] == "html"

    def test_long_body_capped_and_flagged(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body=self._data("x" * 20000))):
            result = wrapper.get_message("m1")
        assert len(result["body"]) == 8000
        assert result["body_truncated"] is True and result["body_total_chars"] == 20000

    def test_max_body_chars_custom_and_unlimited(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body=self._data("x" * 20000))):
            assert len(wrapper.get_message("m1", max_body_chars=100)["body"]) == 100
            full = wrapper.get_message("m1", max_body_chars=0)
        assert len(full["body"]) == 20000 and "body_truncated" not in full

    def test_recipients_capped(self, wrapper):
        with patch("requests.get", return_value=_resp(json_body=self._data("hi"))):
            result = wrapper.get_message("m1", max_recipients=3)
        assert len(result["to"]) == 3 and result["to_more"] == 37
        assert len(result["cc"]) == 3 and result["cc_more"] == 9

    def test_without_body_no_prefer_and_no_body_fields(self, wrapper):
        data = self._data("hi")
        del data["body"]
        with patch("requests.get", return_value=_resp(json_body=data)) as get:
            result = wrapper.get_message("m1", include_body=False)
        assert "body-content-type" not in get.call_args.kwargs["headers"]["Prefer"]
        assert "body" not in result and "body" not in get.call_args.kwargs["params"]["$select"].split(",")

    def test_invalid_body_format_rejected(self, wrapper):
        with pytest.raises(ToolException, match="body_format"):
            wrapper.get_message("m1", body_format="rtf")


class TestThreadCompact:
    def _thread(self, body):
        msgs = [_mass_mail("a"), _mass_mail("b")]
        for m in msgs:
            m["uniqueBody"] = {"content": body}
        return {"value": msgs}

    def test_bodies_capped_per_message(self, wrapper):
        with patch("requests.get", side_effect=[_resp(json_body=self._thread("y" * 5000)), _resp(json_body={"id": "s"})]):
            result = wrapper.get_thread_messages(conversation_id="conv-1", include_body=True)
        for item in result["messages"]:
            assert len(item["body"]) == 2000 and item["body_truncated"] is True
            assert item["to_more"] == 35
            assert "conversationId" not in item
        assert result["conversation_id"] == "conv-1"

    def test_short_bodies_untouched_and_custom_cap(self, wrapper):
        with patch("requests.get", side_effect=[_resp(json_body=self._thread("short")), _resp(json_body={"id": "s"})]):
            result = wrapper.get_thread_messages(conversation_id="conv-1", include_body=True)
        assert all(m["body"] == "short" and "body_truncated" not in m for m in result["messages"])
        with patch("requests.get", side_effect=[_resp(json_body=self._thread("y" * 500)), _resp(json_body={"id": "s"})]):
            result = wrapper.get_thread_messages(conversation_id="conv-1", include_body=True, max_body_chars=100)
        assert all(len(m["body"]) == 100 for m in result["messages"])


class TestToolSchemas:
    def test_new_params_exposed_and_passed_through(self):
        from elitea_sdk.tools.outlook.api_wrapper import OutlookApiWrapper
        api = OutlookApiWrapper(token="t", scopes=["Mail.Read"])
        schemas = {t["name"]: t["args_schema"] for t in api.get_available_tools()}
        assert {"include_preview", "preview_chars", "max_recipients", "include_ids"} <= set(schemas["list_messages"].model_fields)
        assert {"body_format", "max_body_chars", "max_recipients"} <= set(schemas["get_message"].model_fields)
        assert "latest_limit" in schemas["check_new_messages"].model_fields
        assert {"max_body_chars", "max_recipients"} <= set(schemas["get_thread_messages"].model_fields)
        with patch.object(api._backend, "get_message", return_value={}) as gm:
            api.run("get_message", message_id="m1", body_format="html", max_body_chars=10, max_recipients=2)
        gm.assert_called_once_with(message_id="m1", include_body=True, body_format="html",
                                   max_body_chars=10, max_recipients=2)
