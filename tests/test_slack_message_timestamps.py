"""Tests for Slack message timestamp exposure and threaded replies.

Covers fix for issue #6076: read_messages dropped the 'ts' field, which blocked
agents from filtering messages by time or replying inside a thread.
"""

import pytest
from unittest.mock import MagicMock
from slack_sdk.errors import SlackApiError

from elitea_sdk.tools.slack.api_wrapper import SlackApiWrapper, SendMessageModel


@pytest.fixture
def wrapper():
    """Create a SlackApiWrapper bypassing Pydantic validation, with a mocked WebClient."""
    w = SlackApiWrapper.model_construct(channel_id="C12345678")
    w._client = MagicMock()
    return w


@pytest.fixture
def history_payload():
    """Shape of a conversations.history response, per Slack's documented message schema."""
    return [
        {
            "type": "message",
            "user": "U012AB3CDE",
            "text": "I find you punny and would like to smell your nose letter",
            "ts": "1512085950.000216",
        },
        {
            "type": "message",
            "user": "U061F7AUR",
            "text": "So, what is the meaning of life?",
            "ts": "1512104434.000490",
            "thread_ts": "1512085950.000216",
            "parent_user_id": "U012AB3CDE",
        },
        {
            "type": "message",
            "subtype": "bot_message",
            "text": "posted by an app",
            "ts": "1512104900.000100",
            "bot_id": "B123",
            "bot_profile": {"name": "MyApp"},
        },
    ]


class TestExtractSlackMessages:
    """The projection must keep the fields agents need to address messages."""

    def test_ts_present_on_every_message(self, wrapper, history_payload):
        extracted = wrapper.extract_slack_messages(history_payload)
        assert [m["ts"] for m in extracted] == [
            "1512085950.000216",
            "1512104434.000490",
            "1512104900.000100",
        ]

    def test_thread_ts_preserved_for_threaded_message(self, wrapper, history_payload):
        extracted = wrapper.extract_slack_messages(history_payload)
        assert extracted[1]["thread_ts"] == "1512085950.000216"

    def test_thread_ts_omitted_for_top_level_messages(self, wrapper, history_payload):
        extracted = wrapper.extract_slack_messages(history_payload)
        assert "thread_ts" not in extracted[0]
        assert "thread_ts" not in extracted[2]

    def test_existing_fields_unchanged(self, wrapper, history_payload):
        extracted = wrapper.extract_slack_messages(history_payload)
        assert extracted[0]["user"] == "U012AB3CDE"
        assert extracted[0]["message"] == "I find you punny and would like to smell your nose letter"
        assert extracted[2]["app_name"] == "MyApp"

    def test_defaults_for_absent_fields(self, wrapper):
        extracted = wrapper.extract_slack_messages([{"ts": "1512104900.000100"}])
        assert extracted[0] == {
            "ts": "1512104900.000100",
            "user": "Undefined User",
            "message": "No message",
            "app_name": "No App Name",
        }

    def test_null_bot_profile_does_not_raise(self, wrapper):
        extracted = wrapper.extract_slack_messages([{"ts": "1.0", "bot_profile": None}])
        assert extracted[0]["app_name"] == "No App Name"

    def test_blocks_and_attachments_are_dropped(self, wrapper):
        """Raw Slack payloads are heavy — the projection keeps agent context small."""
        extracted = wrapper.extract_slack_messages(
            [{"ts": "1.0", "text": "hi", "blocks": [{"type": "rich_text"}], "attachments": [{}]}]
        )
        assert "blocks" not in extracted[0]
        assert "attachments" not in extracted[0]


class TestReadMessages:
    def test_returns_ts_end_to_end(self, wrapper, history_payload):
        wrapper._client.conversations_history.return_value = {"messages": history_payload}

        messages = wrapper.read_messages(limit=10)

        assert all("ts" in m for m in messages)
        wrapper._client.conversations_history.assert_called_once_with(
            channel="C12345678", limit=10
        )


class TestSendMessageThreadReply:
    def test_thread_ts_forwarded_to_slack(self, wrapper):
        wrapper.send_message(message="a reply", thread_ts="1512085950.000216")

        wrapper._client.chat_postMessage.assert_called_once_with(
            channel="C12345678", text="a reply", thread_ts="1512085950.000216"
        )

    def test_omitting_thread_ts_posts_top_level(self, wrapper):
        """slack_sdk strips None params, so the call stays equivalent to the pre-fix behaviour."""
        wrapper.send_message(message="hello")

        wrapper._client.chat_postMessage.assert_called_once_with(
            channel="C12345678", text="hello", thread_ts=None
        )

    def test_returns_ts_of_posted_message(self, wrapper):
        """Without this an agent cannot thread onto a message it just posted."""
        wrapper._client.chat_postMessage.return_value = {
            "ok": True, "channel": "C12345678", "ts": "1512200000.000999"
        }

        result = wrapper.send_message(message="hello")

        assert result["ts"] == "1512200000.000999"
        assert result["channel_id"] == "C12345678"
        assert result["success"] is True

    def test_thread_reply_echoes_thread_ts(self, wrapper):
        wrapper._client.chat_postMessage.return_value = {
            "ok": True, "channel": "C12345678", "ts": "1512200000.000999"
        }

        result = wrapper.send_message(message="a reply", thread_ts="1512085950.000216")

        assert result["thread_ts"] == "1512085950.000216"

    def test_top_level_send_has_no_thread_ts_key(self, wrapper):
        wrapper._client.chat_postMessage.return_value = {
            "ok": True, "channel": "C12345678", "ts": "1512200000.000999"
        }

        assert "thread_ts" not in wrapper.send_message(message="hello")

    def test_thread_ts_exposed_in_args_schema(self):
        """Agents can only pass thread_ts if the tool's schema advertises it."""
        fields = SendMessageModel.model_fields
        assert "thread_ts" in fields
        assert fields["thread_ts"].default is None


class TestApiErrorHandling:
    """The credential-failure path must report the Slack error, not blow up formatting it."""

    def test_invalid_auth_returns_error_string(self, wrapper):
        wrapper._client.auth_test.side_effect = SlackApiError(
            "invalid_auth", {"error": "invalid_auth"}
        )

        assert "invalid_auth" in wrapper.read_messages(limit=10)

    def test_history_error_returns_error_string(self, wrapper):
        wrapper._client.conversations_history.side_effect = SlackApiError(
            "channel_not_found", {"error": "channel_not_found"}
        )

        assert "channel_not_found" in wrapper.read_messages(limit=10)

    def test_send_message_error_matches_success_shape(self, wrapper):
        """send_message succeeds with a dict, so its failure must be a dict too."""
        wrapper._client.chat_postMessage.side_effect = SlackApiError(
            "not_in_channel", {"error": "not_in_channel"}
        )

        result = wrapper.send_message(message="hi")

        assert result == {"success": False, "error": "not_in_channel"}

