"""Pydantic models for Teams toolkit input schemas."""

from typing import List, Literal, Optional
from pydantic import Field, create_model

_SENDERS = (
    "Only messages from these people: emails, Azure AD user IDs or exact display names. "
    "Omit for messages from anyone"
)
_SINCE = (
    "ISO 8601 date/time (e.g. 2026-09-23T10:15:00Z). Messages created after it count as new; "
    "pass the previous call's watermark here"
)
_AFTER_MESSAGE_ID = (
    "ID of a message already processed; messages created after it count as new. "
    "Pass the previous call's latest_message_id here. If since is also given, the later point wins"
)
_TEXT_CONTAINS = "Only messages whose text contains this substring (case-insensitive)"
_INCLUDE_OWN = "Include messages sent by the signed-in user"
_LOOKBACK = "When neither since nor after_message_id is given, how many hours back to look"
_HTML = "If True, message is HTML (<b>, <i>, <a>, <br>, ...); otherwise plain text"
_MENTIONS = (
    "People to @mention, as emails/UPNs or Azure AD user IDs (display names are not accepted). "
    "Each becomes a real @mention that notifies the person. To place a mention inside the text, write "
    "@[email] where it should appear, e.g. message='Thanks @[anna@contoso.com], please review by 3pm' "
    "posts 'Thanks @Anna Kowalski, please review by 3pm'. People in mentions that are not placed inline "
    "are tagged at the start of the message. Do not type a plain '@name': it is just text and notifies no one"
)
_IMPORTANCE = "Message importance"
_TEAM = "Team ID or exact team display name (see list_teams)"
_CHANNEL = "Channel ID (19:...@thread.tacv2) or exact channel display name (see list_channels)"
_INCLUDE_LINKS = "Include web links (webUrl / webLink / attachment contentUrl); omitted by default to save space"
_TEXT_MAX_CHARS = (
    "Maximum characters of each message text (default 1000, 0 = up to 4000). "
    "Cut messages are flagged with text_truncated"
)


ListTeams = create_model(
    "ListTeams",
    name_contains=(Optional[str], Field(default=None, description="Only teams whose name contains this text")),
)

ListChannels = create_model(
    "ListChannels",
    team=(str, Field(description=_TEAM)),
    name_contains=(Optional[str], Field(default=None, description="Only channels whose name contains this text")),
    limit=(int, Field(default=50, ge=1, le=500, description="Maximum number of channels to return; the response says when results were truncated")),
    include_links=(bool, Field(default=False, description=_INCLUDE_LINKS)),
)

ListChats = create_model(
    "ListChats",
    chat_type=(Optional[Literal["oneOnOne", "group", "meeting"]],
               Field(default=None, description="Only chats of this type")),
    topic_contains=(Optional[str], Field(default=None, description="Only chats whose topic contains this text")),
    member=(Optional[str], Field(default=None, description="Only chats with this member (email or display name)")),
    unread_only=(bool, Field(default=False, description="Only chats with unread messages")),
    limit=(int, Field(default=50, ge=1, le=500, description="Maximum number of chats to return")),
    members=(Literal["none", "summary", "all"], Field(default="summary", description=(
        "How many chat members to list: 'summary' (default) = the first max_members people other than you, "
        "'all' = every member (can be very large for group/meeting chats), 'none' = only member_count"
    ))),
    max_members=(int, Field(default=5, ge=0, le=100, description=(
        "With members='summary', how many members to list per chat; members_truncated is set when more exist"
    ))),
    include_ids=(bool, Field(default=False, description="Include member user IDs and the last message ID")),
    include_links=(bool, Field(default=False, description=_INCLUDE_LINKS)),
    last_message_chars=(int, Field(default=300, ge=0, le=4000, description="Maximum characters of each chat's last message preview")),
)

FindChatMessages = create_model(
    "FindChatMessages",
    chat=(str, Field(description=(
        "Chat ID (19:...), exact group chat topic, or a person's email for your 1:1 chat with them"
    ))),
    senders=(Optional[List[str]], Field(default=None, description=_SENDERS)),
    since=(Optional[str], Field(default=None, description=_SINCE)),
    after_message_id=(Optional[str], Field(default=None, description=_AFTER_MESSAGE_ID)),
    text_contains=(Optional[str], Field(default=None, description=_TEXT_CONTAINS)),
    only_new=(bool, Field(default=True, description=(
        "Only return new messages. When neither since nor after_message_id is given, new means unread"
    ))),
    include_own=(bool, Field(default=False, description=_INCLUDE_OWN)),
    lookback_hours=(int, Field(default=24, ge=1, le=24 * 90, description=_LOOKBACK)),
    limit=(int, Field(default=50, ge=1, le=500, description="Maximum number of messages to return")),
    text_max_chars=(int, Field(default=1000, ge=0, le=4000, description=_TEXT_MAX_CHARS)),
    include_links=(bool, Field(default=False, description=_INCLUDE_LINKS)),
)

SearchTeamsMessages = create_model(
    "SearchTeamsMessages",
    query=(Optional[str], Field(default=None, description=(
        "KQL query over all your chats and channels, e.g. 'release AND deploy', 'IsMentioned:true', "
        "'hasAttachment:true'"
    ))),
    senders=(Optional[List[str]], Field(default=None, description="Only messages from these people (emails or names)")),
    since=(Optional[str], Field(default=None, description="Only messages created after this ISO 8601 date/time")),
    limit=(int, Field(default=25, ge=1, le=200, description="Maximum number of messages to return")),
    include_links=(bool, Field(default=False, description=_INCLUDE_LINKS)),
)

SendChatMessage = create_model(
    "SendChatMessage",
    message=(str, Field(description="Message text")),
    chat=(Optional[str], Field(default=None, description=(
        "Existing chat: chat ID (19:...), exact group chat topic, or a person's email. Omit when using recipients"
    ))),
    recipients=(Optional[List[str]], Field(default=None, description=(
        "People to message (emails). One person: your 1:1 chat. Several: the group chat with exactly these "
        "members (created if it does not exist)"
    ))),
    topic=(Optional[str], Field(default=None, description="Group chat name, used when a new group chat is created")),
    html=(bool, Field(default=False, description=_HTML)),
    mentions=(Optional[List[str]], Field(default=None, description=_MENTIONS)),
    importance=(Optional[Literal["normal", "high", "urgent"]], Field(default=None, description=_IMPORTANCE)),
)

SendChannelMessage = create_model(
    "SendChannelMessage",
    team=(str, Field(description=_TEAM)),
    channel=(str, Field(description=_CHANNEL)),
    message=(str, Field(description="Message text")),
    subject=(Optional[str], Field(default=None, description="Subject of a new post (ignored for replies)")),
    reply_to_message_id=(Optional[str], Field(default=None, description=(
        "ID of the root post to reply to; omit to start a new post"
    ))),
    html=(bool, Field(default=False, description=_HTML)),
    mentions=(Optional[List[str]], Field(default=None, description=_MENTIONS)),
    importance=(Optional[Literal["normal", "high", "urgent"]], Field(default=None, description=_IMPORTANCE)),
)
