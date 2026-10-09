"""Pydantic models for Teams toolkit input schemas."""

from typing import List, Literal, Optional
from pydantic import Field, create_model

_SENDERS = (
    "Only messages from these people: emails, Azure AD user IDs or exact display names. "
    "Omit for messages from anyone. Example: ['anna@contoso.com', 'Ben Miller']"
)
_SINCE = (
    "ISO 8601 date/time (e.g. 2026-09-23T10:15:00Z). Messages created after it count as new; "
    "pass the previous call's watermark here"
)
_AFTER_MESSAGE_ID = (
    "ID of a message already processed; messages created after it count as new. "
    "Pass the previous call's latest_message_id here (e.g. '1727085300123'). "
    "If since is also given, the later point wins"
)
_TEXT_CONTAINS = "Only messages whose text contains this substring (case-insensitive). Example: 'release notes'"
_INCLUDE_OWN = "Include messages sent by the signed-in user (default False: only messages from others)"
_LOOKBACK = (
    "When neither since nor after_message_id is given, how many hours back to look. "
    "Example: 72 = the last three days"
)
_HTML = (
    "If True, message is HTML (<b>, <i>, <a>, <br>, ...); otherwise plain text. "
    "Example: html=True with message='<b>Deploy done</b><br>See <a href=\"https://wiki/x\">notes</a>'"
)
_MENTIONS = (
    "People to @mention, as emails/UPNs or Azure AD user IDs (display names are not accepted). "
    "Each becomes a real @mention that notifies the person. To place a mention inside the text, write "
    "@[email] where it should appear, e.g. message='Thanks @[anna@contoso.com], please review by 3pm' "
    "posts 'Thanks @Anna Kowalski, please review by 3pm'. People in mentions that are not placed inline "
    "are tagged at the start of the message. Do not type a plain '@name': it is just text and notifies no one. "
    "Example: mentions=['anna@contoso.com', 'ben@contoso.com'] with message='Please review' "
    "posts '@Anna Kowalski @Ben Miller Please review'"
)
_IMPORTANCE = (
    "Message priority. Omit (or 'normal') for an ordinary message. 'high' marks it with a red '!' Important "
    "label. 'urgent' marks it Urgent and Teams re-notifies each recipient every 2 minutes for 20 minutes "
    "(a chat feature; may not apply to channel posts). Use 'high' or 'urgent' only when the user asks for "
    "it or the content is time-critical. Example: importance='high' with message='Prod deploy starts at 14:00'"
)
_TEAM = "Team ID or exact team display name (see list_teams). Example: 'Engineering'"
_CHANNEL = (
    "Channel ID (19:...@thread.tacv2) or exact channel display name (see list_channels). "
    "Example: 'General'"
)
_INCLUDE_LINKS = (
    "Include web links (webUrl / webLink / attachment contentUrl); omitted by default to save space. "
    "Example: include_links=True when the user wants a link to open in Teams"
)
_TEXT_MAX_CHARS = (
    "Maximum characters of each message text (default 1000, 0 = up to 4000). "
    "Cut messages are flagged with text_truncated. Example: 4000 to read long announcements in full"
)


ListTeams = create_model(
    "ListTeams",
    name_contains=(Optional[str], Field(
        default=None, description="Only teams whose name contains this text. Example: 'eng'",
    )),
)

ListChannels = create_model(
    "ListChannels",
    team=(str, Field(description=_TEAM)),
    name_contains=(Optional[str], Field(
        default=None, description="Only channels whose name contains this text. Example: 'release'",
    )),
    limit=(int, Field(default=50, ge=1, le=500, description=(
        "Maximum number of channels to return; the response says when results were truncated. Example: 10"
    ))),
    include_links=(bool, Field(default=False, description=_INCLUDE_LINKS)),
)

ListChats = create_model(
    "ListChats",
    chat_type=(Optional[Literal["oneOnOne", "group", "meeting"]], Field(
        default=None,
        description="Only chats of this type. Example: 'group' for group chats, 'oneOnOne' for direct chats",
    )),
    topic_contains=(Optional[str], Field(
        default=None,
        description="Only chats whose topic contains this text (1:1 chats have no topic). Example: 'launch'",
    )),
    member=(Optional[str], Field(
        default=None,
        description="Only chats with this member (email or display name). Example: 'anna@contoso.com'",
    )),
    unread_only=(bool, Field(default=False, description="Only chats with unread messages")),
    limit=(int, Field(default=50, ge=1, le=500, description=(
        "Maximum number of chats to return. Response time grows with it (about 0.4 s per chat when no filter "
        "is set). Example: 10 for a quick look at the latest chats"
    ))),
    members=(Literal["none", "summary", "all"], Field(default="summary", description=(
        "How many chat members to list: 'summary' (default) = the first max_members people other than you, "
        "'all' = every member (can be very large for group/meeting chats), 'none' = only member_count. "
        "Example: 'none' when only chat names and last messages matter"
    ))),
    max_members=(int, Field(default=5, ge=0, le=100, description=(
        "With members='summary', how many members to list per chat; members_truncated is set when more exist. "
        "Example: 10"
    ))),
    include_ids=(bool, Field(default=False, description=(
        "Include member user IDs and the last message ID. Example: True when the IDs are needed for mentions "
        "or for after_message_id"
    ))),
    include_links=(bool, Field(default=False, description=_INCLUDE_LINKS)),
    last_message_chars=(int, Field(default=300, ge=0, le=4000, description=(
        "Maximum characters of each chat's last message preview (0 = no preview). Example: 1000"
    ))),
)

FindChatMessages = create_model(
    "FindChatMessages",
    chat=(str, Field(description=(
        "Chat ID (19:...), exact group chat topic, or a person's email for your 1:1 chat with them. "
        "Examples: '19:abc...@thread.v2', 'Launch team', 'anna@contoso.com'"
    ))),
    senders=(Optional[List[str]], Field(default=None, description=_SENDERS)),
    since=(Optional[str], Field(default=None, description=_SINCE)),
    after_message_id=(Optional[str], Field(default=None, description=_AFTER_MESSAGE_ID)),
    text_contains=(Optional[str], Field(default=None, description=_TEXT_CONTAINS)),
    only_new=(bool, Field(default=True, description=(
        "Only return new messages. When neither since nor after_message_id is given, new means unread. "
        "Example: False to read the chat history of the last lookback_hours regardless of read state"
    ))),
    include_own=(bool, Field(default=False, description=_INCLUDE_OWN)),
    lookback_hours=(int, Field(default=24, ge=1, le=24 * 90, description=_LOOKBACK)),
    limit=(int, Field(default=50, ge=1, le=500, description="Maximum number of messages to return. Example: 20")),
    text_max_chars=(int, Field(default=1000, ge=0, le=4000, description=_TEXT_MAX_CHARS)),
    include_links=(bool, Field(default=False, description=_INCLUDE_LINKS)),
)

SearchTeamsMessages = create_model(
    "SearchTeamsMessages",
    query=(Optional[str], Field(default=None, description=(
        "KQL query over all your chats and channels, e.g. 'release AND deploy', 'IsMentioned:true', "
        "'hasAttachment:true'"
    ))),
    senders=(Optional[List[str]], Field(
        default=None,
        description="Only messages from these people (emails or names). Example: ['anna@contoso.com']",
    )),
    since=(Optional[str], Field(
        default=None,
        description="Only messages created after this ISO 8601 date/time. Example: '2026-09-23T00:00:00Z'",
    )),
    limit=(int, Field(default=25, ge=1, le=200, description="Maximum number of messages to return. Example: 10")),
    include_links=(bool, Field(default=False, description=_INCLUDE_LINKS)),
)

SendChatMessage = create_model(
    "SendChatMessage",
    message=(str, Field(description=(
        "Message text; use @[email] to mention someone in place. "
        "Example: 'Thanks @[anna@contoso.com], the build is green'"
    ))),
    chat=(Optional[str], Field(default=None, description=(
        "Existing chat: chat ID (19:...), exact group chat topic, or a person's email. Omit when using "
        "recipients. Examples: '19:abc...@thread.v2', 'Launch team', 'anna@contoso.com'"
    ))),
    recipients=(Optional[List[str]], Field(default=None, description=(
        "People to message (emails). One person: your 1:1 chat. Several: the group chat with exactly these "
        "members (created if it does not exist). Only chooses the chat, it does not @mention anyone. "
        "Example: ['anna@contoso.com', 'ben@contoso.com']"
    ))),
    topic=(Optional[str], Field(default=None, description=(
        "Group chat name, used when a new group chat is created. Example: 'Release 4.2 coordination'"
    ))),
    html=(bool, Field(default=False, description=_HTML)),
    mentions=(Optional[List[str]], Field(default=None, description=_MENTIONS)),
    importance=(Optional[Literal["normal", "high", "urgent"]], Field(default=None, description=_IMPORTANCE)),
)

SendChannelMessage = create_model(
    "SendChannelMessage",
    team=(str, Field(description=_TEAM)),
    channel=(str, Field(description=_CHANNEL)),
    message=(str, Field(description=(
        "Message text; use @[email] to mention someone in place. "
        "Example: 'Release is out, @[anna@contoso.com] please verify'"
    ))),
    subject=(Optional[str], Field(default=None, description=(
        "Subject of a new post (ignored for replies). Example: 'Release 4.2 is live'"
    ))),
    reply_to_message_id=(Optional[str], Field(default=None, description=(
        "ID of the root post to reply to (the message_id returned when the post was created); omit to start "
        "a new post. Example: '1727085300123'"
    ))),
    html=(bool, Field(default=False, description=_HTML)),
    mentions=(Optional[List[str]], Field(default=None, description=_MENTIONS)),
    importance=(Optional[Literal["normal", "high", "urgent"]], Field(default=None, description=_IMPORTANCE)),
)
