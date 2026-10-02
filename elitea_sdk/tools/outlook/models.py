"""Pydantic models for Outlook toolkit input schemas."""

from typing import List, Literal, Optional
from pydantic import Field, create_model


_MAX_RECIPIENTS = (
    "Maximum To and Cc recipients listed per message (default 5); to_more / cc_more give the number left out. "
    "Example: 100 to see everyone on a mass mail"
)
_PREVIEW_CHARS = "Maximum characters of each body preview (default 200, Graph provides at most 255). Example: 80"
_INCLUDE_IDS = "Include conversationId of each message. Example: True when get_thread_messages will be called next"


def _max_recipients():
    return (int, Field(default=5, ge=0, le=100, description=_MAX_RECIPIENTS))


def _preview_chars():
    return (int, Field(default=200, ge=1, le=255, description=_PREVIEW_CHARS))


ListMessages = create_model(
    "ListMessages",
    folder=(str, Field(default="inbox", description="Well-known name (inbox, sentitems, drafts, etc.), folder path such as 'Inbox/Projects', folder ID from list_folders, or 'all' for the whole mailbox. Example: 'Inbox/Projects'")),
    limit=(int, Field(default=50, ge=1, le=1000, description="Maximum number of messages to return. Example: 10")),
    unread_only=(bool, Field(default=False, description="Only return unread messages")),
    search=(Optional[str], Field(default=None, description="KQL search query to filter messages. Example: 'from:anna@contoso.com subject:invoice'")),
    include_preview=(bool, Field(default=True, description="Include a short body preview of each message. Example: False for a subjects-only overview")),
    preview_chars=_preview_chars(),
    max_recipients=_max_recipients(),
    include_ids=(bool, Field(default=False, description=_INCLUDE_IDS)),
)

GetMessage = create_model(
    "GetMessage",
    message_id=(str, Field(description="The unique identifier of the message (the id from list_messages / find_new_messages). Example: 'AAMkAGI2...'")),
    include_body=(bool, Field(default=True, description="Include message body in response. Example: False to read only headers, recipients and flags")),
    body_format=(Literal["text", "html"], Field(default="text", description=(
        "'text' (default) returns a plain-text body, much smaller than 'html'; use 'html' only when markup matters, "
        "e.g. to copy a formatted table"
    ))),
    max_body_chars=(int, Field(default=8000, ge=0, le=500000, description=(
        "Maximum characters of the body (default 8000, 0 = no limit). A cut body sets body_truncated and body_total_chars. "
        "Example: 0 to read a long newsletter in full"
    ))),
    max_recipients=_max_recipients(),
)

SendMail = create_model(
    "SendMail",
    to=(List[str], Field(description="List of recipient email addresses. Example: ['anna@contoso.com', 'ben@contoso.com']")),
    subject=(str, Field(description="Email subject line. Example: 'Q3 report'")),
    body=(str, Field(description="Email body content. Example: 'Hi Anna,\\n\\nthe report is attached. Thanks!'")),
    cc=(Optional[List[str]], Field(default=None, description="List of CC recipient email addresses. Example: ['manager@contoso.com']")),
    bcc=(Optional[List[str]], Field(default=None, description="List of BCC recipient email addresses (hidden from other recipients). Example: ['archive@contoso.com']")),
    html=(bool, Field(default=False, description="If True, body is treated as HTML content. Example: True with body='<p>Hi <b>Anna</b></p>'")),
)

ReplyToMessage = create_model(
    "ReplyToMessage",
    message_id=(str, Field(description="The unique identifier of the message to reply to (id from list_messages / find_new_messages). Example: 'AAMkAGI2...'")),
    body=(str, Field(description="Reply body content. Example: 'Thanks, I will review it today.'")),
    reply_all=(bool, Field(default=False, description="Reply to all recipients (To and Cc) instead of only the sender. Example: True for a team thread")),
)

MarkAsRead = create_model(
    "MarkAsRead",
    message_id=(str, Field(description="The unique identifier of the message")),
    is_read=(bool, Field(default=True, description="Mark as read (True) or unread (False). Example: False to flag a message for later")),
)

ListFolders = create_model(
    "ListFolders",
    __doc__="List mail folders including nested subfolders",
    parent=(Optional[str], Field(default=None, description="Only list folders below this one: folder path such as 'Inbox/Projects', display name, folder ID or well-known name. Default: the whole mailbox. Example: 'Inbox'")),
    depth=(Optional[int], Field(default=None, ge=1, description="How many levels below the start to include; 1 = direct children only. Default: all levels. Example: 1")),
    name_contains=(Optional[str], Field(default=None, description="Case-insensitive text the folder path must contain. Example: 'project'")),
    sort_by=(Literal["path", "unread", "total"], Field(default="path", description="'path' = alphabetical tree order; 'unread' or 'total' = folders with the most unread/total messages first. Example: 'unread' to find folders needing attention")),
    limit=(int, Field(default=20, ge=1, le=1000, description="Maximum number of folders to return. The response says when results were truncated. Example: 50")),
)

MoveMessage = create_model(
    "MoveMessage",
    message_id=(str, Field(description="The unique identifier of the message to move (id from list_messages). Example: 'AAMkAGI2...'")),
    destination_folder=(str, Field(description="Destination folder: well-known name, folder path such as 'Inbox/Projects', or folder ID. Example: 'archive'")),
)

DeleteMessage = create_model(
    "DeleteMessage",
    message_id=(str, Field(description="The unique identifier of the message to delete (id from list_messages). Example: 'AAMkAGI2...'")),
    permanent=(bool, Field(default=False, description="If True, permanently delete (cannot be undone). If False, move to Deleted Items. Example: False unless the user explicitly asks to erase it")),
)

SearchMessages = create_model(
    "SearchMessages",
    query=(str, Field(description=(
        "KQL search query, e.g. 'from:john@example.com subject:report', 'to:team-dl@example.com', "
        "'received>=2026-09-01 budget'. Dates in KQL have day precision only; for exact "
        "'new since' checks use find_new_messages instead."
    ))),
    folder=(str, Field(default="inbox", description="Folder to search in: well-known name, folder path such as 'Inbox/Projects', folder ID, or 'all' for the whole mailbox. Example: 'all'")),
    limit=(int, Field(default=25, ge=1, le=1000, description="Maximum results to return. Example: 10")),
)

_SINCE_DESCRIPTION = (
    "ISO 8601 UTC timestamp (e.g. 2026-09-23T10:15:00Z). Messages received after it are 'new'. "
    "Pass the 'watermark' returned by a previous call. Example: '2026-09-23T10:15:00Z'"
)
_AFTER_MESSAGE_ID_DESCRIPTION = (
    "Message ID; messages received after this message are 'new'. Use an ID returned by a previous call "
    "(e.g. latest_message_id, or message_id returned by send_mail). If both since and after_message_id "
    "are given, the later point wins. Example: 'AAMkAGI2...'"
)
_SENDERS_DESCRIPTION = "Sender email addresses to match (From). Example: ['anna@contoso.com']"
_RECIPIENTS_DESCRIPTION = (
    "Recipient or distribution list email addresses to match (To or Cc). Example: ['team-dl@contoso.com']"
)

CheckNewMessages = create_model(
    "CheckNewMessages",
    folder=(str, Field(default="inbox", description="Well-known name (inbox, sentitems, etc.), folder path such as 'Inbox/Projects', folder ID, or 'all' for the whole mailbox")),
    since=(Optional[str], Field(default=None, description=_SINCE_DESCRIPTION + " If since and after_message_id are omitted, 'new' means unread.")),
    after_message_id=(Optional[str], Field(default=None, description=_AFTER_MESSAGE_ID_DESCRIPTION)),
    senders=(Optional[List[str]], Field(default=None, description=_SENDERS_DESCRIPTION)),
    recipients=(Optional[List[str]], Field(default=None, description=_RECIPIENTS_DESCRIPTION)),
    latest_limit=(int, Field(default=5, ge=0, le=50, description="How many of the latest messages (id, from, subject, time) to list; 0 = only the counts. Example: 3")),
)

FindNewMessages = create_model(
    "FindNewMessages",
    folder=(str, Field(default="inbox", description="Well-known name (inbox, sentitems, etc.), folder path such as 'Inbox/Projects', folder ID, or 'all' for the whole mailbox")),
    since=(Optional[str], Field(default=None, description=_SINCE_DESCRIPTION)),
    after_message_id=(Optional[str], Field(default=None, description=_AFTER_MESSAGE_ID_DESCRIPTION)),
    senders=(Optional[List[str]], Field(default=None, description=_SENDERS_DESCRIPTION)),
    recipients=(Optional[List[str]], Field(default=None, description=_RECIPIENTS_DESCRIPTION)),
    unread_only=(bool, Field(default=False, description="Only return unread messages")),
    only_new=(bool, Field(default=True, description=(
        "Return only new messages. Set False to return all matching messages, each flagged with is_new. Example: False to see recent context around the new ones"
    ))),
    include_preview=(bool, Field(default=True, description="Include a short body preview of each message. Example: False for a subjects-only overview")),
    limit=(int, Field(default=25, ge=1, le=1000, description="Maximum messages to return. Example: 10")),
    preview_chars=_preview_chars(),
    max_recipients=_max_recipients(),
    include_ids=(bool, Field(default=False, description=_INCLUDE_IDS)),
)

GetThreadMessages = create_model(
    "GetThreadMessages",
    message_id=(Optional[str], Field(default=None, description="ID of any message in the thread (id from list_messages / find_new_messages). Example: 'AAMkAGI2...'")),
    conversation_id=(Optional[str], Field(default=None, description="Conversation ID of the thread (conversation_id returned by send_mail / reply_to_message, or conversationId with include_ids). Example: 'AAQkAGI2...'")),
    since=(Optional[str], Field(default=None, description=_SINCE_DESCRIPTION)),
    after_message_id=(Optional[str], Field(default=None, description=(
        _AFTER_MESSAGE_ID_DESCRIPTION + " Also identifies the thread when message_id/conversation_id are omitted."
    ))),
    only_new=(bool, Field(default=False, description="Return only new messages of the thread (combine with since or after_message_id). Example: True to read just the replies since your last message")),
    include_body=(bool, Field(default=False, description="Include the new (non-quoted) part of each message body. Example: True to read what each person added")),
    limit=(int, Field(default=50, ge=1, le=1000, description="Maximum messages to return (newest are kept). Example: 10")),
    max_body_chars=(int, Field(default=2000, ge=0, le=500000, description=(
        "With include_body, maximum characters of each message body (default 2000, 0 = no limit); "
        "cut bodies are flagged with body_truncated. Example: 500 for short summaries"
    ))),
    max_recipients=_max_recipients(),
    preview_chars=_preview_chars(),
)
