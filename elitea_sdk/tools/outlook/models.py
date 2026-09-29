"""Pydantic models for Outlook toolkit input schemas."""

from typing import List, Optional
from pydantic import Field, create_model


ListMessages = create_model(
    "ListMessages",
    folder=(str, Field(default="inbox", description="Mail folder name (inbox, sentitems, drafts, etc.) or 'all' for the whole mailbox")),
    limit=(int, Field(default=50, ge=1, le=1000, description="Maximum number of messages to return")),
    unread_only=(bool, Field(default=False, description="Only return unread messages")),
    search=(Optional[str], Field(default=None, description="KQL search query to filter messages")),
)

GetMessage = create_model(
    "GetMessage",
    message_id=(str, Field(description="The unique identifier of the message")),
    include_body=(bool, Field(default=True, description="Include message body in response")),
)

SendMail = create_model(
    "SendMail",
    to=(List[str], Field(description="List of recipient email addresses")),
    subject=(str, Field(description="Email subject line")),
    body=(str, Field(description="Email body content")),
    cc=(Optional[List[str]], Field(default=None, description="List of CC recipient email addresses")),
    bcc=(Optional[List[str]], Field(default=None, description="List of BCC recipient email addresses")),
    html=(bool, Field(default=False, description="If True, body is treated as HTML content")),
)

ReplyToMessage = create_model(
    "ReplyToMessage",
    message_id=(str, Field(description="The unique identifier of the message to reply to")),
    body=(str, Field(description="Reply body content")),
    reply_all=(bool, Field(default=False, description="Reply to all recipients")),
)

MarkAsRead = create_model(
    "MarkAsRead",
    message_id=(str, Field(description="The unique identifier of the message")),
    is_read=(bool, Field(default=True, description="Mark as read (True) or unread (False)")),
)

ListFolders = create_model(
    "ListFolders",
    __doc__="List all mail folders in the mailbox",
)

MoveMessage = create_model(
    "MoveMessage",
    message_id=(str, Field(description="The unique identifier of the message to move")),
    destination_folder=(str, Field(description="Destination folder name or ID")),
)

DeleteMessage = create_model(
    "DeleteMessage",
    message_id=(str, Field(description="The unique identifier of the message to delete")),
    permanent=(bool, Field(default=False, description="If True, permanently delete. If False, move to Deleted Items")),
)

SearchMessages = create_model(
    "SearchMessages",
    query=(str, Field(description=(
        "KQL search query, e.g. 'from:john@example.com subject:report', 'to:team-dl@example.com', "
        "'received>=2026-09-01 budget'. Dates in KQL have day precision only; for exact "
        "'new since' checks use find_new_messages instead."
    ))),
    folder=(str, Field(default="inbox", description="Folder to search in, or 'all' for the whole mailbox")),
    limit=(int, Field(default=25, ge=1, le=1000, description="Maximum results to return")),
)

_SINCE_DESCRIPTION = (
    "ISO 8601 UTC timestamp (e.g. 2026-09-23T10:15:00Z). Messages received after it are 'new'. "
    "Pass the 'watermark' returned by a previous call."
)
_AFTER_MESSAGE_ID_DESCRIPTION = (
    "Message ID; messages received after this message are 'new'. Use an ID returned by a previous call "
    "(e.g. latest_message_id, or message_id returned by send_mail). If both since and after_message_id "
    "are given, the later point wins."
)
_SENDERS_DESCRIPTION = "Sender email addresses to match (From)"
_RECIPIENTS_DESCRIPTION = "Recipient or distribution list email addresses to match (To or Cc)"

CheckNewMessages = create_model(
    "CheckNewMessages",
    folder=(str, Field(default="inbox", description="Folder name (inbox, sentitems, etc.) or 'all' for the whole mailbox")),
    since=(Optional[str], Field(default=None, description=_SINCE_DESCRIPTION + " If since and after_message_id are omitted, 'new' means unread.")),
    after_message_id=(Optional[str], Field(default=None, description=_AFTER_MESSAGE_ID_DESCRIPTION)),
    senders=(Optional[List[str]], Field(default=None, description=_SENDERS_DESCRIPTION)),
    recipients=(Optional[List[str]], Field(default=None, description=_RECIPIENTS_DESCRIPTION)),
)

FindNewMessages = create_model(
    "FindNewMessages",
    folder=(str, Field(default="inbox", description="Folder name (inbox, sentitems, etc.) or 'all' for the whole mailbox")),
    since=(Optional[str], Field(default=None, description=_SINCE_DESCRIPTION)),
    after_message_id=(Optional[str], Field(default=None, description=_AFTER_MESSAGE_ID_DESCRIPTION)),
    senders=(Optional[List[str]], Field(default=None, description=_SENDERS_DESCRIPTION)),
    recipients=(Optional[List[str]], Field(default=None, description=_RECIPIENTS_DESCRIPTION)),
    unread_only=(bool, Field(default=False, description="Only return unread messages")),
    only_new=(bool, Field(default=True, description=(
        "Return only new messages. Set False to return all matching messages, each flagged with is_new"
    ))),
    include_preview=(bool, Field(default=True, description="Include a short body preview of each message")),
    limit=(int, Field(default=25, ge=1, le=1000, description="Maximum messages to return")),
)

GetThreadMessages = create_model(
    "GetThreadMessages",
    message_id=(Optional[str], Field(default=None, description="ID of any message in the thread")),
    conversation_id=(Optional[str], Field(default=None, description="Conversation ID of the thread")),
    since=(Optional[str], Field(default=None, description=_SINCE_DESCRIPTION)),
    after_message_id=(Optional[str], Field(default=None, description=(
        _AFTER_MESSAGE_ID_DESCRIPTION + " Also identifies the thread when message_id/conversation_id are omitted."
    ))),
    only_new=(bool, Field(default=False, description="Return only new messages of the thread")),
    include_body=(bool, Field(default=False, description="Include the new (non-quoted) part of each message body")),
    limit=(int, Field(default=50, ge=1, le=1000, description="Maximum messages to return (newest are kept)")),
)
