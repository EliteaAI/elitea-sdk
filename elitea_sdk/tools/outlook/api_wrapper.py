"""Outlook API Wrapper - main entry point for the Outlook toolkit."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, SecretStr, model_validator

from .graph_wrapper import OutlookGraphWrapper
from .models import (
    ListMessages,
    GetMessage,
    SendMail,
    ReplyToMessage,
    MarkAsRead,
    ListFolders,
    MoveMessage,
    DeleteMessage,
    SearchMessages,
    CheckNewMessages,
    FindNewMessages,
    GetThreadMessages,
)

log = logging.getLogger(__name__)


class OutlookApiWrapper(BaseModel):
    """Outlook API wrapper using Microsoft Graph API.

    Supports delegated OAuth authentication for accessing user mailboxes.
    """

    client_id: Optional[str] = Field(default=None, description="Azure AD Client ID")
    client_secret: Optional[SecretStr] = Field(default=None, description="Azure AD Client Secret")
    mailbox: Optional[str] = Field(default=None, description="Target mailbox email (None = /me)")

    token: Optional[str] = Field(default=None, description="OAuth access token")
    refresh_token: Optional[str] = Field(default=None, description="OAuth refresh token")
    scopes: Optional[List[str]] = Field(default=None, description="OAuth scopes")

    oauth_discovery_endpoint: Optional[str] = Field(default=None, description="OAuth discovery endpoint")
    oauth_token_endpoint: Optional[str] = Field(default=None, description="OAuth token endpoint")

    configuration_uuid: Optional[str] = Field(default=None, description="Configuration UUID")
    toolkit_id: Optional[int] = Field(default=None, description="Toolkit ID")
    toolkit_name: Optional[str] = Field(default=None, description="Toolkit name")

    _backend: Optional[OutlookGraphWrapper] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @model_validator(mode="after")
    def validate_and_create_backend(self) -> "OutlookApiWrapper":
        """Create the Graph wrapper backend after validation."""
        if not self.token:
            log.warning("No access token provided - OAuth authorization required")
            return self

        token_endpoint = self.oauth_token_endpoint
        if not token_endpoint and self.oauth_discovery_endpoint:
            base = self.oauth_discovery_endpoint.rstrip("/")
            token_endpoint = f"{base}/oauth2/v2.0/token"

        client_secret_str = None
        if self.client_secret:
            client_secret_str = (
                self.client_secret.get_secret_value()
                if hasattr(self.client_secret, "get_secret_value")
                else str(self.client_secret)
            )

        self._backend = OutlookGraphWrapper(
            token=self.token,
            scopes=self.scopes or [],
            mailbox=self.mailbox,
            refresh_token=self.refresh_token,
            client_id=self.client_id,
            client_secret=client_secret_str,
            oauth_token_endpoint=token_endpoint,
            oauth_discovery_endpoint=self.oauth_discovery_endpoint,
            configuration_uuid=self.configuration_uuid,
            toolkit_id=self.toolkit_id,
            toolkit_name=self.toolkit_name,
        )
        return self

    def _ensure_backend(self) -> OutlookGraphWrapper:
        """Ensure backend is initialized, raise if not."""
        if self._backend is None:
            from ...configurations.outlook import OutlookConfiguration
            raise OutlookConfiguration._build_mcp_authorization_required(
                message="Outlook requires OAuth authorization. Please complete the OAuth flow to obtain an access token.",
                oauth_discovery_endpoint=self.oauth_discovery_endpoint or "https://login.microsoftonline.com/common",
                scopes=self.scopes,
                configuration_uuid=self.configuration_uuid,
                toolkit_id=self.toolkit_id,
                toolkit_name=self.toolkit_name,
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
        return self._backend

    # ------------------------------------------------------------------ #
    #  Tool Methods                                                        #
    # ------------------------------------------------------------------ #

    def list_messages(
        self,
        folder: str = "inbox",
        limit: int = 50,
        unread_only: bool = False,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List messages from a mail folder."""
        return self._ensure_backend().list_messages(
            folder=folder,
            limit=limit,
            unread_only=unread_only,
            search=search,
        )

    def get_message(self, message_id: str, include_body: bool = True) -> Dict[str, Any]:
        """Get a specific message by ID."""
        return self._ensure_backend().get_message(
            message_id=message_id,
            include_body=include_body,
        )

    def send_mail(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        html: bool = False,
    ) -> Dict[str, Any]:
        """Send an email."""
        return self._ensure_backend().send_mail(
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            html=html,
        )

    def reply_to_message(
        self,
        message_id: str,
        body: str,
        reply_all: bool = False,
    ) -> Dict[str, Any]:
        """Reply to a message."""
        return self._ensure_backend().reply_to_message(
            message_id=message_id,
            body=body,
            reply_all=reply_all,
        )

    def mark_as_read(self, message_id: str, is_read: bool = True) -> str:
        """Mark a message as read or unread."""
        return self._ensure_backend().mark_as_read(
            message_id=message_id,
            is_read=is_read,
        )

    def list_folders(self) -> List[Dict[str, Any]]:
        """List all mail folders."""
        return self._ensure_backend().list_folders()

    def move_message(self, message_id: str, destination_folder: str) -> Dict[str, Any]:
        """Move a message to another folder."""
        return self._ensure_backend().move_message(
            message_id=message_id,
            destination_folder=destination_folder,
        )

    def delete_message(self, message_id: str, permanent: bool = False) -> str:
        """Delete a message."""
        return self._ensure_backend().delete_message(
            message_id=message_id,
            permanent=permanent,
        )

    def search_messages(
        self,
        query: str,
        folder: str = "inbox",
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Search messages using query."""
        return self._ensure_backend().search_messages(
            query=query,
            folder=folder,
            limit=limit,
        )

    def check_new_messages(
        self,
        folder: str = "inbox",
        since: Optional[str] = None,
        after_message_id: Optional[str] = None,
        senders: Optional[List[str]] = None,
        recipients: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check whether new / unread messages are present."""
        return self._ensure_backend().check_new_messages(
            folder=folder,
            since=since,
            after_message_id=after_message_id,
            senders=senders,
            recipients=recipients,
        )

    def find_new_messages(
        self,
        folder: str = "inbox",
        since: Optional[str] = None,
        after_message_id: Optional[str] = None,
        senders: Optional[List[str]] = None,
        recipients: Optional[List[str]] = None,
        unread_only: bool = False,
        only_new: bool = True,
        include_preview: bool = True,
        limit: int = 25,
    ) -> Dict[str, Any]:
        """Find messages from people / DLs and flag which are new."""
        return self._ensure_backend().find_new_messages(
            folder=folder,
            since=since,
            after_message_id=after_message_id,
            senders=senders,
            recipients=recipients,
            unread_only=unread_only,
            only_new=only_new,
            include_preview=include_preview,
            limit=limit,
        )

    def get_thread_messages(
        self,
        message_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        since: Optional[str] = None,
        after_message_id: Optional[str] = None,
        only_new: bool = False,
        include_body: bool = False,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get the messages of a conversation and flag new ones."""
        return self._ensure_backend().get_thread_messages(
            message_id=message_id,
            conversation_id=conversation_id,
            since=since,
            after_message_id=after_message_id,
            only_new=only_new,
            include_body=include_body,
            limit=limit,
        )

    # ------------------------------------------------------------------ #
    #  Tool Discovery                                                      #
    # ------------------------------------------------------------------ #

    def run(self, action: str, **kwargs) -> Any:
        """Run a tool action by name."""
        action_map = {
            "list_messages": self.list_messages,
            "get_message": self.get_message,
            "send_mail": self.send_mail,
            "reply_to_message": self.reply_to_message,
            "mark_as_read": self.mark_as_read,
            "list_folders": self.list_folders,
            "move_message": self.move_message,
            "delete_message": self.delete_message,
            "search_messages": self.search_messages,
            "check_new_messages": self.check_new_messages,
            "find_new_messages": self.find_new_messages,
            "get_thread_messages": self.get_thread_messages,
        }
        if action not in action_map:
            raise ValueError(f"Unknown action: {action}")
        return action_map[action](**kwargs)

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools with their schemas."""
        return [
            {
                "name": "list_messages",
                "description": "List email messages from a mail folder (inbox, sentitems, drafts, etc.)",
                "args_schema": ListMessages,
                "ref": self.list_messages,
            },
            {
                "name": "get_message",
                "description": "Get a specific email message by its ID with full details",
                "args_schema": GetMessage,
                "ref": self.get_message,
            },
            {
                "name": "send_mail",
                "description": (
                    "Send a new email message. Returns message_id and conversation_id of the sent message; "
                    "store them to later detect replies with get_thread_messages"
                ),
                "args_schema": SendMail,
                "ref": self.send_mail,
            },
            {
                "name": "reply_to_message",
                "description": "Reply to an email message. Returns message_id and conversation_id of the sent reply",
                "args_schema": ReplyToMessage,
                "ref": self.reply_to_message,
            },
            {
                "name": "mark_as_read",
                "description": "Mark an email as read or unread",
                "args_schema": MarkAsRead,
                "ref": self.mark_as_read,
            },
            {
                "name": "list_folders",
                "description": "List all mail folders in the mailbox",
                "args_schema": ListFolders,
                "ref": self.list_folders,
            },
            {
                "name": "move_message",
                "description": "Move an email to a different folder",
                "args_schema": MoveMessage,
                "ref": self.move_message,
            },
            {
                "name": "delete_message",
                "description": "Delete an email message",
                "args_schema": DeleteMessage,
                "ref": self.delete_message,
            },
            {
                "name": "search_messages",
                "description": "Search for email messages using a KQL query (from:, to:, subject:, body:, received:, ...)",
                "args_schema": SearchMessages,
                "ref": self.search_messages,
            },
            {
                "name": "check_new_messages",
                "description": (
                    "Quickly check whether new or unread emails are present, optionally only from given senders "
                    "or sent to given distribution lists. Returns has_new, counts, a few latest messages and a "
                    "watermark / latest_message_id to pass as since / after_message_id next time"
                ),
                "args_schema": CheckNewMessages,
                "ref": self.check_new_messages,
            },
            {
                "name": "find_new_messages",
                "description": (
                    "Find emails from given people and/or sent to given distribution lists and identify which are "
                    "new (received after since / after_message_id, or unread when neither is given). Each message "
                    "has is_new and matched_by; the result has a watermark / latest_message_id for the next call"
                ),
                "args_schema": FindNewMessages,
                "ref": self.find_new_messages,
            },
            {
                "name": "get_thread_messages",
                "description": (
                    "Get all messages of an email thread (conversation) across folders, oldest first, and identify "
                    "new messages in it (after since / after_message_id, or unread). Reports new_replies_count "
                    "(new messages not sent by the mailbox owner)"
                ),
                "args_schema": GetThreadMessages,
                "ref": self.get_thread_messages,
            },
        ]
