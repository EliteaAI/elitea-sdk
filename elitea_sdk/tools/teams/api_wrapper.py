"""Teams API Wrapper - main entry point for the Teams toolkit."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, SecretStr, model_validator

from .graph_wrapper import TeamsGraphWrapper
from .models import (
    ListTeams,
    ListChannels,
    ListChats,
    FindChatMessages,
    SearchTeamsMessages,
    SendChatMessage,
    SendChannelMessage,
)

log = logging.getLogger(__name__)


class TeamsApiWrapper(BaseModel):
    """Microsoft Teams API wrapper using Microsoft Graph API.

    Supports delegated OAuth authentication for reading and sending chat and
    channel messages as the signed-in user.
    """

    client_id: Optional[str] = Field(default=None, description="Azure AD Client ID")
    client_secret: Optional[SecretStr] = Field(default=None, description="Azure AD Client Secret")

    token: Optional[str] = Field(default=None, description="OAuth access token")
    refresh_token: Optional[str] = Field(default=None, description="OAuth refresh token")
    scopes: Optional[List[str]] = Field(default=None, description="OAuth scopes")

    oauth_discovery_endpoint: Optional[str] = Field(default=None, description="OAuth discovery endpoint")
    oauth_token_endpoint: Optional[str] = Field(default=None, description="OAuth token endpoint")

    configuration_uuid: Optional[str] = Field(default=None, description="Configuration UUID")
    toolkit_id: Optional[int] = Field(default=None, description="Toolkit ID")
    toolkit_name: Optional[str] = Field(default=None, description="Toolkit name")

    _backend: Optional[TeamsGraphWrapper] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @model_validator(mode="after")
    def validate_and_create_backend(self) -> "TeamsApiWrapper":
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

        self._backend = TeamsGraphWrapper(
            token=self.token,
            scopes=self.scopes or [],
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

    def _ensure_backend(self) -> TeamsGraphWrapper:
        """Ensure backend is initialized, raise if not."""
        if self._backend is None:
            from ...configurations.teams import TeamsConfiguration
            raise TeamsConfiguration._build_mcp_authorization_required(
                message="Teams requires OAuth authorization. Please complete the OAuth flow to obtain an access token.",
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

    def list_teams(self, name_contains: Optional[str] = None) -> List[Dict[str, Any]]:
        """List teams the user is a member of."""
        return self._ensure_backend().list_teams(name_contains=name_contains)

    def list_channels(self, team: str) -> Dict[str, Any]:
        """List channels of a team."""
        return self._ensure_backend().list_channels(team=team)

    def list_chats(
        self,
        chat_type: Optional[str] = None,
        topic_contains: Optional[str] = None,
        member: Optional[str] = None,
        unread_only: bool = False,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List the user's chats, most recently active first."""
        return self._ensure_backend().list_chats(
            chat_type=chat_type,
            topic_contains=topic_contains,
            member=member,
            unread_only=unread_only,
            limit=limit,
        )

    def find_chat_messages(
        self,
        chat: str,
        senders: Optional[List[str]] = None,
        since: Optional[str] = None,
        after_message_id: Optional[str] = None,
        text_contains: Optional[str] = None,
        only_new: bool = True,
        include_own: bool = False,
        lookback_hours: int = 24,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Find messages in a chat, optionally from given people, and flag new ones."""
        return self._ensure_backend().find_chat_messages(
            chat=chat,
            senders=senders,
            since=since,
            after_message_id=after_message_id,
            text_contains=text_contains,
            only_new=only_new,
            include_own=include_own,
            lookback_hours=lookback_hours,
            limit=limit,
        )

    def search_teams_messages(
        self,
        query: Optional[str] = None,
        senders: Optional[List[str]] = None,
        since: Optional[str] = None,
        limit: int = 25,
    ) -> Dict[str, Any]:
        """Search the user's Teams messages across chats and channels."""
        return self._ensure_backend().search_teams_messages(
            query=query,
            senders=senders,
            since=since,
            limit=limit,
        )

    def send_chat_message(
        self,
        message: str,
        chat: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        topic: Optional[str] = None,
        html: bool = False,
        mentions: Optional[List[str]] = None,
        importance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a message to a chat or to people."""
        return self._ensure_backend().send_chat_message(
            message=message,
            chat=chat,
            recipients=recipients,
            topic=topic,
            html=html,
            mentions=mentions,
            importance=importance,
        )

    def send_channel_message(
        self,
        team: str,
        channel: str,
        message: str,
        subject: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        html: bool = False,
        mentions: Optional[List[str]] = None,
        importance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Post to a channel or reply to a channel post."""
        return self._ensure_backend().send_channel_message(
            team=team,
            channel=channel,
            message=message,
            subject=subject,
            reply_to_message_id=reply_to_message_id,
            html=html,
            mentions=mentions,
            importance=importance,
        )

    # ------------------------------------------------------------------ #
    #  Tool Discovery                                                      #
    # ------------------------------------------------------------------ #

    def run(self, action: str, **kwargs) -> Any:
        """Run a tool action by name."""
        action_map = {tool["name"]: tool["ref"] for tool in self.get_available_tools()}
        if action not in action_map:
            raise ValueError(f"Unknown action: {action}")
        return action_map[action](**kwargs)

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools with their schemas."""
        return [
            {
                "name": "list_teams",
                "description": "List Microsoft Teams teams the signed-in user is a member of (id, displayName)",
                "args_schema": ListTeams,
                "ref": self.list_teams,
            },
            {
                "name": "list_channels",
                "description": "List the channels of a team (id, displayName, membershipType)",
                "args_schema": ListChannels,
                "ref": self.list_channels,
            },
            {
                "name": "list_chats",
                "description": (
                    "List the signed-in user's Teams chats (1:1, group and meeting chats), most recently active "
                    "first, with members, last message preview and has_unread. Filter by type, topic or member"
                ),
                "args_schema": ListChats,
                "ref": self.list_chats,
            },
            {
                "name": "find_chat_messages",
                "description": (
                    "Read messages of one Teams chat (by chat ID, group chat topic or a person's email for the 1:1 "
                    "chat), optionally only from given people or containing text. Returns messages oldest first "
                    "with is_new and matched_by, plus watermark / latest_message_id to pass as since / "
                    "after_message_id next time. Without since / after_message_id, new means unread"
                ),
                "args_schema": FindChatMessages,
                "ref": self.find_chat_messages,
            },
            {
                "name": "search_teams_messages",
                "description": (
                    "Search Teams messages across all of the signed-in user's chats and channels (KQL), "
                    "optionally only from given people and after a date. Returns chatId (read the chat with "
                    "find_chat_messages) or teamId / channelId per hit"
                ),
                "args_schema": SearchTeamsMessages,
                "ref": self.search_teams_messages,
            },
            {
                "name": "send_chat_message",
                "description": (
                    "Send a Teams chat message to an existing chat, or to people by email: one person uses the "
                    "1:1 chat, several people use the group chat with exactly those members (created if needed). "
                    "Returns message_id and chat_id; store them to read replies later with find_chat_messages"
                ),
                "args_schema": SendChatMessage,
                "ref": self.send_chat_message,
            },
            {
                "name": "send_channel_message",
                "description": (
                    "Post a message to a Teams channel, or reply to a post with reply_to_message_id. "
                    "Returns message_id and thread_id; replies can be found later with search_teams_messages"
                ),
                "args_schema": SendChannelMessage,
                "ref": self.send_channel_message,
            },
        ]
