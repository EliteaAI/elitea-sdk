"""Microsoft Teams Graph API wrapper - delegated access only.

Uses OAuth tokens obtained via the Azure AD OAuth flow to communicate
with https://graph.microsoft.com/v1.0 Teams (chat / channel) endpoints.

Required OAuth scopes (delegated):
    User.Read, User.ReadBasic.All             resolve people by email
    Chat.Read (or Chat.ReadWrite)             read chats and chat messages
    ChatMessage.Send, Chat.Create             send to people / group chats
    Team.ReadBasic.All, Channel.ReadBasic.All list teams and channels
    ChannelMessage.Send                       post / reply in channels
    (offline_access for token refresh)

Graph has no delegated "all my chats" message feed, so messages are read per
chat, or found across Teams with the Search API. Reading channel messages
needs ChannelMessage.Read.All (admin consent) and is intentionally not supported.
"""
from __future__ import annotations

import html as html_lib
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

import requests
from langchain_core.tools import ToolException

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# Teams endpoints accept at most $top=50; hard caps on items scanned per call.
_PAGE_SIZE = 50
_MAX_SCAN = 1000
_MAX_CHAT_SCAN = 500
_SEARCH_PAGE_SIZE = 25
_MAX_TEXT = 4000

# Throttling (429 / 503) retries, honouring Retry-After up to _MAX_RETRY_AFTER seconds.
_MAX_RETRIES = 3
_MAX_RETRY_AFTER = 30

_GUID_RE = re.compile(r"^[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$")
_BREAK_RE = re.compile(r"<\s*(?:br\s*/?|/p|/div|/li)\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

log = logging.getLogger(__name__)


class _GraphError(ToolException):
    """Graph API error carrying the HTTP status."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class _Bound(NamedTuple):
    """Lower time bound that defines which messages count as "new"."""
    dt: Optional[datetime]              # None: no explicit bound
    inclusive: bool                     # True when taken from an anchor message (>= and exclude anchor)
    exclude_ids: Set[str]


def _parse_dt(value: str) -> datetime:
    """Parse an ISO 8601 date/datetime into an aware UTC datetime (millisecond precision)."""
    text = (value or "").strip()
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    # Python < 3.11 only accepts 3 or 6 fractional digits
    text = re.sub(r"\.(\d+)", lambda m: "." + (m.group(1) + "000000")[:6], text)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise ToolException(
            f"Invalid date/time '{value}'. Use ISO 8601, e.g. 2026-09-23T10:15:00Z or 2026-09-23."
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=dt.microsecond // 1000 * 1000)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _odata_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _is_guid(value: str) -> bool:
    return bool(_GUID_RE.match(value or ""))


def _is_thread_id(value: str) -> bool:
    """Chat / channel IDs look like 19:...@thread.v2, 19:..@unq.gbl.spaces, 48:notes."""
    return bool(re.match(r"^\d{2}:", value or ""))


def _html_to_text(content: str) -> str:
    text = _BREAK_RE.sub("\n", content or "")
    text = html_lib.unescape(_TAG_RE.sub("", text))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\s*\n\s*", "\n", text).strip()


def _text_to_html(text: str) -> str:
    return html_lib.escape(text or "").replace("\n", "<br>")


def _message_text(msg: dict) -> str:
    body = msg.get("body") or {}
    content = body.get("content") or ""
    text = _html_to_text(content) if (body.get("contentType") or "").lower() == "html" else content.strip()
    return text[:_MAX_TEXT]


def _is_user_message(msg: dict) -> bool:
    """Skip system events (member added, call started, ...) and deleted messages."""
    return (msg.get("messageType") or "message") == "message" and not msg.get("deletedDateTime")


def _created(msg: dict) -> Optional[datetime]:
    value = msg.get("createdDateTime")
    return _parse_dt(value) if value else None


def _sender_id(msg: dict) -> str:
    return (((msg.get("from") or {}).get("user") or {}).get("id") or "").lower()


def _sender_name(msg: dict) -> str:
    sender = msg.get("from") or {}
    return ((sender.get("user") or {}).get("displayName")
            or (sender.get("application") or {}).get("displayName") or "")


def _after(bound: _Bound, msg: dict) -> bool:
    """True if the message was created after the bound (bound must be set)."""
    created = _created(msg)
    if created is None or msg.get("id") in bound.exclude_ids:
        return False
    return created >= bound.dt if bound.inclusive else created > bound.dt


def _watermark(bound: _Bound, newest: Optional[dict]) -> Optional[str]:
    """Newest createdDateTime seen, never older than the bound that was applied."""
    candidates = []
    if newest and _created(newest):
        candidates.append(_created(newest))
    if bound.dt:
        candidates.append(bound.dt)
    return _fmt_dt(max(candidates)) if candidates else None


def _newest(messages: List[dict]) -> Optional[dict]:
    dated = [m for m in messages if m.get("createdDateTime")]
    return max(dated, key=_created, default=None)


def _pick_by_name(items: List[dict], name: str, kind: str, name_key: str = "displayName") -> dict:
    """Pick one item by case-insensitive exact name, raising a helpful error otherwise."""
    wanted = name.strip().lower()
    matches = [i for i in items if (i.get(name_key) or "").strip().lower() == wanted]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        known = ", ".join(sorted({i.get(name_key) or "" for i in items if i.get(name_key)})[:30])
        raise ToolException(f"{kind} '{name}' not found. Known: {known or 'none'}")
    ids = ", ".join(i.get("id") for i in matches)
    raise ToolException(f"{kind} name '{name}' is ambiguous; use one of these IDs instead: {ids}")


class TeamsGraphWrapper:
    """Graph-API-backed Microsoft Teams wrapper for delegated (user) access.

    Args:
        token: Plain-text OAuth bearer token (delegated, user-context).
        scopes: List of OAuth scopes that were granted for this token.
        refresh_token: Optional refresh token for auto-renewal.
        client_id: Client ID for token refresh.
        client_secret: Client secret for token refresh.
        oauth_token_endpoint: Token endpoint URL for refresh.
        oauth_discovery_endpoint: Discovery endpoint for auth errors.
        configuration_uuid: Configuration UUID for auth errors.
        toolkit_id: Toolkit ID for auth errors.
        toolkit_name: Toolkit name for auth errors.
    """

    def __init__(
        self,
        token: str,
        scopes: List[str],
        refresh_token: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        oauth_token_endpoint: Optional[str] = None,
        oauth_discovery_endpoint: Optional[str] = None,
        configuration_uuid: Optional[str] = None,
        toolkit_id: Optional[int] = None,
        toolkit_name: Optional[str] = None,
    ):
        self._token = token
        self._scopes = scopes
        self._refresh_token = refresh_token
        self._client_id = client_id
        self._client_secret = client_secret
        self._oauth_token_endpoint = oauth_token_endpoint
        self._oauth_discovery_endpoint = oauth_discovery_endpoint
        self._configuration_uuid = configuration_uuid
        self._toolkit_id = toolkit_id
        self._toolkit_name = toolkit_name
        self._me: Optional[dict] = None
        self._users: Dict[str, dict] = {}         # lower(email / id) -> user
        self._emails_by_id: Dict[str, str] = {}   # lower(user id) -> email
        self._team_ids: Dict[str, str] = {}
        self._channel_ids: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------------ #
    #  HTTP helpers                                                        #
    # ------------------------------------------------------------------ #

    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _raise_authorization_required(self, resp: requests.Response) -> None:
        """Turn a rejected token into the standard auth signal."""
        from ...configurations.teams import TeamsConfiguration

        api_message = TeamsConfiguration._extract_api_error_message(resp)
        if not self._oauth_discovery_endpoint:
            raise _GraphError(f"Teams access token is invalid or expired: {api_message}", resp.status_code)

        auth_error = TeamsConfiguration._build_mcp_authorization_required(
            message=f"Teams access token is invalid or expired: {api_message}. Please re-authorize.",
            oauth_discovery_endpoint=self._oauth_discovery_endpoint,
            scopes=self._scopes,
            status=resp.status_code,
            configuration_uuid=self._configuration_uuid,
            toolkit_id=self._toolkit_id,
            toolkit_name=self._toolkit_name,
            client_id=self._client_id,
            client_secret=self._client_secret,
        )
        auth_error.tool_name = None
        raise auth_error

    def _raise_with_body(self, resp: requests.Response) -> None:
        """Raise a ToolException carrying Graph's error message."""
        if resp.ok:
            return
        if resp.status_code == 401:
            self._raise_authorization_required(resp)
        from ...configurations.teams import TeamsConfiguration

        api_message = TeamsConfiguration._extract_api_error_message(resp)
        log.error("Graph API HTTP %s for %s: %s", resp.status_code, resp.url, api_message)
        hint = ""
        if resp.status_code == 403:
            hint = " Check that the Teams configuration scopes include the permission this tool needs."
        raise _GraphError(f"Microsoft Graph returned HTTP {resp.status_code}: {api_message}.{hint}", resp.status_code)

    def _try_refresh_token(self) -> bool:
        """Attempt to refresh the access token using stored refresh_token."""
        if not self._refresh_token or not self._oauth_token_endpoint:
            return False
        try:
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': self._refresh_token,
                'scope': ' '.join(self._scopes) if self._scopes else 'https://graph.microsoft.com/.default',
            }
            if self._client_id:
                data['client_id'] = self._client_id
            if self._client_secret:
                data['client_secret'] = self._client_secret
            resp = requests.post(self._oauth_token_endpoint, data=data, timeout=10)
            if resp.status_code == 200:
                token_data = resp.json()
                self._token = token_data['access_token']
                if token_data.get('refresh_token'):
                    self._refresh_token = token_data['refresh_token']
                log.info("Teams: access token refreshed successfully")
                return True
            log.warning("Teams: token refresh failed with HTTP %s", resp.status_code)
            return False
        except Exception as e:
            log.warning("Teams: token refresh error: %s", e)
            return False

    @staticmethod
    def _retry_delay(resp: requests.Response, attempt: int) -> float:
        try:
            delay = float((resp.headers or {}).get("Retry-After"))
        except (TypeError, ValueError):
            delay = 2 ** attempt
        return max(0.0, min(delay, _MAX_RETRY_AFTER))

    def _request(self, method: str, url: str, params: Optional[dict] = None,
                 payload: Optional[dict] = None) -> requests.Response:
        """Send a request, refreshing the token once on 401 and retrying on throttling."""
        refreshed = False
        attempt = 0
        while True:
            resp = requests.request(method, url, headers=self._auth_headers(), params=params,
                                    json=payload, timeout=60)
            if resp.status_code == 401 and not refreshed:
                refreshed = True
                if self._try_refresh_token():
                    continue
            if resp.status_code in (429, 503) and attempt < _MAX_RETRIES:
                delay = self._retry_delay(resp, attempt)
                attempt += 1
                log.warning("Teams: Graph throttled (HTTP %s), retrying in %.0fs", resp.status_code, delay)
                time.sleep(delay)
                continue
            self._raise_with_body(resp)
            return resp

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        return self._request("GET", url, params=params).json()

    def _post(self, url: str, payload: dict) -> dict:
        resp = self._request("POST", url, payload=payload)
        if resp.status_code in (202, 204):
            return {}
        return resp.json()

    def _iter_pages(self, url: str, params: Optional[dict] = None) -> Iterator[Tuple[List[dict], bool]]:
        """Yield (items, has_next_page) following @odata.nextLink."""
        data = self._get(url, params=params)
        while True:
            next_link = data.get("@odata.nextLink")
            yield data.get("value", []), bool(next_link)
            if not next_link:
                return
            # nextLink already carries the original query parameters
            data = self._get(next_link)

    def _collect(self, url: str, params: Optional[dict] = None, max_items: int = _MAX_SCAN) -> Tuple[List[dict], bool]:
        """Collect up to max_items; returns (items, has_more)."""
        items: List[dict] = []
        for page, has_next in self._iter_pages(url, params):
            for pos, item in enumerate(page):
                items.append(item)
                if len(items) >= max_items:
                    return items, pos < len(page) - 1 or has_next
        return items, False

    # ------------------------------------------------------------------ #
    #  Resolution helpers (people, teams, channels, chats)                #
    # ------------------------------------------------------------------ #

    def _get_me(self) -> dict:
        if self._me is None:
            self._me = self._get(f"{_GRAPH_BASE}/me", params={"$select": "id,displayName,mail,userPrincipalName"})
            self._remember_user(self._me)
        return self._me

    def _my_id(self) -> str:
        return (self._get_me().get("id") or "").lower()

    def _remember_user(self, user: dict, *keys: str) -> dict:
        email = user.get("mail") or user.get("userPrincipalName")
        uid = (user.get("id") or "").lower()
        if uid and email:
            self._emails_by_id[uid] = email
        for key in (uid, (email or "").lower(), *[k.lower() for k in keys]):
            if key:
                self._users[key] = user
        return user

    def _resolve_user(self, identifier: str) -> dict:
        """Resolve an email / UPN / user ID to a directory user (id, displayName, mail)."""
        ident = (identifier or "").strip()
        if not ident:
            raise ToolException("Empty user identifier")
        cached = self._users.get(ident.lower())
        if cached:
            return cached
        select = {"$select": "id,displayName,mail,userPrincipalName"}
        try:
            user = self._get(f"{_GRAPH_BASE}/users/{ident}", params=select)
        except _GraphError as exc:
            if exc.status not in (400, 404) or "@" not in ident:
                raise ToolException(f"User '{ident}' not found: {exc}")
            # Primary SMTP address can differ from the UPN
            found = self._get(f"{_GRAPH_BASE}/users",
                              params={**select, "$filter": f"mail eq {_odata_str(ident)}"}).get("value", [])
            if not found:
                raise ToolException(f"User '{ident}' not found in the directory")
            user = found[0]
        return self._remember_user(user, ident)

    def _resolve_team(self, team: str) -> str:
        team = (team or "").strip()
        if not team:
            raise ToolException("team is required (team ID or display name)")
        if _is_guid(team):
            return team
        if team.lower() not in self._team_ids:
            teams, _ = self._collect(f"{_GRAPH_BASE}/me/joinedTeams", params={"$select": "id,displayName"})
            self._team_ids[team.lower()] = _pick_by_name(teams, team, "Team")["id"]
        return self._team_ids[team.lower()]

    def _resolve_channel(self, team_id: str, channel: str) -> str:
        channel = (channel or "").strip()
        if not channel:
            raise ToolException("channel is required (channel ID or display name)")
        if _is_thread_id(channel):
            return channel
        key = (team_id, channel.lower())
        if key not in self._channel_ids:
            channels = self._get(f"{_GRAPH_BASE}/teams/{team_id}/channels",
                                 params={"$select": "id,displayName"}).get("value", [])
            self._channel_ids[key] = _pick_by_name(channels, channel, "Channel")["id"]
        return self._channel_ids[key]

    def _iter_chats(self, expand_members: bool = False) -> Iterator[dict]:
        params: Dict[str, Any] = {"$top": _PAGE_SIZE}
        if expand_members:
            params["$expand"] = "members"
        scanned = 0
        for page, _ in self._iter_pages(f"{_GRAPH_BASE}/me/chats", params):
            for chat in page:
                yield chat
                scanned += 1
                if scanned >= _MAX_CHAT_SCAN:
                    return

    @staticmethod
    def _member_ids(chat: dict) -> Set[str]:
        return {(m.get("userId") or "").lower() for m in chat.get("members") or [] if m.get("userId")}

    def _find_one_on_one(self, user_id: str) -> Optional[dict]:
        me = self._my_id()
        for chat in self._iter_chats(expand_members=True):
            if chat.get("chatType") == "oneOnOne" and self._member_ids(chat) == {me, user_id.lower()}:
                return chat
        return None

    def _find_group_chat(self, member_ids: Set[str], topic: Optional[str]) -> Optional[dict]:
        wanted_topic = (topic or "").strip().lower()
        for chat in self._iter_chats(expand_members=True):
            if chat.get("chatType") != "group" or self._member_ids(chat) != member_ids:
                continue
            if not wanted_topic or (chat.get("topic") or "").strip().lower() == wanted_topic:
                return chat
        return None

    def _resolve_chat(self, chat: str) -> str:
        """Resolve a chat ID, a group chat topic, or a person's email (their 1:1 chat)."""
        chat = (chat or "").strip()
        if not chat:
            raise ToolException("chat is required (chat ID, group chat topic or a person's email)")
        if _is_thread_id(chat):
            return chat
        if "@" in chat and " " not in chat:
            user = self._resolve_user(chat)
            found = self._find_one_on_one(user["id"])
            if not found:
                raise ToolException(f"No 1:1 chat with '{chat}' found. Use send_chat_message to start one.")
            return found["id"]
        chats = [c for c in self._iter_chats() if c.get("topic")]
        return _pick_by_name(chats, chat, "Chat", name_key="topic")["id"]

    # ------------------------------------------------------------------ #
    #  Formatting / matching                                               #
    # ------------------------------------------------------------------ #

    def _format_message(self, msg: dict) -> Dict[str, Any]:
        uid = _sender_id(msg)
        item = {
            "id": msg.get("id"),
            "replyToId": msg.get("replyToId"),
            "createdDateTime": msg.get("createdDateTime"),
            "lastModifiedDateTime": msg.get("lastModifiedDateTime"),
            "from": {
                "id": ((msg.get("from") or {}).get("user") or {}).get("id"),
                "displayName": _sender_name(msg),
                "email": self._emails_by_id.get(uid),
            },
            "subject": msg.get("subject"),
            "text": _message_text(msg),
            "importance": msg.get("importance"),
            "webUrl": msg.get("webUrl"),
            "chatId": msg.get("chatId"),
            "mentions": [m.get("mentionText") for m in msg.get("mentions") or [] if m.get("mentionText")],
            "attachments": [
                {"name": a.get("name"), "contentType": a.get("contentType"), "contentUrl": a.get("contentUrl")}
                for a in msg.get("attachments") or []
            ],
        }
        channel = msg.get("channelIdentity") or {}
        if channel.get("channelId"):
            item["teamId"] = channel.get("teamId")
            item["channelId"] = channel.get("channelId")
        item["from"] = {k: v for k, v in item["from"].items() if v}
        return {k: v for k, v in item.items() if v not in (None, "", [], {})}

    def _sender_matcher(self, senders: Optional[List[str]]) -> Optional[Callable[[dict], Optional[str]]]:
        """Build msg -> "from:<sender>" (or None) for emails, user IDs or display names."""
        idents = [s.strip() for s in senders or [] if s and s.strip()]
        if not idents:
            return None
        by_id: Dict[str, str] = {}
        by_name: Dict[str, str] = {}
        for ident in idents:
            if _is_guid(ident):
                by_id[ident.lower()] = ident
            elif "@" in ident:
                by_id[(self._resolve_user(ident).get("id") or "").lower()] = ident
            else:
                by_name[ident.lower()] = ident

        def match(msg: dict) -> Optional[str]:
            label = by_id.get(_sender_id(msg)) or by_name.get(_sender_name(msg).strip().lower())
            return f"from:{label}" if label else None

        return match

    def _resolve_bound(self, since: Optional[str], after_message_id: Optional[str],
                       load_anchor: Callable[[str], Optional[dict]]) -> _Bound:
        """Combine since / after_message_id; the later point wins."""
        candidates: List[_Bound] = []
        if since:
            candidates.append(_Bound(_parse_dt(since), False, set()))
        if after_message_id:
            anchor = load_anchor(after_message_id)
            if not anchor or not _created(anchor):
                raise ToolException(f"Message '{after_message_id}' not found; pass since instead.")
            candidates.append(_Bound(_created(anchor), True, {after_message_id}))
        if not candidates:
            return _Bound(None, False, set())
        return max(candidates, key=lambda b: (b.dt, not b.inclusive))

    def _select(
        self,
        messages: List[dict],
        window: _Bound,
        bound: _Bound,
        read_marker: Optional[datetime],
        matcher: Optional[Callable[[dict], Optional[str]]],
        text_contains: Optional[str],
        only_new: bool,
        include_own: bool,
    ) -> Tuple[List[Dict[str, Any]], Optional[dict]]:
        """Filter raw messages; returns (formatted items oldest first, newest scanned message)."""
        me = self._my_id()
        needle = (text_contains or "").strip().lower()
        in_window = [m for m in messages if _is_user_message(m) and _after(window, m)]
        selected = []
        for msg in sorted(in_window, key=_created):
            from_me = bool(me) and _sender_id(msg) == me
            if from_me and not include_own:
                continue
            if bound.dt:
                is_new = _after(bound, msg)
            elif read_marker is not None:
                is_new = not from_me and _created(msg) > read_marker
            else:
                is_new = True
            if only_new and not is_new:
                continue
            reasons = []
            if matcher:
                reason = matcher(msg)
                if not reason:
                    continue
                reasons.append(reason)
            if needle:
                if needle not in _message_text(msg).lower():
                    continue
                reasons.append(f"text:{text_contains.strip()}")
            item = self._format_message(msg)
            item["is_new"] = is_new
            item["from_me"] = from_me
            if reasons:
                item["matched_by"] = reasons
            selected.append(item)
        return selected, _newest(in_window)

    @staticmethod
    def _page_result(selected: List[Dict[str, Any]], newest: Optional[dict], bound: _Bound,
                     limit: int, scan_truncated: bool) -> Dict[str, Any]:
        """Return the oldest `limit` items; the cursor points at the last returned one when cut."""
        returned = selected[:limit]
        cut = len(selected) > limit
        if cut:
            last = returned[-1]
            watermark = _fmt_dt(_parse_dt(last["createdDateTime"]))
            latest_id = last["id"]
        else:
            watermark = _watermark(bound, newest)
            latest_id = newest.get("id") if newest else None
        return {
            "count": len(returned),
            "new_count": sum(1 for m in returned if m.get("is_new")),
            "truncated": cut or scan_truncated,
            "watermark": watermark,
            "latest_message_id": latest_id,
            "messages": returned,
        }

    def _window(self, bound: _Bound, lookback_hours: int, only_new: bool) -> _Bound:
        lookback = _now().replace(microsecond=0) - timedelta(hours=lookback_hours)
        if bound.dt is None:
            return _Bound(lookback, True, set())
        if not only_new and lookback < bound.dt:
            return _Bound(lookback, True, set())
        return bound

    # ------------------------------------------------------------------ #
    #  Discovery                                                           #
    # ------------------------------------------------------------------ #

    def list_teams(self, name_contains: Optional[str] = None) -> List[Dict[str, Any]]:
        teams, _ = self._collect(f"{_GRAPH_BASE}/me/joinedTeams", params={"$select": "id,displayName,description"})
        needle = (name_contains or "").strip().lower()
        return [
            {"id": t.get("id"), "displayName": t.get("displayName"), "description": t.get("description")}
            for t in teams if needle in (t.get("displayName") or "").lower()
        ]

    def list_channels(self, team: str) -> Dict[str, Any]:
        team_id = self._resolve_team(team)
        channels = self._get(f"{_GRAPH_BASE}/teams/{team_id}/channels",
                             params={"$select": "id,displayName,description,membershipType,webUrl"}).get("value", [])
        return {
            "team_id": team_id,
            "channels": [
                {k: c.get(k) for k in ("id", "displayName", "description", "membershipType", "webUrl")}
                for c in channels
            ],
        }

    def list_chats(
        self,
        chat_type: Optional[str] = None,
        topic_contains: Optional[str] = None,
        member: Optional[str] = None,
        unread_only: bool = False,
        limit: int = 50,
    ) -> Dict[str, Any]:
        url = f"{_GRAPH_BASE}/me/chats"
        params = {"$top": _PAGE_SIZE, "$expand": "members,lastMessagePreview",
                  "$orderby": "lastMessagePreview/createdDateTime desc"}
        try:
            pages = self._iter_pages(url, params)
            first = next(pages)
        except _GraphError as exc:
            if exc.status != 400:
                raise
            params.pop("$orderby")
            pages = self._iter_pages(url, params)
            first = next(pages)

        member_id = None
        member_text = (member or "").strip().lower()
        if member_text and "@" in member_text:
            member_id = (self._resolve_user(member_text).get("id") or "").lower()
        topic_text = (topic_contains or "").strip().lower()
        wanted_type = (chat_type or "").strip().lower()

        result: List[Dict[str, Any]] = []
        scanned = 0
        truncated = False

        def _pages():
            yield first
            yield from pages

        for page, has_next in _pages():
            for pos, chat in enumerate(page):
                scanned += 1
                item = self._format_chat(chat)
                keep = (
                    (not wanted_type or (chat.get("chatType") or "").lower() == wanted_type)
                    and (not topic_text or topic_text in (chat.get("topic") or "").lower())
                    and (not unread_only or item.get("has_unread"))
                    and (not member_text or self._chat_has_member(chat, member_id, member_text))
                )
                if keep:
                    result.append(item)
                if len(result) >= limit or scanned >= _MAX_CHAT_SCAN:
                    truncated = pos < len(page) - 1 or has_next
                    return {"count": len(result), "truncated": truncated, "chats": result}
        return {"count": len(result), "truncated": truncated, "chats": result}

    @staticmethod
    def _chat_has_member(chat: dict, member_id: Optional[str], member_text: str) -> bool:
        for m in chat.get("members") or []:
            if member_id and (m.get("userId") or "").lower() == member_id:
                return True
            if member_text in ((m.get("email") or "").lower(), (m.get("displayName") or "").lower()):
                return True
        return False

    @staticmethod
    def _format_chat(chat: dict) -> Dict[str, Any]:
        preview = chat.get("lastMessagePreview") or {}
        read_at = (chat.get("viewpoint") or {}).get("lastMessageReadDateTime")
        item: Dict[str, Any] = {
            "id": chat.get("id"),
            "chatType": chat.get("chatType"),
            "topic": chat.get("topic"),
            "webUrl": chat.get("webUrl"),
            "members": [
                {k: v for k, v in (("displayName", m.get("displayName")), ("email", m.get("email")),
                                   ("userId", m.get("userId"))) if v}
                for m in chat.get("members") or []
            ],
        }
        if preview:
            body = preview.get("body") or {}
            content = body.get("content") or ""
            item["lastMessage"] = {
                "id": preview.get("id"),
                "createdDateTime": preview.get("createdDateTime"),
                "from": _sender_name(preview),
                "text": (_html_to_text(content) if (body.get("contentType") or "").lower() == "html"
                         else content)[:300],
            }
            if read_at and preview.get("createdDateTime"):
                item["has_unread"] = _parse_dt(preview["createdDateTime"]) > _parse_dt(read_at)
        if read_at:
            item["lastReadDateTime"] = read_at
        return {k: v for k, v in item.items() if v not in (None, "", [])}

    # ------------------------------------------------------------------ #
    #  Reading chats                                                       #
    # ------------------------------------------------------------------ #

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
        """Messages of one chat (1:1, group or meeting), oldest first, with is_new per message.

        "New" means created after since / after_message_id; with neither, it means
        unread (created after the user's last read time in this chat).
        """
        chat_id = self._resolve_chat(chat)
        chat_info = self._get(f"{_GRAPH_BASE}/chats/{chat_id}")
        messages_url = f"{_GRAPH_BASE}/chats/{chat_id}/messages"

        def load_anchor(message_id: str) -> Optional[dict]:
            return self._get(f"{messages_url}/{message_id}")

        bound = self._resolve_bound(since, after_message_id, load_anchor)
        read_at = (chat_info.get("viewpoint") or {}).get("lastMessageReadDateTime")
        read_marker = _parse_dt(read_at) if read_at and bound.dt is None else None
        if read_marker is not None and only_new:
            window = _Bound(read_marker, False, set())
        else:
            window = self._window(bound, lookback_hours, only_new)
        matcher = self._sender_matcher(senders)

        # Only lastModifiedDateTime supports gt; a message's lastModified >= createdDateTime,
        # so every message created in the window passes the server filter.
        params = {
            "$top": _PAGE_SIZE,
            "$orderby": "lastModifiedDateTime desc",
            "$filter": f"lastModifiedDateTime gt {_fmt_dt(window.dt - timedelta(seconds=1))}",
        }
        raw, scan_truncated = self._collect(messages_url, params, _MAX_SCAN)
        selected, newest = self._select(raw, window, bound, read_marker, matcher, text_contains,
                                        only_new, include_own)
        result = {
            "chat_id": chat_id,
            "chat_type": chat_info.get("chatType"),
            "topic": chat_info.get("topic"),
            "new_means": (f"created after {_fmt_dt(bound.dt)}" if bound.dt
                          else "unread" if read_marker else f"created in the last {lookback_hours}h"),
            "window_start": _fmt_dt(window.dt),
        }
        result.update(self._page_result(selected, newest, bound, limit, scan_truncated))
        return {k: v for k, v in result.items() if v is not None}

    # ------------------------------------------------------------------ #
    #  Search                                                              #
    # ------------------------------------------------------------------ #

    def search_teams_messages(
        self,
        query: Optional[str] = None,
        senders: Optional[List[str]] = None,
        since: Optional[str] = None,
        limit: int = 25,
    ) -> Dict[str, Any]:
        """KQL search over the signed-in user's Teams chat and channel messages."""
        parts = [query.strip()] if query and query.strip() else []
        idents = [s.strip() for s in senders or [] if s and s.strip()]
        if idents:
            clause = " OR ".join(f'from:"{s}"' for s in idents)
            parts.append(f"({clause})" if len(idents) > 1 else clause)
        since_dt = _parse_dt(since) if since else None
        if since_dt:
            # KQL date granularity is a day; the exact time is applied to the hits below
            parts.append(f"sent>={since_dt.strftime('%Y-%m-%d')}")
        if not parts:
            raise ToolException("Provide query, senders or since")
        kql = " AND ".join(parts)

        hits: List[Dict[str, Any]] = []
        offset = 0
        more = True
        while more and len(hits) < limit:
            size = min(_SEARCH_PAGE_SIZE, limit - len(hits))
            data = self._post(f"{_GRAPH_BASE}/search/query", {
                "requests": [{
                    "entityTypes": ["chatMessage"],
                    "query": {"queryString": kql},
                    "from": offset,
                    "size": size,
                }]
            })
            containers = [c for r in data.get("value", []) for c in r.get("hitsContainers") or []]
            page = [h for c in containers for h in c.get("hits") or []]
            more = any(c.get("moreResultsAvailable") for c in containers) and bool(page)
            offset += len(page)
            for hit in page:
                item = self._format_search_hit(hit)
                created = item.get("createdDateTime")
                if since_dt and created and _parse_dt(created) <= since_dt:
                    continue
                hits.append(item)
        return {"query": kql, "count": len(hits[:limit]), "more_available": more, "messages": hits[:limit]}

    @staticmethod
    def _format_search_hit(hit: dict) -> Dict[str, Any]:
        res = hit.get("resource") or {}
        sender = (res.get("from") or {}).get("emailAddress") or {}
        channel = res.get("channelIdentity") or {}
        item = {
            "id": res.get("id"),
            "chatId": res.get("chatId"),
            "teamId": channel.get("teamId"),
            "channelId": channel.get("channelId"),
            "createdDateTime": res.get("createdDateTime"),
            "from": {k: v for k, v in (("displayName", sender.get("name")), ("email", sender.get("address"))) if v},
            "subject": res.get("subject"),
            "summary": _html_to_text(hit.get("summary") or ""),
            "webLink": res.get("webLink"),
        }
        return {k: v for k, v in item.items() if v not in (None, "", {})}

    # ------------------------------------------------------------------ #
    #  Sending                                                             #
    # ------------------------------------------------------------------ #

    def _message_payload(self, message: str, html: bool, mentions: Optional[List[str]],
                         subject: Optional[str] = None, importance: Optional[str] = None) -> dict:
        if not (message or "").strip():
            raise ToolException("message must not be empty")
        people = [m.strip() for m in mentions or [] if m and m.strip()]
        content = message if html else (_text_to_html(message) if people else message)
        payload: Dict[str, Any] = {}
        if people:
            tags = []
            mention_items = []
            for idx, ident in enumerate(people):
                user = self._resolve_user(ident)
                name = user.get("displayName") or ident
                tags.append(f'<at id="{idx}">{html_lib.escape(name)}</at>')
                mention_items.append({
                    "id": idx,
                    "mentionText": name,
                    "mentioned": {"user": {"id": user["id"], "displayName": name, "userIdentityType": "aadUser"}},
                })
            content = " ".join(tags) + " " + content
            payload["mentions"] = mention_items
        payload["body"] = {"contentType": "html" if (html or people) else "text", "content": content}
        if subject:
            payload["subject"] = subject
        if importance and importance != "normal":
            payload["importance"] = importance
        return payload

    def _create_chat(self, member_ids: List[str], chat_type: str, topic: Optional[str]) -> dict:
        payload: Dict[str, Any] = {
            "chatType": chat_type,
            "members": [
                {
                    "@odata.type": "#microsoft.graph.aadUserConversationMember",
                    "roles": ["owner"],
                    "user@odata.bind": f"{_GRAPH_BASE}/users('{uid}')",
                }
                for uid in member_ids
            ],
        }
        if chat_type == "group" and topic:
            payload["topic"] = topic
        return self._post(f"{_GRAPH_BASE}/chats", payload)

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
        """Send to an existing chat, or to people (1:1 / group chat, created if needed)."""
        chat_created = False
        if chat:
            chat_id = self._resolve_chat(chat)
        else:
            people = [r.strip() for r in recipients or [] if r and r.strip()]
            if not people:
                raise ToolException("Provide chat (ID / topic) or recipients (emails)")
            me = self._my_id()
            others = []
            for ident in people:
                uid = (self._resolve_user(ident).get("id") or "").lower()
                if uid and uid != me and uid not in others:
                    others.append(uid)
            if not others:
                raise ToolException("recipients must include at least one person other than yourself")
            if len(others) == 1:
                # Graph returns the existing 1:1 chat instead of creating a duplicate
                try:
                    chat_id = self._create_chat([me, others[0]], "oneOnOne", None)["id"]
                except _GraphError as exc:
                    existing = self._find_one_on_one(others[0]) if exc.status == 403 else None
                    if not existing:
                        raise
                    chat_id = existing["id"]
            else:
                existing = self._find_group_chat({me, *others}, topic)
                if existing:
                    chat_id = existing["id"]
                else:
                    chat_id = self._create_chat([me, *others], "group", topic)["id"]
                    chat_created = True

        sent = self._post(f"{_GRAPH_BASE}/chats/{chat_id}/messages",
                          self._message_payload(message, html, mentions, importance=importance))
        result = {
            "status": "sent",
            "message_id": sent.get("id"),
            "chat_id": chat_id,
            "chat_created": chat_created,
            "createdDateTime": sent.get("createdDateTime"),
            "web_url": sent.get("webUrl"),
        }
        return {k: v for k, v in result.items() if v is not None}

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
        """Post a new channel message, or reply to an existing post."""
        team_id = self._resolve_team(team)
        channel_id = self._resolve_channel(team_id, channel)
        url = f"{_GRAPH_BASE}/teams/{team_id}/channels/{channel_id}/messages"
        if reply_to_message_id:
            url = f"{url}/{reply_to_message_id}/replies"
            subject = None
        sent = self._post(url, self._message_payload(message, html, mentions, subject, importance))
        result = {
            "status": "sent",
            "message_id": sent.get("id"),
            "thread_id": reply_to_message_id or sent.get("id"),
            "team_id": team_id,
            "channel_id": channel_id,
            "createdDateTime": sent.get("createdDateTime"),
            "web_url": sent.get("webUrl"),
        }
        return {k: v for k, v in result.items() if v is not None}
