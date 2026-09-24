"""Outlook Microsoft Graph API wrapper - delegated access only.

Uses OAuth tokens obtained via the Azure AD OAuth flow to communicate
with https://graph.microsoft.com/v1.0 Mail endpoints.

Required OAuth scopes (delegated):
    Mail.Read, Mail.ReadWrite, Mail.Send
    (offline_access for token refresh)

Every request asks Graph for immutable item IDs (``Prefer: IdType="ImmutableId"``),
so message IDs returned by this wrapper stay valid when a message is moved between
folders and when a draft is sent (the Sent Items copy keeps the draft's ID).
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

import requests
from langchain_core.tools import ToolException

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# Page size requested from Graph and the hard cap on messages scanned per call.
_PAGE_SIZE = 100
_MAX_SCAN = 1000

# Graph requires the $orderby property to be the first $filter clause; this no-op
# clause satisfies that rule when the caller has no date restriction.
_ANY_DATE_FILTER = "receivedDateTime ge 1900-01-01T00:00:00Z"

_ALL_FOLDERS = {"all", "*"}
_WELL_KNOWN_FOLDERS = {
    "inbox", "sentitems", "drafts", "deleteditems", "archive", "junkemail",
    "outbox", "clutter", "conversationhistory", "msgfolderroot", "scheduled",
}

_LIST_FIELDS = ("id,conversationId,subject,from,toRecipients,ccRecipients,"
                "receivedDateTime,isRead,bodyPreview,hasAttachments")
_SUMMARY_FIELDS = ("id,conversationId,parentFolderId,subject,from,toRecipients,ccRecipients,"
                   "receivedDateTime,isRead,isDraft,hasAttachments")
_CHECK_FIELDS = "id,conversationId,subject,from,toRecipients,ccRecipients,receivedDateTime,isRead"
_SENT_FIELDS = "id,conversationId,internetMessageId,subject,toRecipients,sentDateTime"

log = logging.getLogger(__name__)


class _Bound(NamedTuple):
    """Lower time bound that defines which messages count as "new"."""
    dt: Optional[datetime]              # None: no bound, "new" means unread
    inclusive: bool                     # True when taken from an anchor message (ge + exclude anchor)
    exclude_ids: Set[str]
    anchor: Optional[Dict[str, Any]]


def _parse_dt(value: str) -> datetime:
    """Parse an ISO 8601 date/datetime into an aware UTC datetime with second precision."""
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
    return dt.astimezone(timezone.utc).replace(microsecond=0)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _odata_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _normalize_addresses(values: Optional[List[str]]) -> Set[str]:
    return {v.strip().lower() for v in values or [] if v and v.strip()}


def _address(recipient: Optional[dict]) -> str:
    return ((recipient or {}).get("emailAddress") or {}).get("address") or ""


def _email_addresses(recipients: Optional[List[dict]]) -> List[dict]:
    return [r.get("emailAddress", {}) for r in recipients or []]


def _is_all_folders(folder: Optional[str]) -> bool:
    return not folder or folder.strip().lower() in _ALL_FOLDERS


def _status_of(exc: Exception) -> Optional[int]:
    resp = getattr(exc, "response", None)
    return resp.status_code if resp is not None else None


def _is_inefficient_filter(exc: Exception) -> bool:
    resp = getattr(exc, "response", None)
    if resp is None or resp.status_code != 400:
        return False
    text = resp.text or ""
    return "InefficientFilter" in text or "too complex" in text


def _received_key(msg: dict) -> str:
    # Graph returns UTC "YYYY-MM-DDTHH:MM:SSZ", which sorts lexicographically
    return msg.get("receivedDateTime") or ""


def _newest(messages: List[dict]) -> Optional[dict]:
    return max(messages, key=_received_key, default=None)


def _is_new(bound: _Bound, msg: dict) -> bool:
    """True if the message was received after the bound (bound must be set)."""
    if msg.get("id") in bound.exclude_ids or not msg.get("receivedDateTime"):
        return False
    received = _parse_dt(msg["receivedDateTime"])
    return received >= bound.dt if bound.inclusive else received > bound.dt


def _match_reasons(msg: dict, senders: Set[str], recipients: Set[str]) -> List[str]:
    """Explain why a message matched the sender / recipient (DL) criteria."""
    reasons = []
    sender = _address(msg.get("from")).lower()
    if sender and sender in senders:
        reasons.append(f"from:{sender}")
    for field, label in (("toRecipients", "to"), ("ccRecipients", "cc")):
        for recipient in msg.get(field) or []:
            addr = _address(recipient).lower()
            if addr and addr in recipients:
                reasons.append(f"{label}:{addr}")
    return reasons


def _watermark(bound: _Bound, newest: Optional[dict]) -> Optional[str]:
    """Newest receivedDateTime seen, never older than the bound that was applied."""
    candidates = []
    if newest and newest.get("receivedDateTime"):
        candidates.append(_parse_dt(newest["receivedDateTime"]))
    if bound.dt:
        candidates.append(bound.dt)
    return _fmt_dt(max(candidates)) if candidates else None


def _format_message(msg: dict) -> Dict[str, Any]:
    item = {
        "id": msg.get("id"),
        "conversationId": msg.get("conversationId"),
        "subject": msg.get("subject"),
        "from": (msg.get("from") or {}).get("emailAddress", {}),
        "to": _email_addresses(msg.get("toRecipients")),
        "cc": _email_addresses(msg.get("ccRecipients")),
        "receivedDateTime": msg.get("receivedDateTime"),
        "isRead": msg.get("isRead"),
    }
    if "hasAttachments" in msg:
        item["hasAttachments"] = msg.get("hasAttachments")
    if "bodyPreview" in msg:
        item["bodyPreview"] = msg.get("bodyPreview")
    return item


class OutlookGraphWrapper:
    """Graph-API-backed Outlook wrapper for delegated (user) access.

    Args:
        token: Plain-text OAuth bearer token (delegated, user-context).
        scopes: List of OAuth scopes that were granted for this token.
        mailbox: Optional mailbox address. If None, uses /me endpoint.
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
        mailbox: Optional[str] = None,
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
        self._mailbox = mailbox
        self._refresh_token = refresh_token
        self._client_id = client_id
        self._client_secret = client_secret
        self._oauth_token_endpoint = oauth_token_endpoint
        self._oauth_discovery_endpoint = oauth_discovery_endpoint
        self._configuration_uuid = configuration_uuid
        self._toolkit_id = toolkit_id
        self._toolkit_name = toolkit_name
        self._folder_ids: Dict[str, str] = {}

    @property
    def _user_path(self) -> str:
        """Return the Graph API user path - /me or /users/{mailbox}."""
        if self._mailbox:
            return f"/users/{self._mailbox}"
        return "/me"

    def _auth_headers(self, content_type: str = "application/json", prefer: Optional[str] = None) -> dict:
        preferences = ['IdType="ImmutableId"']
        if prefer:
            preferences.append(prefer)
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": content_type,
            "Prefer": ", ".join(preferences),
        }

    def _raise_authorization_required(self, resp: requests.Response) -> None:
        """Turn a rejected token into the standard auth signal."""
        if not self._oauth_discovery_endpoint:
            resp.raise_for_status()
            return

        from ...configurations.outlook import OutlookConfiguration

        api_message = OutlookConfiguration._extract_api_error_message(resp)

        auth_error = OutlookConfiguration._build_mcp_authorization_required(
            message=f"Outlook access token is invalid or expired: {api_message}. Please re-authorize.",
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
        """Raise with response body included in error message."""
        if resp.ok:
            return
        if resp.status_code == 401:
            self._raise_authorization_required(resp)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        log.error("Graph API HTTP %s for %s: %s", resp.status_code, resp.url, body)
        resp.raise_for_status()

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
                log.info("Outlook: access token refreshed successfully")
                return True
            log.warning("Outlook: token refresh failed with HTTP %s", resp.status_code)
            return False
        except Exception as e:
            log.warning("Outlook: token refresh error: %s", e)
            return False

    def _get(self, url: str, params: Optional[dict] = None, prefer: Optional[str] = None) -> dict:
        resp = requests.get(url, headers=self._auth_headers(prefer=prefer), params=params, timeout=60)
        if resp.status_code == 401 and self._try_refresh_token():
            resp = requests.get(url, headers=self._auth_headers(prefer=prefer), params=params, timeout=60)
        self._raise_with_body(resp)
        return resp.json()

    def _post(self, url: str, payload: Optional[dict]) -> dict:
        resp = requests.post(url, headers=self._auth_headers(), json=payload, timeout=60)
        if resp.status_code == 401 and self._try_refresh_token():
            resp = requests.post(url, headers=self._auth_headers(), json=payload, timeout=60)
        self._raise_with_body(resp)
        if resp.status_code == 202 or resp.status_code == 204:
            return {"status": "success"}
        return resp.json()

    def _patch(self, url: str, payload: dict) -> dict:
        resp = requests.patch(url, headers=self._auth_headers(), json=payload, timeout=60)
        if resp.status_code == 401 and self._try_refresh_token():
            resp = requests.patch(url, headers=self._auth_headers(), json=payload, timeout=60)
        self._raise_with_body(resp)
        return resp.json()

    def _delete(self, url: str) -> None:
        resp = requests.delete(url, headers=self._auth_headers(), timeout=30)
        if resp.status_code == 401 and self._try_refresh_token():
            resp = requests.delete(url, headers=self._auth_headers(), timeout=30)
        self._raise_with_body(resp)

    # ------------------------------------------------------------------ #
    #  Paging / query helpers                                             #
    # ------------------------------------------------------------------ #

    def _iter_pages(
        self, url: str, params: Optional[dict] = None, prefer: Optional[str] = None
    ) -> Iterator[Tuple[List[dict], bool]]:
        """Yield (items, has_next_page) following @odata.nextLink."""
        data = self._get(url, params=params, prefer=prefer)
        while True:
            next_link = data.get("@odata.nextLink")
            yield data.get("value", []), bool(next_link)
            if not next_link:
                return
            # nextLink already carries the original query parameters
            data = self._get(next_link, prefer=prefer)

    def _collect(
        self,
        url: str,
        params: dict,
        max_items: int,
        predicate: Optional[Callable[[dict], bool]] = None,
        prefer: Optional[str] = None,
    ) -> Tuple[List[dict], bool]:
        """Collect up to max_items matching items, scanning at most _MAX_SCAN.

        Returns (items, has_more) where has_more means scanning stopped early.
        """
        matched: List[dict] = []
        scanned = 0
        for page, has_next in self._iter_pages(url, params, prefer):
            for pos, item in enumerate(page):
                scanned += 1
                if predicate is None or predicate(item):
                    matched.append(item)
                if len(matched) >= max_items or scanned >= _MAX_SCAN:
                    return matched, pos < len(page) - 1 or has_next
        return matched, False

    def _messages_url(self, folder: Optional[str]) -> str:
        if _is_all_folders(folder):
            return f"{_GRAPH_BASE}{self._user_path}/messages"
        return f"{_GRAPH_BASE}{self._user_path}/mailFolders/{folder}/messages"

    def _query_messages(
        self,
        folder: Optional[str],
        filters: List[str],
        select: str,
        limit: int,
        predicate: Optional[Callable[[dict], bool]] = None,
        ascending: bool = False,
    ) -> Tuple[List[dict], bool]:
        """Run a $filter query ordered by receivedDateTime.

        Falls back to client-side sorting when Graph rejects the filter/order
        combination as InefficientFilter.
        """
        url = self._messages_url(folder)
        if filters and not filters[0].startswith("receivedDateTime"):
            filters = [_ANY_DATE_FILTER] + filters
        params = {
            "$top": min(max(limit, 1), _PAGE_SIZE),
            "$select": select,
            "$orderby": f"receivedDateTime {'asc' if ascending else 'desc'}",
        }
        if filters:
            params["$filter"] = " and ".join(filters)
        try:
            return self._collect(url, params, limit, predicate)
        except requests.HTTPError as e:
            if not _is_inefficient_filter(e):
                raise
            log.warning("Outlook: Graph rejected $filter+$orderby as too complex; sorting client-side")
            params.pop("$orderby")
            params["$top"] = _PAGE_SIZE
            items, has_more = self._collect(url, params, _MAX_SCAN, predicate)
            items.sort(key=_received_key, reverse=not ascending)
            return items[:limit], has_more or len(items) > limit

    def _get_message_ref(self, message_id: str, not_found_hint: str = "") -> Dict[str, Any]:
        """Fetch the identifying fields of a single message (1 call)."""
        url = f"{_GRAPH_BASE}{self._user_path}/messages/{message_id}"
        try:
            return self._get(url, params={"$select": "id,conversationId,subject,receivedDateTime,sentDateTime"})
        except requests.HTTPError as e:
            if _status_of(e) == 404:
                raise ToolException(
                    f"Message '{message_id}' not found (deleted, or the ID is not an immutable ID "
                    f"and the message was moved).{not_found_hint}"
                )
            raise

    def _resolve_bound(self, since: Optional[str], after_message_id: Optional[str]) -> _Bound:
        """Combine `since` and `after_message_id` into one bound; the later point wins."""
        since_dt = _parse_dt(since) if since else None
        anchor = None
        if after_message_id:
            anchor = self._get_message_ref(after_message_id, " Pass 'since' with a timestamp instead.")
            anchor_ts = anchor.get("receivedDateTime") or anchor.get("sentDateTime")
            anchor_dt = _parse_dt(anchor_ts) if anchor_ts else None
            if anchor_dt and (since_dt is None or anchor_dt >= since_dt):
                # ge + exclude the anchor so messages sharing its timestamp are not lost
                return _Bound(anchor_dt, True, {after_message_id, anchor.get("id")}, anchor)
        return _Bound(since_dt, False, set(), anchor)

    def _find(
        self,
        folder: Optional[str],
        bound: _Bound,
        senders: Set[str],
        recipients: Set[str],
        unread_only: bool,
        only_new: bool,
        select: str,
        limit: int,
    ) -> Tuple[List[dict], bool]:
        """Shared query for check_new_messages / find_new_messages.

        Time, read state and senders are filtered server-side; recipients
        (distribution lists on To/Cc) are matched client-side.
        """
        apply_bound = only_new and bound.dt is not None
        filters = []
        if apply_bound:
            filters.append(f"receivedDateTime {'ge' if bound.inclusive else 'gt'} {_fmt_dt(bound.dt)}")
        if unread_only:
            filters.append("isRead eq false")
        if senders and not recipients:
            # With recipients present, senders OR recipients is evaluated client-side
            filters.append("(" + " or ".join(
                f"from/emailAddress/address eq {_odata_str(addr)}" for addr in sorted(senders)
            ) + ")")

        def predicate(msg: dict) -> bool:
            if apply_bound and not _is_new(bound, msg):
                return False
            if senders or recipients:
                return bool(_match_reasons(msg, senders, recipients))
            return True

        # Oldest-first when looking for new mail, so a truncated batch can be
        # continued with since=watermark / after_message_id=latest_message_id.
        return self._query_messages(folder, filters, select, limit, predicate, ascending=apply_bound)

    def _folder_id(self, well_known_name: str) -> Optional[str]:
        if well_known_name not in self._folder_ids:
            data = self._get(f"{_GRAPH_BASE}{self._user_path}/mailFolders/{well_known_name}",
                             params={"$select": "id"})
            self._folder_ids[well_known_name] = data.get("id")
        return self._folder_ids[well_known_name]

    # ------------------------------------------------------------------ #
    #  Mail Operations                                                     #
    # ------------------------------------------------------------------ #

    def list_messages(
        self,
        folder: str = "inbox",
        limit: int = 50,
        unread_only: bool = False,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List messages from a mail folder.

        Args:
            folder: Mail folder name (inbox, sentitems, drafts, deleteditems, etc.) or "all"
            limit: Maximum number of messages to return
            unread_only: Only return unread messages
            search: Optional search query (KQL)

        Returns:
            List of message objects with id, conversationId, subject, from, receivedDateTime, isRead, bodyPreview
        """
        try:
            if search:
                # Graph does not support $orderby or $filter together with $search on messages;
                # search results come back ordered by date, unread filtering is applied client-side.
                escaped = search.replace("\\", "\\\\").replace('"', '\\"')
                params = {
                    "$top": min(limit, _PAGE_SIZE),
                    "$select": _LIST_FIELDS,
                    "$search": f'"{escaped}"',
                }
                predicate = (lambda m: m.get("isRead") is False) if unread_only else None
                messages, _ = self._collect(self._messages_url(folder), params, limit, predicate)
            else:
                filters = ["isRead eq false"] if unread_only else []
                messages, _ = self._query_messages(folder, filters, _LIST_FIELDS, limit)

            return [_format_message(msg) for msg in messages]
        except ToolException:
            raise
        except Exception as e:
            log.error("list_messages failed: %s", e)
            raise ToolException(f"Failed to list messages: {e}")

    def check_new_messages(
        self,
        folder: str = "inbox",
        since: Optional[str] = None,
        after_message_id: Optional[str] = None,
        senders: Optional[List[str]] = None,
        recipients: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Cheap "is there anything new?" check.

        Without since/after_message_id, "new" means unread. With them, "new"
        means received after that point (read or not).
        """
        try:
            bound = self._resolve_bound(since, after_message_id)
            senders_set = _normalize_addresses(senders)
            recipients_set = _normalize_addresses(recipients)

            folder_info = None
            if not _is_all_folders(folder):
                folder_info = self._get(
                    f"{_GRAPH_BASE}{self._user_path}/mailFolders/{folder}",
                    params={"$select": "id,displayName,unreadItemCount,totalItemCount"},
                )

            unread_mode = bound.dt is None
            # Plain unread check on one folder: the folder counter is exact, only fetch a preview
            counts_from_folder = unread_mode and folder_info is not None and not (senders_set or recipients_set)
            messages, truncated = self._find(
                folder, bound, senders_set, recipients_set,
                unread_only=unread_mode, only_new=True, select=_CHECK_FIELDS,
                limit=5 if counts_from_folder else _MAX_SCAN,
            )

            if counts_from_folder:
                new_count = folder_info.get("unreadItemCount") or 0
                unread_count = new_count
                truncated = False
            else:
                new_count = len(messages)
                unread_count = sum(1 for m in messages if m.get("isRead") is False)

            newest = _newest(messages)
            latest = sorted(messages, key=_received_key, reverse=True)[:5]
            latest_id = newest.get("id") if newest else (bound.anchor or {}).get("id")
            return {
                "has_new": new_count > 0,
                "new_count": new_count,
                "unread_count": unread_count,
                "folder": folder_info.get("displayName") if folder_info else "all",
                "folder_unread_count": folder_info.get("unreadItemCount") if folder_info else None,
                "new_since": _fmt_dt(bound.dt) if bound.dt else None,
                "latest": [_format_message(m) for m in latest],
                "latest_message_id": latest_id,
                "latest_received_at": newest.get("receivedDateTime") if newest else None,
                "watermark": _watermark(bound, newest),
                "truncated": truncated,
            }
        except ToolException:
            raise
        except Exception as e:
            log.error("check_new_messages failed: %s", e)
            raise ToolException(f"Failed to check new messages: {e}")

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
        """Find messages from people / sent to DLs and flag which of them are new.

        Args:
            folder: Folder name or "all" for the whole mailbox
            since: ISO 8601 timestamp; messages received after it are new
            after_message_id: Messages received after this message are new
            senders: Sender addresses (From)
            recipients: Recipient / distribution list addresses (To or Cc)
            unread_only: Only return unread messages
            only_new: Return only new messages; False returns all matches with an is_new flag
            include_preview: Include bodyPreview (requires Mail.Read)
            limit: Maximum messages to return
        """
        try:
            bound = self._resolve_bound(since, after_message_id)
            senders_set = _normalize_addresses(senders)
            recipients_set = _normalize_addresses(recipients)
            select = _SUMMARY_FIELDS + (",bodyPreview" if include_preview else "")

            messages, truncated = self._find(
                folder, bound, senders_set, recipients_set,
                unread_only=unread_only, only_new=only_new, select=select, limit=limit,
            )

            items = []
            for msg in messages:
                item = _format_message(msg)
                # Without a time bound, "new" means unread
                item["is_new"] = _is_new(bound, msg) if bound.dt else msg.get("isRead") is False
                if senders_set or recipients_set:
                    item["matched_by"] = _match_reasons(msg, senders_set, recipients_set)
                items.append(item)

            newest = _newest(messages)
            return {
                "messages": items,
                "count": len(items),
                "new_count": sum(1 for i in items if i["is_new"]),
                "unread_count": sum(1 for i in items if i["isRead"] is False),
                "new_since": _fmt_dt(bound.dt) if bound.dt else None,
                "order": "oldest_first" if only_new and bound.dt else "newest_first",
                "latest_message_id": newest.get("id") if newest else (bound.anchor or {}).get("id"),
                "watermark": _watermark(bound, newest),
                "truncated": truncated,
            }
        except ToolException:
            raise
        except Exception as e:
            log.error("find_new_messages failed: %s", e)
            raise ToolException(f"Failed to find new messages: {e}")

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
        """Get all messages of a conversation (all folders), oldest first, flagging new ones.

        Args:
            message_id: Any message of the thread (its conversationId is used)
            conversation_id: Conversation ID of the thread
            since: ISO 8601 timestamp; messages received after it are new
            after_message_id: Messages received after this message are new
                (also identifies the thread when message_id/conversation_id are omitted)
            only_new: Return only new messages
            include_body: Include the unique (non-quoted) part of each message body
            limit: Maximum messages to return (newest are kept)
        """
        try:
            if not (message_id or conversation_id or after_message_id):
                raise ToolException("Provide message_id, conversation_id or after_message_id.")

            bound = self._resolve_bound(since, after_message_id)
            if not conversation_id:
                if bound.anchor and (not message_id or message_id == after_message_id):
                    ref = bound.anchor
                else:
                    ref = self._get_message_ref(message_id)
                conversation_id = ref.get("conversationId")
                if not conversation_id:
                    raise ToolException("The message has no conversationId.")

            prefer = 'outlook.body-content-type="text"' if include_body else None
            params = {
                "$top": _PAGE_SIZE,
                "$select": _SUMMARY_FIELDS + (",uniqueBody" if include_body else ",bodyPreview"),
                "$filter": f"conversationId eq {_odata_str(conversation_id)}",
            }
            # conversationId + $orderby is rejected by Graph as too complex; sort client-side
            messages, truncated = self._collect(self._messages_url("all"), params, _MAX_SCAN, prefer=prefer)
            messages.sort(key=_received_key)

            try:
                sent_items_id = self._folder_id("sentitems")
            except Exception as e:
                log.warning("Outlook: could not resolve Sent Items folder id: %s", e)
                sent_items_id = None
            mailbox = (self._mailbox or "").lower()
            items = []
            for msg in messages:
                item = _format_message(msg)
                item["isDraft"] = msg.get("isDraft")
                item["from_me"] = bool(sent_items_id and msg.get("parentFolderId") == sent_items_id) or bool(
                    mailbox and _address(msg.get("from")).lower() == mailbox
                )
                item["is_new"] = _is_new(bound, msg) if bound.dt else msg.get("isRead") is False
                if include_body:
                    item["body"] = (msg.get("uniqueBody") or {}).get("content", "")
                items.append(item)

            new_items = [i for i in items if i["is_new"]]
            new_replies = [i for i in new_items if not i["from_me"] and not i["isDraft"]]
            newest = _newest(messages)

            if only_new:
                items = new_items
            if len(items) > limit:
                items = items[-limit:]
                truncated = True

            return {
                "conversation_id": conversation_id,
                "total_in_thread": len(messages),
                "count": len(items),
                "messages": items,
                "new_count": len(new_items),
                "new_replies_count": len(new_replies),
                "new_since": _fmt_dt(bound.dt) if bound.dt else None,
                "latest_message_id": newest.get("id") if newest else None,
                "latest_received_at": newest.get("receivedDateTime") if newest else None,
                "watermark": _watermark(bound, newest),
                "truncated": truncated,
            }
        except ToolException:
            raise
        except Exception as e:
            log.error("get_thread_messages failed: %s", e)
            raise ToolException(f"Failed to get thread messages: {e}")

    def get_message(self, message_id: str, include_body: bool = True) -> Dict[str, Any]:
        """Get a specific message by ID.

        Args:
            message_id: The unique message identifier
            include_body: Whether to include the full message body

        Returns:
            Message object with full details
        """
        try:
            url = f"{_GRAPH_BASE}{self._user_path}/messages/{message_id}"
            select_fields = ("id,conversationId,subject,from,toRecipients,ccRecipients,"
                             "receivedDateTime,sentDateTime,isRead,importance,hasAttachments")
            if include_body:
                select_fields += ",body"
            params = {"$select": select_fields}

            data = self._get(url, params=params)

            result = {
                "id": data.get("id"),
                "conversationId": data.get("conversationId"),
                "subject": data.get("subject"),
                "from": data.get("from", {}).get("emailAddress", {}),
                "to": [r.get("emailAddress", {}) for r in data.get("toRecipients", [])],
                "cc": [r.get("emailAddress", {}) for r in data.get("ccRecipients", [])],
                "receivedDateTime": data.get("receivedDateTime"),
                "sentDateTime": data.get("sentDateTime"),
                "isRead": data.get("isRead"),
                "importance": data.get("importance"),
                "hasAttachments": data.get("hasAttachments"),
            }
            if include_body:
                body = data.get("body", {})
                result["body"] = body.get("content", "")
                result["bodyContentType"] = body.get("contentType", "text")

            return result
        except ToolException:
            raise
        except Exception as e:
            log.error("get_message failed: %s", e)
            raise ToolException(f"Failed to get message: {e}")

    # ------------------------------------------------------------------ #
    #  Sending                                                             #
    # ------------------------------------------------------------------ #

    def _send_draft(self, draft_id: str) -> None:
        """Send a draft; delete it if Graph rejects the send."""
        try:
            self._post(f"{_GRAPH_BASE}{self._user_path}/messages/{draft_id}/send", None)
        except requests.HTTPError as e:
            status = _status_of(e) or 0
            if 400 <= status < 500:
                try:
                    self._delete(f"{_GRAPH_BASE}{self._user_path}/messages/{draft_id}")
                except Exception as cleanup_error:
                    log.warning("Outlook: failed to delete unsent draft: %s", cleanup_error)
            raise

    def _find_sent_copy(
        self,
        started: datetime,
        subject: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Locate the Sent Items copy of a message sent via sendMail / reply (which return no ID).

        Never raises: the mail is already sent, a failure here must not trigger a resend.
        """
        url = self._messages_url("sentitems")
        if conversation_id:
            params = {"$top": 50, "$select": _SENT_FIELDS,
                      "$filter": f"conversationId eq {_odata_str(conversation_id)}"}
        else:
            params = {"$top": 25, "$select": _SENT_FIELDS,
                      "$filter": f"sentDateTime ge {_fmt_dt(started)}",
                      "$orderby": "sentDateTime desc"}
        try:
            for delay in (1, 2, 4):
                # The Sent Items copy appears asynchronously
                time.sleep(delay)
                candidates = [
                    m for m in self._get(url, params=params).get("value", [])
                    if m.get("sentDateTime") and _parse_dt(m["sentDateTime"]) >= started
                    and (subject is None or (m.get("subject") or "") == subject)
                ]
                if candidates:
                    return max(candidates, key=lambda m: m["sentDateTime"])
        except Exception as e:
            log.warning("Outlook: failed to look up the sent message: %s", e)
        return None

    @staticmethod
    def _sent_result(sent: Optional[dict], fallback_to: Optional[List[str]] = None) -> Dict[str, Any]:
        result = {
            "status": "sent",
            "message_id": sent.get("id") if sent else None,
            "conversation_id": sent.get("conversationId") if sent else None,
            "internet_message_id": sent.get("internetMessageId") if sent else None,
            "subject": sent.get("subject") if sent else None,
            "to": [_address(r) for r in sent.get("toRecipients") or []] if sent else (fallback_to or []),
        }
        if not sent:
            result["note"] = ("The message was sent, but its Sent Items copy was not found yet, "
                              "so message_id / conversation_id are unavailable. Do not resend.")
        return result

    def send_mail(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        html: bool = False,
    ) -> Dict[str, Any]:
        """Send an email.

        Creates a draft and sends it, so the returned message_id (immutable, points to
        the Sent Items copy) and conversation_id can be used to track replies. Falls
        back to /sendMail + Sent Items lookup when drafts are not permitted (no Mail.ReadWrite).

        Args:
            to: List of recipient email addresses
            subject: Email subject
            body: Email body content
            cc: Optional list of CC recipients
            bcc: Optional list of BCC recipients
            html: If True, body is HTML content

        Returns:
            Dict with status, message_id, conversation_id, internet_message_id, subject, to
        """
        try:
            message = {
                "subject": subject,
                "body": {
                    "contentType": "HTML" if html else "Text",
                    "content": body,
                },
                "toRecipients": [{"emailAddress": {"address": addr}} for addr in to],
            }

            if cc:
                message["ccRecipients"] = [{"emailAddress": {"address": addr}} for addr in cc]
            if bcc:
                message["bccRecipients"] = [{"emailAddress": {"address": addr}} for addr in bcc]

            try:
                draft = self._post(f"{_GRAPH_BASE}{self._user_path}/messages", message)
            except requests.HTTPError as e:
                if _status_of(e) != 403:
                    raise
                log.info("Outlook: draft creation forbidden (Mail.ReadWrite missing?), falling back to sendMail")
                started = datetime.now(timezone.utc) - timedelta(minutes=2)
                self._post(f"{_GRAPH_BASE}{self._user_path}/sendMail",
                           {"message": message, "saveToSentItems": True})
                return self._sent_result(self._find_sent_copy(started, subject=subject), fallback_to=to)

            self._send_draft(draft["id"])
            return self._sent_result(draft, fallback_to=to)

        except ToolException:
            raise
        except Exception as e:
            log.error("send_mail failed: %s", e)
            raise ToolException(f"Failed to send email: {e}")

    def reply_to_message(
        self,
        message_id: str,
        body: str,
        reply_all: bool = False,
    ) -> Dict[str, Any]:
        """Reply to a message.

        Args:
            message_id: ID of the message to reply to
            body: Reply body content
            reply_all: If True, reply to all recipients

        Returns:
            Dict with status, message_id, conversation_id, internet_message_id, in_reply_to
        """
        try:
            base = f"{_GRAPH_BASE}{self._user_path}/messages/{message_id}"
            payload = {"comment": body}
            try:
                draft = self._post(f"{base}/{'createReplyAll' if reply_all else 'createReply'}", payload)
            except requests.HTTPError as e:
                if _status_of(e) != 403:
                    raise
                log.info("Outlook: reply draft forbidden (Mail.ReadWrite missing?), falling back to direct reply")
                original = self._get_message_ref(message_id)
                started = datetime.now(timezone.utc) - timedelta(minutes=2)
                self._post(f"{base}/{'replyAll' if reply_all else 'reply'}", payload)
                sent = self._find_sent_copy(started, conversation_id=original.get("conversationId"))
                result = self._sent_result(sent)
                result["in_reply_to"] = message_id
                return result

            self._send_draft(draft["id"])
            result = self._sent_result(draft)
            result["in_reply_to"] = message_id
            return result

        except ToolException:
            raise
        except Exception as e:
            log.error("reply_to_message failed: %s", e)
            raise ToolException(f"Failed to reply to message: {e}")

    def mark_as_read(self, message_id: str, is_read: bool = True) -> str:
        """Mark a message as read or unread.

        Args:
            message_id: ID of the message
            is_read: True to mark as read, False for unread

        Returns:
            Success message
        """
        try:
            url = f"{_GRAPH_BASE}{self._user_path}/messages/{message_id}"
            payload = {"isRead": is_read}

            self._patch(url, payload)
            status = "read" if is_read else "unread"
            return f"Message marked as {status}"

        except ToolException:
            raise
        except Exception as e:
            log.error("mark_as_read failed: %s", e)
            raise ToolException(f"Failed to update message: {e}")

    def list_folders(self) -> List[Dict[str, Any]]:
        """List all top-level mail folders.

        Returns:
            List of folder objects with id, displayName, totalItemCount, unreadItemCount
        """
        try:
            url = f"{_GRAPH_BASE}{self._user_path}/mailFolders"
            params = {"$top": _PAGE_SIZE, "$select": "id,displayName,totalItemCount,unreadItemCount"}
            folders, _ = self._collect(url, params, _MAX_SCAN)

            return [
                {
                    "id": f.get("id"),
                    "displayName": f.get("displayName"),
                    "totalItemCount": f.get("totalItemCount"),
                    "unreadItemCount": f.get("unreadItemCount"),
                }
                for f in folders
            ]
        except ToolException:
            raise
        except Exception as e:
            log.error("list_folders failed: %s", e)
            raise ToolException(f"Failed to list folders: {e}")

    def move_message(self, message_id: str, destination_folder: str) -> Dict[str, Any]:
        """Move a message to another folder.

        Args:
            message_id: ID of the message to move
            destination_folder: Destination folder: well-known name (e.g. deleteditems), display name or ID

        Returns:
            Updated message object
        """
        try:
            folder_id = destination_folder
            if destination_folder.lower() in _WELL_KNOWN_FOLDERS:
                # Graph accepts well-known folder names as destinationId
                folder_id = destination_folder.lower()
            else:
                folders = self.list_folders()
                match = next(
                    (f for f in folders if (f["displayName"] or "").lower() == destination_folder.lower()),
                    None
                )
                if match:
                    folder_id = match["id"]
                elif len(destination_folder) < 40:
                    # Too short to be a Graph folder ID, so it was meant as a display name
                    raise ToolException(f"Folder '{destination_folder}' not found")

            url = f"{_GRAPH_BASE}{self._user_path}/messages/{message_id}/move"
            payload = {"destinationId": folder_id}

            result = self._post(url, payload)
            return {
                "id": result.get("id"),
                "subject": result.get("subject"),
                "message": f"Message moved to {destination_folder}",
            }

        except ToolException:
            raise
        except Exception as e:
            log.error("move_message failed: %s", e)
            raise ToolException(f"Failed to move message: {e}")

    def delete_message(self, message_id: str, permanent: bool = False) -> str:
        """Delete a message.

        Args:
            message_id: ID of the message to delete
            permanent: If True, permanently delete. If False, move to Deleted Items

        Returns:
            Success message
        """
        try:
            if permanent:
                url = f"{_GRAPH_BASE}{self._user_path}/messages/{message_id}"
                self._delete(url)
                return "Message permanently deleted"
            else:
                self.move_message(message_id, "deleteditems")
                return "Message moved to Deleted Items"

        except ToolException:
            raise
        except Exception as e:
            log.error("delete_message failed: %s", e)
            raise ToolException(f"Failed to delete message: {e}")

    def search_messages(
        self,
        query: str,
        folder: str = "inbox",
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Search messages using KQL query.

        Args:
            query: Search query (KQL syntax)
            folder: Folder to search in, or "all" for the whole mailbox
            limit: Maximum results (Graph caps $search at 1000)

        Returns:
            List of matching messages
        """
        return self.list_messages(folder=folder, limit=limit, search=query)
