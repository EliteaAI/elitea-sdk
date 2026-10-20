"""Resolve a SharePoint/OneDrive URL to the single file it points at.

Browser URLs come in a few shapes, each carrying enough to address the file
without enumerating its folder:

* direct file path   ``/sites/Site/Shared Documents/a/report.pdf``
* library viewer     ``/sites/Site/Shared Documents/Forms/AllItems.aspx?id=/sites/Site/...``
* OneDrive viewer    ``/personal/user/_layouts/15/onedrive.aspx?id=/personal/user/...``
* Office for the web ``/sites/Site/_layouts/15/Doc.aspx?sourcedoc={GUID}&file=a.docx``
* existing-access    ``/:w:/r/sites/Site/_layouts/15/Doc.aspx?sourcedoc={GUID}``
* sharing link       ``/:w:/s/Site/EaBcD...?e=xyz`` (resolved by the Graph /shares API)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from langchain_core.tools import ToolException

_EXISTING_ACCESS_LINK_PREFIX = re.compile(r'^/:[a-z]+:/r(?=/)', re.IGNORECASE)
_SHARING_LINK_PREFIX = re.compile(r'^/:[a-z]+:/', re.IGNORECASE)
_LAYOUTS_SEGMENT = re.compile(r'/_layouts/', re.IGNORECASE)
_GUID = re.compile(
    r'^\{?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\}?$',
    re.IGNORECASE)

NOT_A_FILE_URL_MESSAGE = (
    "The URL does not identify a single file (it looks like a site, library, "
    "folder or list view). Open the file in the browser and copy its URL, or "
    "use a sharing link to the file."
)


@dataclass(frozen=True)
class SharepointFileUrl:
    host: str
    site_path: str = ''
    server_relative_paths: Tuple[str, ...] = ()
    unique_id: Optional[str] = None
    is_sharing_link: bool = False


def _id_path_candidates(raw_query: str) -> Tuple[str, ...]:
    """SharePoint (ASP.NET) decodes '+' in the query as a space, so that reading
    goes first; the literal-'+' reading covers URLs whose '+' was decoded from
    %2B before they reached the tool."""
    raw_id = next((value for key, _, value in (pair.partition('=') for pair in raw_query.split('&'))
                   if unquote(key).lower() == 'id'), '')
    candidates = (unquote(raw_id.replace('+', ' ')).rstrip('/'), unquote(raw_id).rstrip('/'))
    return tuple(dict.fromkeys(path for path in candidates if path.startswith('/')))


def parse_sharepoint_file_url(url: Optional[str]) -> SharepointFileUrl:
    parsed = urlparse((url or '').strip())
    if parsed.scheme != 'https' or not parsed.hostname:
        raise ToolException("Invalid SharePoint URL. Must be a full HTTPS URL.")

    host = parsed.hostname
    path = _EXISTING_ACCESS_LINK_PREFIX.sub('', unquote(parsed.path))
    if _SHARING_LINK_PREFIX.match(path):
        return SharepointFileUrl(host=host, is_sharing_link=True)

    query = {key.lower(): values[0] for key, values in parse_qs(parsed.query).items()}

    if 'sourcedoc' in query:
        guid = _GUID.match(query['sourcedoc'].strip())
        if not guid:
            raise ToolException(
                "Invalid SharePoint URL: the 'sourcedoc' parameter is not a document id.")
        site_path = _LAYOUTS_SEGMENT.split(path, maxsplit=1)[0].rstrip('/')
        return SharepointFileUrl(host=host, site_path=site_path, unique_id=guid.group(1))

    id_paths = _id_path_candidates(parsed.query)
    if id_paths:
        return SharepointFileUrl(host=host, server_relative_paths=id_paths)

    file_path = path.rstrip('/')
    if not file_path or _LAYOUTS_SEGMENT.search(file_path + '/') or file_path.lower().endswith('.aspx'):
        raise ToolException(NOT_A_FILE_URL_MESSAGE)
    return SharepointFileUrl(host=host, server_relative_paths=(file_path,))
