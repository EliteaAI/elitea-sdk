"""Where an indexed item came from: the `source_info` object on index reports and search results."""
from typing import Dict, Iterable, Optional
from urllib.parse import SplitResult, parse_qsl, quote, urlencode, urlsplit, urlunsplit

SOURCE_INFO_KEY = "source_info"
SOURCE_TARGET_KEYS = ("repository", "branch")

_DEFAULT_PORTS = {"http": 80, "https": 443}


def single_line(text: str) -> str:
    """Flatten text to one line of at most 512 characters, so it cannot forge lines in rendered output."""
    flattened = "".join(" " if ord(char) < 32 or ord(char) == 127 else char for char in str(text))
    return flattened.strip()[:512]


def build_source_info(source_type: str, location: Optional[str] = None, url: Optional[str] = None,
                      item_id: Optional[object] = None) -> Dict[str, str]:
    """Assemble the object from the fields that are known, leaving the unknown ones out."""
    fields = {
        "type": source_type,
        "location": single_line(location) if location else None,
        "url": url,
        "id": str(item_id) if item_id not in (None, "") else None,
    }
    return {key: value for key, value in fields.items() if value}


def join_location(*parts: object) -> Optional[str]:
    """The known parts of a location, broadest first, joined with " / "."""
    return " / ".join(str(part) for part in parts if part not in (None, "")) or None


def _web_parts(url: Optional[str]) -> Optional[SplitResult]:
    try:
        parts = urlsplit(str(url or "").strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname:
        return None
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    netloc = f"{host}:{port}" if port and port != _DEFAULT_PORTS[scheme] else host
    return parts._replace(scheme=scheme, netloc=netloc, fragment="")


def web_base(configured_url: Optional[str]) -> Optional[str]:
    """The configured URL without credentials, query or fragment; None unless it is http(s)."""
    parts = _web_parts(configured_url)
    if parts is None:
        return None
    return urlunsplit(parts._replace(path=parts.path.rstrip("/"), query=""))


def web_host(configured_url: Optional[str]) -> Optional[str]:
    """The host (and non-default port) of the configured URL; None unless it is http(s)."""
    parts = _web_parts(configured_url)
    return parts.netloc if parts else None


def reuse_url(url: Optional[str], configured_url: Optional[str], kept_query: Iterable[str] = ()) -> Optional[str]:
    """A stored URL, kept only when it has the configured scheme and host, without credentials or fragment."""
    parts, base = _web_parts(url), _web_parts(configured_url)
    if parts is None or base is None or (parts.scheme, parts.netloc) != (base.scheme, base.netloc):
        return None
    allowed = set(kept_query)
    query = urlencode([(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
                       if key in allowed])
    return urlunsplit(parts._replace(query=query))


def encode_path(path: str) -> str:
    """Percent-encode a slash-separated path, keeping its slashes."""
    return quote(path.strip("/"), safe="/")


def encode_ref(ref: str) -> str:
    """Percent-encode a single URL segment such as a branch, slashes included."""
    return quote(ref, safe="")


def web_url(base: Optional[str], *encoded_parts: str, query: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Join a sanitized base with already-encoded path parts and an encoded query."""
    if not base:
        return None
    url = "/".join([base.rstrip("/"), *(part for part in encoded_parts if part)])
    return f"{url}?{urlencode(query)}" if query else url
