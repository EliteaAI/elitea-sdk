"""Shared utilities for Atlassian (Jira / Confluence) credential configurations."""
from typing import Optional
from urllib.parse import urlparse


def url_host_matches_domain(url: Optional[str], domain: str) -> bool:
    """Return True if *url*'s host is exactly *domain* or a subdomain of it.

    Host-based matching, not substring matching: this is robust against
    spoofing such as ``https://evil.atlassian.net.attacker.com`` or
    ``https://attacker.com/?x=bitbucket.org`` — both of which would pass a naive
    ``domain in url`` check but resolve to a non-matching host here.

    The URL may omit its scheme (e.g. a bare ``bitbucket.org/foo``); in that
    case the host is recovered by reparsing with a ``//`` netloc marker.
    """
    if not url:
        return False
    candidate = url.strip()
    host = urlparse(candidate).hostname
    if not host:
        # No scheme: urlparse put the authority in the path. Reparse so that
        # 'bitbucket.org/foo' yields hostname 'bitbucket.org'.
        host = urlparse('//' + candidate).hostname
    host = (host or '').lower()
    domain = domain.lower().lstrip('.')
    return host == domain or host.endswith('.' + domain)


_ATLASSIAN_HOSTING_TOOLTIP = (
    "Hosting defines how the API client connects to your instance. "
    "Auto detects hosting type from the Base URL automatically. "
    "Select Cloud for Atlassian-hosted instances (*.atlassian.net) or "
    "Server for self-hosted / Data Center deployments. "
    "Incorrect hosting type may cause authentication or API failures."
)


def _hosting_to_cloud(hosting: Optional[str], base_url: Optional[str]) -> bool:
    """Resolve a hosting string to the boolean cloud flag used by Atlassian clients.

    Accepted values are case-insensitive UI literals such as ``'Auto'``,
    ``'Cloud'``, and ``'Server'`` (lower-case forms are also accepted).

    Resolution order:
    1. ``'Cloud'`` / ``'cloud'``  → ``True``
    2. ``'Server'`` / ``'server'`` → ``False``
    3. ``'Auto'`` / ``'auto'`` / ``None`` — auto-detect from *base_url*:
       - URL contains ``.atlassian.net`` → ``True`` (Cloud)
       - Otherwise → ``False`` (Server / Data Center)
    """
    normalized_hosting = (hosting or '').strip().lower()

    if normalized_hosting == 'cloud':
        return True
    if normalized_hosting == 'server':
        return False
    # 'auto' or None — detect from URL
    if url_host_matches_domain(base_url, 'atlassian.net'):
        return True
    return False


def _validate_atlassian_hosting_selection(
    hosting: Optional[str],
    base_url: Optional[str],
    product_name: str,
) -> Optional[str]:
    """Validate that an explicit Atlassian hosting selection matches the URL.

    Returns an error message for explicit Cloud/Server mismatches and ``None``
    when hosting is Auto/empty or the URL matches the selected hosting type.
    """
    normalized_hosting = (hosting or '').strip().lower()
    if normalized_hosting in {'', 'auto'}:
        return None

    parsed = urlparse((base_url or '').strip())
    if not parsed.hostname:
        return None

    host = (parsed.hostname or '').lower()
    is_cloud_url = host.endswith('.atlassian.net')

    if normalized_hosting == 'cloud' and not is_cloud_url:
        return (
            f"Hosting is set to Cloud, but the {product_name} Base URL does not match "
            "Atlassian Cloud (*.atlassian.net)."
        )

    if normalized_hosting == 'server' and is_cloud_url:
        return (
            f"Hosting is set to Server, but the {product_name} Base URL points to "
            "Atlassian Cloud (*.atlassian.net)."
        )

    return None


def _resolve_api_version(api_version: Optional[str], cloud: Optional[bool], base_url: Optional[str]) -> str:
    """Resolve ``'auto'`` api_version to ``'2'`` or ``'3'`` based on cloud/hosting setting.

    Used by Jira (which has REST API v2 and v3).

    Resolution order:
    1. If *api_version* is explicitly ``'2'`` or ``'3'``, return as-is.
    2. If *api_version* is ``'auto'`` (or ``None``/empty):
       - *cloud* is ``True``               → ``'3'``
       - *base_url* contains ``.atlassian.net`` → ``'3'``
       - Otherwise (Server / Data Center)  → ``'2'``
    """
    if api_version and api_version in ('2', '3'):
        return api_version
    # Auto-resolve
    if cloud is True:
        return '3'
    if url_host_matches_domain(base_url, 'atlassian.net'):
        return '3'
    return '2'


def _resolve_confluence_api_version(api_version: Optional[str], cloud: Optional[bool], base_url: Optional[str]) -> str:
    """Resolve ``'auto'`` api_version to ``'1'`` or ``'2'`` for Confluence.

    Confluence offers REST API v1 (legacy ``/rest/api``) and v2 (``/api/v2``,
    Cloud only). There is no public Confluence REST API v3.

    Resolution order:
    1. If *api_version* is explicitly ``'1'`` or ``'2'``, return as-is.
    2. If *api_version* is ``'auto'`` (or ``None``/empty):
       - *cloud* is ``True``               → ``'2'``
       - *base_url* contains ``.atlassian.net`` → ``'2'``
       - Otherwise (Server / Data Center)  → ``'1'`` (v2 is Cloud-only)
    """
    if api_version and api_version in ('1', '2'):
        return api_version
    # Auto-resolve
    if cloud is True:
        return '2'
    if url_host_matches_domain(base_url, 'atlassian.net'):
        return '2'
    return '1'


