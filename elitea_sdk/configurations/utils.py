"""Shared utilities for Atlassian (Jira / Confluence) credential configurations."""
from typing import Optional
from urllib.parse import urlparse


_ATLASSIAN_HOSTING_TOOLTIP = (
    "Hosting defines how the API client connects to your instance. "
    "Auto detects hosting type from the Base URL automatically. "
    "Select Cloud for Atlassian-hosted instances (*.atlassian.net) or "
    "Server for self-hosted / Data Center deployments. "
    "Incorrect hosting type may cause authentication or API failures."
)


def _hosting_to_cloud(hosting: Optional[str], base_url: Optional[str]) -> bool:
    """Resolve a hosting string (``'auto'`` | ``'cloud'`` | ``'server'``) to a
    boolean *cloud* flag used by the ``atlassian-python-api`` client.

    Resolution order:
    1. ``'cloud'``  → ``True``
    2. ``'server'`` → ``False``
    3. ``'auto'`` / ``None`` — auto-detect from *base_url*:
       - URL contains ``.atlassian.net`` → ``True`` (Cloud)
       - Otherwise → ``False`` (Server / Data Center)
    """
    normalized_hosting = (hosting or '').strip().lower()

    if normalized_hosting == 'cloud':
        return True
    if normalized_hosting == 'server':
        return False
    # 'auto' or None — detect from URL
    if base_url and '.atlassian.net' in base_url.lower():
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
    if base_url and '.atlassian.net' in base_url.lower():
        return '3'
    return '2'


