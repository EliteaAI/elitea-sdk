"""Shared utilities for Atlassian (Jira / Confluence) credential configurations."""
from typing import Optional


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
    if hosting == 'cloud':
        return True
    if hosting == 'server':
        return False
    # 'auto' or None — detect from URL
    if base_url and '.atlassian.net' in base_url.lower():
        return True
    return False


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


