"""
Runtime clients package.
"""

try:
    from .client import EliteAClient
    __all__ = ['EliteAClient']
except ImportError as e:
    # Handle case where dependencies are not available
    import logging
    logging.getLogger(__name__).debug(f"Failed to import EliteAClient: {e}")
    __all__ = []