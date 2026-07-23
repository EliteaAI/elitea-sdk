"""Enable use of the operating system trust store for TLS verification.

Python HTTP libraries (``requests``, ``urllib3``, ``aiohttp``, ``httpcore``/``httpx``,
the ``openai``/``anthropic`` clients, ...) default to the :mod:`certifi` CA bundle
instead of the operating system trust store. In corporate DLP / TLS-interception
environments the proxy re-signs TLS traffic with a private root CA that the org
installs into the *system* trust store. Because the SDK would otherwise never consult
that store, outbound HTTPS calls fail with ``CERTIFICATE_VERIFY_FAILED``.

Calling :func:`enable_system_ca` injects :mod:`truststore` into the standard library
``ssl`` module so Python's default ``SSLContext`` validates against the system trust
store. This is enabled by default and can be disabled by setting the environment
variable ``ELITEA_DISABLE_SYSTEM_CA`` to a truthy value (``1``/``true``/``yes``).

See https://kharkevich.org/2024/12/07/i-hate-dlp/ for background.
"""

import logging
import os

logger = logging.getLogger(__name__)

_DISABLE_ENV_VAR = "ELITEA_DISABLE_SYSTEM_CA"

# Guard so repeated calls (library import + CLI entry point) inject only once.
_injected = False


def enable_system_ca() -> bool:
    """Inject the OS trust store into Python's default TLS verification.

    Idempotent and best-effort: any failure (including a missing ``truststore``
    package) is logged at DEBUG level and swallowed so that importing the SDK is
    never broken by this step.

    Returns:
        ``True`` if injection is now active (either performed by this call or a
        previous one), ``False`` if it was skipped or failed.
    """
    global _injected

    if _injected:
        return True

    if os.environ.get(_DISABLE_ENV_VAR, "").strip().lower() in ("1", "true", "yes"):
        logger.debug(
            "System CA trust store injection disabled via %s", _DISABLE_ENV_VAR
        )
        return False

    try:
        import truststore  # pylint: disable=import-outside-toplevel

        truststore.inject_into_ssl()
        _injected = True
        logger.debug("Injected system CA trust store via truststore")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Could not enable system CA trust store: %s", exc)
        return False
