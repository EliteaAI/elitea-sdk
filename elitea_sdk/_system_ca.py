"""Enable use of the operating system trust store for TLS verification.

Python HTTP libraries (``requests``, ``urllib3``, ``aiohttp``, ``httpcore``/``httpx``,
the ``openai``/``anthropic`` clients, ...) default to the :mod:`certifi` CA bundle
instead of the operating system trust store. In corporate DLP / TLS-interception
environments the proxy re-signs TLS traffic with a private root CA that the org
installs into the *system* trust store. Because the SDK would otherwise never consult
that store, outbound HTTPS calls fail with ``CERTIFICATE_VERIFY_FAILED``.

Calling :func:`enable_system_ca` does two things, both best-effort and idempotent:

1. Injects :mod:`truststore` into the standard library ``ssl`` module so **this**
   Python process's default ``SSLContext`` validates against the system trust store.
2. Exports CA-related environment variables so **child processes** spawned by the SDK
   (the Deno/Pyodide sandbox and Node-based stdio MCP servers) also trust the system
   store. An in-process ``ssl`` monkeypatch does not cross a process boundary, so
   these subprocesses need their own runtime-specific configuration:
     - Deno:  ``DENO_TLS_CA_STORE=system`` (native, macOS/Windows/Linux) and/or
              ``DENO_CERT=<pem>`` (a real PEM file, primarily Linux).
     - Node:  ``NODE_EXTRA_CA_CERTS=<pem>`` (a real PEM file) or, on Node >= 22.15,
              ``NODE_OPTIONS=--use-system-ca`` (native system store).

Both behaviours are enabled by default and can be disabled together by setting the
environment variable ``ELITEA_DISABLE_SYSTEM_CA`` to a truthy value
(``1``/``true``/``yes``).

Known limitation: on macOS/Windows with Node < 22.15 there is no way to hand Node the
system roots automatically (they are not available as a PEM file, and Node lacks the
native flag). Set ``SSL_CERT_FILE`` to a PEM containing your org root CA to cover that
case (it is propagated to every runtime).

See https://kharkevich.org/2024/12/07/i-hate-dlp/ for background.
"""

import logging
import os
import shlex
import ssl
import subprocess
import sys

logger = logging.getLogger(__name__)

_DISABLE_ENV_VAR = "ELITEA_DISABLE_SYSTEM_CA"

# Capture the OS default verify paths *now*, at import time. On Linux the ``cafile``
# (e.g. /etc/ssl/certs/ca-certificates.crt, maintained by update-ca-certificates) is
# the real file that contains org DLP roots. This must be read before truststore
# replaces ``ssl.SSLContext``; it does not depend on truststore either way.
try:
    _DEFAULT_VERIFY_PATHS = ssl.get_default_verify_paths()
    _SSL_CAFILE = _DEFAULT_VERIFY_PATHS.cafile
except Exception:  # pragma: no cover - defensive
    _SSL_CAFILE = None

# Candidate system CA bundles on Linux distros, in preference order. Used only if the
# stdlib-reported cafile is missing (e.g. minimal containers with a stale symlink).
_LINUX_CA_CANDIDATES = (
    "/etc/ssl/certs/ca-certificates.crt",       # Debian/Ubuntu/Alpine
    "/etc/pki/tls/certs/ca-bundle.crt",         # RHEL/CentOS/Fedora
    "/etc/ssl/ca-bundle.pem",                    # openSUSE
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
    "/etc/ssl/cert.pem",                         # some minimal distros
)

_NODE_SYSTEM_CA_FLAG = "--use-system-ca"

# Guards so repeated calls inject only once.
_injected = False
_child_env_cache = None          # None = not computed, dict = computed
_node_gte_22_15_cache = None     # None = not probed, bool = probed


def _is_disabled() -> bool:
    return os.environ.get(_DISABLE_ENV_VAR, "").strip().lower() in ("1", "true", "yes")


def _node_version_gte_22_15() -> bool:
    """Return True iff a ``node`` binary on PATH reports version >= 22.15.

    Cached for the process lifetime. Any failure (node absent, times out, unparsable)
    is swallowed and returns False, so we never enable a flag an older Node rejects.
    """
    global _node_gte_22_15_cache
    if _node_gte_22_15_cache is not None:
        return _node_gte_22_15_cache

    result = False
    try:
        proc = subprocess.run(
            ["node", "-e", "process.stdout.write(process.versions.node)"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        raw = (proc.stdout or "").strip()
        major_str, _, rest = raw.partition(".")
        minor_str = rest.partition(".")[0]
        major, minor = int(major_str), int(minor_str)
        result = (major, minor) >= (22, 15)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Could not probe node version: %s", exc)
        result = False

    _node_gte_22_15_cache = result
    return result


def _find_linux_ca_pem():
    """Return a path to an existing system CA PEM on Linux, or None.

    Falls back to certifi's bundle only as a last resort (it does NOT contain org DLP
    roots), logging a warning so the situation is diagnosable.
    """
    if _SSL_CAFILE and os.path.isfile(_SSL_CAFILE):
        return _SSL_CAFILE
    for candidate in _LINUX_CA_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    try:
        import certifi  # pylint: disable=import-outside-toplevel

        logger.warning(
            "No system CA bundle found; falling back to certifi (%s). This bundle "
            "does NOT contain corporate/DLP root CAs. Set SSL_CERT_FILE to your org "
            "CA bundle if TLS interception is in use.",
            certifi.where(),
        )
        return certifi.where()
    except Exception:  # pylint: disable=broad-except
        return None


def _merge_node_options(flag: str) -> str:
    """Return a NODE_OPTIONS value with ``flag`` present exactly once.

    Preserves any existing NODE_OPTIONS and de-duplicates at the token level (so a
    lookalike such as ``--no-use-system-ca`` does not suppress ``--use-system-ca``).
    """
    existing = os.environ.get("NODE_OPTIONS", "")
    try:
        tokens = shlex.split(existing)
    except ValueError:
        tokens = existing.split()
    if flag not in tokens:
        tokens.append(flag)
    return shlex.join(tokens) if hasattr(shlex, "join") else " ".join(tokens)


def get_child_process_ca_env():
    """Compute CA-related env vars for child processes (Deno sandbox, Node MCP).

    Returns a ``dict[str, str]`` of variables to layer onto a subprocess environment.
    Empty when disabled or when nothing useful can be determined. Never raises; result
    is cached for the process lifetime.

    Tiers:
      0. If ``SSL_CERT_FILE`` is set and points at a real file, trust it on every
         runtime (all platforms).
      1. macOS/Windows: use each runtime's native system-store mode
         (``DENO_TLS_CA_STORE=system`` always; ``NODE_OPTIONS=--use-system-ca`` only
         when Node >= 22.15).
      2. Linux: point Deno/Node at the real system PEM bundle; also set
         ``DENO_TLS_CA_STORE=system`` as belt-and-suspenders.
    """
    global _child_env_cache
    if _child_env_cache is not None:
        return _child_env_cache

    env: dict = {}
    try:
        if _is_disabled():
            _child_env_cache = env
            return env

        # Tier 0: explicit user-provided bundle wins on all platforms.
        user_bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get(
            "REQUESTS_CA_BUNDLE"
        )
        if user_bundle and os.path.isfile(user_bundle):
            env["DENO_CERT"] = user_bundle
            env["NODE_EXTRA_CA_CERTS"] = user_bundle
            _child_env_cache = env
            return env

        if sys.platform in ("darwin", "win32"):
            # Tier 1: native OS store. Deno reads DENO_TLS_CA_STORE at startup and
            # ignores it on old versions; harmless to set unconditionally.
            env["DENO_TLS_CA_STORE"] = "system"
            # Node's flag is fatal on versions that don't know it (exit code 9), so
            # only enable it when we've confirmed Node >= 22.15.
            if _node_version_gte_22_15():
                env["NODE_OPTIONS"] = _merge_node_options(_NODE_SYSTEM_CA_FLAG)
        else:
            # Tier 2: Linux (and other POSIX). Use the real system PEM.
            pem = _find_linux_ca_pem()
            if pem:
                env["DENO_CERT"] = pem
                env["NODE_EXTRA_CA_CERTS"] = pem
            env["DENO_TLS_CA_STORE"] = "system"
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Could not compute child-process CA env: %s", exc)
        env = {}

    _child_env_cache = env
    return env


def _inject_child_process_ca_env() -> None:
    """Export child-process CA env vars into ``os.environ`` (without overwriting).

    Uses ``setdefault`` so an explicit user value is always preserved. This is what
    lets the Deno sandbox and the ``dict(os.environ)``-based MCP spawn sites pick up
    the settings for free.
    """
    for key, value in get_child_process_ca_env().items():
        os.environ.setdefault(key, value)


def enable_system_ca() -> bool:
    """Route TLS verification (this process and child processes) through the OS store.

    Idempotent and best-effort: any failure (including a missing ``truststore``
    package) is logged at DEBUG level and swallowed so that importing the SDK is
    never broken by this step. The child-process env export runs even when the
    in-process ``truststore`` injection is skipped or fails, because the two are
    independent.

    Returns:
        ``True`` if in-process injection is now active (this call or a previous one),
        ``False`` if it was skipped or failed. The return value reflects only the
        in-process injection; child-process env export is always attempted.
    """
    global _injected

    if _injected:
        return True

    if _is_disabled():
        logger.debug(
            "System CA trust store injection disabled via %s", _DISABLE_ENV_VAR
        )
        return False

    # Independent of truststore: give child processes their CA config regardless.
    try:
        _inject_child_process_ca_env()
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Could not export child-process CA env: %s", exc)

    try:
        import truststore  # pylint: disable=import-outside-toplevel

        truststore.inject_into_ssl()
        _injected = True
        logger.debug("Injected system CA trust store via truststore")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Could not enable system CA trust store: %s", exc)
        return False
