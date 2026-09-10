"""Tests for canonical_resource URL normalization.

Regression coverage for the trailing-slash mismatch bug where tokens stored by
the frontend under "/mcp" (no trailing slash) were not found by the backend
when looking up under "/mcp/" (with trailing slash).

The fix: canonical_resource strips trailing slashes from all paths, not just
the root path, so both sides always agree on the canonical form.
"""
import pytest

from elitea_sdk.runtime.utils.mcp_oauth import canonical_resource


# ---------------------------------------------------------------------------
# Trailing-slash normalization (core regression cases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url, expected", [
    # Root URL — trailing slash was already stripped before fix
    ("https://example.com/", "https://example.com"),
    ("http://example.com/", "http://example.com"),
    # Non-root path with trailing slash — the regression case (GH Copilot MCP)
    ("https://api.githubcopilot.com/mcp/", "https://api.githubcopilot.com/mcp"),
    ("https://mcp.atlassian.com/v1/mcp/authv2/", "https://mcp.atlassian.com/v1/mcp/authv2"),
    ("https://example.com/api/v1/mcp/", "https://example.com/api/v1/mcp"),
    # Already canonical — must be idempotent
    ("https://api.githubcopilot.com/mcp", "https://api.githubcopilot.com/mcp"),
    ("https://example.com", "https://example.com"),
])
def test_trailing_slash_stripped(url, expected):
    assert canonical_resource(url) == expected


def test_idempotent():
    url = "https://api.githubcopilot.com/mcp/"
    once = canonical_resource(url)
    twice = canonical_resource(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Scheme and host case normalization
# ---------------------------------------------------------------------------

def test_scheme_lowercased():
    assert canonical_resource("HTTPS://Example.COM/mcp/") == "https://example.com/mcp"


def test_host_lowercased():
    assert canonical_resource("https://API.GitHub.COM/mcp") == "https://api.github.com/mcp"


# ---------------------------------------------------------------------------
# Frontend/backend key agreement (the concrete bug scenario)
# ---------------------------------------------------------------------------

def test_frontend_backend_key_agreement():
    """Token stored by frontend under no-trailing-slash key must match canonical form."""
    server_url_from_backend = "https://api.githubcopilot.com/mcp/"   # from interrupt message
    token_key_from_frontend = "https://api.githubcopilot.com/mcp"    # stored by canonicalizeServerUrl

    assert canonical_resource(server_url_from_backend) == token_key_from_frontend


def test_mcp_tokens_lookup_finds_token_after_canonicalization():
    """Simulate the mcp_tokens lookup: token stored without trailing slash must be found
    when the server_url in the interrupt has a trailing slash."""
    mcp_tokens = {
        "https://api.githubcopilot.com/mcp": {"access_token": "gho_abc123", "session_id": None},
    }
    server_url = "https://api.githubcopilot.com/mcp/"
    canonical = canonical_resource(server_url)
    assert mcp_tokens.get(canonical) is not None
    assert mcp_tokens[canonical]["access_token"] == "gho_abc123"


# ---------------------------------------------------------------------------
# Path preservation (non-trailing parts must be unchanged)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url, expected", [
    ("https://example.com/v1/mcp", "https://example.com/v1/mcp"),
    ("https://example.com/api/v2/endpoint", "https://example.com/api/v2/endpoint"),
    ("https://example.com/path/to/resource/", "https://example.com/path/to/resource"),
])
def test_path_preserved(url, expected):
    assert canonical_resource(url) == expected


# ---------------------------------------------------------------------------
# Port handling
# ---------------------------------------------------------------------------

def test_port_preserved():
    assert canonical_resource("https://example.com:8443/mcp/") == "https://example.com:8443/mcp"


def test_standard_port_preserved():
    assert canonical_resource("https://example.com:443/mcp/") == "https://example.com:443/mcp"
