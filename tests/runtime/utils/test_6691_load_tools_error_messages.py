"""Load Tools must surface the same credential messages the agent run does (#6691).

With a configured PAT the adapter raises curated ValueErrors; the sync task passes them
through `extract_user_friendly_mcp_error`, which used to remap the 400 text to a generic
"Malformed request" message. Re-breaks if either message leaves the curated list or its
text drifts from the adapter's.
"""

import pytest

from elitea_sdk.runtime.utils import mcp_adapter, mcp_transport_negotiation
from elitea_sdk.runtime.utils.mcp_oauth import (
    GITHUB_BAD_TOKEN_MESSAGE,
    INVALID_CONFIGURED_CREDENTIALS_MESSAGE,
    extract_user_friendly_mcp_error,
)

PAT_HEADERS = {"Authorization": "Bearer configured-pat"}


@pytest.mark.parametrize("message", [GITHUB_BAD_TOKEN_MESSAGE, INVALID_CONFIGURED_CREDENTIALS_MESSAGE])
def test_configured_credential_messages_reach_the_user_verbatim(message):
    assert extract_user_friendly_mcp_error(ValueError(message), PAT_HEADERS) == message


def test_a_generic_400_is_still_mapped_for_unconfigured_auth():
    mapped = extract_user_friendly_mcp_error(ValueError("HTTP 400 Bad Request"), None)

    assert mapped != GITHUB_BAD_TOKEN_MESSAGE
    assert mapped.startswith("Bad request (400).")


def test_negotiator_and_adapter_share_the_single_message_definitions():
    assert mcp_transport_negotiation.GITHUB_BAD_TOKEN_MESSAGE is GITHUB_BAD_TOKEN_MESSAGE
    assert mcp_adapter.GITHUB_BAD_TOKEN_MESSAGE is GITHUB_BAD_TOKEN_MESSAGE
    assert mcp_adapter.INVALID_CONFIGURED_CREDENTIALS_MESSAGE is INVALID_CONFIGURED_CREDENTIALS_MESSAGE
