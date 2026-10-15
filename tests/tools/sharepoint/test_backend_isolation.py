"""Per-instance backend isolation for SharepointApiWrapper.

The backend used to be assigned on the class during validation, so every
wrapper instance in the process ended up sharing the last-built backend.
"""

from unittest.mock import patch

import pytest

from elitea_sdk.tools.sharepoint.api_wrapper import SharepointApiWrapper

MODULE = "elitea_sdk.tools.sharepoint.api_wrapper"
SITE = "https://tenant.sharepoint.com/sites/x"


class _Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_graph_backends_are_per_instance():
    with patch(f"{MODULE}.SharepointGraphWrapper", _Recorder):
        a = SharepointApiWrapper(site_url=SITE, llm=None, token="token-a", scopes=["Sites.Read.All"])
        b = SharepointApiWrapper(site_url=SITE, llm=None, token="token-b", scopes=["Sites.Read.All"])

    assert a._backend is not b._backend
    assert a._backend.kwargs["token"] == "token-a"
    assert b._backend.kwargs["token"] == "token-b"


def test_rest_app_credential_backends_are_per_instance():
    with patch(f"{MODULE}.SharepointRestWrapper", _Recorder):
        a = SharepointApiWrapper(site_url=SITE, llm=None, client_id="id-a", client_secret="secret-a")
        b = SharepointApiWrapper(site_url=SITE, llm=None, client_id="id-b", client_secret="secret-b")

    assert a._backend is not b._backend
    assert a._backend.kwargs["client_id"] == "id-a"
    assert b._backend.kwargs["client_id"] == "id-b"


def test_instance_backend_not_leaked_onto_class():
    with patch(f"{MODULE}.SharepointGraphWrapper", _Recorder):
        SharepointApiWrapper(site_url=SITE, llm=None, token="token-a", scopes=["Sites.Read.All"])

    assert "_backend" not in SharepointApiWrapper.__dict__


def test_missing_credentials_still_refused():
    from langchain_core.tools import ToolException

    with pytest.raises(ToolException, match="Authentication error"):
        SharepointApiWrapper(site_url=SITE, llm=None)
