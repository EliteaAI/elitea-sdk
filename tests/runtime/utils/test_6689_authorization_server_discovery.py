import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import requests

from elitea_sdk.runtime.utils import mcp_oauth
from elitea_sdk.runtime.utils.mcp_adapter import UnifiedMcpClient
from elitea_sdk.runtime.utils.mcp_oauth import (
    McpAuthorizationRequired,
    _is_issued_by,
    authorization_server_metadata_urls,
    fetch_oauth_authorization_server_metadata,
    legacy_appended_metadata_urls,
)
from tests.runtime.utils.mcp_rig_server import RunningRig

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "as_metadata_6689.json"
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def load_recorded_issuers() -> Dict[str, Dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text())["issuers"]


def build_url_cassette(issuers: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[int, Any]]:
    cassette = {}
    for info in issuers.values():
        for probe in info["probes"]:
            cassette[probe["url"]] = (probe["status"], probe["body"])
    return cassette


RECORDED_ISSUERS = load_recorded_issuers()
URL_CASSETTE = build_url_cassette(RECORDED_ISSUERS)


def recorded_document(issuer_name: str) -> Dict[str, Any]:
    for probe in RECORDED_ISSUERS[issuer_name]["probes"]:
        if probe["status"] == 200:
            return probe["body"]
    raise AssertionError(f"No recorded 200 document for {issuer_name}")


def build_replayed_response(status_code: int, body: Any) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(body).encode() if body is not None else b""
    return response


def patch_requests_get(monkeypatch, cassette: Dict[str, Tuple[int, Any]], calls: List[str]) -> None:
    def replay(url, timeout=None):
        calls.append(url)
        status_code, body = cassette.get(url, (404, None))
        return build_replayed_response(status_code, body)

    monkeypatch.setattr(mcp_oauth.requests, "get", replay)


def path_has_doubled_slash(url: str) -> bool:
    _, _, remainder = url.partition("://")
    return "//" in remainder


@pytest.fixture
def recorded_calls() -> List[str]:
    return []


@pytest.fixture
def replay_cassette(monkeypatch, recorded_calls):
    patch_requests_get(monkeypatch, URL_CASSETTE, recorded_calls)
    return recorded_calls


def test_stripe_resolves_the_inserted_candidate_in_one_request(replay_cassette):
    issuer = RECORDED_ISSUERS["Stripe"]["issuer"]
    document = recorded_document("Stripe")

    result = fetch_oauth_authorization_server_metadata(issuer, timeout=5)

    assert replay_cassette == ["https://access.stripe.com/.well-known/oauth-authorization-server/mcp"]
    assert result["authorization_endpoint"] == document["authorization_endpoint"]
    assert result["token_endpoint"] == document["token_endpoint"]
    assert result["registration_endpoint"] == document["registration_endpoint"]


def test_monday_resolves_the_inserted_candidate_in_one_request(replay_cassette):
    issuer = RECORDED_ISSUERS["monday"]["issuer"]
    document = recorded_document("monday")

    result = fetch_oauth_authorization_server_metadata(issuer, timeout=5)

    assert replay_cassette == ["https://auth.monday.com/.well-known/oauth-authorization-server/mcp"]
    assert result["authorization_endpoint"] == document["authorization_endpoint"]
    assert result["token_endpoint"] == document["token_endpoint"]
    assert result["registration_endpoint"] == document["registration_endpoint"]


def test_box_resolves_without_ever_requesting_a_doubled_slash_url(replay_cassette):
    issuer = RECORDED_ISSUERS["Box"]["issuer"]
    document = recorded_document("Box")

    result = fetch_oauth_authorization_server_metadata(issuer, timeout=5)

    assert not any(path_has_doubled_slash(url) for url in replay_cassette)
    assert result["authorization_endpoint"] == document["authorization_endpoint"]
    assert result["token_endpoint"] == document["token_endpoint"]


def test_github_resolves_the_inserted_rfc8414_candidate_with_s256(replay_cassette):
    issuer = RECORDED_ISSUERS["GitHub"]["issuer"]
    document = recorded_document("GitHub")

    result = fetch_oauth_authorization_server_metadata(issuer, timeout=5)

    assert replay_cassette == ["https://github.com/.well-known/oauth-authorization-server/login/oauth"]
    assert result["authorization_endpoint"] == document["authorization_endpoint"]
    assert result["token_endpoint"] == document["token_endpoint"]
    assert "S256" in result["code_challenge_methods_supported"]


def test_atlassian_tenant_resolves_the_inserted_candidate_first(replay_cassette):
    issuer = RECORDED_ISSUERS["Atlassian authv2"]["issuer"]
    document = recorded_document("Atlassian authv2")

    result = fetch_oauth_authorization_server_metadata(issuer, timeout=5)

    assert replay_cassette == [
        "https://auth.atlassian.com/.well-known/oauth-authorization-server/VCeDsk8ZHncYF1g234fKtc4lNipbBhu3"
    ]
    assert result["authorization_endpoint"] == document["authorization_endpoint"]
    assert result["registration_endpoint"] == document["registration_endpoint"]


def test_entra_v2_tenant_falls_through_to_the_appended_oidc_candidate(replay_cassette):
    issuer = RECORDED_ISSUERS["Entra v2 tenant"]["issuer"]
    document = recorded_document("Entra v2 tenant")
    tenant_path = "72f988bf-86f1-41af-91ab-2d7cd011db47/v2.0"

    result = fetch_oauth_authorization_server_metadata(issuer, timeout=5)

    assert replay_cassette == [
        f"https://login.microsoftonline.com/.well-known/oauth-authorization-server/{tenant_path}",
        f"https://login.microsoftonline.com/.well-known/openid-configuration/{tenant_path}",
        f"https://login.microsoftonline.com/{tenant_path}/.well-known/openid-configuration",
    ]
    assert result["authorization_endpoint"] == document["authorization_endpoint"]
    assert result["token_endpoint"] == document["token_endpoint"]


@pytest.mark.parametrize("issuer_name", ["Entra common v2", "Entra organizations v2", "Entra tenant v1"])
def test_entra_issuers_whose_documents_name_another_issuer_fall_back_to_the_legacy_appended_document(
    issuer_name, replay_cassette
):
    issuer = RECORDED_ISSUERS[issuer_name]["issuer"]
    document = recorded_document(issuer_name)
    inserted_rfc8414, inserted_oidc, appended_oidc = authorization_server_metadata_urls(issuer)
    appended_rfc8414, _ = legacy_appended_metadata_urls(issuer)

    result = fetch_oauth_authorization_server_metadata(issuer, timeout=5)

    assert not _is_issued_by(document, issuer)
    assert replay_cassette == [inserted_rfc8414, inserted_oidc, appended_oidc, appended_rfc8414]
    assert result == document


def test_a_mismatched_document_served_only_at_a_path_inserted_url_is_rejected(monkeypatch, recorded_calls):
    github_issuer = RECORDED_ISSUERS["GitHub"]["issuer"]
    miro_document = recorded_document("Miro")
    candidates = authorization_server_metadata_urls(github_issuer)
    patch_requests_get(monkeypatch, {candidates[0]: (200, miro_document)}, recorded_calls)

    result = fetch_oauth_authorization_server_metadata(github_issuer, timeout=5)

    assert result is None
    assert recorded_calls == candidates + [legacy_appended_metadata_urls(github_issuer)[0]]


def test_legacy_appended_urls_never_double_a_trailing_slash():
    assert legacy_appended_metadata_urls("https://api.box.com/") == [
        "https://api.box.com/.well-known/oauth-authorization-server",
        "https://api.box.com/.well-known/openid-configuration",
    ]


@pytest.mark.parametrize("issuer_name", ["Miro", "Notion"])
def test_origin_only_issuers_keep_the_old_two_url_candidate_list(issuer_name):
    issuer = RECORDED_ISSUERS[issuer_name]["issuer"]
    stripped = issuer.rstrip("/")

    assert authorization_server_metadata_urls(issuer) == [
        f"{stripped}/.well-known/oauth-authorization-server",
        f"{stripped}/.well-known/openid-configuration",
    ]


@pytest.mark.parametrize("issuer_name", ["Miro", "Notion"])
def test_origin_only_issuers_resolve_from_the_first_candidate(issuer_name, replay_cassette):
    issuer = RECORDED_ISSUERS[issuer_name]["issuer"]
    document = recorded_document(issuer_name)

    result = fetch_oauth_authorization_server_metadata(issuer, timeout=5)

    assert replay_cassette == [f"{issuer.rstrip('/')}/.well-known/oauth-authorization-server"]
    assert result["authorization_endpoint"] == document["authorization_endpoint"]
    assert result["token_endpoint"] == document["token_endpoint"]
    assert result["registration_endpoint"] == document["registration_endpoint"]


def test_is_issued_by_tolerates_trailing_slash_differences():
    box_document = recorded_document("Box")
    miro_document = recorded_document("Miro")

    assert _is_issued_by(box_document, "https://api.box.com/")
    assert _is_issued_by(miro_document, "https://mcp.miro.com")
    assert not _is_issued_by(box_document, "https://api.box.com/mcp")
    assert not _is_issued_by(miro_document, "https://mcp.notion.com")


def test_a_document_naming_a_different_issuer_is_skipped_for_the_next_candidate(monkeypatch, recorded_calls):
    github_issuer = RECORDED_ISSUERS["GitHub"]["issuer"]
    github_document = recorded_document("GitHub")
    miro_document = recorded_document("Miro")
    candidates = authorization_server_metadata_urls(github_issuer)

    overridden_cassette = dict(URL_CASSETTE)
    overridden_cassette[candidates[0]] = (200, miro_document)
    patch_requests_get(monkeypatch, overridden_cassette, recorded_calls)

    result = fetch_oauth_authorization_server_metadata(github_issuer, timeout=5)

    assert recorded_calls == candidates
    assert result["authorization_endpoint"] == github_document["authorization_endpoint"]
    assert result["token_endpoint"] == github_document["token_endpoint"]
    assert result != miro_document


def test_extra_endpoints_are_tried_first_and_accepted_without_issuer_check(monkeypatch, recorded_calls):
    tenant_url = "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47"
    entra_document = recorded_document("Entra v2 tenant")
    extra_endpoint = f"{tenant_url}/v2.0/.well-known/openid-configuration"
    assert extra_endpoint in URL_CASSETTE
    patch_requests_get(monkeypatch, URL_CASSETTE, recorded_calls)

    result = fetch_oauth_authorization_server_metadata(tenant_url, timeout=5, extra_endpoints=[extra_endpoint])

    assert recorded_calls == [extra_endpoint]
    assert result["issuer"] != tenant_url
    assert result["token_endpoint"] == entra_document["token_endpoint"]


def test_a_direct_well_known_url_is_fetched_exactly_once(replay_cassette):
    direct_url = "https://access.stripe.com/.well-known/oauth-authorization-server/mcp"
    document = recorded_document("Stripe")

    result = fetch_oauth_authorization_server_metadata(direct_url, timeout=5)

    assert replay_cassette == [direct_url]
    assert result == document


def test_a_direct_well_known_url_skips_issuer_validation(monkeypatch, recorded_calls):
    direct_url = "https://figma.example/.well-known/oauth-authorization-server"
    miro_document = recorded_document("Miro")
    patch_requests_get(monkeypatch, {direct_url: (200, miro_document)}, recorded_calls)

    result = fetch_oauth_authorization_server_metadata(direct_url, timeout=5)

    assert recorded_calls == [direct_url]
    assert result == miro_document


@pytest.fixture
def running_rig(monkeypatch):
    for variable in PROXY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with RunningRig() as running:
        yield running


def connect_and_capture_authorization_required(base_url: str) -> McpAuthorizationRequired:
    async def session():
        async with UnifiedMcpClient(url=f"{base_url}/big401/mcp", timeout=20):
            pass

    with pytest.raises(McpAuthorizationRequired) as raised:
        asyncio.run(session())
    return raised.value


def test_handle_401_discovers_the_rig_authorization_server_without_stubbing(running_rig):
    error = connect_and_capture_authorization_required(running_rig.base_url)

    issuer = f"{running_rig.base_url}/as"
    oauth_authorization_server = error.resource_metadata["oauth_authorization_server"]
    assert oauth_authorization_server["authorization_endpoint"] == f"{issuer}/authorize"
    assert oauth_authorization_server["token_endpoint"] == f"{issuer}/token"
    assert oauth_authorization_server["registration_endpoint"] == f"{issuer}/register"
    assert running_rig.requests == [
        ("POST", "/big401/mcp"),
        ("GET", "/.well-known/oauth-protected-resource/big401/mcp"),
        ("GET", "/.well-known/oauth-authorization-server/as"),
    ]
