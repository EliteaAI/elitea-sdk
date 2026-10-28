"""#7149: a browser token is used only for the OAuth client it was issued to."""
from elitea_sdk.configurations.oauth_app_mode import token_for_credential

ENDPOINT = "https://login.microsoftonline.com/contoso"
CONFIG = {"configuration_uuid": "u1", "oauth_discovery_endpoint": ENDPOINT, "client_id": "admin-app"}


def test_token_of_the_credential_client_is_used():
    tokens = {f"u1:{ENDPOINT}": {"access_token": "t", "client_id": "admin-app"}}
    assert token_for_credential(tokens, CONFIG)["access_token"] == "t"


def test_token_of_another_client_is_ignored():
    tokens = {f"u1:{ENDPOINT}": {"access_token": "t", "client_id": "own-app"}}
    assert token_for_credential(tokens, CONFIG) is None


def test_older_tokens_without_a_client_still_work():
    assert token_for_credential({ENDPOINT: {"access_token": "t"}}, CONFIG)["access_token"] == "t"


def test_no_tokens():
    assert token_for_credential(None, CONFIG) is None
