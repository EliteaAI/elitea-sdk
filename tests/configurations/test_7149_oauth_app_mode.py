"""#7149: Teams, Outlook and SharePoint credentials work with an admin app (Default) or their own client (Custom)."""
import pytest

from elitea_sdk.configurations.outlook import OutlookConfiguration
from elitea_sdk.configurations.sharepoint import SharepointConfiguration
from elitea_sdk.configurations.teams import TeamsConfiguration

ENDPOINT = "https://login.microsoftonline.com/contoso"
EXTRA = {SharepointConfiguration: {"site_url": "https://contoso.sharepoint.com/sites/a"}}
MODELS = [TeamsConfiguration, OutlookConfiguration, SharepointConfiguration]


@pytest.mark.parametrize("model", MODELS)
def test_default_app_needs_no_client(model):
    config = model(oauth_app_id="app-uuid", **EXTRA.get(model, {}))
    assert config.client_id is None and config.client_secret is None


@pytest.mark.parametrize("model", MODELS)
def test_existing_custom_credentials_still_validate(model):
    config = model(client_id="c", client_secret="s", oauth_discovery_endpoint=ENDPOINT, **EXTRA.get(model, {}))
    assert config.oauth_app_id is None


@pytest.mark.parametrize("model", MODELS)
def test_neither_app_nor_client_is_rejected(model):
    with pytest.raises(ValueError, match="Default app"):
        model(**EXTRA.get(model, {}))


@pytest.mark.parametrize("model", MODELS)
def test_schema_offers_default_and_custom_app(model):
    schema = model.model_json_schema()
    names = [s["name"] for s in schema["metadata"]["sections"]["auth"]["subsections"]]
    assert names[:2] == ["Default app", "Custom app"]
    assert schema["properties"]["oauth_app_id"]["ui_component"] == "oauth_app_select"
    assert "client_id" not in (schema.get("required") or [])
