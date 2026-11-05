"""Default app / Custom app modes shared by the Teams, Outlook and SharePoint credentials (#7149).

In Default app mode a credential stores only ``oauth_app_id``, a reference to an OAuth app
an administrator configured in the Admin Portal; the platform offers only apps of the
credential type's provider (Microsoft Entra ID for these three). The platform fills in the
app's client ID, secret, OAuth authority, scopes and token refresh server-side whenever the
credential is used, so the secret never reaches the browser and an admin change needs no
user action. Custom app mode keeps every option on the credential.
"""
from pydantic import Field

OAUTH_APP_ID_FIELD = Field(
    default=None,
    title="App",
    description="OAuth app configured by your administrator",
    json_schema_extra={"ui_component": "oauth_app_select"},
)


def sign_in_scopes(scopes, auto_refresh_token=None) -> list:
    """Scopes a Microsoft sign-in asks for (#7415).

    ``offline_access`` makes Microsoft return a refresh token, so it is asked for unless token
    refresh is switched off (Auto refresh token off on the credential or the admin's app). A
    missing setting keeps it, as before: older credentials never stored the switch.
    """
    wanted = [scope for scope in (scopes or []) if scope and scope != "offline_access"]
    return wanted if auto_refresh_token is False else ["offline_access", *wanted]


def require_app_client(config) -> None:
    """A credential needs either an admin app or its own client (Custom app)."""
    if getattr(config, "oauth_app_id", None):
        return
    missing = [name for name in ("client_id", "client_secret") if not getattr(config, name, None)]
    if missing:
        raise ValueError(
            f"Select a Default app or enter {' and '.join(missing)} for a Custom app"
        )


def token_for_credential(tokens: dict | None, config: dict) -> dict | str | None:
    """The browser token for this credential, only when it was issued to the credential's client.

    A sign-in belongs to one OAuth client: once the platform moves a credential to another app,
    a token made with the previous client must not be used (#7149). Tokens without a recorded
    client (older sign-ins) are kept as before.
    """
    if not tokens:
        return None
    endpoint = config.get("oauth_discovery_endpoint")
    config_uuid = config.get("configuration_uuid")
    token = tokens.get(f"{config_uuid}:{endpoint}") if config_uuid else None
    if token is None:
        token = tokens.get(endpoint)
    if isinstance(token, dict) and token.get("client_id") and config.get("client_id"):
        if token["client_id"] != config["client_id"]:
            return None
    return token
