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


def require_app_client(config) -> None:
    """A credential needs either an admin app or its own client (Custom app)."""
    if getattr(config, "oauth_app_id", None):
        return
    missing = [name for name in ("client_id", "client_secret") if not getattr(config, name, None)]
    if missing:
        raise ValueError(
            f"Select a Default app or enter {' and '.join(missing)} for a Custom app"
        )
