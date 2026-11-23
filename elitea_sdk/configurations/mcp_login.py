from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class McpLoginConfiguration(BaseModel):
    """A remote MCP server login saved for unattended runs; Main refreshes it, the runtime only sees access tokens."""
    model_config = ConfigDict(
        json_schema_extra={
            "metadata": {
                "label": "MCP login",
                "icon_url": "mcp.svg",
                "section": "credentials",
                "type": "mcp_login",
                "categories": ["other"],
                "extra_categories": ["mcp", "oauth", "login"],
            }
        }
    )
    url: str = Field(description="MCP server URL this login is for, e.g. https://api.githubcopilot.com/mcp/")
    client_id: Optional[str] = Field(default=None, description="OAuth client ID (empty: registered automatically)")
    client_secret: Optional[SecretStr] = Field(default=None, description="OAuth client secret, if the client has one")
    scopes: Optional[str] = Field(default=None, description="Space-separated scopes to request (optional)")
    # Everything below is set by the Login button, never typed by the user
    token_endpoint: Optional[str] = Field(default=None, json_schema_extra={"hidden": True})
    resource: Optional[str] = Field(default=None, json_schema_extra={"hidden": True})
    used_dcr: Optional[bool] = Field(default=None, json_schema_extra={"hidden": True})
    account: Optional[str] = Field(default=None, description="Signed-in account", json_schema_extra={"hidden": True})
    refresh_token: Optional[SecretStr] = Field(
        default=None, description="Refresh token of the saved login (set by Login)",
        json_schema_extra={"hidden": True},
    )

    @staticmethod
    def check_connection(settings: dict) -> str | None:
        """Only the URL can be checked here; the login itself is made and checked from the MCP toolkit page."""
        url = (settings.get("url") or "").strip()
        if not url.startswith(("https://", "http://")):
            return "MCP server URL must start with http:// or https://"
        return None
