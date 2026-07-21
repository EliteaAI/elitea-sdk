from pydantic import BaseModel, ConfigDict, Field, SecretStr


class AhaConfiguration(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "metadata": {
                "label": "Aha!",
                "icon_url": "aha.svg",
                "section": "credentials",
                "type": "aha",
                "categories": ["project management"],
                "extra_categories": [
                    "aha",
                    "roadmap",
                    "requirements management",
                    "ideas",
                    "product management",
                ],
            }
        }
    )

    base_url: str = Field(
        description="Aha! base URL (e.g. https://mycompany.aha.io)"
    )
    api_key: SecretStr = Field(
        description="Aha! API token",
        json_schema_extra={"secret": True},
    )

    @staticmethod
    def check_connection(settings: dict) -> str | None:
        """Validate the Aha! credentials by calling GET /api/v1/me.

        Returns None on success, an error message string otherwise.
        """
        import requests

        base_url = (settings.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            return "Aha! base URL is required"
        if not base_url.startswith(("http://", "https://")):
            return "Aha! base URL must start with http:// or https://"

        api_key = settings.get("api_key")
        if not api_key:
            return "Aha! API token is required"
        token = api_key.get_secret_value() if hasattr(api_key, "get_secret_value") else api_key
        if not token or not str(token).strip():
            return "Aha! API token cannot be empty"

        try:
            response = requests.get(
                f"{base_url}/api/v1/me",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
        except requests.exceptions.SSLError as exc:
            return f"SSL certificate verification failed: {exc}"
        except requests.exceptions.ConnectionError:
            return f"Cannot connect to Aha! at {base_url}: connection refused"
        except requests.exceptions.Timeout:
            return f"Connection to Aha! at {base_url} timed out"
        except requests.exceptions.RequestException as exc:
            return f"Error connecting to Aha!: {exc}"

        if response.status_code == 200:
            return None
        if response.status_code == 401:
            return "Authentication failed: invalid Aha! API token"
        if response.status_code == 403:
            return "Access forbidden: check API token permissions"
        if response.status_code == 404:
            return "Aha! API endpoint not found: verify the base URL"
        return f"Aha! API returned status code {response.status_code}"
