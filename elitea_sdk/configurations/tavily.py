import os

from pydantic import BaseModel, ConfigDict, Field, SecretStr

DEFAULT_TAVILY_BASE_URL = "https://api.tavily.com"
# Deployment-level only (tests, egress proxies): a tenant-set URL would capture the bearer key
TAVILY_BASE_URL_ENV = "TAVILY_API_BASE_URL"
CHECK_TIMEOUT_SECONDS = 10


def tavily_base_url() -> str:
    return (os.getenv(TAVILY_BASE_URL_ENV) or DEFAULT_TAVILY_BASE_URL).rstrip("/")


class TavilyConfiguration(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "metadata": {
                "label": "Tavily",
                "icon_url": "tavily.svg",
                "section": "credentials",
                "type": "tavily",
                "categories": ["other"],
                "extra_categories": ["tavily", "web", "search", "internet", "extract", "crawl"],
            }
        }
    )
    api_key: SecretStr = Field(description="Tavily API Key")

    @staticmethod
    def check_connection(settings: dict) -> str | None:
        """Validate the key via the free GET /usage (spends no credits)."""
        import requests

        api_key = settings.get("api_key")
        if hasattr(api_key, "get_secret_value"):
            api_key = api_key.get_secret_value()
        if not api_key or not str(api_key).strip():
            return "Tavily API key is required"
        try:
            response = requests.get(
                f"{tavily_base_url()}/usage",
                headers={"Authorization": f"Bearer {str(api_key).strip()}"},
                timeout=CHECK_TIMEOUT_SECONDS,
            )
        except requests.exceptions.Timeout:
            return "Tavily did not respond in time, please try again later"
        except requests.exceptions.RequestException:
            return "Could not connect to Tavily"
        if response.status_code == 200:
            return None
        if response.status_code == 401:
            return "Invalid Tavily API key"
        if response.status_code in (403, 432, 433):
            return "Tavily rejected the key: plan or usage limit reached"
        if response.status_code == 429:
            return "Tavily rate limit reached, please try again later"
        return f"Tavily connection check failed (HTTP {response.status_code})"
