import json
import logging
import time
from typing import Any, Dict, Optional

import requests
from FigmaPy import FigmaPy
from langchain_core.tools import ToolException

logger = logging.getLogger(__name__)

# Status codes eligible for retry
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_DEFAULT_BACKOFF = (2, 4, 8)
# Do not retry if Retry-After exceeds this (likely monthly quota)
_MAX_RETRY_AFTER_S = 60
_MAX_WALL_CLOCK_S = 30


class EliteAFigmaPy(FigmaPy):
    """Thin wrapper over FigmaPy with retry logic and structured error reporting."""

    def api_request(self, endpoint: str, method: str = "get", payload: Optional[str] = None) -> Dict[str, Any]:
        """Send an API request with automatic retry on transient errors."""
        method = method.lower()

        if payload is None:
            payload = ""

        if self.oauth2:
            header = {"Authorization": f"Bearer {self.api_token}"}
        else:
            header = {"X-Figma-Token": f"{self.api_token}"}

        header["Content-Type"] = "application/json"

        url = f"{self.api_uri}{endpoint}"

        attempt = 0
        started = time.monotonic()
        last_response = None

        while True:
            try:
                response = self._send_http(method, url, header, payload)
            except (requests.HTTPError, requests.exceptions.SSLError, requests.RequestException) as e:
                raise ToolException(f"Figma API request failed: {e}") from e

            # Happy path
            if 200 <= response.status_code < 300:
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    return {"raw": response.text}

            last_response = response

            # Decide whether to retry
            if (response.status_code in _RETRYABLE_STATUS
                    and attempt < _MAX_RETRIES
                    and (time.monotonic() - started) < _MAX_WALL_CLOCK_S):
                sleep_s = self._compute_retry_sleep(response, attempt)
                if sleep_s is not None:
                    logger.info(
                        "Figma API %d on %s — retry %d/%d in %.1fs",
                        response.status_code, endpoint, attempt + 1, _MAX_RETRIES, sleep_s,
                    )
                    time.sleep(sleep_s)
                    attempt += 1
                    continue

            # Not retryable or retries exhausted
            break

        raise self._build_error(last_response)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _send_http(method: str, url: str, header: dict, payload: str) -> requests.Response:
        """Dispatch a single HTTP request."""
        if method == "head":
            return requests.head(url, headers=header)
        elif method == "delete":
            return requests.delete(url, headers=header)
        elif method == "get":
            return requests.get(url, headers=header, data=payload)
        elif method == "options":
            return requests.options(url, headers=header)
        elif method == "post":
            return requests.post(url, headers=header, data=payload)
        elif method == "put":
            return requests.put(url, headers=header, data=payload)
        else:
            raise ToolException(f"Unsupported HTTP method: {method}")

    @staticmethod
    def _compute_retry_sleep(response: requests.Response, attempt: int) -> Optional[float]:
        """Return seconds to sleep before retrying, or None to stop retrying."""
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait = int(retry_after)
                except (ValueError, TypeError):
                    wait = _DEFAULT_BACKOFF[min(attempt, len(_DEFAULT_BACKOFF) - 1)]
                if wait > _MAX_RETRY_AFTER_S:
                    # Monthly quota — do not retry
                    return None
                return float(wait)
        # Exponential backoff for 5xx or 429 without Retry-After
        return float(_DEFAULT_BACKOFF[min(attempt, len(_DEFAULT_BACKOFF) - 1)])

    @staticmethod
    def _build_error(response: requests.Response) -> ToolException:
        """Build a ToolException with upgrade link for 429 responses."""
        try:
            data: Any = response.json()
        except ValueError:
            data = None

        message = response.text or f"HTTP {response.status_code} {response.reason}"
        details = data or {"status": response.status_code, "message": message}
        error_msg = f"Figma API error {response.status_code}: {message}. Details: {details}"

        # Surface upgrade link for rate-limited users
        if response.status_code == 429:
            upgrade_link = response.headers.get("X-Figma-Upgrade-Link")
            rate_type = response.headers.get("X-Figma-Rate-Limit-Type", "")
            if upgrade_link:
                error_msg += f" Upgrade your Figma plan/seat to increase limits: {upgrade_link}"
            if rate_type:
                error_msg += f" (rate-limit-type: {rate_type})"

        return ToolException(error_msg)
