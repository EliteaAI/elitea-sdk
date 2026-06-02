from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class GithubConfiguration(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "metadata": {
                "label": "GitHub",
                "icon_url": None,
                "sections": {
                    "auth": {
                        "required": False,
                        "subsections": [
                            {
                                "name": "Token",
                                "fields": ["access_token"]
                            },
                            {
                                "name": "Password",
                                "fields": ["username", "password"]
                            },
                            {
                                "name": "App private key",
                                "fields": ["app_id", "app_private_key"]
                            }
                        ]
                    },
                },
                "section": "credentials",
                "type": "github",
                "categories": ["code repositories"],
                "extra_categories": ["github", "git", "repository", "code", "version control"],
            }
        }
    )

    # prefill_value: UI hint for pre-filling required fields.
    # Unlike 'default', this keeps the field truly required in Pydantic validation
    # while allowing UI to show a sensible initial value. The API schema stays
    # Pydantic-compliant (field is in 'required' array, no 'default' attribute).
    base_url: str = Field(
        description="Base API URL",
        json_schema_extra={"prefill_value": "https://api.github.com"}
    )
    app_id: Optional[str] = Field(description="Github APP ID", default=None)
    app_private_key: Optional[SecretStr] = Field(description="Github APP private key", default=None)

    access_token: Optional[SecretStr] = Field(description="Github Access Token", default=None)

    username: Optional[str] = Field(description="Github Username", default=None)
    password: Optional[SecretStr] = Field(description="Github Password", default=None)

    @model_validator(mode='before')
    @classmethod
    def validate_auth_sections(cls, data):
        if not isinstance(data, dict):
            return data

        has_token = bool(data.get('access_token') and str(data.get('access_token')).strip())
        has_password = bool(
            data.get('username') and str(data.get('username')).strip() and
            data.get('password') and str(data.get('password')).strip()
        )
        has_app_key = bool(
            data.get('app_id') and str(data.get('app_id')).strip() and
            data.get('app_private_key') and str(data.get('app_private_key')).strip()
        )

        # If any method is partially configured, raise exception
        if (
                (data.get('username') and not data.get('password')) or
                (data.get('password') and not data.get('username')) or
                (data.get('app_id') and not data.get('app_private_key')) or
                (data.get('app_private_key') and not data.get('app_id'))
        ):
            raise ValueError(
                "Authentication is misconfigured: both username and password, or both app_id and app_private_key, must be provided together."
            )

        # If all are missing, allow anonymous
        if not (has_token or has_password or has_app_key):
            return data

        # If any method is fully configured
        if has_token or has_password or has_app_key:
            return data

        raise ValueError(
            "Authentication is misconfigured: provide either Token (access_token), "
            "Password (username + password), App private key (app_id + app_private_key), "
            "or leave all blank for anonymous access."
        )

    @staticmethod
    def _normalize_private_key(private_key: str) -> str:
        """
        Normalize private key to proper PEM format.
        Supports both PKCS#1 (RSA PRIVATE KEY) and PKCS#8 (PRIVATE KEY).
        Handles keys with or without headers, single-line formatted keys, etc.

        Args:
            private_key: Raw private key string in any format

        Returns:
            Normalized PEM-formatted private key string
        """
        # Supported PEM formats
        pkcs1_header = "-----BEGIN RSA PRIVATE KEY-----"
        pkcs1_footer = "-----END RSA PRIVATE KEY-----"
        pkcs8_header = "-----BEGIN PRIVATE KEY-----"
        pkcs8_footer = "-----END PRIVATE KEY-----"

        key = private_key.strip()

        # Detect format and extract body
        detected_header = None
        detected_footer = None

        if pkcs1_header in key:
            detected_header = pkcs1_header
            detected_footer = pkcs1_footer
            key = key.replace(pkcs1_header, "").replace(pkcs1_footer, "").strip()
        elif pkcs8_header in key:
            detected_header = pkcs8_header
            detected_footer = pkcs8_footer
            key = key.replace(pkcs8_header, "").replace(pkcs8_footer, "").strip()

        # Normalize whitespace: replace spaces with newlines, collapse multiple newlines
        key_body = key.replace(" ", "\n")
        # Remove any blank lines
        key_lines = [line.strip() for line in key_body.split("\n") if line.strip()]
        key_body = "\n".join(key_lines)

        # Reconstruct with original format (default to PKCS#1 if no headers)
        if detected_header is None:
            detected_header = pkcs1_header
            detected_footer = pkcs1_footer

        return f"{detected_header}\n{key_body}\n{detected_footer}"

    @staticmethod
    def check_connection(settings: dict) -> str | None:
        """
        Check GitHub connection using provided settings.
        Returns None if connection is successful, error message otherwise.
        """
        import requests
        from requests.auth import HTTPBasicAuth
        import jwt
        import time

        base_url = settings.get('base_url', 'https://api.github.com')
        access_token = settings.get('access_token')
        username = settings.get('username')
        password = settings.get('password')
        app_id = settings.get('app_id')
        app_private_key = settings.get('app_private_key')

        # Check for partial auth configuration (one field provided but not the other)
        if (username and not password) or (password and not username):
            return "Authentication misconfigured: both username and password must be provided together"
        if (app_id and not app_private_key) or (app_private_key and not app_id):
            return "Authentication misconfigured: both app_id and app_private_key must be provided together"

        # if all auth methods are None or empty, allow anonymous access
        if not any([access_token, (username and password), (app_id and app_private_key)]):
            return None

        headers = {'Accept': 'application/vnd.github.v3+json'}
        auth = None

        try:
            # Determine authentication method
            if access_token:
                headers['Authorization'] = f'token {access_token}'
                response = requests.get(f'{base_url}/user', headers=headers, timeout=10)
            elif username and password:
                auth = HTTPBasicAuth(username, password)
                response = requests.get(f'{base_url}/user', headers=headers, auth=auth, timeout=10)
            elif app_id and app_private_key:
                # Normalize the private key to proper PEM format
                app_private_key = GithubConfiguration._normalize_private_key(app_private_key)

                # Generate JWT for GitHub App authentication
                payload = {
                    'iat': int(time.time()),
                    'exp': int(time.time()) + 600,  # 10 minutes
                    'iss': app_id
                }
                jwt_token = jwt.encode(payload, app_private_key, algorithm='RS256')
                headers['Authorization'] = f'Bearer {jwt_token}'

                # GitHub App JWT tokens must use /app endpoint, not /user
                # The /user endpoint requires user-level authentication
                response = requests.get(f'{base_url}/app', headers=headers, timeout=10)

                if response.status_code == 200:
                    # /app returning 200 proves credentials are valid
                    return None
                elif response.status_code == 401:
                    return "Authentication failed: Invalid GitHub App credentials (app_id or private_key)"
                elif response.status_code == 403:
                    return "Access forbidden: Check your GitHub App permissions"
                elif response.status_code == 404:
                    return "GitHub API endpoint not found"
                else:
                    return f"Connection failed with status {response.status_code}: {response.text}"

            if response.status_code == 200:
                return None
            elif response.status_code == 401:
                return "Authentication failed: Invalid credentials"
            elif response.status_code == 403:
                return "Access forbidden: Check your permissions"
            elif response.status_code == 404:
                return "GitHub API endpoint not found"
            else:
                return f"Connection failed with status {response.status_code}: {response.text}"

        except requests.exceptions.ConnectionError:
            return "Connection error: Unable to reach GitHub API"
        except requests.exceptions.Timeout:
            return "Connection timeout: GitHub API did not respond in time"
        except jwt.InvalidKeyError:
            return "Invalid private key format for GitHub App authentication"
        except Exception as e:
            return f"Unexpected error: {str(e)}"