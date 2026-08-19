"""
elitea_issues#6246: EliteAClient/SandboxClient issued outbound requests.* calls
with no timeout=, so a stalled endpoint could hang a worker forever. Both
clients now route every call through a `_request()` helper backed by a
`requests.Session`, which injects a default (connect, read) timeout unless
the call site already passes one.
"""
from unittest.mock import MagicMock, patch

BASE = "https://platform.example.com"
PROJECT_ID = 42


def make_elitea_client(**kwargs):
    from elitea_sdk.runtime.clients.client import EliteAClient
    return EliteAClient(base_url=BASE, project_id=PROJECT_ID, auth_token="tok", **kwargs)


def make_sandbox_client(**kwargs):
    from elitea_sdk.runtime.clients.sandbox_client import SandboxClient
    return SandboxClient(base_url=BASE, project_id=PROJECT_ID, auth_token="tok", **kwargs)


class TestEliteAClientTimeoutDefault:
    def test_default_timeout_is_5_30(self):
        client = make_elitea_client()
        assert client.timeout == (5, 30)

    def test_custom_timeout_kwarg_is_honored(self):
        client = make_elitea_client(timeout=(2, 10))
        assert client.timeout == (2, 10)

    def test_unsecret_passes_default_timeout(self):
        client = make_elitea_client()
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(json=lambda: {"value": "secret-value"})
            client.unsecret("some_secret")
        assert mock_request.call_args[1]["timeout"] == (5, 30)

    def test_unsecret_passes_custom_timeout(self):
        client = make_elitea_client(timeout=(1, 3))
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(json=lambda: {"value": "secret-value"})
            client.unsecret("some_secret")
        assert mock_request.call_args[1]["timeout"] == (1, 3)

    def test_get_mcp_toolkits_passes_timeout(self):
        client = make_elitea_client()
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(json=lambda: [])
            client.get_mcp_toolkits()
        assert mock_request.call_args[1]["timeout"] == (5, 30)


class TestEliteAClientArtifactTransferTimeout:
    def test_create_artifact_uses_longer_timeout(self):
        client = make_elitea_client()
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})
            client.create_artifact("bucket", "file.txt", b"data")
        assert mock_request.call_args[1]["timeout"] == (5, 120)

    def test_download_artifact_uses_longer_timeout(self):
        client = make_elitea_client()
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(status_code=200, content=b"data")
            client.download_artifact("bucket", "file.txt")
        assert mock_request.call_args[1]["timeout"] == (5, 120)

    def test_upload_artifact_s3_uses_longer_timeout(self):
        client = make_elitea_client()
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(status_code=200)
            client.upload_artifact_s3("bucket", "key.txt", b"data")
        assert mock_request.call_args[1]["timeout"] == (5, 120)

    def test_download_artifact_s3_uses_longer_timeout(self):
        client = make_elitea_client()
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(status_code=200, content=b"data")
            client.download_artifact_s3("bucket", "key.txt")
        assert mock_request.call_args[1]["timeout"] == (5, 120)


class TestEliteAClientSessionReuse:
    def test_uses_a_single_session_instance(self):
        client = make_elitea_client()
        assert hasattr(client, "_session")
        with patch.object(client._session, "request") as mock_request:
            mock_request.return_value = MagicMock(json=lambda: {"value": None})
            client.unsecret("a")
            client.unsecret("b")
        assert mock_request.call_count == 2


class TestSandboxClientTimeoutDefault:
    def test_default_timeout_is_5_30(self):
        client = make_sandbox_client()
        assert client.timeout == (5, 30)

    def test_custom_timeout_kwarg_is_honored(self):
        client = make_sandbox_client(timeout=(2, 10))
        assert client.timeout == (2, 10)

    def test_unsecret_passes_default_timeout(self):
        client = make_sandbox_client()
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(json=lambda: {"value": "secret-value"})
            client.unsecret("some_secret")
        assert mock_request.call_args[1]["timeout"] == (5, 30)


class TestSandboxClientArtifactTransferTimeout:
    def test_download_artifact_uses_longer_timeout(self):
        client = make_sandbox_client()
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(status_code=200, content=b"data")
            client.download_artifact("bucket", "file.txt")
        assert mock_request.call_args[1]["timeout"] == (5, 120)

    def test_upload_artifact_s3_uses_longer_timeout(self):
        client = make_sandbox_client()
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(status_code=200)
            client.upload_artifact_s3("bucket", "key.txt", b"data")
        assert mock_request.call_args[1]["timeout"] == (5, 120)
