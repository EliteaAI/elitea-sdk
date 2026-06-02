"""
Tests for GitHub configuration check_connection.

Issue #3986: GitHub App private key credential test fails with error message
"""
import pytest
from unittest.mock import patch, MagicMock
from urllib.parse import urlparse

from elitea_sdk.configurations.github import GithubConfiguration


# Valid test RSA private key (generated for testing only, not used anywhere else)
TEST_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAmY8p/k5MQWFxY+BKBc3B6bTIcdd879KZUh+G0H44xRDn/ds9
HH10geU1vLp7zMn+RgPepvZ+3qcJcAjV/3SUhzDSenSg3ziwMwhp71AHi4qnNIz+
J9mickGA8KiZP1JqQ7iYytOG5tMa5BvK+TbvsZuAY2R2wMZlgYhvIvoCxTkDhNue
q/tFbb33vN4CbxGFfHeBQRR9FwaiQMqTPjFj8YDxOPLdeSOiCyGvur16gIqPoE0f
z8IqFu7VXVPE88yzzFZs2jflBHXcHOHOY5iEvQWPxfOMks9XJ91sxsVl2yLpNe0T
XoWl4djHKHGcaAktG0f9NEDOtZPsGYbkDqutHQIDAQABAoIBABiJIv0StqV1s9/w
/PLbcwHsFGD4POq23C+egPR8TobSUjciGiwcsYp1vLXYmCJbHglC4gcMeK4Lw3rG
tqa4EmldeCv/yZqRHLoyvHZz85iswVWot41XfVjDNZ5+WeofuYHzk1pJHRNxpKjQ
9ggG6pqWzZdT/JOFv79INPXCO8IWQBiO/knZpm7OkfwotkdwPWTWJD+rXTTzG4+x
6CuyjtSsPfHFg8spDYS0DS9++ZtnFSP5GipuyMQn6byLKK6h1tpXQYej2XEKIfnk
dArDctXR1H/h7MJRBRkSstqtNvTZit+yG/LQTejCFRP9ZKLuoS6k+idnnvemgxyL
whH5+YECgYEAz2QfIxwFV1boNDR/I4K6PFVyce4RWA7YT1eDCFRI0Z0DfFl+iCWE
lHoumZiw9AYJk0qJF8qzupewzAWp+N0dc/6Wrweuiyfj+Z2KEL1yiXK31JlQv4R9
4/RhgVev5mpRhe3V0oeI+h7kiu1lvSIIPOu+MckOAsymkzFlTD+mysECgYEAvY0F
/9zflyr3oN9++LxjGRAM92tfC4zZS72kTIvLAHvluLB62/EETzNy+RnFA7A8Aazm
+bFZmyxyd2SfdnRtYFazPWc6XzsCDHLKb91WoRwbPVngpjQIlj56ZLt3yPPB+wUt
0Ir/osaPayfOmqd00/E+HFR661FqWioUvjvuRV0CgYEApM57r/rkg7Occ2AEaMPK
G4gLml4FimTBoMt6ZXQVKf4MdxTnaGnoIdW2kni0pjbmBRaGO1Hp4a4J4RffKtUM
QtFeDVmdaxgYIpT+0q66BmATle8ALDGtmSjrE04Lip+SiUunT9ZFE/7Yv05IOzSA
N2lfi1Cqwa6/8NigFye99AECgYAKVYbvNSaHglMv1R+CBhtNAYADeTocUhiCtZsg
hTqTEy4qDI0WMqSugLqS9CG2msQav0d0c4PUHu86rSS4e45/AxsQjPE0we3Rqex5
ftK7Q+IETUMfLJUPQ+a+WS4lqYx42AZwaTOYt0SYbfoomlqXN37QYpa0/6JRuhuZ
Z4ENDQKBgCmSkMoEwCfYmirGXlSz3bm8SRBRuw6rigy2+9eeyfAE1MgPYVYTj9Ba
kYWNdLCJdoN9db6x/5J+5vFsCKrooFxdNLdT8y/IXwN7Toau6oWfvAPey4ecIuwQ
PzbRu77QKe3K92EMPjzNCR6zZi/l7Wdw1UUjHXuUDXCkSkfk8siL
-----END RSA PRIVATE KEY-----"""

# Key body without headers (for testing key formatting)
TEST_KEY_BODY = """MIIEowIBAAKCAQEAmY8p/k5MQWFxY+BKBc3B6bTIcdd879KZUh+G0H44xRDn/ds9
HH10geU1vLp7zMn+RgPepvZ+3qcJcAjV/3SUhzDSenSg3ziwMwhp71AHi4qnNIz+
J9mickGA8KiZP1JqQ7iYytOG5tMa5BvK+TbvsZuAY2R2wMZlgYhvIvoCxTkDhNue
q/tFbb33vN4CbxGFfHeBQRR9FwaiQMqTPjFj8YDxOPLdeSOiCyGvur16gIqPoE0f
z8IqFu7VXVPE88yzzFZs2jflBHXcHOHOY5iEvQWPxfOMks9XJ91sxsVl2yLpNe0T
XoWl4djHKHGcaAktG0f9NEDOtZPsGYbkDqutHQIDAQABAoIBABiJIv0StqV1s9/w
/PLbcwHsFGD4POq23C+egPR8TobSUjciGiwcsYp1vLXYmCJbHglC4gcMeK4Lw3rG
tqa4EmldeCv/yZqRHLoyvHZz85iswVWot41XfVjDNZ5+WeofuYHzk1pJHRNxpKjQ
9ggG6pqWzZdT/JOFv79INPXCO8IWQBiO/knZpm7OkfwotkdwPWTWJD+rXTTzG4+x
6CuyjtSsPfHFg8spDYS0DS9++ZtnFSP5GipuyMQn6byLKK6h1tpXQYej2XEKIfnk
dArDctXR1H/h7MJRBRkSstqtNvTZit+yG/LQTejCFRP9ZKLuoS6k+idnnvemgxyL
whH5+YECgYEAz2QfIxwFV1boNDR/I4K6PFVyce4RWA7YT1eDCFRI0Z0DfFl+iCWE
lHoumZiw9AYJk0qJF8qzupewzAWp+N0dc/6Wrweuiyfj+Z2KEL1yiXK31JlQv4R9
4/RhgVev5mpRhe3V0oeI+h7kiu1lvSIIPOu+MckOAsymkzFlTD+mysECgYEAvY0F
/9zflyr3oN9++LxjGRAM92tfC4zZS72kTIvLAHvluLB62/EETzNy+RnFA7A8Aazm
+bFZmyxyd2SfdnRtYFazPWc6XzsCDHLKb91WoRwbPVngpjQIlj56ZLt3yPPB+wUt
0Ir/osaPayfOmqd00/E+HFR661FqWioUvjvuRV0CgYEApM57r/rkg7Occ2AEaMPK
G4gLml4FimTBoMt6ZXQVKf4MdxTnaGnoIdW2kni0pjbmBRaGO1Hp4a4J4RffKtUM
QtFeDVmdaxgYIpT+0q66BmATle8ALDGtmSjrE04Lip+SiUunT9ZFE/7Yv05IOzSA
N2lfi1Cqwa6/8NigFye99AECgYAKVYbvNSaHglMv1R+CBhtNAYADeTocUhiCtZsg
hTqTEy4qDI0WMqSugLqS9CG2msQav0d0c4PUHu86rSS4e45/AxsQjPE0we3Rqex5
ftK7Q+IETUMfLJUPQ+a+WS4lqYx42AZwaTOYt0SYbfoomlqXN37QYpa0/6JRuhuZ
Z4ENDQKBgCmSkMoEwCfYmirGXlSz3bm8SRBRuw6rigy2+9eeyfAE1MgPYVYTj9Ba
kYWNdLCJdoN9db6x/5J+5vFsCKrooFxdNLdT8y/IXwN7Toau6oWfvAPey4ecIuwQ
PzbRu77QKe3K92EMPjzNCR6zZi/l7Wdw1UUjHXuUDXCkSkfk8siL"""

# PKCS#8 format key for testing (converted from TEST_PRIVATE_KEY)
TEST_PKCS8_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCZjyn+TkxBYXFj
4EoFzcHptMhx13zv0plSH4bQfjjFEOf92z0cfXSB5TW8unvMyf5GA96m9n7epwlw
CNX/dJSHMNJ6dKDfOLAzCGnvUAeLiqc0jP4n2aJyQYDwqJk/UmpDuJjK04bm0xrk
G8r5Nu+xm4BjZHbAxmWBiG8i+gLFOQOE256r+0Vtvfe83gJvEYV8d4FBFH0XBqJA
ypM+MWPxgPE48t15I6ILIa+6vXqAio+gTR/PwioW7tVdU8TzzLPMVmzaN+UEddwc
4c5jmIS9BY/F84ySz1cn3WzGxWXbIuk17RNehaXh2McocZxoCS0bR/00QM61k+wZ
huQOq60dAgMBAAECggEAGIki/RK2pXWz3/D88ttzAewUYPg86rbcL56A9HxOhtJS
NyIaLByxinW8tdiYIlseCULiBwx4rgvDesa2prgSaV14K//JmpEcujK8dnPzmKzB
Vai3jVd9WMM1nn5Z6h+5gfOTWkkdE3GkqND2CAbqmpbNl1P8k4W/v0g09cI7whZA
GI7+Sdmmbs6R/Ci2R3A9ZNYkP6tdNPMbj7HoK7KO1Kw98cWDyykNhLQNL375m2cV
I/kaKm7IxCfpvIsorqHW2ldBh6PZcQoh+eR0CsNy1dHUf+HswlEFGRKy2q029NmK
37Ib8tBN6MIVE/1kou6hLqT6J2ee96aDHIvCEfn5gQKBgQDPZB8jHAVXVug0NH8j
gro8VXJx7hFYDthPV4MIVEjRnQN8WX6IJYSUei6ZmLD0BgmTSokXyrO6l7DMBan4
3R1z/pavB66LJ+P5nYoQvXKJcrfUmVC/hH3j9GGBV6/malGF7dXSh4j6HuSK7WW9
Igg8674xyQ4CzKaTMWVMP6bKwQKBgQC9jQX/3N+XKveg3374vGMZEAz3a18LjNlL
vaRMi8sAe+W4sHrb8QRPM3L5GcUDsDwBrOb5sVmbLHJ3ZJ92dG1gVrM9ZzpfOwIM
cspv3VahHBs9WeCmNAiWPnpku3fI88H7BS3Qiv+ixo9rJ86ap3TT8T4cVHrrUWpa
KhS+O+5FXQKBgQCkznuv+uSDs5xzYARow8obiAuaXgWKZMGgy3pldBUp/gx3FOdo
aegh1baSeLSmNuYFFoY7UenhrgnhF98q1QxC0V4NWZ1rGBgilP7SrroGYBOV7wAs
Ma2ZKOsTTguKn5KJS6dP1kUT/ti/Tkg7NIA3aV+LUKrBrr/w2KAXJ730AQKBgApV
hu81JoeCUy/VH4IGG00BgAN5OhxSGIK1myCFOpMTLioMjRYypK6AupL0IbaaxBq/
R3Rzg9Qe7zqtJLh7jn8DGxCM8TTB7dGp7Hl+0rtD4gRNQx8slQ9D5r5ZLiWpjHjY
BnBpM5i3RJht+iiaWpc3ftBilrT/olG6G5lngQ0NAoGAKZKQygTAJ9iaKsZeVLPd
ubxJEFG7DquKDLb7157J8ATUyA9hVhOP0FqRhY10sIl2g311vrH/kn7m8WwIquig
XF00t1PzL8hfA3tOhq7qhZ+8A97Lh5wi7BA/NtG7vtAp7cr3YQw+PM0JHrNmL+Xt
Z3DVRSMde5QNcKRKR+TyyIs=
-----END PRIVATE KEY-----"""


class TestGithubConfigurationCheckConnectionPartialAuth:
    """Test that partial auth configurations are rejected early."""

    def test_check_connection_rejects_app_id_without_private_key(self):
        """check_connection should reject app_id without app_private_key."""
        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'app_id': '123456',
            # app_private_key missing
        })
        assert error is not None
        assert 'app_id' in error.lower() and 'app_private_key' in error.lower()

    def test_check_connection_rejects_private_key_without_app_id(self):
        """check_connection should reject app_private_key without app_id."""
        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'app_private_key': 'some-key',
            # app_id missing
        })
        assert error is not None
        assert 'app_id' in error.lower() and 'app_private_key' in error.lower()

    def test_check_connection_rejects_username_without_password(self):
        """check_connection should reject username without password."""
        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'username': 'testuser',
            # password missing
        })
        assert error is not None
        assert 'username' in error.lower() and 'password' in error.lower()

    def test_check_connection_rejects_password_without_username(self):
        """check_connection should reject password without username."""
        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'password': 'testpass',
            # username missing
        })
        assert error is not None
        assert 'username' in error.lower() and 'password' in error.lower()


class TestGithubConfigurationCheckConnectionAnonymous:
    """Test anonymous access (no credentials)."""

    def test_check_connection_allows_anonymous_access(self):
        """check_connection should allow anonymous access when no credentials provided."""
        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
        })
        assert error is None

    def test_check_connection_allows_anonymous_with_empty_values(self):
        """check_connection should allow anonymous access with empty credential values."""
        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'access_token': '',
            'username': '',
            'password': '',
            'app_id': '',
            'app_private_key': '',
        })
        assert error is None


class TestGithubConfigurationCheckConnectionToken:
    """Test access token authentication."""

    @patch('requests.get')
    def test_check_connection_token_uses_user_endpoint(self, mock_get):
        """Token auth should use /user endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'access_token': 'ghp_test_token',
        })

        assert error is None
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert call_url == 'https://api.github.com/user'

    @patch('requests.get')
    def test_check_connection_token_invalid_returns_error(self, mock_get):
        """Invalid token should return authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'access_token': 'invalid_token',
        })

        assert error is not None
        assert 'Authentication failed' in error or 'Invalid' in error


class TestGithubConfigurationCheckConnectionUsernamePassword:
    """Test username/password authentication."""

    @patch('requests.get')
    def test_check_connection_password_uses_user_endpoint(self, mock_get):
        """Username/password auth should use /user endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'username': 'testuser',
            'password': 'testpass',
        })

        assert error is None
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert call_url == 'https://api.github.com/user'
        # Verify auth was passed
        assert mock_get.call_args[1].get('auth') is not None


class TestGithubConfigurationCheckConnectionGitHubApp:
    """Test GitHub App private key authentication."""

    @patch('requests.get')
    def test_check_connection_app_uses_app_endpoint(self, mock_get):
        """GitHub App auth should use /app endpoint, not /user."""
        mock_app_response = MagicMock()
        mock_app_response.status_code = 200
        mock_get.return_value = mock_app_response

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'app_id': '123456',
            'app_private_key': TEST_PRIVATE_KEY,
        })

        assert error is None
        # Verify /app endpoint was called (not /user)
        assert mock_get.call_count == 1
        call_url = mock_get.call_args[0][0]
        assert call_url == 'https://api.github.com/app'

    @patch('requests.get')
    def test_check_connection_app_invalid_credentials_returns_error(self, mock_get):
        """Invalid GitHub App credentials should return authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'app_id': '123456',
            'app_private_key': TEST_PRIVATE_KEY,
        })

        assert error is not None
        assert 'Invalid' in error or 'Authentication failed' in error

    @patch('requests.get')
    def test_check_connection_app_valid_even_without_installations(self, mock_get):
        """GitHub App credentials are valid if /app returns 200, regardless of installations.

        Issue #3986: credential check should confirm credentials work, not check installations.
        """
        mock_app_response = MagicMock()
        mock_app_response.status_code = 200
        mock_get.return_value = mock_app_response

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'app_id': '123456',
            'app_private_key': TEST_PRIVATE_KEY,
        })

        # /app returning 200 = credentials are valid
        assert error is None
        # Should only call /app, not /app/installations
        assert mock_get.call_count == 1

    def test_check_connection_app_invalid_private_key_format(self):
        """Invalid private key format should return error."""
        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'app_id': '123456',
            'app_private_key': 'not-a-valid-key',
        })

        assert error is not None
        assert 'private key' in error.lower() or 'error' in error.lower()

    @patch('requests.get')
    def test_check_connection_app_formats_key_without_headers(self, mock_get):
        """Private key without headers should be formatted correctly."""
        mock_app_response = MagicMock()
        mock_app_response.status_code = 200
        mock_get.return_value = mock_app_response

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'app_id': '123456',
            'app_private_key': TEST_KEY_BODY,
        })

        # Should succeed because the code adds the headers
        assert error is None

    @patch('requests.get')
    def test_check_connection_app_formats_single_line_key_with_headers(self, mock_get):
        """Private key as single line WITH headers should be normalized correctly.

        Issue #3986: keys should work with or without begin/end sections.
        A key pasted as single line like:
        '-----BEGIN RSA PRIVATE KEY----- MIIEp... -----END RSA PRIVATE KEY-----'
        should be normalized to proper multi-line PEM format.
        """
        mock_app_response = MagicMock()
        mock_app_response.status_code = 200
        mock_get.return_value = mock_app_response

        # Simulate a key pasted as a single line with headers
        single_line_key = TEST_PRIVATE_KEY.replace("\n", " ")

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'app_id': '123456',
            'app_private_key': single_line_key,
        })

        # Should succeed because the code normalizes the key format
        assert error is None

    @patch('requests.get')
    def test_check_connection_app_accepts_pkcs8_format(self, mock_get):
        """GitHub App auth should accept PKCS#8 format private keys.

        Issue #3986: PKCS#8 keys (-----BEGIN PRIVATE KEY-----) should work
        without being corrupted by the normalization code.
        """
        mock_app_response = MagicMock()
        mock_app_response.status_code = 200
        mock_get.return_value = mock_app_response

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'app_id': '123456',
            'app_private_key': TEST_PKCS8_KEY,
        })

        # Should succeed with PKCS#8 format key
        assert error is None

    @patch('requests.get')
    def test_check_connection_app_formats_single_line_pkcs8_key(self, mock_get):
        """PKCS#8 key as single line should be normalized correctly."""
        mock_app_response = MagicMock()
        mock_app_response.status_code = 200
        mock_get.return_value = mock_app_response

        # Simulate PKCS#8 key pasted as a single line with headers
        single_line_pkcs8_key = TEST_PKCS8_KEY.replace("\n", " ")

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'app_id': '123456',
            'app_private_key': single_line_pkcs8_key,
        })

        # Should succeed because the code normalizes the key format
        assert error is None


class TestGithubConfigurationNormalizePrivateKey:
    """Test _normalize_private_key method directly."""

    def test_normalize_pkcs1_key_with_headers(self):
        """PKCS#1 key with headers should be preserved."""
        normalized = GithubConfiguration._normalize_private_key(TEST_PRIVATE_KEY)
        assert normalized.startswith("-----BEGIN RSA PRIVATE KEY-----")
        assert normalized.endswith("-----END RSA PRIVATE KEY-----")
        assert "-----BEGIN PRIVATE KEY-----" not in normalized

    def test_normalize_pkcs8_key_with_headers(self):
        """PKCS#8 key with headers should be preserved."""
        normalized = GithubConfiguration._normalize_private_key(TEST_PKCS8_KEY)
        assert normalized.startswith("-----BEGIN PRIVATE KEY-----")
        assert normalized.endswith("-----END PRIVATE KEY-----")
        assert "-----BEGIN RSA PRIVATE KEY-----" not in normalized

    def test_normalize_key_body_without_headers(self):
        """Key body without headers should default to PKCS#1."""
        normalized = GithubConfiguration._normalize_private_key(TEST_KEY_BODY)
        assert normalized.startswith("-----BEGIN RSA PRIVATE KEY-----")
        assert normalized.endswith("-----END RSA PRIVATE KEY-----")

    def test_normalize_single_line_pkcs1_key(self):
        """Single-line PKCS#1 key should be normalized to multi-line."""
        single_line = TEST_PRIVATE_KEY.replace("\n", " ")
        normalized = GithubConfiguration._normalize_private_key(single_line)
        assert normalized.startswith("-----BEGIN RSA PRIVATE KEY-----")
        assert normalized.endswith("-----END RSA PRIVATE KEY-----")
        assert "\n" in normalized  # Should have newlines

    def test_normalize_single_line_pkcs8_key(self):
        """Single-line PKCS#8 key should be normalized to multi-line."""
        single_line = TEST_PKCS8_KEY.replace("\n", " ")
        normalized = GithubConfiguration._normalize_private_key(single_line)
        assert normalized.startswith("-----BEGIN PRIVATE KEY-----")
        assert normalized.endswith("-----END PRIVATE KEY-----")
        assert "\n" in normalized  # Should have newlines


class TestGithubConfigurationCheckConnectionErrors:
    """Test error handling scenarios."""

    @patch('requests.get')
    def test_check_connection_handles_connection_error(self, mock_get):
        """Connection errors should return appropriate message."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'access_token': 'test_token',
        })

        assert error is not None
        assert 'Connection error' in error

    @patch('requests.get')
    def test_check_connection_handles_timeout(self, mock_get):
        """Timeout errors should return appropriate message."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'access_token': 'test_token',
        })

        assert error is not None
        assert 'timeout' in error.lower()

    @patch('requests.get')
    def test_check_connection_handles_403_forbidden(self, mock_get):
        """403 response should return forbidden message."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response

        error = GithubConfiguration.check_connection({
            'base_url': 'https://api.github.com',
            'access_token': 'test_token',
        })

        assert error is not None
        assert 'forbidden' in error.lower()

    @patch('requests.get')
    def test_check_connection_custom_base_url(self, mock_get):
        """Custom base URL (GitHub Enterprise) should be used."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        error = GithubConfiguration.check_connection({
            'base_url': 'https://github.mycompany.com/api/v3',
            'access_token': 'test_token',
        })

        assert error is None
        call_url = mock_get.call_args[0][0]
        parsed_url = urlparse(call_url)
        assert parsed_url.hostname == 'github.mycompany.com'
