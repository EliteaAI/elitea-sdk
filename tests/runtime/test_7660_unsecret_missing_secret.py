"""
elitea_issues#7660: Main answers an unknown secret with 404 and a JSON ``null`` body. ``unsecret`` parsed it and
crashed on ``None.get``, so callers saw a transient-looking AttributeError instead of "no such secret".
"""
from unittest.mock import patch

import pytest
import requests


def make_client():
    from elitea_sdk.runtime.clients.client import EliteAClient
    return EliteAClient(base_url="https://platform.example.com", project_id=42, auth_token="tok")


def make_response(status_code, body):
    response = requests.Response()
    response.status_code = status_code
    response._content = body
    return response


def test_missing_secret_is_none_and_named_in_the_log(caplog):
    with patch("requests.Session.request", return_value=make_response(404, b"null\n")):
        assert make_client().unsecret("mcp_token") is None
    assert "Secret 'mcp_token' not available from project vault (HTTP 404)" in caplog.text


def test_existing_secret_returns_value():
    with patch("requests.Session.request", return_value=make_response(200, b'{"value": "postgresql://x"}')):
        assert make_client().unsecret("pgvector_project_connstr") == "postgresql://x"


def test_a_default_key_the_api_withholds_is_none():
    # ignore_default_secret_api answers a default key such as pgvector_project_connstr with `None, 400`
    with patch("requests.Session.request", return_value=make_response(400, b"null\n")):
        assert make_client().unsecret("pgvector_project_connstr") is None


def test_a_route_level_404_still_raises():
    with patch("requests.Session.request", return_value=make_response(404, b"404 page not found\n")):
        with pytest.raises(ValueError):
            make_client().unsecret("pgvector_project_connstr")


def test_server_error_still_raises():
    with patch("requests.Session.request", return_value=make_response(500, b"<html>boom</html>")):
        with pytest.raises(ValueError):
            make_client().unsecret("pgvector_project_connstr")
