"""
Tests for PgVectorConfiguration.check_connection URI routing.

Covers the fix for: psycopg2 cannot parse SQLAlchemy-format URIs
(e.g. postgresql+psycopg://...) — those must be routed to SQLAlchemy first.

Note: create_engine and text are imported lazily inside check_connection,
so the correct patch target is 'sqlalchemy.create_engine' / 'sqlalchemy.text'.
"""
import pytest
from unittest.mock import MagicMock, patch

from elitea_sdk.configurations.pgvector import PgVectorConfiguration


class TestPgVectorCheckConnectionUriRouting:
    """SQLAlchemy-format URIs must be routed to SQLAlchemy, not psycopg2."""

    @patch('sqlalchemy.text')
    @patch('sqlalchemy.create_engine')
    def test_sqlalchemy_uri_with_psycopg_driver_succeeds(self, mock_create_engine, mock_text):
        """postgresql+psycopg://... is a SQLAlchemy URI — must use create_engine."""
        mock_conn = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value.connect.return_value = mock_ctx

        result = PgVectorConfiguration.check_connection(
            {'connection_string': 'postgresql+psycopg://centry:changeme@postgres:5432/db'}
        )

        assert result is None
        mock_create_engine.assert_called_once_with(
            'postgresql+psycopg://centry:changeme@postgres:5432/db'
        )

    @patch('sqlalchemy.text')
    @patch('sqlalchemy.create_engine')
    def test_standard_postgresql_uri_uses_sqlalchemy(self, mock_create_engine, mock_text):
        """postgresql://... is also a URI — must use SQLAlchemy path."""
        mock_conn = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value.connect.return_value = mock_ctx

        result = PgVectorConfiguration.check_connection(
            {'connection_string': 'postgresql://user:pass@localhost:5432/mydb'}
        )

        assert result is None
        mock_create_engine.assert_called_once()

    @patch('sqlalchemy.text')
    @patch('sqlalchemy.create_engine')
    def test_sqlalchemy_uri_connection_failure_returns_error(self, mock_create_engine, mock_text):
        """A failed SQLAlchemy connection returns an error string."""
        mock_create_engine.return_value.connect.side_effect = Exception('connection refused')

        result = PgVectorConfiguration.check_connection(
            {'connection_string': 'postgresql+psycopg://user:wrong@host:5432/db'}
        )

        assert result is not None
        assert 'connection refused' in result

    @patch('sqlalchemy.text')
    @patch('sqlalchemy.create_engine')
    def test_sqlalchemy_psycopg2_driver_uri_uses_sqlalchemy(self, mock_create_engine, mock_text):
        """postgresql+psycopg2://... is also a URI — must not be passed to psycopg2.connect."""
        mock_conn = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value.connect.return_value = mock_ctx

        result = PgVectorConfiguration.check_connection(
            {'connection_string': 'postgresql+psycopg2://user:pass@localhost/db'}
        )

        assert result is None
        mock_create_engine.assert_called_once()


class TestPgVectorCheckConnectionPlainDsn:
    """Plain DSN (key=value) strings must be routed to psycopg2."""

    @patch('psycopg2.connect')
    def test_plain_dsn_uses_psycopg2(self, mock_connect):
        """host=... dbname=... format is a plain DSN — must use psycopg2."""
        mock_connect.return_value = MagicMock()

        result = PgVectorConfiguration.check_connection(
            {'connection_string': 'host=localhost dbname=mydb user=user password=pass'}
        )

        assert result is None
        mock_connect.assert_called_once_with(
            dsn='host=localhost dbname=mydb user=user password=pass'
        )

    @patch('psycopg2.connect')
    def test_plain_dsn_connection_failure_returns_error(self, mock_connect):
        """A failed psycopg2 connection returns an error string."""
        mock_connect.side_effect = Exception('password authentication failed')

        result = PgVectorConfiguration.check_connection(
            {'connection_string': 'host=localhost dbname=mydb user=user password=wrong'}
        )

        assert result is not None
        assert 'password authentication failed' in result


class TestPgVectorCheckConnectionValidation:
    """Missing or empty connection strings are rejected immediately."""

    def test_missing_connection_string_returns_error(self):
        result = PgVectorConfiguration.check_connection({})
        assert result == 'Connection string is required'

    def test_none_connection_string_returns_error(self):
        result = PgVectorConfiguration.check_connection({'connection_string': None})
        assert result == 'Connection string is required'

    def test_empty_connection_string_returns_error(self):
        result = PgVectorConfiguration.check_connection({'connection_string': '   '})
        assert result == 'Connection string cannot be empty'
