"""
Tests for SQL toolkit connection string handling.
"""
import pytest
from unittest.mock import patch, MagicMock

from elitea_sdk.tools.sql.api_wrapper import SQLApiWrapper
from elitea_sdk.tools.sql.models import SQLDialect


class TestSQLConnectionStringEncoding:
    """Test URL encoding of credentials in connection strings."""

    @patch('elitea_sdk.tools.sql.api_wrapper.create_engine')
    def test_password_with_at_symbol_is_encoded(self, mock_create_engine):
        """Bug #2601: Password containing '@' should be URL-encoded."""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        wrapper = SQLApiWrapper(
            dialect=SQLDialect.POSTGRES,
            host="localhost",
            port="5432",
            username="testuser",
            password="secret@123",
            database_name="testdb"
        )

        _ = wrapper.client

        mock_create_engine.assert_called_once()
        connection_string = mock_create_engine.call_args[0][0]
        assert "secret%40123" in connection_string
        assert "secret@123@" not in connection_string

    @patch('elitea_sdk.tools.sql.api_wrapper.create_engine')
    def test_password_with_multiple_special_chars(self, mock_create_engine):
        """Password with multiple special characters should be fully encoded."""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        wrapper = SQLApiWrapper(
            dialect=SQLDialect.POSTGRES,
            host="localhost",
            port="5432",
            username="testuser",
            password="p@ss:word/test",
            database_name="testdb"
        )

        _ = wrapper.client

        connection_string = mock_create_engine.call_args[0][0]
        assert "%40" in connection_string  # @ encoded
        assert "%3A" in connection_string  # : encoded
        assert "%2F" in connection_string  # / encoded

    @patch('elitea_sdk.tools.sql.api_wrapper.create_engine')
    def test_mysql_password_encoding(self, mock_create_engine):
        """MySQL dialect should also encode special characters."""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        wrapper = SQLApiWrapper(
            dialect=SQLDialect.MYSQL,
            host="localhost",
            port="3306",
            username="testuser",
            password="pass@word",
            database_name="testdb"
        )

        _ = wrapper.client

        connection_string = mock_create_engine.call_args[0][0]
        assert "mysql+pymysql://" in connection_string
        assert "pass%40word" in connection_string

    @patch('elitea_sdk.tools.sql.api_wrapper.create_engine')
    def test_plain_password_unchanged(self, mock_create_engine):
        """Password without special characters should work normally."""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        wrapper = SQLApiWrapper(
            dialect=SQLDialect.POSTGRES,
            host="localhost",
            port="5432",
            username="testuser",
            password="simplepassword",
            database_name="testdb"
        )

        _ = wrapper.client

        connection_string = mock_create_engine.call_args[0][0]
        assert "simplepassword@localhost" in connection_string
