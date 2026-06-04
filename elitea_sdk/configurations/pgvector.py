from pydantic import BaseModel, ConfigDict, Field, SecretStr


class PgVectorConfiguration(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "metadata": {
                "label": "PgVector",
                "icon_url": None,
                "section": "vectorstorage",
                "type": "pgvector"
            }
        }
    )
    connection_string: SecretStr = Field(
        description="Connection string for PgVector database",
        default=None
    )

    @staticmethod
    def check_connection(settings: dict) -> str | None:
        """
        Check the connection to a PgVector database.

        Args:
            settings: Dictionary containing pgvector configuration
                - connection_string: PostgreSQL connection string (required)

        Returns:
            None if connection successful, error message string if failed
        """
        connection_string = settings.get("connection_string")
        if not connection_string:
            return "Connection string is required"

        if hasattr(connection_string, "get_secret_value"):
            connection_string = connection_string.get_secret_value()

        connection_string = str(connection_string).strip()
        if not connection_string:
            return "Connection string cannot be empty"

        if "://" in connection_string:
            try:
                from sqlalchemy import create_engine, text
                engine = create_engine(connection_string)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return None
            except Exception as e:
                return f"Cannot connect to PgVector database: {str(e)}"

        try:
            import psycopg2
            conn = psycopg2.connect(dsn=connection_string)
            conn.close()
            return None
        except ImportError:
            pass
        except Exception as e:
            return f"Cannot connect to PgVector database: {str(e)}"

        return "Cannot connect to PgVector database: no suitable driver found"
