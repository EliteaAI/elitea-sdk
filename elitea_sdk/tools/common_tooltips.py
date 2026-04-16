# Common tooltip descriptions for toolkit configuration fields
# These provide consistent help text across all toolkits

def get_credentials_tooltip(toolkit_name: str) -> str:
    """
    Get tooltip description for toolkit credentials configuration field.

    Args:
        toolkit_name: Display name of the toolkit (e.g., "Jira", "Confluence", "GitHub")

    Returns:
        Formatted tooltip description string
    """
    return (
        f"Select the credentials used to connect this toolkit to {toolkit_name}.\n\n"
        "**Project credentials** are shared with the project team.\n"
        "**Private credentials** are visible only to you."
    )


PGVECTOR_CONFIGURATION_TOOLTIP = (
    "PgVector is required to store indexed data for RAG (search + retrieval).\n\n"
    "Select an existing PgVector config or create a new one (Private or Project), "
    "then use refresh to update the list."
)


EMBEDDING_MODEL_TOOLTIP = (
    "Generates embeddings (vectors) used for indexing and semantic search (RAG). "
    "This affects retrieval quality and performance.\n\n"
    "Common models: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large."
)
