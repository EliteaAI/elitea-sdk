#!/usr/bin/env python3
"""
Generate SQL dump with pre-computed embeddings for RAG test corpus.

This utility pre-computes embeddings for the KNOWN_ANSWER_CORPUS and saves them
as SQL INSERT statements. RAG tests then load this dump directly into pgvector
instead of calling the embedding API, ensuring:
  - Deterministic test results (same vectors every run)
  - Faster test execution (no API calls during RAG tests)
  - Reduced API costs

Usage:
    # From elitea-sdk directory, with credentials in environment:
    python tests/tools/index/generate_rag_embeddings.py

    # With custom embedding model:
    python tests/tools/index/generate_rag_embeddings.py --model text-embedding-3-small

    # Verify existing dump:
    python tests/tools/index/generate_rag_embeddings.py --verify

Environment Variables Required:
    ELITEA_DEPLOYMENT_URL - EliteA platform URL
    ELITEA_PROJECT_ID - Project ID
    ELITEA_TOKEN - API token

When to Regenerate:
    - When KNOWN_ANSWER_CORPUS content changes
    - When switching to a different embedding model
    - When embedding model version changes
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Script paths
SCRIPT_DIR = Path(__file__).parent
FIXTURES_DIR = SCRIPT_DIR / "fixtures"
SQL_DUMP_FILE = FIXTURES_DIR / "rag_corpus_embeddings.sql"


def get_corpus() -> List[Dict[str, Any]]:
    """Import and return the test corpus from the RAG quality test module."""
    # Import here to avoid circular dependencies when running standalone
    sys.path.insert(0, str(SCRIPT_DIR))
    from test_indexer_rag_quality import KNOWN_ANSWER_CORPUS
    return KNOWN_ANSWER_CORPUS


def create_elitea_client():
    """Create EliteA client from environment variables."""
    from pydantic import SecretStr
    from elitea_sdk.runtime.clients.client import EliteAClient

    deployment_url = os.getenv("ELITEA_DEPLOYMENT_URL")
    project_id = os.getenv("ELITEA_PROJECT_ID")
    api_token = os.getenv("ELITEA_TOKEN")

    if not all([deployment_url, project_id, api_token]):
        missing = []
        if not deployment_url:
            missing.append("ELITEA_DEPLOYMENT_URL")
        if not project_id:
            missing.append("ELITEA_PROJECT_ID")
        if not api_token:
            missing.append("ELITEA_TOKEN")
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Set these variables before running the embedding generator."
        )

    return EliteAClient(
        base_url=deployment_url,
        project_id=int(project_id),
        auth_token=SecretStr(api_token),
    )


def escape_sql_string(s: str) -> str:
    """Escape a string for use in SQL INSERT statement."""
    if s is None:
        return "NULL"
    # Escape single quotes by doubling them
    escaped = s.replace("'", "''")
    # Escape backslashes
    escaped = escaped.replace("\\", "\\\\")
    return f"'{escaped}'"


def format_embedding_for_sql(embedding: List[float]) -> str:
    """Format embedding vector for pgvector SQL insertion."""
    # pgvector expects format: '[0.1,0.2,0.3,...]'
    vector_str = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
    return f"'{vector_str}'"


def generate_sql_dump(
    corpus: List[Dict[str, Any]],
    embedding_model: str = "text-embedding-ada-002",
    collection_name: str = "rettest",
    schema_name: str = "{{SCHEMA}}",
) -> str:
    """
    Generate SQL INSERT statements for the corpus with embeddings.

    Args:
        corpus: List of document dicts with 'page_content' and 'metadata' keys
        embedding_model: Name of the embedding model to use
        collection_name: Index/collection name for metadata
        schema_name: Schema placeholder (replaced at runtime)

    Returns:
        SQL string with INSERT statements
    """
    print(f"Creating EliteA client...")
    client = create_elitea_client()

    print(f"Getting embedding model: {embedding_model}")
    embeddings_model = client.get_embeddings(embedding_model=embedding_model)

    # Generate a deterministic UUID for the collection (based on collection name)
    # This ensures the same collection_id is used across regenerations
    collection_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"elitea.rag.{collection_name}"))

    # Build SQL header
    sql_lines = [
        f"-- RAG Test Corpus Embeddings Dump",
        f"-- Generated: {datetime.utcnow().isoformat()}Z",
        f"-- Model: {embedding_model}",
        f"-- Documents: {len(corpus)}",
        f"-- Collection: {collection_name}",
        f"-- Collection UUID: {collection_uuid}",
        f"--",
        f"-- Usage: Replace {{{{SCHEMA}}}} with actual schema name before execution",
        f"--",
        f"",
        f"-- Create collection record first (langchain_postgres requires this)",
        f"INSERT INTO {schema_name}.langchain_pg_collection (uuid, name, cmetadata) VALUES ('{collection_uuid}', {escape_sql_string(collection_name)}, '{{}}'::json) ON CONFLICT (name) DO UPDATE SET cmetadata = EXCLUDED.cmetadata;",
        f"",
        f"-- Embedding records with collection_id foreign key",
    ]

    print(f"Generating embeddings for {len(corpus)} documents...")

    for i, doc in enumerate(corpus):
        content = doc["page_content"]
        metadata = dict(doc["metadata"])

        # Generate embedding
        vector = embeddings_model.embed_query(content)
        dimension = len(vector)

        # Build metadata JSON (matching what index_data would produce)
        metadata["id"] = f"known_doc_{i}"
        metadata["collection"] = collection_name

        # Format SQL INSERT
        # Table: langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
        # collection_id references langchain_pg_collection.uuid
        doc_id = f"rag_cached_{i:03d}"

        sql = (
            f"INSERT INTO {schema_name}.langchain_pg_embedding "
            f"(id, collection_id, embedding, document, cmetadata) VALUES ("
            f"{escape_sql_string(doc_id)}, "
            f"'{collection_uuid}', "
            f"{format_embedding_for_sql(vector)}, "
            f"{escape_sql_string(content)}, "
            f"{escape_sql_string(json.dumps(metadata))}::jsonb"
            f");"
        )
        sql_lines.append(sql)

        source = metadata.get("source", "unknown")
        print(f"  [{i + 1}/{len(corpus)}] {source} - {dimension} dims")

    sql_lines.append("")
    sql_lines.append(f"-- End of dump ({len(corpus)} documents)")

    print(f"\nGenerated {len(corpus)} INSERT statements + 1 collection record")
    return "\n".join(sql_lines)


def save_sql_dump(sql: str, output_path: Path) -> None:
    """Save SQL dump to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(sql)

    size_kb = output_path.stat().st_size / 1024
    print(f"Saved to: {output_path} ({size_kb:.1f} KB)")


def verify_sql_dump(input_path: Path, expected_corpus_size: int) -> bool:
    """
    Verify the integrity of a saved SQL dump file.

    Returns True if valid, False otherwise.
    """
    print(f"Verifying: {input_path}")

    if not input_path.exists():
        print(f"  ERROR: File does not exist")
        return False

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()
    except IOError as e:
        print(f"  ERROR: Cannot read file - {e}")
        return False

    # Count INSERT statements (embeddings + 1 collection record)
    embedding_insert_count = content.count("INSERT INTO {{SCHEMA}}.langchain_pg_embedding")
    collection_insert_count = content.count("INSERT INTO {{SCHEMA}}.langchain_pg_collection")

    if embedding_insert_count == 0:
        print(f"  ERROR: No embedding INSERT statements found")
        return False

    if collection_insert_count == 0:
        print(f"  ERROR: No collection INSERT statement found")
        return False

    if embedding_insert_count != expected_corpus_size:
        print(f"  WARNING: Embedding INSERT count ({embedding_insert_count}) != corpus size ({expected_corpus_size})")
        print(f"           Regenerate dump to sync with current corpus.")
        return False

    # Extract metadata from header
    model_line = [l for l in content.split("\n") if l.startswith("-- Model:")]
    generated_line = [l for l in content.split("\n") if l.startswith("-- Generated:")]

    model = model_line[0].split(": ", 1)[1] if model_line else "unknown"
    generated = generated_line[0].split(": ", 1)[1] if generated_line else "unknown"

    print(f"  Model: {model}")
    print(f"  Embedding records: {embedding_insert_count}")
    print(f"  Collection records: {collection_insert_count}")
    print(f"  Generated: {generated}")
    print(f"  Status: VALID")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL dump with embeddings for RAG test corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="text-embedding-ada-002",
        help="Embedding model name (default: text-embedding-ada-002)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SQL_DUMP_FILE,
        help=f"Output file path (default: {SQL_DUMP_FILE})",
    )
    parser.add_argument(
        "--collection",
        default="rettest",
        help="Collection/index name for metadata (default: rettest)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing dump without regenerating",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dump without prompting",
    )

    args = parser.parse_args()

    # Load corpus for size check
    try:
        corpus = get_corpus()
        corpus_size = len(corpus)
        print(f"Loaded corpus with {corpus_size} documents")
    except ImportError as e:
        print(f"ERROR: Cannot import corpus - {e}", file=sys.stderr)
        sys.exit(1)

    if args.verify:
        success = verify_sql_dump(args.output, corpus_size)
        sys.exit(0 if success else 1)

    # Check if output exists
    if args.output.exists() and not args.force:
        print(f"Output file already exists: {args.output}")
        response = input("Overwrite? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    # Generate SQL dump
    try:
        sql = generate_sql_dump(
            corpus,
            embedding_model=args.model,
            collection_name=args.collection,
        )
        save_sql_dump(sql, args.output)

        # Verify the saved file
        print("\nVerifying saved file...")
        verify_sql_dump(args.output, corpus_size)

        print("\nDone! RAG tests will now use cached embeddings.")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
