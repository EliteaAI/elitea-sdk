"""Shared Pydantic schemas for index-related tools.

Consolidates the RemoveIndex / BaseSearch / BaseStepbackSearch schemas that were
previously duplicated across ``elitea_base`` and ``base_indexer_toolkit``.
Callers that need a different ``cut_off`` default or the optional
``output_fields`` field use the factory functions below.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field, create_model

INDEX_NAME_MAX_LENGTH = 32

RemoveIndexParams = create_model(
    "RemoveIndexParams",
    index_name=(
        Optional[str],
        Field(
            description=f"Optional index name (max {INDEX_NAME_MAX_LENGTH} characters)",
            default="",
            max_length=INDEX_NAME_MAX_LENGTH,
        ),
    ),
)


def _search_common_fields(cut_off_default: float) -> dict:
    return {
        "filter": (
            Optional[dict | str],
            Field(
                description="Filter to apply to the search results. Can be a dictionary or a JSON string.",
                default={},
                examples=['{"key": "value"}', '{"status": "active"}'],
            ),
        ),
        "cut_off": (
            Optional[float],
            Field(description="Cut-off score for search results", default=cut_off_default, ge=0, le=1),
        ),
        "search_top": (
            Optional[int],
            Field(description="Number of top results to return", default=10, gt=0),
        ),
        "full_text_search": (
            Optional[Dict[str, Any]],
            Field(
                description="Full text search parameters. Can be a dictionary with search options.",
                default=None,
            ),
        ),
        "extended_search": (
            Optional[List[str]],
            Field(
                description="List of additional fields to include in the search results.",
                default=None,
            ),
        ),
        "reranker": (
            Optional[dict],
            Field(
                description="Reranker configuration. Can be a dictionary with reranking parameters.",
                default={},
            ),
        ),
        "reranking_config": (
            Optional[Dict[str, Dict[str, Any]]],
            Field(
                description="Reranking configuration. Can be a dictionary with reranking settings.",
                default=None,
            ),
        ),
    }


def build_base_search_params(*, cut_off_default: float, include_output_fields: bool = False):
    fields = {
        "query": (str, Field(description="Query text to search in the index")),
        "index_name": (
            Optional[str],
            Field(
                description=(
                    f"Optional index name (max {INDEX_NAME_MAX_LENGTH} characters). "
                    "Leave empty to search across all datasets"
                ),
                default="",
                max_length=INDEX_NAME_MAX_LENGTH,
            ),
        ),
        **_search_common_fields(cut_off_default),
    }
    if include_output_fields:
        fields["output_fields"] = (
            Optional[List[str]],
            Field(
                description=(
                    "Fields to include in output. Supports: 'page_content', 'score', 'metadata' (all metadata), "
                    "or 'metadata.<field>' for specific metadata fields (e.g., 'metadata.source'). "
                    "If None or empty, returns all fields."
                ),
                default=None,
                examples=[
                    ["metadata", "score"],
                    ["page_content", "metadata.source"],
                    ["metadata.id", "metadata.source"],
                ],
            ),
        )
    return create_model("BaseSearchParams", **fields)


def build_base_stepback_search_params(*, cut_off_default: float):
    fields = {
        "query": (str, Field(description="Query text to search in the index")),
        "index_name": (
            Optional[str],
            Field(
                description=f"Optional index name (max {INDEX_NAME_MAX_LENGTH} characters)",
                default="",
                max_length=INDEX_NAME_MAX_LENGTH,
            ),
        ),
        "messages": (
            Optional[List],
            Field(description="Chat messages for stepback search context", default=[]),
        ),
        **_search_common_fields(cut_off_default),
    }
    return create_model("BaseStepbackSearchParams", **fields)
