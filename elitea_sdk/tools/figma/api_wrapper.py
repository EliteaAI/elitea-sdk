import functools
import json
import logging
import re
from enum import Enum
from typing import Callable, ClassVar, Dict, List, Generator, Optional, Tuple, TypeVar, Union
from urllib.parse import urlparse, parse_qs

import requests

T = TypeVar('T')
from langchain_core.documents import Document
from langchain_core.tools import ToolException
from pydantic import Field, PrivateAttr, create_model, model_validator, SecretStr

from elitea_sdk.runtime.utils.utils import IndexerKeywords


# User-friendly error messages for common Figma API errors
FIGMA_ERROR_MESSAGES = {
    429: "Figma API rate limit exceeded. Please wait a moment and try again.",
    403: "Access denied. Please check your Figma API token has access to this file.",
    404: "File or node not found. Please verify the file key or node ID is correct.",
    401: "Authentication failed. Please check your Figma API token is valid.",
    500: "Figma server error. Please try again later.",
    503: "Figma service temporarily unavailable. Please try again later.",
}


def _handle_figma_error(e: ToolException) -> str:
    """
    Convert a ToolException from Figma API into a user-friendly error message.
    Returns a clean error string without technical details.
    """
    error_str = str(e)

    # Extract status code from error message
    for code, message in FIGMA_ERROR_MESSAGES.items():
        if f"error {code}:" in error_str.lower() or f"status\": {code}" in error_str:
            return message

    # Handle other common patterns
    if "rate limit" in error_str.lower():
        return FIGMA_ERROR_MESSAGES[429]
    if "not found" in error_str.lower():
        return FIGMA_ERROR_MESSAGES[404]
    if "forbidden" in error_str.lower() or "access denied" in error_str.lower():
        return FIGMA_ERROR_MESSAGES[403]
    if "unauthorized" in error_str.lower():
        return FIGMA_ERROR_MESSAGES[401]

    # Fallback: return a generic but clean message
    return f"Figma API request failed. Please try again or check your file key and permissions."

from ..non_code_indexer_toolkit import NonCodeIndexerToolkit
from ..utils.available_tools_decorator import extend_with_parent_available_tools
from ..utils.content_parser import _load_content_from_bytes_with_prompt
from .figma_client import EliteAFigmaPy
from .toon_tools import (
    TOONSerializer,
    process_page_to_toon_data,
    process_frame_to_toon_data,
    detect_sequences,
    group_variants,
    infer_cta_destination,
    AnalyzeFileSchema,
    analyze_frame_with_llm,
)
from .chunking_strategy import (
    BisectionChunkingStrategy,
)
from ..utils.retry import (
    is_server_error_retriable,
    is_volume_error_retriable,
    retry_on_server_error,
    retry_on_volume_error,
)

GLOBAL_LIMIT = 1000000
GLOBAL_RETAIN = ['id', 'name', 'type', 'document', 'children']
GLOBAL_REMOVE = []
GLOBAL_DEPTH_START = 1
GLOBAL_DEPTH_END = 6
DEFAULT_NUMBER_OF_THREADS = 5  # valid range for number_of_threads is 1..5
# Default prompts for image analysis and summarization reused across toolkit and wrapper
DEFAULT_FIGMA_IMAGES_PROMPT: Dict[str, str] = {
    "prompt": (
        "You are an AI model for image analysis. For each image, first identify its type "
        "(diagram, screenshot, photograph, illustration/drawing, text-centric, or mixed), "
        "then describe all visible elements and extract any readable text. For diagrams, "
        "capture titles, labels, legends, axes, and all numerical values, and summarize key "
        "patterns or trends. For screenshots, describe the interface or page, key UI elements, "
        "and any conversations or messages with participants and timestamps if visible. For "
        "photos and illustrations, describe the setting, main objects/people, their actions, "
        "style, colors, and composition. Be precise and thorough; when something is unclear or "
        "illegible, state that explicitly instead of guessing."
    )
}
DEFAULT_FIGMA_SUMMARY_PROMPT: Dict[str, str] = {
    "prompt": (
        "You are summarizing a visual design document exported from Figma as a sequence of images and text. "
        "Provide a clear, concise overview of the main purpose, key elements, and notable changes or variations in the screens. "
        "Infer a likely user flow or sequence of steps across the screens, calling out entry points, decisions, and outcomes. "
        "Explain how this design could impact planning, development, testing, and review activities in a typical software lifecycle. "
        "Return the result as structured Markdown with headings and bullet lists so it can be reused in SDLC documentation."
    )
}
EXTRA_PARAMS = (
    Optional[Dict[str, Union[str, int, List, None]]],
    Field(
        description=(
            "Optional output controls: `limit` (max characters, always applied), `regexp` (regex cleanup on text), "
            "`fields_retain`/`fields_remove` (which keys to keep or drop), and `depth_start`/`depth_end` (depth range "
            "where that key filtering is applied). Field/depth filters are only used when the serialized JSON result "
            "exceeds `limit` to reduce its size."
        ),
        default={
            "limit": GLOBAL_LIMIT, "regexp": None,
            "fields_retain": GLOBAL_RETAIN, "fields_remove": GLOBAL_REMOVE,
            "depth_start": GLOBAL_DEPTH_START, "depth_end": GLOBAL_DEPTH_END,
        },
        examples=[
            {
                "limit": "1000",
                "regexp": r'("strokes"|"fills")\s*:\s*("[^"]*"|[^\s,}\[]+)\s*(?=,|\}|\n)',
                "fields_retain": GLOBAL_RETAIN, "fields_remove": GLOBAL_REMOVE,
                "depth_start": GLOBAL_DEPTH_START, "depth_end": GLOBAL_DEPTH_END,
            }
        ],
    ),
)


class ArgsSchema(Enum):
    NoInput = create_model("NoInput")
    FileNodes = create_model(
        "FileNodes",
        file_key=(
            str,
            Field(
                description="Specifies file key id", examples=["Fp24FuzPwH0L74ODSrCnQo"]
            ),
        ),
        ids=(
            str,
            Field(
                description="Specifies id of file nodes separated by comma",
                examples=["8:6,1:7"],
            ),
        ),
        extra_params=EXTRA_PARAMS,
    )
    File = create_model(
        "FileNodes",
        file_key=(
            str,
            Field(
                description="Specifies file key id.",
                examples=["Fp24FuzPwH0L74ODSrCnQo"],
            ),
        ),
        geometry=(
            Optional[str],
            Field(description="Sets to 'paths' to export vector data", default=None),
        ),
        version=(
            Optional[str],
            Field(description="Sets version of file", default=None),
        ),
        extra_params=EXTRA_PARAMS,
    )
    FileKey = create_model(
        "FileKey",
        file_key=(
            str,
            Field(
                description="Specifies file key id.",
                examples=["Fp24FuzPwH0L74ODSrCnQo"],
            ),
        ),
        extra_params=EXTRA_PARAMS,
    )
    FileComment = create_model(
        "FileComment",
        file_key=(
            str,
            Field(
                description="Specifies file key id.",
                examples=["Fp24FuzPwH0L74ODSrCnQo"],
            ),
        ),
        message=(
            str,
            Field(description="Message for the comment."),
        ),
        client_meta=(
            Optional[dict],
            Field(
                description="Positioning information of the comment (Vector, FrameOffset, Region, FrameOffsetRegion)",
                default=None,
            ),
        ),
        extra_params=EXTRA_PARAMS,
    )
    FileImages = create_model(
        "FileImages",
        file_key=(
            str,
            Field(
                description="Specifies file key id.",
                examples=["Fp24FuzPwH0L74ODSrCnQo"],
            ),
        ),
        ids=(
            str,
            Field(
                description="Specifies id of file images separated by comma",
                examples=["8:6,1:7"],
            ),
        ),
        scale=(
            Optional[str],
            Field(description="A number between 0.01 and 4, the image scaling factor", default=None),
        ),
        format=(
            Optional[str],
            Field(
                description="A string enum for the image output format",
                examples=["jpg", "png", "svg", "pdf"],
                default=None,
            ),
        ),
        version=(
            Optional[str],
            Field(description="A specific version ID to use", default=None),
        ),
        extra_params=EXTRA_PARAMS,
    )
    TeamProjects = create_model(
        "TeamProjects",
        team_id=(
            str,
            Field(
                description="ID of the team to list projects from",
                examples=["1101853299713989222"],
            ),
        ),
        extra_params=EXTRA_PARAMS,
    )
    ProjectFiles = create_model(
        "ProjectFiles",
        project_id=(
            str,
            Field(
                description="ID of the project to list files from",
                examples=["55391681"],
            ),
        ),
        extra_params=EXTRA_PARAMS,
    )
    ExtractDesignTokens = create_model(
        "ExtractDesignTokens",
        file_key=(
            str,
            Field(
                description="Figma file key.",
                examples=["Fp24FuzPwH0L74ODSrCnQo"],
            ),
        ),
        node_id=(
            str,
            Field(
                description=(
                    "ID of the node to extract design tokens from. "
                    "Must be a FRAME, COMPONENT, COMPONENT_SET, SECTION, or GROUP — not a PAGE (CANVAS). "
                    "Hyphens are automatically converted to colons (e.g. '169-14446' → '169:14446')."
                ),
                examples=["169:14446"],
            ),
        ),
        depth=(
            Optional[int],
            Field(
                description=(
                    "How deep to traverse the node tree when fetching from Figma. "
                    "Use 4 for most frames and components (default). "
                    "Use 6 for COMPONENT_SET nodes with many variants. "
                    "Valid range: 1–8."
                ),
                default=4,
                ge=1,
                le=8,
            ),
        ),
    )
    ExtractDesignTokensBatch = create_model(
        "ExtractDesignTokensBatch",
        entries=(
            List[Dict],
            Field(
                description=(
                    "List of design token extraction requests. "
                    "Each entry must contain 'file_key' and 'node_id'. "
                    "Optional 'depth' per entry (defaults to 4). "
                    "Supports mixed depths across entries. "
                    "Node IDs with hyphens are auto-converted to colons."
                ),
                examples=[
                    [
                        {"file_key": "Fp24FuzPwH0L74ODSrCnQo", "node_id": "169:14446"},
                        {"file_key": "URzQWqsJejoMYDDz9TmyY1", "node_id": "1262:67108", "depth": 6},
                    ]
                ],
            ),
        ),
        parallel=(
            Optional[bool],
            Field(
                description="Enable parallel extraction for faster batch processing (default: True).",
                default=True,
            ),
        ),
        output_format=(
            Optional[str],
            Field(
                description=(
                    "Control output verbosity for LLM optimization: "
                    "'full' (all data), 'compact' (no full token lists), "
                    "'summary' (counts only, fastest), 'style_guide' (detailed tokens + component references). "
                    "Default: 'full'. Use 'compact'/'summary' for smallest LLM payload, "
                    "or 'style_guide' to generate a screen style guide."
                ),
                default="full",
            ),
        ),
    )


class FigmaApiWrapper(NonCodeIndexerToolkit):
    # Threshold for subframe extraction: frames larger than this will have their
    # children extracted for better image quality instead of rendering the whole frame
    SUBFRAME_EXTRACT_THRESHOLD: ClassVar[int] = 15000

    # Max dimension for scaling images to fit Claude's 8000px limit.
    # User sets the desired limit (e.g., 8000), and the effective value is calculated
    # as 98% to account for Figma API sometimes returning images slightly larger than requested.
    SCALE_TO_LIMIT_THRESHOLD: ClassVar[int] = 8000
    SCALE_MARGIN_PERCENT: ClassVar[float] = 0.02  # 2% safety margin

    @classmethod
    def get_effective_scale_threshold(cls) -> int:
        """Return effective scale threshold with 2% safety margin."""
        return int(cls.SCALE_TO_LIMIT_THRESHOLD * (1 - cls.SCALE_MARGIN_PERCENT))

    token: Optional[SecretStr] = Field(default=None)
    oauth2: Optional[SecretStr] = Field(default=None)
    global_limit: Optional[int] = Field(default=GLOBAL_LIMIT)
    global_regexp: Optional[str] = Field(default=None)
    global_fields_retain: Optional[List[str]] = GLOBAL_RETAIN
    global_fields_remove: Optional[List[str]] = GLOBAL_REMOVE
    global_depth_start: Optional[int] = Field(default=GLOBAL_DEPTH_START)
    global_depth_end: Optional[int] = Field(default=GLOBAL_DEPTH_END)
    # prompt-related configuration, populated from FigmaToolkit.toolkit_config_schema
    apply_images_prompt: Optional[bool] = Field(default=True)
    images_prompt: Optional[Dict[str, str]] = Field(default=DEFAULT_FIGMA_IMAGES_PROMPT)
    apply_summary_prompt: Optional[bool] = Field(default=True)
    summary_prompt: Optional[Dict[str, str]] = Field(default=DEFAULT_FIGMA_SUMMARY_PROMPT)
    # concurrency configuration, populated from toolkit config like images_prompt
    number_of_threads: Optional[int] = Field(default=DEFAULT_NUMBER_OF_THREADS, ge=1, le=5)
    _client: Optional[EliteAFigmaPy] = PrivateAttr()

    def _parse_figma_url(self, url: str) -> tuple[str, Optional[List[str]]]:
        """Parse and validate a Figma URL.

        Returns a tuple of (file_key, node_ids_from_url or None).
        Raises ToolException with a clear message if the URL is malformed.
        """
        try:
            parsed = urlparse(url)

            # Basic structural validation
            if not parsed.scheme or not parsed.netloc:
                raise ToolException(
                    "Figma URL must include protocol and host (e.g., https://www.figma.com/file/...). "
                    f"Got: {url}"
                )

            path_parts = parsed.path.strip('/').split('/') if parsed.path else []

            # Supported URL patterns:
            #  - /file/<file_key>/...
            #  - /design/<file_key>/... (older / embedded variant)
            if len(path_parts) < 2 or path_parts[0] not in {"file", "design"}:
                raise ToolException(
                    "Unsupported Figma URL format. Expected path like '/file/<FILE_KEY>/...' or "
                    "'/design/<FILE_KEY>/...'. "
                    f"Got path: '{parsed.path}' from URL: {url}"
                )

            file_key = path_parts[1]
            if not file_key:
                raise ToolException(
                    "Figma URL is missing the file key segment after '/file/' or '/design/'. "
                    f"Got path: '{parsed.path}' from URL: {url}"
                )

            # Optional node-id is passed via query parameter
            query_params = parse_qs(parsed.query or "")
            node_ids_from_url = query_params.get("node-id", []) or None

            return file_key, node_ids_from_url

        except ToolException:
            # Re-raise our own clear ToolException as-is
            raise
        except Exception as e:
            # Catch any unexpected parsing issues and wrap them clearly
            raise ToolException(
                "Unexpected error while processing Figma URL. "
                "Please provide a valid Figma file or page URL, for example: "
                "'https://www.figma.com/file/<FILE_KEY>/...'? "
                f"Original error: {e}"
            )

    def _remove_metadata_keys(self) -> List[str]:
        """Remove internal keys from document metadata before indexing.

        Extends base class to remove Figma-specific keys that are only used
        during document loading and should not be persisted to vectorstore.
        """
        base_keys = super()._remove_metadata_keys()
        figma_keys = [
            'number_of_threads_override', # Internal concurrency setting
            'figma_nodes_include',       # Already processed during loading
            'figma_nodes_exclude',       # Already processed during loading
            'image_max_dimension',       # Adaptive scaling setting
            'frame_spatial_order',       # Spatial ordering setting
        ]
        return base_keys + figma_keys

    def _base_loader(
        self,
        urls_or_file_keys: Optional[str] = None,
        node_ids_include: Optional[List[str]] = None,
        node_ids_exclude: Optional[List[str]] = None,
        node_types_include: Optional[List[str]] = None,
        node_types_exclude: Optional[List[str]] = None,
        number_of_threads: Optional[int] = None,
        image_max_dimension: Optional[int] = None,
        frame_spatial_order: Optional[bool] = None,
        subframe_extract_threshold: Optional[int] = None,
        scale_to_limit_threshold: Optional[int] = None,
        **kwargs,
    ) -> Generator[Document, None, None]:
        """Base loader used by the indexer tool.

        Args:
            urls_or_file_keys: Comma-separated list of Figma file URLs or raw file keys. Each
                entry can be:
                  - a full Figma URL (https://www.figma.com/file/... or /design/...) optionally
                    with a node-id query parameter, or
                  - a bare file key string.
                URL entries are parsed via _parse_figma_url; raw keys are used as-is.
            node_ids_include: Optional list of top-level node IDs (pages) to include when an
                entry does not specify node-id in the URL and is not otherwise constrained.
            node_ids_exclude: Optional list of top-level node IDs (pages) to exclude when
                node_ids_include is not provided.
            node_types_include: Optional list of node types to include within each page.
            node_types_exclude: Optional list of node types to exclude when node_types_include
                is not provided.
            number_of_threads: Optional override for number of worker threads to use when
                processing images.
            image_max_dimension: Maximum image dimension in pixels for LLM analysis.
                Images are scaled down to fit this limit. Default: 2000.
            frame_spatial_order: When True, processes frames in spatial order (top-to-bottom,
                left-to-right). Default: True.
        """
        self._init_indexing_stats()

        # Log model name used for indexing
        model_name = getattr(self.llm, 'model_name', None) or getattr(self.llm, 'model', None) or 'unknown'
        logging.info(f"Starting Figma index_data with LLM model: {model_name}")

        if not urls_or_file_keys:
            raise ValueError("You must provide urls_or_file_keys with at least one URL or file key.")

        # Parse the comma-separated entries into concrete (file_key, per_entry_node_ids_include)
        entries = [item.strip() for item in urls_or_file_keys.split(',') if item.strip()]
        if not entries:
            raise ValueError("You must provide urls_or_file_keys with at least one non-empty value.")

        # Validate number_of_threads override once and pass via metadata
        metadata_threads_override: Optional[int] = None
        if isinstance(number_of_threads, int) and 1 <= number_of_threads <= 5:
            metadata_threads_override = number_of_threads

        for entry in entries:
            per_file_node_ids_include: Optional[List[str]] = None
            file_key: Optional[str] = None

            # Heuristic: treat as URL if it has a scheme and figma.com host
            if entry.startswith("http://") or entry.startswith("https://"):
                file_key, node_ids_from_url = self._parse_figma_url(entry)
                per_file_node_ids_include = node_ids_from_url
            else:
                # Assume this is a raw file key
                file_key = entry

            if not file_key:
                continue

            # If URL-derived node IDs exist, they take precedence over global include list
            effective_node_ids_include = per_file_node_ids_include or node_ids_include or []

            self._log_tool_event(f"Loading file `{file_key}`")
            try:
                file = self._client.get_file(file_key, geometry='depth=1')
            except ToolException as e:
                # Enrich the error message with the file_key for easier troubleshooting
                raise ToolException(
                    f"Failed to retrieve Figma file '{file_key}'. Original error: {e}"
                ) from e

            if not file:
                raise ToolException(
                    f"Unexpected error while retrieving file {file_key}. Please try specifying the node-id of an inner page."
                )

            metadata = {
                'id': file_key,
                'file_key': file_key,
                'name': file.name,
                'updated_on': file.last_modified,
                'figma_pages_include': effective_node_ids_include,
                'figma_pages_exclude': node_ids_exclude or [],
                'figma_nodes_include': node_types_include or [],
                'figma_nodes_exclude': node_types_exclude or [],
                'image_max_dimension': image_max_dimension if image_max_dimension is not None else 2000,
                'frame_spatial_order': frame_spatial_order if frame_spatial_order is not None else True,
                'subframe_extract_threshold': subframe_extract_threshold if subframe_extract_threshold is not None else self.SUBFRAME_EXTRACT_THRESHOLD,
                'scale_to_limit_threshold': scale_to_limit_threshold if scale_to_limit_threshold is not None else self.SCALE_TO_LIMIT_THRESHOLD,
            }

            if metadata_threads_override is not None:
                metadata['number_of_threads_override'] = metadata_threads_override

            self._track_processed_item()
            yield Document(page_content=json.dumps(metadata), metadata=metadata)

    def has_image_representation(self, node):
        node_type = node.get('type', '').lower()
        default_images_types = [
            'image', 'canvas', 'frame', 'vector', 'table', 'slice', 'sticky', 'shape_with_text', 'connector'
        ]
        # filter nodes of type which has image representation
        # or rectangles with image as background
        if (node_type in default_images_types
                or (node_type == 'rectangle' and 'fills' in node and any(
                    fill.get('type') == 'IMAGE' for fill in node['fills'] if isinstance(fill, dict)))):
            return True
        return False

    def get_texts_recursive(self, node):
        texts = []
        node_type = node.get('type', '').lower()
        if node_type == 'text':
            texts.append(node.get('characters', ''))
        if 'children' in node:
            for child in node['children']:
                texts.extend(self.get_texts_recursive(child))
        return texts
    
    def _load_pages(
        self,
        document: Document,
    ) -> List[dict]:
        """
        Load pages from a Figma file using bisection chunking strategy.

        Args:
            document: Document with file metadata

        Returns:
            List of page content dictionaries
        """
        file_key = document.metadata.get('id', '')
        node_ids_include = document.metadata.pop('figma_pages_include', [])
        node_ids_exclude = document.metadata.pop('figma_pages_exclude', [])
        self._log_tool_event(f"Included pages: {node_ids_include}. Excluded pages: {node_ids_exclude}.")

        # Eager split at 10 pages - larger batches almost always timeout
        strategy = BisectionChunkingStrategy(eager_split_threshold=10)

        # First, get shallow file structure to know all page IDs
        shallow_file = self._client.get_file(file_key, geometry='depth=1')
        if not shallow_file:
            raise ToolException(
                f"Unexpected error while retrieving file {file_key}. "
                "Please try specifying the node-id of an inner page."
            )

        # Build shallow page lookup for fallback on failed pages
        shallow_pages_by_id: Dict[str, dict] = {}
        for page in shallow_file.document.get('children', []):
            if 'id' in page:
                shallow_pages_by_id[page['id']] = page

        all_page_ids = list(shallow_pages_by_id.keys())

        # Determine which pages to fetch
        if node_ids_include:
            pages_to_fetch = node_ids_include
        else:
            # Exclude specified pages
            pages_to_fetch = [
                pid for pid in all_page_ids
                if pid.replace(':', '-') not in node_ids_exclude
            ]

        if not pages_to_fetch:
            logging.info(f"File {file_key}: no pages to fetch after filtering")
            return []

        # Define fetch function for pages with retry for transient errors
        def fetch_pages(page_ids: List[str]) -> Dict[str, dict]:
            """Fetch page content for given page IDs with retry for transient server errors."""
            logging.debug(f"fetch_pages: requesting {len(page_ids)} pages: {page_ids}")

            @retry_on_server_error(max_attempts=3, wait_seconds=(1, 4, 10))
            def do_fetch() -> Dict[str, dict]:
                result = self._get_file_nodes(file_key, ','.join(page_ids))
                if not result:
                    logging.debug("fetch_pages: result is empty/None")
                    return {}
                nodes = result.get("nodes") or {}
                logging.debug(f"fetch_pages: got {len(nodes)} nodes in response")
                output = {}
                for page_id, node in nodes.items():
                    if node is None:
                        logging.debug(f"Page {page_id}: node is None (skipped by Figma)")
                        continue
                    if "document" not in node:
                        logging.debug(f"Page {page_id}: no 'document' key in node: {list(node.keys())}")
                        continue
                    output[page_id] = node["document"]
                logging.debug(f"fetch_pages: returning {len(output)} pages")
                return output

            # Retry transient server errors (5xx), but let volume errors propagate to bisection
            try:
                return do_fetch()
            except Exception as e:
                # Re-raise for bisection to handle (volume errors will trigger split)
                logging.debug(f"fetch_pages: error after retries: {type(e).__name__}: {e}")
                raise

        # Execute with bisection strategy (splits on volume errors)
        chunk_result = strategy.execute(
            items=pages_to_fetch,
            fetch_fn=fetch_pages,
            is_retriable_error=is_volume_error_retriable,
        )

        # Log summary at INFO level
        logging.info(
            f"File {file_key}: loaded {len(chunk_result.successful)}/{len(pages_to_fetch)} pages"
            + (f", {len(chunk_result.failed)} failed: {chunk_result.failed}" if chunk_result.failed else "")
        )

        # Return page content in order (preserve original order where possible)
        # For failed pages, use shallow page data as fallback (allows frame extraction via individual fetches)
        pages = []
        for page_id in pages_to_fetch:
            if page_id in chunk_result.successful:
                pages.append(chunk_result.successful[page_id])
            elif page_id in chunk_result.failed and page_id in shallow_pages_by_id:
                # Use shallow page data as fallback - process_page_streaming will try to fetch full content
                logging.debug(f"File {file_key}: using shallow data fallback for failed page {page_id}")
                pages.append(shallow_pages_by_id[page_id])

        return pages

    def _describe_image(
        self,
        image_url: str,
        prompt: str,
        node_id: str = "",
    ) -> Optional[str]:
        """
        Download an image from URL and describe it using the LLM.

        Args:
            image_url: URL to download the image from
            prompt: Prompt to guide the LLM's image description
            node_id: Node ID for logging

        Returns:
            LLM-generated description of the image, or None if processing fails
        """
        log = logging.getLogger(__name__)

        if not image_url:
            return None

        if not self.llm:
            return None

        try:
            response = requests.get(image_url, timeout=60)
        except Exception as exc:
            log.warning(f"Download failed for node {node_id}: {exc}")
            return None

        if response.status_code != 200:
            return None

        content_type = response.headers.get('Content-Type', '')

        if 'text/html' in content_type.lower():
            return None

        extension = f".{content_type.split('/')[-1]}" if content_type.startswith('image') else '.png'

        try:
            description = _load_content_from_bytes_with_prompt(
                file_content=response.content,
                extension=extension,
                llm=self.llm,
                prompt=prompt,
            )
            return description
        except Exception as exc:
            log.warning(f"LLM processing failed for node {node_id}: {exc}")
            return None

    def _calculate_optimal_scale(
        self,
        width: float,
        height: float,
        target_max_dimension: int = 2000,
    ) -> float:
        """
        Calculate optimal scale to keep image dimensions within target.

        Figma API supports scale from 0.01 to 4.0. This method calculates
        the scale needed to ensure the largest dimension doesn't exceed
        the target, while staying within Figma's bounds.

        Args:
            width: Original width in pixels
            height: Original height in pixels
            target_max_dimension: Target maximum for either dimension (default 2000)

        Returns:
            Scale factor between 0.01 and 4.0 (Figma API limits)
        """
        if width <= 0 or height <= 0:
            return 1.0

        max_dim = max(width, height)
        if max_dim <= target_max_dimension:
            return 1.0  # No scaling needed

        optimal_scale = target_max_dimension / max_dim
        # Clamp to Figma API bounds (0.01 to 4.0)
        return max(0.01, min(optimal_scale, 4.0))

    def _sort_frames_spatially(
        self,
        frames: List[Dict],
    ) -> List[Dict]:
        """
        Sort frames by spatial position: top-to-bottom, then left-to-right.

        This preserves the visual reading order of UI components, which is
        important for LLM analysis to understand the flow of a design.

        The y-coordinate is rounded to the nearest 50px to group frames
        that are roughly on the same "row" together.

        Args:
            frames: List of frame dicts with 'bounds' containing x, y coordinates

        Returns:
            Sorted list of frames in spatial reading order
        """
        def sort_key(frame: Dict) -> Tuple[float, float]:
            bounds = frame.get('bounds', {})
            y = bounds.get('y', 0)
            x = bounds.get('x', 0)
            # Round y to nearest 50px to group frames on same "row"
            y_rounded = round(y / 50) * 50
            return (y_rounded, x)

        return sorted(frames, key=sort_key)

    def _get_large_frame_strategy(
        self,
        width: float,
        height: float,
        scale_threshold: Optional[int] = None,
        extract_threshold: Optional[int] = None,
    ) -> str:
        """
        Determine the strategy for handling a large frame.

        Strategies:
        - 'normal': Frame fits within limits, use adaptive scaling
        - 'scale_to_limit': Frame is 7800-15000px, scale down to fit Claude's 8000px limit
        - 'extract_subframes': Frame is >15000px, extract children for better quality

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            scale_threshold: Max dimension for normal processing (default SCALE_TO_LIMIT_THRESHOLD)
            extract_threshold: Above this, extract subframes (default SUBFRAME_EXTRACT_THRESHOLD)

        Returns:
            Strategy string: 'normal', 'scale_to_limit', or 'extract_subframes'
        """
        if scale_threshold is None:
            scale_threshold = self.get_effective_scale_threshold()
        if extract_threshold is None:
            extract_threshold = self.SUBFRAME_EXTRACT_THRESHOLD

        if width <= 0 or height <= 0:
            return 'normal'

        max_dim = max(width, height)

        if max_dim <= scale_threshold:
            return 'normal'
        elif max_dim <= extract_threshold:
            return 'scale_to_limit'
        else:
            return 'extract_subframes'

    def _extract_subframes(
        self,
        file_key: str,
        parent_node_id: str,
        parent_info: Dict,
        scale_threshold: Optional[int] = None,
        extract_threshold: Optional[int] = None,
        debug_logger: Optional[logging.Logger] = None,
    ) -> List[Dict]:
        """
        Extract child frames from a large parent frame for individual processing.

        When a frame is too large (>SUBFRAME_EXTRACT_THRESHOLD) to render at good quality,
        we fetch its children and process them individually. This preserves semantic context
        (each child is a meaningful UI element) while allowing higher quality rendering.

        Args:
            file_key: Figma file key
            parent_node_id: ID of the large parent frame
            parent_info: Parent frame metadata dict
            scale_threshold: Max dimension for normal processing (default SCALE_TO_LIMIT_THRESHOLD)
            extract_threshold: Above this, extract subframes (default SUBFRAME_EXTRACT_THRESHOLD)
            debug_logger: Optional logger for debug output

        Returns:
            List of frame_info dicts for each child, similar to frames_to_process format.
            Each child has a 'strategy' field indicating how it should be processed.
        """
        if scale_threshold is None:
            scale_threshold = self.get_effective_scale_threshold()
        if extract_threshold is None:
            extract_threshold = self.SUBFRAME_EXTRACT_THRESHOLD

        log = debug_logger or logging.getLogger(__name__)
        parent_name = parent_info.get('node_name', parent_node_id)

        try:
            # Fetch the parent node with its children
            nodes_data = self._get_file_nodes(file_key, parent_node_id)
            if not nodes_data or 'nodes' not in nodes_data:
                log.warning(f"Failed to fetch children for {parent_node_id}")
                return []

            parent_node = nodes_data['nodes'].get(parent_node_id, {}).get('document', {})
            children = parent_node.get('children', [])

            if not children:
                log.debug(f"No children found in '{parent_name}'")
                return []

            log.debug(f"Extracting {len(children)} subframes from '{parent_name}'")

            subframes = []
            for child in children:
                child_id = child.get('id')
                child_name = child.get('name', '')
                child_type = child.get('type', '').lower()

                if not child_id:
                    continue

                # Extract bounds
                bounds = child.get('absoluteBoundingBox', {})
                width = bounds.get('width', 0)
                height = bounds.get('height', 0)
                x = bounds.get('x', 0)
                y = bounds.get('y', 0)

                # Skip if no dimensions
                if width <= 0 or height <= 0:
                    continue

                # Collect text from this child recursively
                texts = self.get_texts_recursive(child)

                # Skip vectors with no text content - they're just connectors/arrows
                # The connection info is typically in the name (e.g., "CTA --> content")
                is_vector = child_type in ('vector', 'line', 'arrow')
                if is_vector and not texts:
                    log.debug(f"SKIP {child_type}: {child_name} ({child_id})")
                    continue

                # Text nodes: no image needed, just text extraction
                is_text_node = child_type == 'text'

                # Determine strategy for this child (only relevant for non-text nodes)
                if is_text_node:
                    strategy = 'text_only'
                else:
                    strategy = self._get_large_frame_strategy(
                        width, height, scale_threshold, extract_threshold
                    )

                subframe_info = {
                    'node_id': child_id,
                    'node_name': child_name,
                    'node_type': child_type,
                    'page_id': parent_info.get('page_id', ''),
                    'page_name': parent_info.get('page_name', ''),
                    'parent_frame_id': parent_node_id,
                    'parent_frame_name': parent_name,
                    'texts': texts,
                    'has_image': not is_text_node,  # Text nodes don't need image
                    'bounds': {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                    },
                    'is_subframe': True,  # Mark as extracted subframe
                    'strategy': strategy,  # 'normal', 'scale_to_limit', 'extract_subframes', or 'text_only'
                    'node': child,  # Preserve original Figma node for TOON processing
                }

                subframes.append(subframe_info)

                log.debug(
                    f"  Subframe: {child_name} ({child_id}), "
                    f"size: {width:.0f}x{height:.0f}, strategy: {strategy}"
                )

            return subframes

        except Exception as e:
            log.warning(f"Error extracting subframes from {parent_node_id}: {e}")
            return []

    def _add_subframes_to_scale_groups(
        self,
        subframes: List[Dict],
        file_key: str,
        scale_threshold: int,
        extract_threshold: int,
        add_to_scale_group: Callable[[str, float], None],
        log: logging.Logger,
    ) -> None:
        """
        Recursively add subframes to scale groups for image fetching.

        Handles nested subframes that also need extraction, ensuring all
        leaf frames get their images fetched.
        """
        for sf in subframes:
            sf_strategy = sf.get('strategy', 'normal')
            sf_id = sf['node_id']
            sf_bounds = sf['bounds']
            sf_width = sf_bounds.get('width', 0)
            sf_height = sf_bounds.get('height', 0)

            if sf_strategy == 'extract_subframes':
                # This subframe needs further extraction
                nested = self._extract_subframes(
                    file_key, sf_id, sf,
                    scale_threshold, extract_threshold, log
                )
                sf['subframes'] = nested
                # Recursively process nested subframes
                self._add_subframes_to_scale_groups(
                    nested, file_key, scale_threshold, extract_threshold,
                    add_to_scale_group, log
                )
            elif sf_strategy == 'text_only':
                pass  # No image needed
            else:
                # Normal or scale_to_limit - add to scale group
                if sf.get('has_image', True):
                    sf_scale = self._calculate_scale_for_strategy(
                        sf_strategy, sf_width, sf_height, scale_threshold
                    )
                    add_to_scale_group(sf_id, sf_scale)

    def _collect_frames_for_analysis(
        self,
        file_key: str,
        pages: List[Dict],
        max_frames: int = 50,
        debug_logger: Optional[logging.Logger] = None,
        subframe_extract_threshold: Optional[int] = None,
        scale_to_limit_threshold: Optional[int] = None,
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Collect frames with subframe extraction and compute scaled image URLs.

        This method applies frame-level optimizations for analyze_file:
        - Subframe extraction for large frames (>extract_threshold px)
        - Adaptive scaling for images
        - Vector/connector skipping
        - Spatial ordering

        Args:
            file_key: Figma file key
            pages: List of page nodes to process
            max_frames: Maximum frames to collect (default 50)
            debug_logger: Optional logger for debug output
            subframe_extract_threshold: Override for SUBFRAME_EXTRACT_THRESHOLD
            scale_to_limit_threshold: Override for SCALE_TO_LIMIT_THRESHOLD

        Returns:
            Tuple of:
            - frames_list: List of frame dicts with nesting preserved (subframes field)
            - image_urls_dict: {frame_id: image_url} for all frames needing images
        """
        log = debug_logger or logging.getLogger(__name__)

        all_frames: List[Dict] = []
        scale_groups: Dict[str, List[str]] = {}  # scale_str -> [node_ids]
        frame_scales: Dict[str, float] = {}  # node_id -> scale

        # Use provided thresholds or fall back to class constants
        effective_scale_limit = scale_to_limit_threshold if scale_to_limit_threshold is not None else self.SCALE_TO_LIMIT_THRESHOLD
        scale_threshold = int(effective_scale_limit * (1 - self.SCALE_MARGIN_PERCENT))
        extract_threshold = subframe_extract_threshold if subframe_extract_threshold is not None else self.SUBFRAME_EXTRACT_THRESHOLD
        log.debug(f"Scale threshold: {scale_threshold}px, Extract threshold: {extract_threshold}px")

        def add_to_scale_group(node_id: str, scale: float):
            """Helper to add a frame to scale groups for batch fetching."""
            frame_scales[node_id] = scale
            scale_str = f"{scale:.2f}"
            if scale_str not in scale_groups:
                scale_groups[scale_str] = []
            scale_groups[scale_str].append(node_id)

        # Phase 1: Collect frames from all pages
        for page in pages:
            page_id = page.get('id', '')
            page_name = page.get('name', '')
            children = page.get('children', [])

            log.debug(f"Page: {page_name} ({page_id}), children: {len(children)}")

            for node in children:
                node_type = node.get('type', '').lower()
                node_id = node.get('id')
                node_name = node.get('name', '')

                if not node_id:
                    continue

                # Extract bounds
                bounds = node.get('absoluteBoundingBox', {})
                width = bounds.get('width', 0)
                height = bounds.get('height', 0)
                x = bounds.get('x', 0)
                y = bounds.get('y', 0)

                # Skip if no dimensions
                if width <= 0 or height <= 0:
                    continue

                # Collect text recursively
                texts = self.get_texts_recursive(node)

                # Skip vectors with no text content
                is_vector = node_type in ('vector', 'line', 'arrow')
                if is_vector and not texts:
                    log.debug(f"  SKIP {node_type}: {node_name} ({node_id})")
                    continue

                # Determine strategy
                strategy = self._get_large_frame_strategy(width, height, scale_threshold, extract_threshold)

                frame_info = {
                    'node_id': node_id,
                    'node_name': node_name,
                    'node_type': node_type,
                    'page_id': page_id,
                    'page_name': page_name,
                    'texts': texts,
                    'bounds': {'x': x, 'y': y, 'width': width, 'height': height},
                    'strategy': strategy,
                    'subframes': [],  # Will be populated if strategy == 'extract_subframes'
                    'is_subframe': False,
                    'node': node,  # Preserve original Figma node for TOON processing
                }

                log.debug(
                    f"  Frame: {node_name} ({node_id}) - {node_type}, "
                    f"size: {width:.0f}x{height:.0f}, strategy: {strategy}"
                )

                # Handle based on strategy
                if strategy == 'extract_subframes':
                    # Extract subframes for large frames
                    subframes = self._extract_subframes(
                        file_key, node_id, frame_info,
                        scale_threshold, extract_threshold, log
                    )

                    # Process subframes recursively for nested large frames
                    processed_subframes = []
                    for sf in subframes:
                        sf_strategy = sf.get('strategy', 'normal')
                        sf_id = sf['node_id']
                        sf_bounds = sf['bounds']
                        sf_width = sf_bounds.get('width', 0)
                        sf_height = sf_bounds.get('height', 0)

                        if sf_strategy == 'extract_subframes':
                            # Recursively extract nested subframes
                            nested = self._extract_subframes(
                                file_key, sf_id, sf,
                                scale_threshold, extract_threshold, log
                            )
                            sf['subframes'] = nested
                            # Add nested subframes to scale groups (recursively)
                            self._add_subframes_to_scale_groups(
                                nested, file_key, scale_threshold, extract_threshold,
                                add_to_scale_group, log
                            )
                        elif sf_strategy == 'text_only':
                            pass  # No image needed
                        else:
                            # Calculate scale and add to group
                            sf_scale = self._calculate_scale_for_strategy(
                                sf_strategy, sf_width, sf_height, scale_threshold
                            )
                            add_to_scale_group(sf_id, sf_scale)

                        processed_subframes.append(sf)

                    frame_info['subframes'] = processed_subframes
                    # Parent frame doesn't need its own image when subframes are extracted
                    frame_info['has_image'] = False

                elif strategy == 'scale_to_limit':
                    # Scale down to fit Claude's 8000px limit
                    max_dim = max(width, height)
                    scale = scale_threshold / max_dim
                    add_to_scale_group(node_id, scale)
                    frame_info['has_image'] = True

                else:  # 'normal'
                    # Use adaptive scaling
                    scale = self._calculate_optimal_scale(width, height, target_max_dimension=2000)
                    add_to_scale_group(node_id, scale)
                    frame_info['has_image'] = True

                all_frames.append(frame_info)

                # Check frame limit
                total_frames = len(all_frames) + sum(
                    len(f.get('subframes', [])) for f in all_frames
                )
                if total_frames >= max_frames:
                    log.debug(f"Reached max_frames limit ({max_frames})")
                    break

            if len(all_frames) >= max_frames:
                break

        # Sort frames spatially
        all_frames = self._sort_frames_spatially(all_frames)

        # Count total including subframes
        total_with_subframes = len(all_frames)
        for f in all_frames:
            total_with_subframes += len(f.get('subframes', []))
            for sf in f.get('subframes', []):
                total_with_subframes += len(sf.get('subframes', []))
        log.debug(f"Collected {len(all_frames)} top-level frames, {total_with_subframes} total with subframes")

        # Phase 2: Batch fetch image URLs by scale (parallel for multiple scale groups)
        all_image_urls: Dict[str, str] = {}

        if len(scale_groups) <= 1:
            # Zero or one scale group - no need for threading overhead
            for scale_str, node_ids in scale_groups.items():
                scale = float(scale_str)
                try:
                    images = self._get_file_images_with_scale(
                        file_key, node_ids, scale=scale, debug_logger=log
                    )
                    all_image_urls.update(images)
                except Exception as e:
                    log.warning(f"Failed to fetch images at scale {scale_str}: {e}")
        else:
            # Multiple scale groups - fetch in parallel
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def fetch_for_scale(scale_str: str, node_ids: List[str]) -> Dict[str, str]:
                scale = float(scale_str)
                try:
                    return self._get_file_images_with_scale(
                        file_key, node_ids, scale=scale, debug_logger=log
                    )
                except Exception as e:
                    log.warning(f"Failed to fetch images at scale {scale_str}: {e}")
                    return {}

            with ThreadPoolExecutor(max_workers=len(scale_groups)) as executor:
                futures = {
                    executor.submit(fetch_for_scale, scale_str, node_ids): scale_str
                    for scale_str, node_ids in scale_groups.items()
                }
                for future in as_completed(futures):
                    try:
                        images = future.result()
                        all_image_urls.update(images)
                    except Exception as e:
                        log.warning(f"Image fetch future failed: {e}")

        return all_frames, all_image_urls

    def _calculate_scale_for_strategy(
        self,
        strategy: str,
        width: float,
        height: float,
        scale_threshold: int,
    ) -> float:
        """Calculate the appropriate scale based on strategy."""
        if strategy == 'scale_to_limit':
            max_dim = max(width, height)
            return scale_threshold / max_dim if max_dim > 0 else 1.0
        elif strategy == 'normal':
            return self._calculate_optimal_scale(width, height, target_max_dimension=2000)
        else:
            return 1.0

    def _split_image_into_tiles(
        self,
        image_data: bytes,
        cols: int,
        rows: int,
        overlap: int = 50,
    ) -> List[Tuple[bytes, int, int, Dict]]:
        """
        Split an image into a grid of tiles with optional overlap.

        Args:
            image_data: PNG image bytes
            cols: Number of columns
            rows: Number of rows
            overlap: Overlap in pixels between adjacent tiles (default 50)

        Returns:
            List of (tile_bytes, col_index, row_index, tile_info) tuples,
            ordered left-to-right, top-to-bottom.
            tile_info contains: width, height, left, top, right, bottom
        """
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_data))
        img_width, img_height = img.size

        # Calculate base tile dimensions
        base_tile_width = img_width // cols
        base_tile_height = img_height // rows

        tiles = []
        for row in range(rows):
            for col in range(cols):
                # Calculate tile boundaries with overlap
                left = max(0, col * base_tile_width - (overlap if col > 0 else 0))
                upper = max(0, row * base_tile_height - (overlap if row > 0 else 0))

                # For rightmost/bottom tiles, extend to image edge
                if col == cols - 1:
                    right = img_width
                else:
                    right = min(img_width, (col + 1) * base_tile_width + overlap)

                if row == rows - 1:
                    lower = img_height
                else:
                    lower = min(img_height, (row + 1) * base_tile_height + overlap)

                tile = img.crop((left, upper, right, lower))

                buffer = io.BytesIO()
                tile.save(buffer, format='PNG')
                tile_info = {
                    'width': right - left,
                    'height': lower - upper,
                    'left': left,
                    'top': upper,
                    'right': right,
                    'bottom': lower,
                }
                tiles.append((buffer.getvalue(), col, row, tile_info))

        return tiles

    def _describe_image_from_bytes(
        self,
        image_data: bytes,
        prompt: str = "",
        debug_logger: Optional[logging.Logger] = None,
    ) -> Optional[str]:
        """
        Get LLM description for image from raw bytes.

        Similar to _describe_image but takes bytes instead of URL.
        Used for analyzing tiles that were split locally.

        Args:
            image_data: PNG image bytes
            prompt: Prompt for the LLM
            debug_logger: Optional logger for debug output

        Returns:
            LLM-generated description or None if no LLM configured
        """
        from langchain_core.messages import HumanMessage
        import base64

        log = debug_logger or logging.getLogger(__name__)

        if not self.llm:
            log.debug("No LLM configured, skipping image description")
            return None

        try:
            image_b64 = base64.b64encode(image_data).decode('utf-8')

            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": prompt or "Describe this UI element in detail."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ])
            ]

            response = self.llm.invoke(messages)
            if response and response.content:
                return response.content
            return None
        except Exception as e:
            log.warning(f"Failed to get LLM description from bytes: {e}")
            return None

    def _get_file_images_with_scale(
        self,
        file_key: str,
        node_ids: List[str],
        scale: float = 1.0,
        debug_logger: Optional[logging.Logger] = None,
        max_retries: int = 3,
    ) -> Dict[str, str]:
        """
        Fetch image URLs with specified scale, with retry and bisection for large batches.

        This method is used by frame-level processing to request images
        at a specific scale factor for optimal text readability.

        Uses bisection strategy when Figma returns "render timeout" (400) for batches
        that are too large to render at once.

        Args:
            file_key: Figma file key
            node_ids: List of node IDs to render
            scale: Scale factor (0.01-4.0)
            debug_logger: Optional logger for debug output
            max_retries: Maximum retry attempts for transient errors (default 3)

        Returns:
            Dict mapping node_id -> image_url
        """
        log = debug_logger or logging.getLogger(__name__)

        if not node_ids:
            return {}

        scale_str = f"{scale:.2f}"

        def is_render_timeout(e: BaseException) -> bool:
            """Check if error is Figma render timeout (needs bisection)."""
            error_str = str(e).lower()
            return 'render timeout' in error_str or 'too large' in error_str

        @retry_on_server_error(max_attempts=max_retries + 1, wait_seconds=(1, 4, 10, 30), log=log)
        def fetch_images_batch(ids: List[str]) -> Dict[str, str]:
            """Fetch a batch of images, raising on error."""
            id_list = ','.join(ids)
            endpoint = f'images/{file_key}?ids={id_list}&scale={scale_str}&format=png'
            data = self._client.api_request(endpoint, method='get')
            if data and 'images' in data:
                return data['images']
            return {}

        # Eager split threshold - batches >50 images almost always timeout
        EAGER_SPLIT_THRESHOLD = 50

        def fetch_with_bisection(ids: List[str]) -> Dict[str, str]:
            """Fetch images with bisection fallback for render timeouts."""
            # Eager split: skip first try if batch exceeds threshold (avoids ~30s timeout)
            if len(ids) > EAGER_SPLIT_THRESHOLD:
                mid = len(ids) // 2
                result = {}
                if ids[:mid]:
                    result.update(fetch_with_bisection(ids[:mid]))
                if ids[mid:]:
                    result.update(fetch_with_bisection(ids[mid:]))
                return result

            try:
                return fetch_images_batch(ids)
            except Exception as e:
                if is_render_timeout(e) and len(ids) > 1:
                    # Split batch in half and try each half
                    mid = len(ids) // 2
                    result = {}
                    if ids[:mid]:
                        result.update(fetch_with_bisection(ids[:mid]))
                    if ids[mid:]:
                        result.update(fetch_with_bisection(ids[mid:]))
                    return result
                elif is_render_timeout(e) and len(ids) == 1:
                    # Single item still times out - skip it
                    log.warning(f"Frame {ids[0]} render timeout, skipping")
                    return {}
                else:
                    raise

        try:
            images = fetch_with_bisection(list(node_ids))
            return images
        except Exception as e:
            log.warning(f"Failed to fetch images at scale {scale_str} after retries: {e}")
            return {}

    def _process_document(
        self,
        document: Document,
        prompt: str = "",
    ) -> Generator[Document, None, None]:
        file_key = document.metadata.get('id', '')
        self._log_tool_event(f"Loading details (images) for `{file_key}`")

        # # If prompt not provided, retrieve from toolkit configuration
        # if not prompt:
        #     apply_images_prompt = getattr(self, "apply_images_prompt", True)
        #     images_prompt = getattr(self, "images_prompt", None)
        #     if (
        #         apply_images_prompt
        #         and isinstance(images_prompt, dict)
        #         and isinstance(images_prompt.get("prompt"), str)
        #         and images_prompt["prompt"].strip()
        #     ):
        #         prompt = images_prompt["prompt"].strip()

        figma_pages = self._load_pages(document)
        node_types_include = [t.strip().lower() for t in document.metadata.pop('figma_nodes_include', [])]
        node_types_exclude = [t.strip().lower() for t in document.metadata.pop('figma_nodes_exclude', [])]

        # Extract threshold parameters (with fallback to class constants)
        subframe_extract_threshold = document.metadata.pop('subframe_extract_threshold', self.SUBFRAME_EXTRACT_THRESHOLD)
        scale_to_limit_threshold = document.metadata.pop('scale_to_limit_threshold', self.SCALE_TO_LIMIT_THRESHOLD)

        yield from self._process_toon_level(
            document, figma_pages, node_types_include, node_types_exclude, prompt,
            subframe_extract_threshold=subframe_extract_threshold,
            scale_to_limit_threshold=scale_to_limit_threshold,
        )

    def _process_toon_level(
        self,
        document: Document,
        figma_pages: List[dict],
        node_types_include: List[str],
        node_types_exclude: List[str],
        prompt: str = "",
        subframe_extract_threshold: Optional[int] = None,
        scale_to_limit_threshold: Optional[int] = None,
    ) -> Generator[Document, None, None]:
        """
        Process document at TOON level - structured TOON format with LLM analysis.

        Outputs:
        - 1 document for file: List of pages overview
        - 1 document per page: FLOWS, VARIANTS, and DESIGN INSIGHTS blocks
        - 1 document per top-level frame: Full TOON content including nested subframes

        Features:
        - Parallel page processing for faster indexing
        - Subframe extraction for large frames (>15000px)
        - Adaptive scaling for image URLs
        - LLM analysis for each frame with vision support
        - Hierarchical frame structure with level indicators
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from queue import Queue
        from threading import Thread

        file_key = document.metadata.get('id', '')
        file_name = document.metadata.get('name', 'Untitled')
        log = logging.getLogger(__name__)

        # Resolve number of threads
        override_threads = document.metadata.get('number_of_threads_override')
        if isinstance(override_threads, int) and 1 <= override_threads <= 5:
            number_of_threads = override_threads
        else:
            threads_cfg = getattr(self, "number_of_threads", DEFAULT_NUMBER_OF_THREADS)
            number_of_threads = threads_cfg if isinstance(threads_cfg, int) and 1 <= threads_cfg <= 5 else DEFAULT_NUMBER_OF_THREADS

        serializer = TOONSerializer()
        max_frames_per_page = 50

        # === Helper to serialize a single leaf frame ===
        def serialize_leaf_frame(frame: Dict, explanation: Optional[object] = None) -> List[str]:
            """Serialize a leaf frame (no subframes) with LLM explanation."""
            from .toon_tools import format_inputs_list
            lines = []
            frame_id = frame.get('id', '')
            frame_name = frame.get('name', 'Untitled')
            frame_type = frame.get('type', 'screen')
            frame_state = frame.get('state', 'default')

            pos = frame.get('position', {})
            size = frame.get('size', {})
            pos_str = f"[{int(pos.get('x', 0))},{int(pos.get('y', 0))} {int(size.get('w', 0))}x{int(size.get('h', 0))}]"

            lines.append(f"FRAME: {frame_name} {pos_str} {frame_type}/{frame_state} #{frame_id}")

            if explanation:
                lines.append(f"  Purpose: {explanation.purpose}")
                goal_line = f"Goal: {explanation.user_goal}"
                if explanation.primary_action:
                    goal_line += f" | Action: \"{explanation.primary_action}\""
                lines.append(f"  {goal_line}")

                visual_parts = []
                if explanation.visual_focus:
                    visual_parts.append(f"[focus] {explanation.visual_focus}")
                if explanation.layout_pattern:
                    visual_parts.append(f"[layout] {explanation.layout_pattern}")
                if visual_parts:
                    lines.append(f"  Visual: {' | '.join(visual_parts)}")

            # Content from TOON data
            headings = frame.get('headings', [])
            buttons = frame.get('buttons', [])
            inputs = frame.get('inputs', [])

            if headings:
                lines.append(f"  Headings: {' | '.join(headings[:5])}")
            if buttons:
                btn_strs = []
                for btn in buttons[:8]:
                    dest = infer_cta_destination(btn) if isinstance(btn, str) else btn.get('destination', '')
                    btn_text = btn if isinstance(btn, str) else btn.get('text', '')
                    btn_strs.append(f"{btn_text} > {dest}" if dest else btn_text)
                lines.append(f"  Buttons: {' | '.join(btn_strs)}")
            if inputs:
                inputs_str = format_inputs_list(inputs[:10])
                if inputs_str:
                    lines.append(f"  Inputs: {inputs_str}")

            return lines

        # === Helper to find leaf frames (frames with no subframes) ===
        def get_leaf_frames(frames: List[Dict], with_image_only: bool = True) -> List[Dict]:
            """Recursively find all leaf frames (frames with no subframes).

            Args:
                frames: List of frame dicts from TOON processing
                with_image_only: If True, only return frames with has_image=True
            """
            leaves = []
            for f in frames:
                subframes = f.get('subframes', [])
                if not subframes:
                    # Check if this frame needs an image
                    if with_image_only and not f.get('has_image', True):
                        continue  # Skip text-only frames
                    leaves.append(f)
                else:
                    leaves.extend(get_leaf_frames(subframes, with_image_only))
            return leaves

        # Result queue for streaming documents as they complete
        result_queue: Queue = Queue()
        SENTINEL = object()

        def process_page_streaming(page: Dict):
            """Process a page and stream leaf frame docs to result_queue as LLM completes.

            Optimization: Fetch images for first scale group and start LLM immediately,
            while fetching remaining scale groups in parallel.
            """
            page_id = page.get('id', '')
            page_name = page.get('name', 'Untitled Page')

            # Fetch full page content with retry for transient errors
            page_node = page

            @retry_on_volume_error(max_attempts=4, wait_seconds=(1, 4, 10, 30), log=log)
            def fetch_full_page() -> Optional[Dict]:
                """Fetch full page content."""
                data = self._get_file_nodes(file_key, page_id)
                if data:
                    return data.get('nodes', {}).get(page_id, {}).get('document')
                return None

            try:
                full_page_node = fetch_full_page()
                if full_page_node:
                    page_node = full_page_node
            except Exception as e:
                log.warning(f"Error fetching full page {page_id} after retries: {e}")

            # Collect frames with subframe extraction (includes image URL fetching)
            collected_frames, frame_image_urls = self._collect_frames_for_analysis(
                file_key=file_key,
                pages=[page_node],
                max_frames=max_frames_per_page,
                debug_logger=log,
                subframe_extract_threshold=subframe_extract_threshold,
                scale_to_limit_threshold=scale_to_limit_threshold,
            )

            # Process frames to TOON data
            page_data = process_page_to_toon_data(
                page_node,
                max_frames=max_frames_per_page,
                collected_frames=collected_frames,
            )
            all_frame_data = page_data.get('frames', [])

            # Find all leaf frames (no subframes), including text-only frames
            leaf_frames = get_leaf_frames(all_frame_data, with_image_only=False)

            if not leaf_frames:
                # Page has no frames - just skip silently (not an error, just empty content)
                log.debug(f"Page '{page_name}' ({page_id}): no frames, skipping")
                return

            # Use pre-fetched image URLs (only for frames with has_image=True)
            leaf_frame_ids = [f.get('id', '') for f in leaf_frames if f.get('id') and f.get('has_image', True)]
            leaf_image_urls = {fid: frame_image_urls.get(fid, '') for fid in leaf_frame_ids if fid in frame_image_urls}

            # Track which image frames are missing URLs (unexpected - should be rare)
            missing_image_frames = [fid for fid in leaf_frame_ids if fid not in frame_image_urls]
            if missing_image_frames:
                log.warning(
                    f"Page '{page_name}': {len(missing_image_frames)}/{len(leaf_frame_ids)} image frames "
                    f"missing from frame_image_urls (IDs: {missing_image_frames[:5]})"
                )

            def process_leaf_frame(frame: Dict) -> Optional[Document]:
                """Process a single leaf frame with optional LLM analysis."""
                frame_id = frame.get('id', '')
                frame_name = frame.get('name', 'Untitled')
                has_image = frame.get('has_image', True)

                frame_size = frame.get('size', {})
                image_width = int(frame_size.get('w', 0))
                image_height = int(frame_size.get('h', 0))

                # Text-only frames: skip LLM analysis, just serialize
                if not has_image:
                    frame_lines = serialize_leaf_frame(frame, explanation=None)
                    metadata = {
                        **document.metadata,
                        'id': frame_id,
                        'file_key': file_key,
                        'file_name': file_name,
                        'page_id': page_id,
                        'page_name': page_name,
                        'node_id': frame_id,
                        'node_name': frame_name,
                        'image_url': '',  # No image for text-only frames
                        'image_width': image_width,
                        'image_height': image_height,
                        'type': 'text',  # Mark as text-only element
                    }
                    return Document(
                        page_content='\n'.join(frame_lines),
                        metadata=metadata,
                    )

                # Image frames: full LLM analysis
                image_url = leaf_image_urls.get(frame_id, '')
                analysis_result = analyze_frame_with_llm(
                    frame, self.llm, serializer, image_url=image_url
                )
                explanation = analysis_result.explanation
                llm_status = analysis_result.llm_status

                frame_lines = serialize_leaf_frame(frame, explanation)
                metadata = {
                    **document.metadata,
                    'id': frame_id,
                    'file_key': file_key,
                    'file_name': file_name,
                    'page_id': page_id,
                    'page_name': page_name,
                    'node_id': frame_id,
                    'node_name': frame_name,
                    'image_url': image_url,
                    'image_width': image_width,
                    'image_height': image_height,
                    'type': 'image',
                }
                # Only include llm_issue if there was a problem
                if llm_status != 'success':
                    metadata['llm_issue'] = llm_status
                return Document(
                    page_content='\n'.join(frame_lines),
                    metadata=metadata,
                )

            # Process leaf frames in parallel, queue docs as they complete
            with ThreadPoolExecutor(max_workers=number_of_threads) as executor:
                futures = {executor.submit(process_leaf_frame, f): f for f in leaf_frames}
                for future in as_completed(futures):
                    try:
                        doc = future.result()
                        if doc:
                            result_queue.put(doc)
                    except Exception as e:
                        log.warning(f"Frame processing failed: {e}")

        # Process all pages in parallel, streaming docs as they complete
        def page_worker(page: Dict):
            try:
                process_page_streaming(page)
            except Exception as e:
                log.error(f"Page processing failed: {e}")

        # Start page processing threads
        page_threads = []
        for page in figma_pages:
            t = Thread(target=page_worker, args=(page,))
            t.start()
            page_threads.append(t)

        # Monitor thread to signal completion
        def completion_monitor():
            for t in page_threads:
                t.join()
            result_queue.put(SENTINEL)

        monitor_thread = Thread(target=completion_monitor)
        monitor_thread.start()

        # Yield documents as they arrive (streaming)
        while True:
            item = result_queue.get()
            if item is SENTINEL:
                break
            yield item

        monitor_thread.join()

    def _index_tool_params(self):
        """Return the parameters for indexing data."""
        return {
            "urls_or_file_keys": (str, Field(
                description=(
                    "Comma-separated list of Figma file URLs or raw file keys to index. "
                    "Each entry may be a full Figma URL (with optional node-id query) or a file key. "
                    "Example: 'https://www.figma.com/file/<FILE_KEY>/...?node-id=<NODE_ID>,Fp24FuzPwH0L74ODSrCnQo'."
                ))),
            'number_of_threads': (Optional[int], Field(
                description=(
                    "Optional override for the number of worker threads used when indexing Figma images. "
                    f"Valid values are from 1 to 5. Default is {DEFAULT_NUMBER_OF_THREADS}."
                ),
                default=DEFAULT_NUMBER_OF_THREADS,
                ge=1,
                le=5,
            )),
            'node_ids_include': (Optional[List[str]], Field(
                description=(
                    "List of top-level node IDs (pages) to include in the index. Values should match "
                    'Figma node-id format like ["123-56", "7651-9230"]. These include rules are applied '
                    "for each entry in urls_or_file_keys when the URL does not specify node-id and for "
                    "each raw file_key entry."
                ),
                default=None,
            )),
            'node_ids_exclude': (Optional[List[str]], Field(
                description=(
                    "List of top-level node IDs (pages) to exclude from the index when node_ids_include "
                    "is not provided. Values should match Figma node-id format. These exclude rules are "
                    "applied for each entry in urls_or_file_keys (URLs without node-id and raw fileKey "
                    "entries)."
                ),
                default=None,
            )),
            'node_types_include': (Optional[List[str]], Field(
                description=(
                    'List of node types to include in the index, e.g. ["FRAME", "COMPONENT", "RECTANGLE", '
                    '"COMPONENT_SET", "INSTANCE", "VECTOR", ...]. If provided, only these types are indexed '
                    "for each page loaded from each urls_or_file_keys entry."
                ),
                default=None,
            )),
            'node_types_exclude': (Optional[List[str]], Field(
                description=(
                    "List of node types to exclude from the index when node_types_include is not provided. "
                    "These exclude rules are applied to nodes within each page loaded from each "
                    "urls_or_file_keys entry."
                ),
                default=None,
            )),
            'scale_to_limit_threshold': (Optional[int], Field(
                description=(
                    "Maximum image dimension (in pixels) to meet LLM vision requirements. "
                    "Frames larger than this will be scaled down. A 2% safety margin is applied automatically."
                ),
                default=self.SCALE_TO_LIMIT_THRESHOLD,
            )),
            'subframe_extract_threshold': (Optional[int], Field(
                description=(
                    "Frames larger than this dimension (in pixels) will have their subframes extracted "
                    "and processed individually. Frames between this size and scale_to_limit_threshold "
                    "will be scaled down to fit the limit."
                ),
                default=self.SUBFRAME_EXTRACT_THRESHOLD,
            )),
        }

    @model_validator(mode='before')
    @classmethod
    def check_before(cls, values):
        return super().validate_toolkit(values)

    @model_validator(mode="after")
    @classmethod
    def validate_toolkit(cls, values):
        token = values.token.get_secret_value() if values.token else None
        oauth2 = values.oauth2.get_secret_value() if values.oauth2 else None
        global_regexp = values.global_regexp

        if global_regexp is None:
            logging.warning("No regex pattern provided. Skipping regex compilation.")
            cls.global_regexp = None
        else:
            try:
                re.compile(global_regexp)
                cls.global_regexp = global_regexp
            except re.error as e:
                msg = f"Failed to compile regex pattern: {str(e)}"
                logging.error(msg)
                raise ToolException(msg)

        try:
            if token:
                cls._client = EliteAFigmaPy(token=token, oauth2=False)
                logging.info("Authenticated with Figma token")
            elif oauth2:
                cls._client = EliteAFigmaPy(token=oauth2, oauth2=True)
                logging.info("Authenticated with OAuth2 token")
            else:
                raise ToolException("You have to define Figma token.")
            logging.info("Successfully authenticated to Figma.")
        except Exception as e:
            msg = f"Failed to authenticate with Figma: {str(e)}"
            logging.error(msg)
            raise ToolException(msg)

        return values

    @staticmethod
    def process_output(func):
        def simplified_dict(obj, depth=1, max_depth=3, seen=None):
            """Convert object to a dictionary, limit recursion depth and manage cyclic references."""
            if seen is None:
                seen = set()

            if id(obj) in seen:
                pass
            seen.add(id(obj))

            if depth > max_depth:
                return str(obj)

            if isinstance(obj, list):
                return [
                    simplified_dict(item, depth + 1, max_depth, seen) for item in obj
                ]
            elif hasattr(obj, "__dict__"):
                return {
                    key: simplified_dict(getattr(obj, key), depth + 1, max_depth, seen)
                    for key in obj.__dict__
                    if not key.startswith("__") and not callable(getattr(obj, key))
                }
            elif isinstance(obj, dict):
                return {
                    k: simplified_dict(v, depth + 1, max_depth, seen)
                    for k, v in obj.items()
                }
            return obj

        def process_fields(obj, fields_retain=None, fields_remove=None, depth_start=1, depth_end=2, depth=1):
            """
            Reduces a nested dictionary or list by retaining or removing specified fields at certain depths.

            - At each level, starting from `depth_start`, only fields in `fields_retain` are kept; fields in `fields_remove` are excluded unless also retained.
            - Recursion stops at `depth_end`, ignoring all fields at or beyond this depth.
            - Tracks which fields were retained and removed during processing.
            - Returns a JSON string of the reduced object, plus lists of retained and removed fields.
            """
            fields_retain = set(fields_retain or [])
            fields_remove = set(fields_remove or []) - fields_retain # fields in remove have lower priority than in retain

            retained = set()
            removed = set()

            def _process(o, d):
                if depth_end is not None and d >= depth_end:
                    return None  # Ignore keys at or beyond cut_depth
                if isinstance(o, dict):
                    result = {}
                    for k, v in o.items():
                        if k in fields_remove:
                            removed.add(k)
                            continue
                        if d >= depth_start:
                            if k in fields_retain:
                                retained.add(k)
                                result[k] = _process(v, d + 1)  # process recursively
                            else:
                                # else: skip keys not in retain/default/to_process
                                removed.add(k) # remember skipped keys
                        else:
                            # retained.add(k) # remember retained keys
                            result[k] = _process(v, d + 1)
                    return result
                elif isinstance(o, list):
                    return [_process(item, d + 1) for item in o]
                else:
                    return o

            new_obj = _process(obj, depth)
            return {
                "result": json.dumps(new_obj),
                "retained_fields": list(retained),
                "removed_fields": list(removed)
            }

        def fix_trailing_commas(json_string):
            json_string = re.sub(r",\s*,+", ",", json_string)
            json_string = re.sub(r",\s*([\]}])", r"\1", json_string)
            json_string = re.sub(r"([\[{])\s*,", r"\1", json_string)
            return json_string

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            extra_params = kwargs.pop("extra_params", {})
            limit = extra_params.get("limit", self.global_limit)
            regexp = extra_params.get("regexp", self.global_regexp)
            fields_retain = extra_params.get("fields_retain", self.global_fields_retain)
            fields_remove = extra_params.get("fields_remove", self.global_fields_remove)
            depth_start = extra_params.get("depth_start", self.global_depth_start)
            depth_end = extra_params.get("depth_end", self.global_depth_end)
            try:
                limit = int(limit)
                result = func(self, *args, **kwargs)
                if result and "__dict__" in dir(result):
                    result = result.__dict__
                elif not result:
                    return ToolException(
                        "Response result is empty. Check your input parameters or credentials"
                    )
                if isinstance(result, (dict, list)):
                    raw_result = result
                    processed_result = simplified_dict(raw_result)
                    raw_str_result = json.dumps(processed_result)
                    str_result = raw_str_result
                    if regexp:
                        regexp = re.compile(regexp)
                        str_result = re.sub(regexp, "", raw_str_result)
                        str_result = fix_trailing_commas(str_result)
                    if len(str_result) > limit:
                        reduced = process_fields(raw_result, fields_retain=fields_retain, fields_remove=fields_remove, depth_start=depth_start, depth_end=depth_end)
                        note = (f"Size of the output exceeds limit {limit}. Data reducing has been applied. "
                                f"Starting from the depth_start = {depth_start} the following object fields were removed: {reduced['removed_fields']}. "
                                f"The following fields were retained: {reduced['retained_fields']}. "
                                f"Starting from depth_end = {depth_end} all fields were ignored. "
                                f"You can adjust fields_retain, fields_remove, depth_start, depth_end, limit and regexp parameters to get more precise output")
                        return f"## NOTE:\n{note}.\n## Result: {reduced['result']}"[:limit]
                    return str_result
                else:
                    result = json.dumps(result)
                if regexp:
                    regexp = re.compile(regexp)
                    result = re.sub(regexp, "", result)
                    result = fix_trailing_commas(result)
                result = result[:limit]
                return result
            except Exception as e:
                msg = f"Error in '{func.__name__}': {str(e)}"
                logging.error(msg)
                return ToolException(msg)

        return wrapper

    @process_output
    def get_file_nodes(self, file_key: str, ids: str, **kwargs):
        """Reads a specified file nodes by field key from Figma."""
        return self._client.api_request(
            f"files/{file_key}/nodes?ids={str(ids)}", method="get"
        )

    def _get_file_nodes(self, file_key: str, ids: str, **kwargs):
        """Reads a specified file nodes by field key from Figma."""
        return self._client.api_request(
            f"files/{file_key}/nodes?ids={str(ids)}", method="get"
        )

    @process_output
    def get_file(
        self,
        file_key: str,
        geometry: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ):
        """Reads a specified file by field key from Figma."""
        return self._client.get_file(file_key, geometry, version)

    @process_output
    def get_file_versions(self, file_key: str, **kwargs):
        """Retrieves the version history of a specified file from Figma."""
        return self._client.get_file_versions(file_key)

    @process_output
    def get_file_comments(self, file_key: str, **kwargs):
        """Retrieves comments on a specified file from Figma."""
        return self._client.get_comments(file_key)

    @process_output
    def post_file_comment(
        self, file_key: str, message: str, client_meta: Optional[dict] = None, **kwargs
    ):
        """Posts a comment to a specific file in Figma."""
        return self._client.post_comment(file_key, message, client_meta)

    @process_output
    def get_file_images(
        self,
        file_key: str,
        ids: str,
        scale: Optional[str] = None,
        format: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ):
        """Fetches URLs for server-rendered images from a Figma file based on node IDs."""
        ids_list = ids.split(",")
        return self._client.get_file_images(
            file_key=file_key, ids=ids_list, scale=scale, format=format, version=version
        )

    @process_output
    def get_team_projects(self, team_id: str, **kwargs):
        """Retrieves all projects for a specified team ID from Figma."""
        return self._client.get_team_projects(team_id)

    @process_output
    def get_project_files(self, project_id: str, **kwargs):
        """Retrieves all files for a specified project ID from Figma."""
        return self._client.get_project_files(project_id)

    # -------------------------------------------------------------------------
    # TOON Format Tools (Token-Optimized Output)
    # -------------------------------------------------------------------------

    def get_file_structure_toon(
        self,
        url: Optional[str] = None,
        file_key: Optional[str] = None,
        include_pages: Optional[str] = None,
        exclude_pages: Optional[str] = None,
        max_frames: int = 50,
        **kwargs,
    ) -> str:
        """
        Get file structure in TOON format - optimized for LLM token consumption.

        Returns a compact, human-readable format with:
        - Page and frame hierarchy
        - Text content categorized (headings, labels, buttons, body, errors)
        - Component usage
        - Inferred screen types and states
        - Flow analysis (sequences, variants, CTA destinations)

        TOON format uses ~70% fewer tokens than JSON for the same data.

        Use this tool when you need to:
        - Understand overall file structure quickly
        - Generate user journey documentation
        - Analyze screen flows and navigation
        - Identify UI patterns and components
        """
        self._log_tool_event("Getting file structure in TOON format")

        # Parse URL or use file_key
        if url:
            file_key, node_ids_from_url = self._parse_figma_url(url)
            if node_ids_from_url and not include_pages:
                include_pages = ','.join(node_ids_from_url)

        if not file_key:
            raise ToolException("Either url or file_key must be provided")

        # Parse include/exclude pages
        include_ids = [p.strip() for p in include_pages.split(',')] if include_pages else None
        exclude_ids = [p.strip() for p in exclude_pages.split(',')] if exclude_pages else None

        # Get file structure (shallow fetch - only top-level pages, not full content)
        # This avoids "Request too large" errors for big files
        self._log_tool_event(f"Fetching file structure for {file_key}")
        file_data = self._client.get_file(file_key, geometry='depth=1')

        if not file_data:
            raise ToolException(f"Failed to retrieve file {file_key}")

        # Process pages
        pages_data = []
        all_pages = file_data.document.get('children', [])

        for page_node in all_pages:
            page_id = page_node.get('id', '')

            # Apply page filters
            if include_ids and page_id not in include_ids and page_id.replace(':', '-') not in include_ids:
                continue
            if exclude_ids and not include_ids:
                if page_id in exclude_ids or page_id.replace(':', '-') in exclude_ids:
                    continue

            self._log_tool_event(f"Processing page: {page_node.get('name', 'Untitled')}")

            # Fetch full page content individually (avoids large single request)
            try:
                page_full = self._get_file_nodes(file_key, page_id)
                if page_full:
                    page_content = page_full.get('nodes', {}).get(page_id, {}).get('document', page_node)
                else:
                    page_content = page_node
            except Exception as e:
                self._log_tool_event(f"Warning: Could not fetch full page content for {page_id}: {e}")
                page_content = page_node

            page_data = process_page_to_toon_data(page_content)

            # Limit frames per page
            if len(page_data['frames']) > max_frames:
                page_data['frames'] = page_data['frames'][:max_frames]
                page_data['truncated'] = True

            pages_data.append(page_data)

        # Build file data structure
        toon_data = {
            'name': file_data.name,
            'key': file_key,
            'pages': pages_data,
        }

        # Serialize to TOON format
        serializer = TOONSerializer()
        result = serializer.serialize_file(toon_data)

        self._log_tool_event("File structure extracted in TOON format")
        return result

    def get_page_flows_toon(
        self,
        url: Optional[str] = None,
        file_key: Optional[str] = None,
        page_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Analyze a single page for user flows in TOON format.

        Returns detailed flow analysis:
        - Frame sequence detection (from naming: 01_, Step 1, etc.)
        - Screen variant grouping (Login, Login_Error, Login_Loading)
        - CTA/button destination mapping
        - Spatial ordering hints

        Use this for in-depth flow analysis of a specific page.
        Requires a PAGE ID (not a frame ID). Use get_file_structure_toon to find page IDs.
        """
        self._log_tool_event("Analyzing page flows in TOON format")

        # Parse URL
        if url:
            file_key, node_ids_from_url = self._parse_figma_url(url)
            if node_ids_from_url:
                page_id = node_ids_from_url[0]

        if not file_key:
            raise ToolException("Either url or file_key must be provided")
        if not page_id:
            raise ToolException("page_id must be provided (or include node-id in URL)")

        # Fetch node content
        self._log_tool_event(f"Fetching node {page_id} from file {file_key}")
        node_full = self._get_file_nodes(file_key, page_id)

        if not node_full:
            raise ToolException(f"Failed to retrieve node {page_id}")

        node_content = node_full.get('nodes', {}).get(page_id, {}).get('document', {})
        if not node_content:
            raise ToolException(f"Node {page_id} has no content")

        # Check if this is a page (CANVAS) or a frame
        node_type = node_content.get('type', '').upper()
        if node_type != 'CANVAS':
            # This is a frame, not a page - provide helpful error
            raise ToolException(
                f"Node {page_id} is a {node_type}, not a PAGE. "
                f"This tool requires a page ID. Use get_file_structure_toon first to find page IDs "
                f"(look for PAGE: ... #<page_id>)"
            )

        page_content = node_content

        # Process page
        page_data = process_page_to_toon_data(page_content)
        frames = page_data.get('frames', [])

        # Build detailed flow analysis
        lines = []
        lines.append(f"PAGE: {page_data.get('name', 'Untitled')} [id:{page_id}]")
        lines.append(f"  frames: {len(frames)}")
        lines.append("")

        # Sequence analysis
        sequences = detect_sequences(frames)
        if sequences:
            lines.append("SEQUENCES (by naming):")
            for seq in sequences:
                lines.append(f"  {' > '.join(seq)}")
            lines.append("")

        # Variant analysis
        variants = group_variants(frames)
        if variants:
            lines.append("VARIANTS (grouped screens):")
            for base, variant_list in variants.items():
                lines.append(f"  {base}:")
                for v in variant_list:
                    v_name = v.get('name', '')
                    v_id = v.get('id', '')
                    state = next((f.get('state', 'default') for f in frames if f.get('name') == v_name), 'default')
                    lines.append(f"    - {v_name} [{state}] #{v_id}")
            lines.append("")

        # CTA mapping
        lines.append("CTA DESTINATIONS:")
        cta_map = {}
        for frame in frames:
            frame_name = frame.get('name', '')
            for btn in frame.get('buttons', []):
                dest = infer_cta_destination(btn)
                if dest not in cta_map:
                    cta_map[dest] = []
                cta_map[dest].append(f'"{btn}" in {frame_name}')

        for dest, ctas in cta_map.items():
            lines.append(f"  > {dest}:")
            for cta in ctas[:5]:  # Limit per destination
                lines.append(f"      {cta}")
        lines.append("")

        # Spatial ordering
        lines.append("SPATIAL ORDER (canvas position):")
        sorted_frames = sorted(frames, key=lambda f: (f['position']['y'], f['position']['x']))
        for i, frame in enumerate(sorted_frames[:20], 1):
            pos = frame.get('position', {})
            lines.append(f"  {i}. {frame.get('name', '')} [{int(pos.get('x', 0))},{int(pos.get('y', 0))}]")

        # Frame details
        lines.append("")
        lines.append("FRAME DETAILS:")

        serializer = TOONSerializer()
        for frame in frames[:30]:  # Limit frames
            frame_lines = serializer.serialize_frame(frame, level=1)
            lines.extend(frame_lines)

        self._log_tool_event("Page flow analysis complete")
        return '\n'.join(lines)

    def _get_frame_detail_toon_internal(
        self,
        file_key: str,
        frame_ids: str,
        **kwargs,
    ) -> str:
        """Get detailed information for specific frames in TOON format."""
        self._log_tool_event("Getting frame details in TOON format")

        ids_list = [fid.strip() for fid in frame_ids.split(',') if fid.strip()]
        if not ids_list:
            raise ToolException("frame_ids must contain at least one frame ID")

        # Fetch frames
        self._log_tool_event(f"Fetching {len(ids_list)} frames from file {file_key}")
        nodes_data = self._get_file_nodes(file_key, ','.join(ids_list))

        if not nodes_data:
            raise ToolException(f"Failed to retrieve frames from file {file_key}")

        # Process each frame
        lines = [f"FRAMES [{len(ids_list)} requested]", ""]

        serializer = TOONSerializer()

        for frame_id in ids_list:
            node_data = nodes_data.get('nodes', {}).get(frame_id, {})
            frame_node = node_data.get('document', {})

            if not frame_node:
                lines.append(f"FRAME: {frame_id} [NOT FOUND]")
                lines.append("")
                continue

            frame_data = process_frame_to_toon_data(frame_node)
            frame_lines = serializer.serialize_frame(frame_data, level=0)
            lines.extend(frame_lines)

            # Add extra details for individual frames
            lines.append(f"  ID: {frame_id}")

            # Component breakdown
            components = frame_data.get('components', [])
            if components:
                # Count component usage
                from collections import Counter
                comp_counts = Counter(components)
                lines.append(f"  COMPONENT_COUNTS:")
                for comp, count in comp_counts.most_common(10):
                    lines.append(f"    {comp}: {count}")

            lines.append("")

        self._log_tool_event("Frame details extracted")
        return '\n'.join(lines)

    def analyze_file(
        self,
        url: Optional[str] = None,
        file_key: Optional[str] = None,
        node_id: Optional[str] = None,
        include_pages: Optional[str] = None,
        exclude_pages: Optional[str] = None,
        max_frames: int = 50,
        **kwargs,
    ) -> str:
        """
        Comprehensive Figma file analyzer with LLM-powered insights.

        Returns detailed analysis including:
        - File/page/frame structure with all content (text, buttons, components)
        - LLM-powered screen explanations with visual insights (using frame images)
        - LLM-powered user flow analysis identifying key user journeys
        - Design insights (patterns, gaps, recommendations)

        Drill-Down:
          - No node_id: Analyzes entire file (respecting include/exclude pages)
          - node_id=page_id: Focuses on specific page
          - node_id=frame_id: Returns detailed frame analysis
        """
        try:
            return self._analyze_file_internal(
                url=url,
                file_key=file_key,
                node_id=node_id,
                include_pages=include_pages,
                exclude_pages=exclude_pages,
                max_frames=max_frames,
                **kwargs,
            )
        except ToolException as e:
            raise ToolException(_handle_figma_error(e))

    def _analyze_file_internal(
        self,
        url: Optional[str] = None,
        file_key: Optional[str] = None,
        node_id: Optional[str] = None,
        include_pages: Optional[str] = None,
        exclude_pages: Optional[str] = None,
        max_frames: int = 50,
        **kwargs,
    ) -> str:
        """Internal implementation of analyze_file without error handling wrapper."""
        # Always use maximum detail level and LLM analysis
        detail_level = 3
        llm_analysis = 'detailed' if self.llm else 'none'
        self._log_tool_event(f"Getting file in TOON format (detail_level={detail_level}, llm_analysis={llm_analysis})")

        # Parse URL if provided
        if url:
            file_key, node_ids_from_url = self._parse_figma_url(url)
            if node_ids_from_url and not node_id:
                node_id = node_ids_from_url[0]

        if not file_key:
            raise ToolException("Either url or file_key must be provided")

        # Convert node_id from URL format (hyphen) to API format (colon)
        if node_id:
            node_id = node_id.replace('-', ':')

        # Check if node_id is a frame or page (for drill-down)
        node_id_is_page = False
        if node_id:
            try:
                nodes_data = self._get_file_nodes(file_key, node_id)
                if nodes_data:
                    node_info = nodes_data.get('nodes', {}).get(node_id, {})
                    node_doc = node_info.get('document', {})
                    node_type = node_doc.get('type', '').upper()

                    if node_type == 'FRAME':
                        # It's a frame - use frame detail tool (internal to avoid double-wrapping)
                        return self._get_frame_detail_toon_internal(file_key=file_key, frame_ids=node_id)
                    elif node_type == 'CANVAS':
                        # It's a page - we'll filter to this page
                        node_id_is_page = True
            except Exception:
                pass  # Fall through to page/file analysis

        # Get file structure
        file_data = self._client.get_file(file_key, geometry='depth=1')
        if not file_data:
            raise ToolException(f"Failed to retrieve file {file_key}")

        # Determine which pages to process
        # Check if document exists and has the expected structure
        if not hasattr(file_data, 'document') or file_data.document is None:
            self._log_tool_event(f"Warning: file_data has no document attribute. Type: {type(file_data)}")
            all_pages = []
        else:
            all_pages = file_data.document.get('children', [])
        self._log_tool_event(f"File has {len(all_pages)} pages, node_id={node_id}, node_id_is_page={node_id_is_page}")

        # Only filter by node_id if it's confirmed to be a page ID
        if node_id and node_id_is_page:
            include_pages = node_id

        include_ids = [p.strip() for p in include_pages.split(',')] if include_pages else None
        exclude_ids = [p.strip() for p in exclude_pages.split(',')] if exclude_pages else None

        pages_to_process = []
        for page_node in all_pages:
            page_id = page_node.get('id', '')
            if include_ids and page_id not in include_ids:
                continue
            if exclude_ids and page_id in exclude_ids:
                continue
            pages_to_process.append(page_node)

        # Build output based on detail level
        lines = [f"FILE: {file_data.name} [key:{file_key}]"]
        serializer = TOONSerializer()

        all_frames_for_flows = []  # Collect frames for flow analysis at Level 2+

        if not pages_to_process:
            if not all_pages:
                lines.append("  [No pages found in file - file may be empty or access restricted]")
            else:
                lines.append(f"  [All {len(all_pages)} pages filtered out by include/exclude settings]")
            self._log_tool_event(f"No pages to process. all_pages={len(all_pages)}, include_ids={include_ids}, exclude_ids={exclude_ids}")

        self._log_tool_event(f"Processing {len(pages_to_process)} pages at detail_level={detail_level}")

        # Track collected data for LLM analysis (avoid refetching)
        all_page_data_for_llm = []
        all_frame_image_urls = {}  # {frame_id: image_url}

        for page_node in pages_to_process:
            page_id = page_node.get('id', '')
            page_name = page_node.get('name', 'Untitled')

            if detail_level == 1:
                # Level 1: Structure only - just hierarchy with IDs
                lines.append(f"  PAGE: {page_name} #{page_id}")
                frames = page_node.get('children', [])[:max_frames]
                for frame in frames:
                    if frame.get('type', '').upper() == 'FRAME':
                        frame_id = frame.get('id', '')
                        frame_name = frame.get('name', 'Untitled')
                        lines.append(f"    FRAME: {frame_name} #{frame_id}")
            else:
                # Level 2+: Need full page content - fetch via nodes API
                page_fetch_error = None
                try:
                    nodes_data = self._get_file_nodes(file_key, page_id)
                    if nodes_data:
                        full_page_node = nodes_data.get('nodes', {}).get(page_id, {}).get('document', {})
                        if full_page_node:
                            page_node = full_page_node
                except ToolException as e:
                    page_fetch_error = _handle_figma_error(e)
                    self._log_tool_event(f"Error fetching page {page_id}: {page_fetch_error}")
                except Exception as e:
                    page_fetch_error = str(e)
                    self._log_tool_event(f"Error fetching page {page_id}: {e}")

                # Use frame collection with subframe extraction for better coverage
                collected_frames, frame_image_urls = self._collect_frames_for_analysis(
                    file_key=file_key,
                    pages=[page_node],
                    max_frames=max_frames,
                    debug_logger=None,
                )
                self._log_tool_event(f"Collected {len(collected_frames)} frames (with subframe extraction)")

                # Merge image URLs for LLM analysis
                all_frame_image_urls.update(frame_image_urls)

                # Process with pre-collected frames that include subframes
                page_data = process_page_to_toon_data(
                    page_node,
                    max_frames=max_frames,
                    collected_frames=collected_frames,
                )
                frames = page_data.get('frames', [])

                # Store for LLM analysis
                all_page_data_for_llm.append(page_data)

                # If we had an error and got no frames, show the error
                if page_fetch_error and not frames:
                    lines.append(f"  PAGE: {page_name} #{page_id}")
                    lines.append(f"    [Error: {page_fetch_error}]")
                    continue

                if detail_level == 2:
                    # Level 2: Standard - content via serialize_page
                    page_lines = serializer.serialize_page(page_data, level=0)
                    lines.extend(page_lines)
                else:
                    # Level 3: Detailed - content + per-frame component counts
                    lines.append(f"PAGE: {page_data.get('name', 'Untitled')} #{page_data.get('id', '')}")
                    for frame_data in frames:
                        frame_lines = serializer.serialize_frame(frame_data, level=1)
                        lines.extend(frame_lines)

                        # Add detailed component counts
                        components = frame_data.get('components', [])
                        if components:
                            from collections import Counter
                            comp_counts = Counter(components)
                            lines.append(f"    COMPONENT_COUNTS:")
                            for comp, count in comp_counts.most_common(10):
                                lines.append(f"      {comp}: {count}")

                # Collect frames for flow analysis
                all_frames_for_flows.extend(frames)

            lines.append("")

        # Level 2+: Add global flow analysis at the end
        if detail_level >= 2 and all_frames_for_flows:
            flow_lines = serializer.serialize_flows(all_frames_for_flows, level=0)
            if flow_lines:
                lines.append("FLOWS:")
                lines.extend(flow_lines)

        toon_output = '\n'.join(lines)

        # Add LLM analysis if requested
        if llm_analysis and llm_analysis != 'none' and self.llm:
            self._log_tool_event(f"Running LLM analysis (level={llm_analysis})")
            try:
                # Build file_data structure for LLM analysis (reuse collected page data)
                file_data_for_llm = {
                    'name': file_data.name,
                    'key': file_key,
                    'pages': all_page_data_for_llm,  # Already collected with subframes
                }

                # Helper to recursively collect all frame IDs including subframes
                def collect_all_frame_ids(frames: List[Dict]) -> List[str]:
                    """Recursively collect frame IDs from frames and their subframes."""
                    ids = []
                    for frame in frames:
                        frame_id = frame.get('id')
                        if frame_id:
                            ids.append(frame_id)
                        # Recursively collect subframe IDs
                        subframes = frame.get('subframes', [])
                        if subframes:
                            ids.extend(collect_all_frame_ids(subframes))
                    return ids

                # Collect all frame IDs (including subframes) for vision analysis
                all_frame_ids = []
                for page_data in all_page_data_for_llm:
                    all_frame_ids.extend(collect_all_frame_ids(page_data.get('frames', [])))

                # Use pre-collected image URLs from frame collection
                # These already have adaptive scaling applied
                frame_images = all_frame_image_urls
                frames_to_analyze = min(max_frames, len(all_frame_ids))
                self._log_tool_event(f"Using {len(frame_images)} pre-collected image URLs for {frames_to_analyze} frames")

                # Create status callback for progress updates
                def _status_callback(msg: str):
                    self._log_tool_event(msg)

                # Import here to avoid circular imports
                from .toon_tools import enrich_toon_with_llm_analysis

                # Check if design insights should be included (default True)
                include_design_insights = kwargs.get('include_design_insights', True)

                # Get parallel workers from toolkit config (or default)
                parallel_workers = getattr(self, "number_of_threads", DEFAULT_NUMBER_OF_THREADS)
                if parallel_workers is None or not isinstance(parallel_workers, int):
                    parallel_workers = DEFAULT_NUMBER_OF_THREADS
                parallel_workers = max(1, min(parallel_workers, 5))

                self._log_tool_event(f"Starting LLM analysis of {frames_to_analyze} frames with {parallel_workers} parallel workers...")
                toon_output = enrich_toon_with_llm_analysis(
                    toon_output=toon_output,
                    file_data=file_data_for_llm,
                    llm=self.llm,
                    analysis_level=llm_analysis,
                    frame_images=frame_images,
                    status_callback=_status_callback,
                    include_design_insights=include_design_insights,
                    parallel_workers=parallel_workers,
                    max_frames_to_analyze=frames_to_analyze,
                )
                self._log_tool_event("LLM analysis complete")
            except Exception as e:
                self._log_tool_event(f"LLM analysis failed: {e}")
                # Return TOON output without LLM analysis on error
                toon_output += f"\n\n[LLM analysis failed: {e}]"

        self._log_tool_event(f"File analysis complete (detail_level={detail_level})")
        return toon_output

    # -------------------------------------------------------------------------
    # Design Token Extraction
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_node_styles_recursive(
        node: dict,
        path: str = "",
        results: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Recursively walk a Figma node tree and collect raw style data.

        Collects fills, strokes, effects, typography (style), and local style
        references (styles) from every node in the tree.

        Args:
            node:    A Figma node dict (document field from /files/{key}/nodes).
            path:    Slash-separated path built during recursion (for traceability).
            results: Accumulator list; a new list is created when None.

        Returns:
            List of dicts, one per node that carries at least one style property:
            {
                "path":       "Parent/Child/Name",
                "type":       "FRAME" | "TEXT" | "VECTOR" | ...,
                "fills":      [ {type, color, opacity, ...} ],   # IMAGE fills excluded
                "strokes":    [ {type, color, ...} ],
                "effects":    [ {type, visible, color, offset, radius, ...} ],
                "typography": { fontFamily, fontSize, fontWeight, lineHeightPx,
                                letterSpacing, textAlignHorizontal },
                "style_refs": { fill: "S:id", text: "S:id", ... },
            }
        """
        if results is None:
            results = []

        name = node.get("name", "")
        ntype = node.get("type", "")
        current_path = f"{path}/{name}" if path else name

        # Fills — exclude IMAGE type and alpha=0 (invisible)
        raw_fills = node.get("fills", [])
        fills = [
            f for f in raw_fills
            if f.get("type") != "IMAGE"
            and (f.get("color", {}).get("a", 1.0) > 0 or f.get("type") != "SOLID")
        ]

        # Strokes
        strokes = node.get("strokes", [])

        # Effects — exclude invisible
        effects = [e for e in node.get("effects", []) if e.get("visible", True)]

        # Typography — only when fontFamily is present
        style = node.get("style", {})
        typography: dict = {}
        if style.get("fontFamily"):
            typography = {
                k: style[k]
                for k in (
                    "fontFamily", "fontSize", "fontWeight",
                    "lineHeightPx", "letterSpacing", "textAlignHorizontal",
                )
                if k in style
            }

        # Local style node ID references
        style_refs = node.get("styles", {})

        has_style = any([fills, strokes, effects, typography, style_refs])
        if has_style:
            results.append({
                "path":       current_path,
                "type":       ntype,
                "fills":      fills,
                "strokes":    strokes,
                "effects":    effects,
                "typography": typography,
                "style_refs": style_refs,
            })

        for child in node.get("children", []):
            FigmaApiWrapper._extract_node_styles_recursive(child, current_path, results)

        return results

    @staticmethod
    def _dedup_colors(entries: List[dict]) -> List[dict]:
        """Deduplicate fill colors across all entries.

        Dedup key is (hex, rounded_alpha) so the same hex at different opacities
        produces separate tokens — e.g. #1C756A at alpha=1.0 and #1C756A at
        alpha=0.3 are distinct design tokens (overlay vs. solid usage).

        Gradient fills are expanded into their individual gradient stops.
        Invisible fills (alpha=0) are always excluded.
        """
        seen: dict = {}  # (hex, alpha_key) -> token dict (keeps first occurrence)
        for entry in entries:
            for fill in entry.get("fills", []):
                fill_type = fill.get("type", "")
                if fill_type == "SOLID":
                    c = fill.get("color", {})
                    alpha = c.get("a", 1.0)
                    if alpha == 0.0:
                        continue
                    r, g, b = round(c.get("r", 0) * 255), round(c.get("g", 0) * 255), round(c.get("b", 0) * 255)
                    hex_color = f"#{r:02X}{g:02X}{b:02X}"
                    # Include alpha in key: same hex at different opacities = different tokens
                    key = (hex_color, round(alpha, 2))
                    if key not in seen:
                        seen[key] = {
                            "hex":         hex_color,
                            "alpha":       round(alpha, 4),
                            "opacity":     round(fill.get("opacity", 1.0), 4),
                            "source_path": entry["path"],
                        }
                elif fill_type in ("GRADIENT_LINEAR", "GRADIENT_RADIAL",
                                   "GRADIENT_ANGULAR", "GRADIENT_DIAMOND"):
                    for stop in fill.get("gradientStops", []):
                        c = stop.get("color", {})
                        alpha = c.get("a", 1.0)
                        if alpha == 0.0:
                            continue
                        r, g, b = round(c.get("r", 0) * 255), round(c.get("g", 0) * 255), round(c.get("b", 0) * 255)
                        hex_color = f"#{r:02X}{g:02X}{b:02X}"
                        key = (hex_color, round(alpha, 2))
                        if key not in seen:
                            seen[key] = {
                                "hex":         hex_color,
                                "alpha":       round(alpha, 4),
                                "opacity":     1.0,
                                "source_path": f"{entry['path']} [{fill_type}]",
                            }
        return list(seen.values())

    @staticmethod
    def _dedup_strokes(entries: List[dict]) -> List[dict]:
        """Deduplicate stroke colors across all entries.

        Handles both SOLID and gradient stroke types. Dedup key includes alpha
        so the same hex at different opacities produces separate tokens.
        """
        seen: dict = {}
        for entry in entries:
            for stroke in entry.get("strokes", []):
                stroke_type = stroke.get("type", "")
                if stroke_type == "SOLID":
                    c = stroke.get("color", {})
                    alpha = c.get("a", 1.0)
                    if alpha == 0.0:
                        continue
                    r, g, b = round(c.get("r", 0) * 255), round(c.get("g", 0) * 255), round(c.get("b", 0) * 255)
                    hex_color = f"#{r:02X}{g:02X}{b:02X}"
                    key = (hex_color, round(alpha, 2))
                    if key not in seen:
                        seen[key] = {
                            "hex":         hex_color,
                            "alpha":       round(alpha, 4),
                            "source_path": entry["path"],
                        }
                elif stroke_type in ("GRADIENT_LINEAR", "GRADIENT_RADIAL",
                                     "GRADIENT_ANGULAR", "GRADIENT_DIAMOND"):
                    for stop in stroke.get("gradientStops", []):
                        c = stop.get("color", {})
                        alpha = c.get("a", 1.0)
                        if alpha == 0.0:
                            continue
                        r, g, b = round(c.get("r", 0) * 255), round(c.get("g", 0) * 255), round(c.get("b", 0) * 255)
                        hex_color = f"#{r:02X}{g:02X}{b:02X}"
                        key = (hex_color, round(alpha, 2))
                        if key not in seen:
                            seen[key] = {
                                "hex":         hex_color,
                                "alpha":       round(alpha, 4),
                                "source_path": f"{entry['path']} [{stroke_type}]",
                            }
        return list(seen.values())

    @staticmethod
    def _dedup_typography(entries: List[dict]) -> List[dict]:
        """Deduplicate typography combinations (family + size + weight) across entries."""
        seen: dict = {}
        for entry in entries:
            t = entry.get("typography", {})
            if not t.get("fontFamily"):
                continue
            key = (t.get("fontFamily"), t.get("fontSize"), t.get("fontWeight"))
            if key not in seen:
                seen[key] = {**t, "source_path": entry["path"]}
        return list(seen.values())

    @staticmethod
    def _dedup_effects(entries: List[dict]) -> List[dict]:
        """Deduplicate effects by their full structural signature across entries.

        Signature includes: type, offsetX, offsetY, radius, spread, and the
        full color RGBA (rounded to 2dp). This prevents collapsing shadows that
        differ only in color RGB or X-offset.
        """
        seen: dict = {}
        for entry in entries:
            for eff in entry.get("effects", []):
                color = eff.get("color") or {}
                offset = eff.get("offset") or {}
                sig = (
                    eff.get("type"),
                    round(offset.get("x", 0), 2),
                    round(offset.get("y", 0), 2),
                    eff.get("radius"),
                    eff.get("spread"),
                    round(color.get("r", 0), 2),
                    round(color.get("g", 0), 2),
                    round(color.get("b", 0), 2),
                    round(color.get("a", 0), 2),
                )
                if sig not in seen:
                    seen[sig] = {**eff, "source_path": entry["path"]}
        return list(seen.values())

    def extract_design_tokens(
        self,
        file_key: str,
        node_id: str,
        depth: int = 4,
        **kwargs,
    ) -> str:
        """Extract deduplicated design tokens (colors, typography, effects, strokes)
        from a Figma node and its entire subtree.

        Fetches the node tree at the requested depth and recursively collects all
        style properties — fills, strokes, effects, and typography — then
        deduplicates them across the subtree.

        Returns a JSON string with the following structure:
        {
          "node_id":   "169:14446",
          "node_name": "Formula Section",
          "node_type": "SECTION",
          "colors":    [{"hex": "#1C756A", "alpha": 1.0, "opacity": 1.0, "source_path": "..."}],
          "strokes":   [{"hex": "#000000", "alpha": 1.0, "source_path": "..."}],
          "typography":[{"fontFamily": "Noto Sans", "fontSize": 14.0, "fontWeight": 400, ...}],
          "effects":   [{"type": "DROP_SHADOW", "radius": 6, ..., "source_path": "..."}],
          "summary":   {"total_style_entries": 177, "unique_colors": 11,
                        "unique_fonts": 5, "unique_effects": 5, "unique_strokes": 4}
        }

        Args:
            file_key: Figma file key (e.g. "Fp24FuzPwH0L74ODSrCnQo").
            node_id:  Node ID of a FRAME, COMPONENT, COMPONENT_SET, or SECTION.
                      Hyphens are auto-converted to colons.
                      Do NOT pass a PAGE (CANVAS) id — it will return an error.
            depth:    Tree fetch depth (1–8). Default 4 covers most components.
                      Use 6–8 for deeply nested COMPONENT_SET nodes.

        Returns:
            JSON string with deduplicated design tokens.

        Raises:
            ToolException: If the node is null (likely a PAGE id), the file is
                           not found, or the API request fails.
        """
        # Normalise node id format (URL uses hyphens, API uses colons)
        node_id = node_id.replace("-", ":")

        try:
            raw = self._client.api_request(
                f"files/{file_key}/nodes?ids={node_id}&depth={depth}",
                method="get",
            )
        except ToolException as e:
            raise ToolException(
                f"Failed to fetch node '{node_id}' from file '{file_key}': {e}"
            ) from e

        nodes = (raw or {}).get("nodes") or {}
        node_wrap = nodes.get(node_id)

        if node_wrap is None:
            raise ToolException(
                f"Node '{node_id}' returned null. "
                "This usually means the ID belongs to a PAGE (CANVAS) node. "
                "Please provide a FRAME, COMPONENT, COMPONENT_SET, or SECTION node ID. "
                "Use get_file_structure_toon to discover valid child frame IDs."
            )

        node_doc = node_wrap.get("document") or {}
        node_name = node_doc.get("name", "")
        node_type = node_doc.get("type", "")

        entries = self._extract_node_styles_recursive(node_doc)

        colors     = self._dedup_colors(entries)
        strokes    = self._dedup_strokes(entries)
        typography = self._dedup_typography(entries)
        effects    = self._dedup_effects(entries)

        result = {
            "node_id":    node_id,
            "node_name":  node_name,
            "node_type":  node_type,
            "colors":     colors,
            "strokes":    strokes,
            "typography": typography,
            "effects":    effects,
            "summary": {
                "total_style_entries": len(entries),
                "unique_colors":       len(colors),
                "unique_fonts":        len(typography),
                "unique_effects":      len(effects),
                "unique_strokes":      len(strokes),
            },
            "raw_entries": entries,
        }
        return json.dumps(result)

    def extract_design_tokens_batch(
        self,
        entries: List[Dict],
        parallel: bool = True,
        output_format: str = "full",
        **kwargs,
    ) -> str:
        """Extract deduplicated design tokens from multiple Figma nodes in batch.

        Efficiently processes multiple file_key + node_id pairs and returns
        deduplicated design tokens for each entry.

        Each entry must contain:
        - file_key (required): Figma file key
        - node_id (required): Node ID (FRAME, COMPONENT, COMPONENT_SET, SECTION, or GROUP)
        - depth (optional): Tree fetch depth for this entry (1–8, default 4)

        Batch processing can be parallelized for better performance.

        Output formats (for LLM optimization):
        - 'full': All data (colors, strokes, typography, effects lists + summary)
        - 'compact': Summary only + counts (no full token lists) - Recommended for LLM
        - 'summary': Batch summary only (fastest, minimal tokens)
        - 'style_guide': Detailed design tokens + per-component token references (recommended for style guide generation)

        Returns a JSON string with the following structure (varies by output_format):

        FULL FORMAT (default):
        {
          "results": [
            {
              "entry_index": 0,
              "file_key": "...",
              "node_id": "...",
              "node_name": "...",
              "status": "success" | "error",
              "colors": [...objects...],
              "strokes": [...objects...],
              "typography": [...objects...],
              "effects": [...objects...],
              "summary": {...}
            },
            ...
          ],
          "batch_summary": {...}
        }

        COMPACT FORMAT (LLM optimized):
        {
          "results": [
            {
              "entry_index": 0,
              "node_id": "...",
              "node_name": "...",
              "status": "success",
              "color_count": 15,
              "stroke_count": 3,
              "font_count": 4,
              "effect_count": 2
            },
            ...
          ],
          "batch_summary": {...}
        }

        SUMMARY FORMAT (minimal):
        {
          "batch_summary": {
            "total_entries": 22,
            "processed_entries": 21,
            "successful": 21,
            "failed": 1,
            "total_colors": 87,
            ...
          }
        }

        Args:
            entries: List of dicts, each with file_key, node_id, and optional depth
            parallel: If True, process entries in parallel (default True).
            output_format: 'full' (all data), 'compact' (counts only), 'summary' (batch only),
                          or 'style_guide' (detailed tokens + component references).
                          Default: 'full'. Use 'compact'/'summary' for smallest LLM payload,
                          or 'style_guide' for screen style guide creation.

        Raises:
            ToolException: If entries list is empty or malformed.
        """
        # Validate output_format
        if output_format not in ("full", "compact", "summary", "style_guide"):
            output_format = "full"

        if not entries:
            raise ToolException("Batch entries list cannot be empty")

        if not isinstance(entries, list):
            raise ToolException("entries must be a list of dicts")

        self._log_tool_event(f"Starting batch design token extraction for {len(entries)} entries (format={output_format})...")

        # ...existing code...

        # Validate and normalize entries
        validated_entries = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ToolException(f"Entry {i} is not a dict: {type(entry)}")

            file_key = entry.get("file_key")
            node_id = entry.get("node_id")

            if node_id is None:
                self._log_tool_event(f"Skipping entry {i}: node_id is None")
                continue

            if not file_key or not node_id:
                raise ToolException(
                    f"Entry {i} missing required fields. "
                    f"Each entry must have 'file_key' and 'node_id'. "
                    f"Got: {list(entry.keys())}"
                )

            depth = entry.get("depth", 4)
            if not isinstance(depth, int) or depth < 1 or depth > 8:
                depth = 4

            validated_entries.append({
                "index": i,
                "file_key": file_key,
                "node_id": node_id,
                "depth": depth,
            })

        if not validated_entries:
            raise ToolException("No valid entries after filtering (all had None node_id)")

        # Process entries
        def process_entry(entry: Dict) -> Dict:
            """Process a single entry and return result dict."""
            entry_index = entry["index"]
            file_key = entry["file_key"]
            node_id = entry["node_id"]
            depth = entry["depth"]

            try:
                result_json = self.extract_design_tokens(
                    file_key=file_key,
                    node_id=node_id,
                    depth=depth,
                )
                result_data = json.loads(result_json)

                return {
                    "entry_index": entry_index,
                    "file_key": file_key,
                    "node_id": node_id,
                    "node_name": result_data.get("node_name", ""),
                    "node_type": result_data.get("node_type", ""),
                    "status": "success",
                    "colors": result_data.get("colors", []),
                    "strokes": result_data.get("strokes", []),
                    "typography": result_data.get("typography", []),
                    "effects": result_data.get("effects", []),
                    "summary": result_data.get("summary", {}),
                }
            except Exception as e:
                return {
                    "entry_index": entry_index,
                    "file_key": file_key,
                    "node_id": node_id,
                    "status": "error",
                    "error": str(e),
                }

        # Execute extraction
        all_results = []
        if parallel and len(validated_entries) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import logging as logging_module

            log = logging_module.getLogger(__name__)
            max_workers = min(5, len(validated_entries))  # Cap at 5 parallel workers

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_entry, entry): entry for entry in validated_entries}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        entry = futures[future]
                        log.warning(f"Entry {entry['index']} failed: {e}")
                        all_results.append({
                            "entry_index": entry["index"],
                            "file_key": entry["file_key"],
                            "node_id": entry["node_id"],
                            "status": "error",
                            "error": str(e),
                        })
        else:
            # Sequential processing
            for entry in validated_entries:
                result = process_entry(entry)
                all_results.append(result)

        # Sort by entry index to maintain order
        all_results.sort(key=lambda r: r["entry_index"])

        # Calculate batch summary
        successful = [r for r in all_results if r["status"] == "success"]
        failed = [r for r in all_results if r["status"] == "error"]

        total_colors = sum(len(r.get("colors", [])) for r in successful)
        total_strokes = sum(len(r.get("strokes", [])) for r in successful)
        total_fonts = sum(len(r.get("typography", [])) for r in successful)
        total_effects = sum(len(r.get("effects", [])) for r in successful)

        batch_summary = {
            "total_entries": len(entries),
            "processed_entries": len(validated_entries),
            "successful": len(successful),
            "failed": len(failed),
            "total_colors": total_colors,
            "total_strokes": total_strokes,
            "total_fonts": total_fonts,
            "total_effects": total_effects,
        }

        # Apply output format filtering for token optimization
        if output_format == "summary":
            # Return ONLY batch summary (minimal tokens for LLM)
            result = {"batch_summary": batch_summary}
        elif output_format == "compact":
            # Return entry summaries without full token lists (compact for LLM)
            compact_results = []
            for r in all_results:
                if r["status"] == "error":
                    compact_results.append({
                        "entry_index": r["entry_index"],
                        "node_id": r["node_id"],
                        "status": "error",
                        "error": r["error"],
                    })
                else:
                    compact_results.append({
                        "entry_index": r["entry_index"],
                        "node_id": r["node_id"],
                        "node_name": r["node_name"],
                        "node_type": r["node_type"],
                        "status": "success",
                        "color_count": len(r.get("colors", [])),
                        "stroke_count": len(r.get("strokes", [])),
                        "font_count": len(r.get("typography", [])),
                        "effect_count": len(r.get("effects", [])),
                    })
            result = {
                "results": compact_results,
                "batch_summary": batch_summary,
            }
        elif output_format == "style_guide":
            # Build normalized token dictionaries and per-component references.
            colors_registry: Dict[tuple, str] = {}
            strokes_registry: Dict[tuple, str] = {}
            typography_registry: Dict[tuple, str] = {}
            effects_registry: Dict[tuple, str] = {}

            colors_definitions: List[Dict] = []
            strokes_definitions: List[Dict] = []
            typography_definitions: List[Dict] = []
            effects_definitions: List[Dict] = []

            color_usage: Dict[str, int] = {}
            stroke_usage: Dict[str, int] = {}
            typography_usage: Dict[str, int] = {}
            effect_usage: Dict[str, int] = {}

            def _register_color(token: Dict) -> str:
                key = (token.get("hex", ""), token.get("alpha", 1.0), token.get("opacity", 1.0))
                token_id = colors_registry.get(key)
                if token_id:
                    return token_id
                token_id = f"color_{len(colors_definitions) + 1:03d}"
                colors_registry[key] = token_id
                colors_definitions.append({
                    "id": token_id,
                    "hex": token.get("hex", ""),
                    "alpha": token.get("alpha", 1.0),
                    "opacity": token.get("opacity", 1.0),
                    "source_path": token.get("source_path", ""),
                })
                color_usage[token_id] = 0
                return token_id

            def _register_stroke(token: Dict) -> str:
                key = (token.get("hex", ""), token.get("alpha", 1.0))
                token_id = strokes_registry.get(key)
                if token_id:
                    return token_id
                token_id = f"stroke_{len(strokes_definitions) + 1:03d}"
                strokes_registry[key] = token_id
                strokes_definitions.append({
                    "id": token_id,
                    "hex": token.get("hex", ""),
                    "alpha": token.get("alpha", 1.0),
                    "source_path": token.get("source_path", ""),
                })
                stroke_usage[token_id] = 0
                return token_id

            def _register_typography(token: Dict) -> str:
                key = (
                    token.get("fontFamily", ""),
                    token.get("fontSize", None),
                    token.get("fontWeight", None),
                    token.get("lineHeightPx", None),
                    token.get("letterSpacing", None),
                    token.get("textAlignHorizontal", None),
                )
                token_id = typography_registry.get(key)
                if token_id:
                    return token_id
                token_id = f"type_{len(typography_definitions) + 1:03d}"
                typography_registry[key] = token_id
                typography_definitions.append({
                    "id": token_id,
                    "fontFamily": token.get("fontFamily", ""),
                    "fontSize": token.get("fontSize", None),
                    "fontWeight": token.get("fontWeight", None),
                    "lineHeightPx": token.get("lineHeightPx", None),
                    "letterSpacing": token.get("letterSpacing", None),
                    "textAlignHorizontal": token.get("textAlignHorizontal", None),
                    "source_path": token.get("source_path", ""),
                })
                typography_usage[token_id] = 0
                return token_id

            def _register_effect(token: Dict) -> str:
                effect_key_payload = {
                    "type": token.get("type", ""),
                    "visible": token.get("visible", True),
                    "radius": token.get("radius", None),
                    "spread": token.get("spread", None),
                    "offset": token.get("offset", {}),
                    "color": token.get("color", {}),
                }
                key = json.dumps(effect_key_payload, sort_keys=True)
                token_id = effects_registry.get(key)
                if token_id:
                    return token_id
                token_id = f"effect_{len(effects_definitions) + 1:03d}"
                effects_registry[key] = token_id
                effects_definitions.append({
                    "id": token_id,
                    **effect_key_payload,
                    "source_path": token.get("source_path", ""),
                })
                effect_usage[token_id] = 0
                return token_id

            components: List[Dict] = []
            for r in all_results:
                if r["status"] == "error":
                    components.append({
                        "entry_index": r["entry_index"],
                        "file_key": r["file_key"],
                        "node_id": r["node_id"],
                        "status": "error",
                        "error": r["error"],
                    })
                    continue

                color_ids = sorted({_register_color(c) for c in r.get("colors", [])})
                stroke_ids = sorted({_register_stroke(s) for s in r.get("strokes", [])})
                typography_ids = sorted({_register_typography(t) for t in r.get("typography", [])})
                effect_ids = sorted({_register_effect(e) for e in r.get("effects", [])})

                for token_id in color_ids:
                    color_usage[token_id] += 1
                for token_id in stroke_ids:
                    stroke_usage[token_id] += 1
                for token_id in typography_ids:
                    typography_usage[token_id] += 1
                for token_id in effect_ids:
                    effect_usage[token_id] += 1

                components.append({
                    "entry_index": r["entry_index"],
                    "file_key": r["file_key"],
                    "node_id": r["node_id"],
                    "node_name": r.get("node_name", ""),
                    "node_type": r.get("node_type", ""),
                    "status": "success",
                    "tokens": {
                        "colors": color_ids,
                        "strokes": stroke_ids,
                        "typography": typography_ids,
                        "effects": effect_ids,
                    },
                })

            for item in colors_definitions:
                item["components_count"] = color_usage.get(item["id"], 0)
            for item in strokes_definitions:
                item["components_count"] = stroke_usage.get(item["id"], 0)
            for item in typography_definitions:
                item["components_count"] = typography_usage.get(item["id"], 0)
            for item in effects_definitions:
                item["components_count"] = effect_usage.get(item["id"], 0)

            result = {
                "style_guide": {
                    "tokens": {
                        "colors": colors_definitions,
                        "strokes": strokes_definitions,
                        "typography": typography_definitions,
                        "effects": effects_definitions,
                    },
                    "components": components,
                },
                "batch_summary": batch_summary,
            }
        else:
            # FULL format (backward compatible, all data)
            result = {
                "results": all_results,
                "batch_summary": batch_summary,
            }

        self._log_tool_event(
            f"Batch extraction complete (format={output_format}): {len(successful)} succeeded, {len(failed)} failed"
        )
        return json.dumps(result, indent=2)

    @extend_with_parent_available_tools
    def get_available_tools(self):
        return [
            {
                "name": "get_file_nodes",
                "description": self.get_file_nodes.__doc__,
                "args_schema": ArgsSchema.FileNodes.value,
                "ref": self.get_file_nodes,
            },
            {
                "name": "get_file",
                "description": self.get_file.__doc__,
                "args_schema": ArgsSchema.File.value,
                "ref": self.get_file,
            },
            {
                "name": "get_file_versions",
                "description": self.get_file_versions.__doc__,
                "args_schema": ArgsSchema.FileKey.value,
                "ref": self.get_file_versions,
            },
            {
                "name": "get_file_comments",
                "description": self.get_file_comments.__doc__,
                "args_schema": ArgsSchema.FileKey.value,
                "ref": self.get_file_comments,
            },
            {
                "name": "post_file_comment",
                "description": self.post_file_comment.__doc__,
                "args_schema": ArgsSchema.FileComment.value,
                "ref": self.post_file_comment,
            },
            {
                "name": "get_file_images",
                "description": self.get_file_images.__doc__,
                "args_schema": ArgsSchema.FileImages.value,
                "ref": self.get_file_images,
            },
            {
                "name": "get_team_projects",
                "description": self.get_team_projects.__doc__,
                "args_schema": ArgsSchema.TeamProjects.value,
                "ref": self.get_team_projects,
            },
            {
                "name": "get_project_files",
                "description": self.get_project_files.__doc__,
                "args_schema": ArgsSchema.ProjectFiles.value,
                "ref": self.get_project_files,
            },
            {
                "name": "analyze_file",
                "description": self.analyze_file.__doc__,
                "args_schema": AnalyzeFileSchema,
                "ref": self.analyze_file,
            },
            {
                "name": "extract_design_tokens",
                "description": self.extract_design_tokens.__doc__,
                "args_schema": ArgsSchema.ExtractDesignTokens.value,
                "ref": self.extract_design_tokens,
            },
            {
                "name": "extract_design_tokens_batch",
                "description": self.extract_design_tokens_batch.__doc__,
                "args_schema": ArgsSchema.ExtractDesignTokensBatch.value,
                "ref": self.extract_design_tokens_batch,
            },
        ]
