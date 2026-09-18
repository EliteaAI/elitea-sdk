import ast
import fnmatch
import json
import logging
from typing import Dict, Optional, List, Generator, Set

from langchain_core.documents import Document
from langchain_core.tools import ToolException
from pydantic import Field

from elitea_sdk.tools.base_indexer_toolkit import (
    _STATS_COUNTER_LOCK,
    BaseIndexerToolkit,
    IndexingStats,
    indexed_rows_of,
)

logger = logging.getLogger(__name__)

PROGRESS_EVENTS_PER_DECADE = 10
PROGRESS_EVENTS_MINIMUM_STEP = 10
PROGRESS_EVENTS_MAXIMUM_STEP = 1000


def progress_step_at(processed: int) -> int:
    step = PROGRESS_EVENTS_MINIMUM_STEP
    while processed >= step * PROGRESS_EVENTS_PER_DECADE:
        step *= PROGRESS_EVENTS_PER_DECADE
    return min(step, PROGRESS_EVENTS_MAXIMUM_STEP)


class CodeIndexerToolkit(BaseIndexerToolkit):
    index_item_labels = ('file', 'files')
    loader_yields_chunks = True
    loader_skips_unchanged_by_identity = True

    def _get_indexed_data(self, index_name: str):
        self._ensure_vectorstore_initialized()
        if not self.vector_adapter:
            raise ToolException("Vector adapter is not initialized. "
                             "Check your configuration: embedding_model and vectorstore_type.")
        return self.vector_adapter.get_code_indexed_data(self, index_name)

    def key_fn(self, document: Document):
        return document.metadata.get("filename")

    def compare_fn(self, document: Document, idx_data):
        return (document.metadata.get('commit_hash') and
            idx_data.get('commit_hashes') and
            document.metadata.get('commit_hash') in idx_data.get('commit_hashes')
        )

    def remove_ids_fn(self, idx_data, key: str):
        return idx_data[key]['ids']

    def _get_files_with_identity(self, path: str, branch: str) -> Optional[Dict[str, str]]:
        return None

    def _note_identity_backfill(self, document: Document, entry):
        run = getattr(self, "_index_run", None)
        identity = document.metadata.get('blob_sha')
        if run is None or not identity:
            return
        key = document.metadata.get('filename')
        if key in run.identity_backfill_visited_keys:
            return
        run.identity_backfill_visited_keys.add(key)
        if self._every_row_carries(entry, identity):
            return
        content_hash = document.metadata.get('commit_hash')
        for row_id, row_content_hash, row_identity in indexed_rows_of(entry):
            if row_content_hash == content_hash and row_identity != identity:
                run.identity_backfill[str(row_id)] = identity

    def _base_loader(
            self,
            branch: Optional[str] = None,
            whitelist: Optional[List[str]] = None,
            blacklist: Optional[List[str]] = None,
            chunking_config: Optional[dict] = None,
            skip_unsupported_extensions: bool = True,
            **kwargs) -> Generator[Document, None, None]:
        """Index repository files in the vector store using code parsing."""
        run = getattr(self, "_index_run", None)
        yield from self.loader(
            branch=branch,
            whitelist=whitelist,
            blacklist=blacklist,
            chunking_config=chunking_config,
            skip_unsupported_extensions=skip_unsupported_extensions,
            index_name=kwargs.get("index_name") if getattr(run, "unchanged_skip_enabled", False) else None,
            preskipped_keys=getattr(run, "preskipped_keys", None),
        )

    def _extend_data(self, documents: Generator[Document, None, None]):
        yield from documents

    def _index_tool_params(self):
        """Return the parameters for indexing data."""
        return {
            "branch": (Optional[str], Field(
                description="Branch to index files from. Defaults to active branch if None.",
                default=None)),
            "whitelist": (Optional[List[str]], Field(
                description='File extensions or paths to include. Defaults to all files if None. Example: `["*.md", "*.java"]`',
                default=None)),
            "blacklist": (Optional[List[str]], Field(
                description='File extensions or paths to exclude. Defaults to no exclusions if None. Example: `["*.md", "*.java"]`',
                default=None)),
            "skip_unsupported_extensions": (Optional[bool], Field(
                description='Skip files with unsupported extensions (default: True). Supported: .py, .js, .ts, .java, .go, .rs, .md, .json, etc.',
                default=True)),
        }

    def loader(self,
               branch: Optional[str] = None,
               whitelist: Optional[List[str]] = None,
               blacklist: Optional[List[str]] = None,
               chunked: bool = True,
               chunking_config: Optional[dict] = None,
               skip_unsupported_extensions: bool = True,
               index_name: Optional[str] = None,
               preskipped_keys: Optional[Set[str]] = None) -> Generator[Document, None, None]:
        """
        Generates Documents from files in a branch, respecting whitelist and blacklist patterns.

        Parameters:
        - branch (Optional[str]): Branch for listing files. Defaults to the current branch if None.
        - whitelist (Optional[List[str]]): File extensions or paths to include. Defaults to all files if None.
        - blacklist (Optional[List[str]]): File extensions or paths to exclude. Defaults to no exclusions if None.
        - chunked (bool): If True (default), applies universal chunker based on file type.
                         If False, returns raw Documents without chunking.
        - chunking_config (Optional[dict]): Chunking configuration by file extension
        - skip_unsupported_extensions (bool): If True (default), skip files with unsupported extensions
                                              and report them. If False, process them with text chunker.

        Returns:
        - generator: Yields Documents from files matching the whitelist but not the blacklist.
                    Each document has exactly the key 'filename' in metadata, which is used as an ID
                    for further operations (indexing, deduplication, and retrieval).

        Example:
        # Use 'feature-branch', include '.py' files, exclude 'test_' files
        for doc in loader(branch='feature-branch', whitelist=['*.py'], blacklist=['*test_*']):
            print(doc.page_content)

        Notes:
        - Whitelist and blacklist use Unix shell-style wildcards.
        - Files must match the whitelist and not the blacklist to be included.
        - Each document MUST have exactly the key 'filename' in metadata. This key is used as an ID
          for further operations such as indexing, deduplication, and retrieval.
        - When chunked=True:
          - .md files → markdown chunker (header-based splitting)
          - .py/.js/.ts/etc → code parser (TreeSitter-based)
          - .json files → JSON chunker
          - other files → skipped (with skip_unsupported_extensions=True) or text chunker
        """
        import hashlib
        import os
        from .chunkers.universal_chunker import (
            MARKDOWN_EXTENSIONS, JSON_EXTENSIONS, CODE_EXTENSIONS,
            CONFIG_EXTENSIONS, TEXT_EXTENSIONS
        )

        # Combined supported extensions
        SUPPORTED_EXTENSIONS = (
            MARKDOWN_EXTENSIONS | JSON_EXTENSIONS | CODE_EXTENSIONS |
            CONFIG_EXTENSIONS | TEXT_EXTENSIONS
        )

        self._init_indexing_stats()

        listing = self.__list_files("", self.__get_branch(branch))
        file_identities = listing if isinstance(listing, dict) else None
        _files = list(file_identities) if file_identities is not None \
            else self.__handle_get_files("", self.__get_branch(branch), listing)

        identity_skip_is_armed = bool(chunked and index_name and file_identities
                                      and preskipped_keys is not None)
        unchanged_identities = None

        def identity_confirmed_by(file_path: str, content: str) -> Optional[str]:
            listed = file_identities.get(file_path) if file_identities else None
            if not listed:
                return None
            body = content.encode("utf-8")
            git_blob = hashlib.sha1(b"blob %d\0" % len(body) + body,
                                    usedforsecurity=False).hexdigest()
            return listed if git_blob == listed else None

        def is_unchanged_since_last_index(file_path: str) -> bool:
            nonlocal unchanged_identities
            if not identity_skip_is_armed:
                return False
            identity = file_identities.get(file_path)
            if not identity:
                return False
            if unchanged_identities is None:
                unchanged_identities = self._collect_unchanged_identities(
                    self._read_indexed_data_once(index_name))
            return unchanged_identities.get(file_path) == identity

        def is_whitelisted(file_path: str) -> bool:
            if whitelist:
                return (any(fnmatch.fnmatch(file_path, pattern) for pattern in whitelist)
                        or any(file_path.endswith(f'.{pattern}') for pattern in whitelist))
            return True

        def is_blacklisted(file_path: str) -> bool:
            if blacklist:
                return (any(fnmatch.fnmatch(file_path, pattern) for pattern in blacklist)
                        or any(file_path.endswith(f'.{pattern}') for pattern in blacklist))
            return False

        def has_supported_extension(file_path: str) -> bool:
            """Check if file has a supported extension for indexing."""
            ext = os.path.splitext(file_path)[-1].lower()
            return ext in SUPPORTED_EXTENSIONS

        yielded_files = set()

        def raw_document_generator() -> Generator[Document, None, None]:
            """Yields raw Documents without chunking - pure generator, no pre-filtering."""
            processed = 0
            total_files = 0
            stats = self.get_indexing_stats()

            downloaded = 0

            def count_processed_file():
                nonlocal processed
                processed += 1
                with _STATS_COUNTER_LOCK:
                    stats.items_processed += 1
                if processed % progress_step_at(processed) == 0:
                    self._log_tool_event(message=f"{processed} files processed",
                                         tool_name="loader")

            for file in _files:
                total_files += 1
                stats.total_fetched = total_files
                # Check whitelist first
                if whitelist and not is_whitelisted(file):
                    stats.files_skipped_whitelist.add(file)
                    continue

                # Check blacklist
                if is_blacklisted(file):
                    stats.files_skipped_blacklist.add(file)
                    continue

                # Check for supported extensions (only when skip_unsupported_extensions is True)
                if skip_unsupported_extensions and not has_supported_extension(file):
                    stats.files_unsupported_extension.add(file)
                    continue

                if is_unchanged_since_last_index(file):
                    preskipped_keys.add(file)
                    count_processed_file()
                    continue

                downloaded += 1
                try:
                    file_content = self._read_file(file, self.__get_branch(branch))
                except Exception as e:
                    logger.error(f"Failed to read file {file}: {e}")
                    stats.files_skipped_read_error.add(file)
                    continue

                if not file_content:
                    stats.files_skipped_empty.add(file)
                    continue

                # Ensure file content is a string
                if isinstance(file_content, bytes):
                    file_content = file_content.decode("utf-8", errors="ignore")
                elif isinstance(file_content, dict) and file.endswith('.json'):
                    file_content = json.dumps(file_content)
                elif not isinstance(file_content, str):
                    file_content = str(file_content)

                # Hash the file content for uniqueness tracking
                file_hash = hashlib.sha256(file_content.encode("utf-8")).hexdigest()
                count_processed_file()
                yielded_files.add(file)

                metadata = {
                    'file_path': file,
                    'filename': file,
                    'source': file,
                    'commit_hash': file_hash,
                }
                blob_sha = identity_confirmed_by(file, file_content)
                if blob_sha:
                    metadata['blob_sha'] = blob_sha

                yield Document(page_content=file_content, metadata=metadata)

            self._log_tool_event(
                message=f"{processed} files processed, {downloaded} downloaded",
                tool_name="loader")

            # Log skipped files summary
            summary = stats.get_summary()
            if summary:
                self._log_tool_event(message=summary, tool_name="loader")

        if not chunked:
            # Return raw documents without chunking
            return raw_document_generator()

        # Apply universal chunker based on file type
        from .chunkers.universal_chunker import universal_chunker

        def chunked_document_generator() -> Generator[Document, None, None]:
            """A chunker can produce nothing for a file the loader already counted. That
            file reaches neither dedup nor the writer, so it can never be recorded
            unchanged, and would otherwise read as freshly indexed on every run.
            """
            chunked_files = set()
            for chunk in universal_chunker(raw_document_generator(), config=chunking_config):
                chunked_files.add(chunk.metadata.get('filename'))
                yield chunk

            dropped = yielded_files - chunked_files
            if dropped:
                stats = self.get_indexing_stats()
                stats.files_skipped_empty.update(dropped)
                # Counted at load time, so the invariant needs them back out.
                with _STATS_COUNTER_LOCK:
                    stats.items_processed = max(stats.items_processed - len(dropped), 0)
                self._log_tool_event(
                    message=f"{len(dropped)} files produced no indexable content",
                    tool_name="loader")

        return chunked_document_generator()

    def __list_files(self, path: str, branch: str):
        listing = self._get_files_with_identity(path=path, branch=branch)
        if not isinstance(listing, dict):
            return listing
        if not all(isinstance(file_path, str) for file_path in listing):
            return list(listing)
        return {
            file_path: identity if isinstance(identity, str) else ""
            for file_path, identity in listing.items()
        }

    def __handle_get_files(self, path: str, branch: str, prefetched=None):
        """
        Handles the retrieval of files from a specific path and branch.
        This method should be implemented in subclasses to provide the actual file retrieval logic.
        """
        _files = prefetched if prefetched is not None else self._get_files(path=path, branch=branch)
        if isinstance(_files, str):
            try:
                # Attempt to convert the string to a list using ast.literal_eval
                _files = ast.literal_eval(_files)
                # Ensure that the result is actually a list of strings
                if not isinstance(_files, list) or not all(isinstance(item, str) for item in _files):
                    raise ValueError("The evaluated result is not a list of strings")
            except (SyntaxError, ValueError):
                # Handle the case where the string cannot be converted to a list
                raise ValueError("Expected a list of strings, but got a string that cannot be converted")

            # Ensure _files is a list of strings
        if not isinstance(_files, list) or not all(isinstance(item, str) for item in _files):
            raise ValueError("Expected a list of strings")
        return _files

    def __get_branch(self, branch):
       return (branch or getattr(self, 'active_branch', None)
               or getattr(self, '_active_branch', None) or getattr(self, 'branch', None))
