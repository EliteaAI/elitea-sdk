import os

from typing import Generator, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

from .constants import (Language, get_langchain_language, get_file_extension,
                        get_programming_language, image_extensions, default_skip)
from .treesitter.treesitter import Treesitter, TreesitterMethodNode

from logging import getLogger

logger = getLogger(__name__)

def parse_code_files_for_db(
    file_content_generator: Generator[str, None, None],
    config: Optional[Dict[str, Any]] = None
) -> Generator[Document, None, None]:
    """
    Parses code files from a generator and returns a generator of Document objects for database storage.

    Args:
        file_content_generator (Generator[str, None, None]): Generator that yields file contents.
        config (dict, optional): Configuration dict with:
            - max_tokens or chunk_size: Maximum tokens per chunk for known languages (default: 1024)
            - token_overlap or chunk_overlap: Token overlap for known languages (default: 128)
            - unknown_chunk_size: Chunk size for unknown file types (default: 256)
            - unknown_chunk_overlap: Overlap for unknown file types (default: 30)

    Returns:
        Generator[Document, None, None]: Generator of Document objects containing parsed code information.
    """
    if config is None:
        config = {}

    chunk_size = config.get('max_tokens', config.get('chunk_size', 1024))
    chunk_overlap = config.get('token_overlap', config.get('chunk_overlap', 128))
    unknown_chunk_size = config.get('unknown_chunk_size', 256)
    unknown_chunk_overlap = config.get('unknown_chunk_overlap', 30)

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 8)
    if unknown_chunk_overlap >= unknown_chunk_size:
        unknown_chunk_overlap = max(0, unknown_chunk_size // 8)

    code_splitter = None
    for data in file_content_generator:
        file_name: str = data.get("file_name")
        file_content: str = data.get("file_content")
        file_bytes = file_content.encode()

        file_extension = get_file_extension(file_name)
        programming_language = get_programming_language(file_extension)
        if len(file_content.strip()) == 0 or file_name in default_skip:
            logger.debug(f"Skipping file: {file_name}")
            continue
        if file_extension in image_extensions:
            logger.debug(f"Skipping image file: {file_name} as it is image")
            continue
        chunk_id = 0
        if programming_language == Language.UNKNOWN:
            documents = TokenTextSplitter(encoding_name="gpt2", chunk_size=unknown_chunk_size, chunk_overlap=unknown_chunk_overlap).split_text(file_content)
            for document in documents:
                metadata = {
                    "filename": file_name,
                    "method_name": 'text',
                    "language": programming_language.value,
                }
                commit_hash = data.get("commit_hash")
                if commit_hash is not None:
                    metadata["commit_hash"] = commit_hash
                chunk_id += 1
                metadata["chunk_id"] = chunk_id
                document = Document(
                    page_content=document,
                    metadata=metadata,
                )
                yield document
        else:
            try:
                langchain_language = get_langchain_language(programming_language)

                if langchain_language:
                    code_splitter = RecursiveCharacterTextSplitter.from_language(
                        language=langchain_language,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                treesitter_parser = Treesitter.create_treesitter(programming_language)
                treesitterNodes: list[TreesitterMethodNode] = treesitter_parser.parse(
                    file_bytes
                )
                for node in treesitterNodes:
                    method_source_code = node.method_source_code

                    if node.doc_comment and programming_language != Language.PYTHON:
                        method_source_code = node.doc_comment + "\n" + method_source_code

                    splitted_documents = [method_source_code]
                    if code_splitter:
                        splitted_documents = code_splitter.split_text(method_source_code)

                    for splitted_document in splitted_documents:
                        metadata = {
                            "filename": file_name,
                            "method_name": node.name if node.name else 'unknown',
                            "language": programming_language.value,
                        }
                        commit_hash = data.get("commit_hash")
                        if commit_hash is not None:
                            metadata["commit_hash"] = commit_hash
                        chunk_id += 1
                        metadata["chunk_id"] = chunk_id
                        document = Document(
                            page_content=splitted_document,
                            metadata=metadata,
                        )
                        yield document
            except Exception as e:
                from traceback import format_exc
                logger.error(f"Error: {format_exc()}")
                raise e