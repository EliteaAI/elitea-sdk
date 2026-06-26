import logging
from pathlib import Path
from typing import Any, List, Union, Generator, Iterator
from langchain_core.documents import Document

from langchain_community.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)

logger = logging.getLogger(__name__)


class EliteAMarkdownLoader(UnstructuredFileLoader):

    @classmethod
    def get_file_metadata(cls, *, filename: str,
                          file_content=None,
                          file_size=None) -> dict:
        """Report total line count and advertise start_line/end_line (PRE-3 #5434)."""
        total_lines = 0
        if file_content:
            try:
                if isinstance(file_content, (bytes, bytearray)):
                    text = file_content.decode("utf-8")
                else:
                    text = file_content
                total_lines = len(text.splitlines())
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Failed to count lines for %s: %s", filename, e)

        range_hint = f"Valid range 1..{total_lines}. " if total_lines else ""
        instruction = {
            "first_class_params": {
                "start_line": (
                    f"integer (1-indexed, inclusive) — first line to read. "
                    f"{range_hint}Omit to read from the beginning."
                ),
                "end_line": (
                    f"integer (1-indexed, inclusive) — last line to read. "
                    f"{range_hint}Omit to read to the end."
                ),
            },
            "notes": (
                "Use start_line/end_line together to read a bounded slice "
                "of a large Markdown file and keep tokens bounded."
            ),
        }
        return {
            "unit": "lines",
            "total_lines": total_lines,
            "instruction_for_readFile": instruction,
        }

    def get_content(self) -> str:
        """Return raw markdown text so read_file line-slicing works correctly."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    def __init__(
        self,
        file_path: Union[str, Path],
        mode: str = "elements",
        **kwargs: Any,
    ):
        """
        Args:
            file_path: The path to the Markdown file to load.
            mode: The mode to use when loading the file. Can be one of "single",
                "multi", or "all". Default is "single".
            **kwargs: Accepts ``max_tokens`` (default 512) and
                ``token_overlap`` (default 10) for chunking configuration.
                Any remaining kwargs are forwarded to UnstructuredFileLoader.
        """
        file_path = str(file_path)
        validate_unstructured_version("0.4.16")
        max_tokens = kwargs.pop("max_tokens", 512)
        token_overlap = kwargs.pop("token_overlap", 10)
        self.chunker_config = {
            "strip_header": False,
            "return_each_line": False,
            "headers_to_split_on": [],
            "max_tokens": max_tokens,
            "token_overlap": token_overlap,
        }
        super().__init__(file_path=file_path, mode=mode, **kwargs)

    def _file_content_generator(self) -> Generator[Document, None, None]:
        """
        Creates a generator that yields a single Document object
        representing the entire content of the Markdown file.
        """
        with open(self.file_path, "r", encoding="utf-8") as file:
            content = file.read()
        yield Document(page_content=content, metadata={"source": self.file_path})

    def _get_elements(self) -> List[Document]:
        """
        Processes the Markdown file using the markdown_chunker and returns the chunks.
        """
        from elitea_sdk.tools.chunkers.sematic.markdown_chunker import markdown_chunker

        # Create a generator for the file content
        file_content_generator = self._file_content_generator()

        # Use the markdown_chunker to process the content
        chunks = markdown_chunker(file_content_generator, config=self.chunker_config)

        # Convert the generator to a list of Document objects
        return list(chunks)

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        elements = self._get_elements()
        self._post_process_elements(elements)
        yield from elements