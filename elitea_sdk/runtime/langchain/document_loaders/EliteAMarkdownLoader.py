from pathlib import Path
from typing import Any, List, Union, Generator, Iterator
from langchain_core.documents import Document

from langchain_community.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)

class EliteAMarkdownLoader(UnstructuredFileLoader):

    @classmethod
    def get_file_metadata(cls, *, filename: str,
                          file_content=None,
                          file_size=None) -> dict:
        """Report total line count and advertise start_line/end_line (PRE-3 #5434)."""
        from elitea_sdk.tools.utils.file_metadata import build_line_range_metadata
        return build_line_range_metadata(file_content, file_type_note="Markdown file")

    def get_content(self) -> str:
        """Return raw markdown text so read_file line-slicing works correctly."""
        if getattr(self, "file_path", None):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return f.read()
        if getattr(self, "file_content", None) is not None:
            content = self.file_content
            if isinstance(content, bytes):
                try:
                    return content.decode("utf-8")
                except UnicodeDecodeError:
                    from elitea_sdk.tools.utils.text_operations import decode_text
                    return decode_text(content)
            return content
        raise ValueError("Neither file_path nor file_content is provided.")

    def __init__(
        self,
        file_path: Union[str, Path, None] = None,
        mode: str = "elements",
        **kwargs: Any,
    ):
        """
        Args:
            file_path: The path to the Markdown file to load. May be ``None``
                when the content is supplied in-memory via ``file_content``.
            mode: The mode to use when loading the file. Can be one of "single",
                "multi", or "all". Default is "single".
            **kwargs: Accepts ``file_content``/``file_name`` for in-memory reads,
                ``max_tokens`` (default 512) and ``token_overlap`` (default 10)
                for chunking configuration. Any remaining kwargs are forwarded
                to UnstructuredFileLoader.
        """
        # Preserve in-memory content so get_content()/_file_content_generator()
        # work when no on-disk path is given (e.g. SharePoint read_document).
        self.file_content = kwargs.pop("file_content", None)
        self.file_name = kwargs.pop("file_name", None)
        # Do NOT stringify None → "None"; that produced open("None") failures.
        file_path = str(file_path) if file_path is not None else None
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
        content = self.get_content()
        source = self.file_path if getattr(self, "file_path", None) else self.file_name
        yield Document(page_content=content, metadata={"source": source})

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