from typing import Generator
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from copy import deepcopy as copy


def text_chunker(
    file_content_generator: Generator[Document, None, None],
    config: dict,
    *args,
    **kwargs,
) -> Generator[Document, None, None]:
    """
    Chunks plain-text documents using token-based splitting.

    Unlike markdown_chunker, this function does NOT apply MarkdownHeaderTextSplitter.
    All output chunks receive method_name='text', making it semantically correct
    for .txt, .yaml, .groovy, and other non-markdown file types.

    Config options:
        max_tokens (int): Maximum tokens per chunk. Default: 512
        token_overlap (int): Token overlap between chunks. Default: 10
    """
    max_tokens = config.get("max_tokens", 512)
    token_overlap = config.get("token_overlap", 10)

    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=max_tokens,
        chunk_overlap=token_overlap,
    )

    for doc in file_content_generator:
        doc_metadata = doc.metadata
        chunk_texts = splitter.split_text(doc.page_content)
        for idx, chunk_text in enumerate(chunk_texts, 1):
            docmeta = copy(doc_metadata)
            docmeta["headers"] = ""
            docmeta["chunk_id"] = idx
            docmeta["chunk_type"] = "document"
            docmeta["method_name"] = "text"
            yield Document(page_content=chunk_text, metadata=docmeta)
