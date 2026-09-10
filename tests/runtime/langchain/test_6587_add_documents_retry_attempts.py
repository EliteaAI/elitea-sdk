import pytest
from langchain_core.documents import Document
from sqlalchemy.exc import DataError, OperationalError
from tenacity import wait_none

from elitea_sdk.runtime.langchain.interfaces.llm_processor import add_documents

_INSERT_STMT = (
    "INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata) "
    "VALUES (%(id__0)s, %(collection_id__0)s, %(embedding__0)s, %(document__0)s, %(cmetadata__0)s) "
    "ON CONFLICT (id) DO UPDATE SET embedding = excluded.embedding"
)
_BOUND_VECTORS_CONTAINING_STATUS_CODE_DIGITS = {
    "embedding__0": "[0.500123, -0.429876, 0.502244, -0.503991, 0.504010]"
}

add_documents_without_backoff = add_documents.retry_with(wait=wait_none())


class CountingVectorStore:
    def __init__(self, exception):
        self.exception = exception
        self.attempts = 0

    def add_texts(self, texts, metadatas=None, ids=None):
        self.attempts += 1
        raise self.exception


def _insert_error(error_class, orig):
    return error_class(_INSERT_STMT, _BOUND_VECTORS_CONTAINING_STATUS_CODE_DIGITS, orig)


def _flush_one_document(vectorstore):
    return add_documents_without_backoff(
        vectorstore=vectorstore,
        documents=[Document(page_content="chunk", metadata={"source": "a.py"})],
    )


def test_deterministic_database_error_is_not_retried():
    vectorstore = CountingVectorStore(
        _insert_error(DataError, Exception("unsupported Unicode escape sequence: \\u0000"))
    )

    with pytest.raises(DataError):
        _flush_one_document(vectorstore)

    assert vectorstore.attempts == 1


def test_transient_database_error_exhausts_the_retry_chain():
    vectorstore = CountingVectorStore(
        _insert_error(OperationalError, Exception("terminating connection due to administrator command"))
    )

    with pytest.raises(OperationalError):
        _flush_one_document(vectorstore)

    assert vectorstore.attempts == 5
