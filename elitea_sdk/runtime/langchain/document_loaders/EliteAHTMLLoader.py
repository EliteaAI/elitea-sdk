import os
import tempfile

from langchain_community.document_loaders import UnstructuredHTMLLoader, UnstructuredXMLLoader


class EliteAHTMLLoader(UnstructuredHTMLLoader):
    """Thin wrapper that adds get_file_metadata and get_content (PRE-9 #5440).

    get_content returns the same extracted text that the indexer always received
    (Unstructured partition_html, tags stripped) — no indexer regression.
    get_file_metadata runs the same extraction so line counts match reality.
    """

    def get_content(self) -> str:
        return "\n".join(doc.page_content for doc in self.load())

    @classmethod
    def get_file_metadata(cls, *, filename: str, file_content=None, file_size=None) -> dict:
        from elitea_sdk.tools.utils.file_metadata import build_line_range_metadata
        extracted = _extract_to_text(cls, filename, file_content)
        return build_line_range_metadata(extracted, file_type_note="HTML file")


class EliteAXMLLoader(UnstructuredXMLLoader):
    """Thin wrapper for XML files — same pattern as EliteAHTMLLoader (PRE-9 #5440)."""

    def get_content(self) -> str:
        return "\n".join(doc.page_content for doc in self.load())

    @classmethod
    def get_file_metadata(cls, *, filename: str, file_content=None, file_size=None) -> dict:
        from elitea_sdk.tools.utils.file_metadata import build_line_range_metadata
        extracted = _extract_to_text(cls, filename, file_content)
        return build_line_range_metadata(extracted, file_type_note="XML file")


def _extract_to_text(loader_cls, filename: str, file_content) -> bytes:
    """Run Unstructured extraction on file_content bytes via a temp file.

    Returns the extracted text encoded as UTF-8 bytes so build_line_range_metadata
    can count lines against the same content that read_file will return.
    Gracefully falls back to raw file_content on any extraction error.
    """
    if not file_content:
        return b""
    ext = os.path.splitext(filename)[1] or ".html"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        text = "\n".join(doc.page_content for doc in loader_cls(file_path=tmp_path).load())
        return text.encode("utf-8", errors="replace")
    except Exception:
        return file_content if isinstance(file_content, (bytes, bytearray)) else b""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
