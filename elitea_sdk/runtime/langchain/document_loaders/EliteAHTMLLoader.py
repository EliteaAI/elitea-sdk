import os
import tempfile
from contextlib import contextmanager

from langchain_community.document_loaders import UnstructuredHTMLLoader, UnstructuredXMLLoader


class _BytesInputMixin:
    """Lets an Unstructured file loader take ``file_content`` bytes instead of a path (#7751).

    Toolkits read downloaded files through ``parse_file_content(file_name=..., file_content=...)``,
    which builds the loader with ``file_path=None``. Unstructured only partitions files on disk,
    so the bytes are written to a temp file (with the original suffix) for the duration of a load.
    """

    _default_suffix = ""

    def __init__(self, file_path=None, file_content=None, file_name=None, mode: str = "single", **unstructured_kwargs):
        if isinstance(file_content, str):
            file_content = file_content.encode("utf-8")
        self._file_content = file_content
        self._file_name = file_name
        self._from_path = bool(file_path)
        super().__init__(file_path=file_path or "", mode=mode, **unstructured_kwargs)
        if not file_path:
            self.file_path = None

    @contextmanager
    def _materialized(self):
        """Yields the path to partition: the given file, or a temp copy of the bytes owned by this load only."""
        if self._from_path or self._file_content is None:
            yield self.file_path
            return
        suffix = os.path.splitext(self._file_name or "")[1] or self._default_suffix
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(self._file_content)
            yield tmp_path
        finally:
            os.unlink(tmp_path)

    def _get_elements(self):
        with self._materialized() as path:
            return self._partition(path)

    def lazy_load(self):
        if not self._from_path and self._file_content is not None and not bytes(self._file_content).strip():
            return  # Unstructured raises on an empty document; an empty file has no text
        yield from super().lazy_load()

    def _get_metadata(self) -> dict:
        if not self._from_path and self._file_content is not None:
            return {"source": self._file_name}
        return super()._get_metadata()


class EliteAHTMLLoader(_BytesInputMixin, UnstructuredHTMLLoader):
    """Thin wrapper that adds get_file_metadata and get_content (PRE-9 #5440).

    get_content returns the same extracted text that the indexer always received
    (Unstructured partition_html, tags stripped) — no indexer regression.
    get_file_metadata runs the same extraction so line counts match reality.
    """

    _default_suffix = ".html"

    def _partition(self, path: str):
        from unstructured.partition.html import partition_html

        return partition_html(filename=path, **self.unstructured_kwargs)

    def get_content(self) -> str:
        return "\n".join(doc.page_content for doc in self.load())

    @classmethod
    def get_file_metadata(cls, *, filename: str, file_content=None, file_size=None) -> dict:
        from elitea_sdk.tools.utils.file_metadata import build_line_range_metadata
        extracted = _extract_to_text(cls, filename, file_content)
        return build_line_range_metadata(extracted, file_type_note="HTML file")


class EliteAXMLLoader(_BytesInputMixin, UnstructuredXMLLoader):
    """Thin wrapper for XML files — same pattern as EliteAHTMLLoader (PRE-9 #5440)."""

    _default_suffix = ".xml"

    def _partition(self, path: str):
        from unstructured.partition.xml import partition_xml

        return partition_xml(filename=path, **self.unstructured_kwargs)

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
