"""Abstract base interface for SharePoint API implementations.

Two concrete implementations exist:
- :class:`SharepointRestWrapper`  : office365-rest-python-client (app credentials or REST token)
- :class:`SharepointGraphWrapper` : Microsoft Graph API (delegated access — requires token + scopes)
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.tools import ToolException


class BaseSharepointWrapper(ABC):
    """Abstract base defining the SharePoint operations contract.

    All concrete wrappers must implement every abstract method so that the
    factory in :class:`SharepointApiWrapper` can transparently swap backends.
    """

    # ------------------------------------------------------------------ #
    #  Lists                                                               #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def read_list(self, list_title: str, limit: int = 1000):
        """Return items (up to *limit*) from the named list.

        Returns:
            list[dict] on success, :class:`ToolException` on failure.
        """

    @abstractmethod
    def get_lists(self):
        """Return all non-hidden lists on the site.

        Returns:
            list[dict] on success, :class:`ToolException` on failure.
        """

    @abstractmethod
    def get_list_columns(self, list_title: str):
        """Return column metadata for the named list.

        Returns:
            list[dict] on success, raises :class:`ToolException` on failure.
        """

    @abstractmethod
    def create_list_item(self, list_title: str, fields: dict):
        """Create a new item in the named list.

        Returns:
            dict with {id, fields, ...} on success, raises :class:`ToolException` on failure.
        """

    # ------------------------------------------------------------------ #
    #  Files                                                               #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_files_list(
        self,
        folder_name: Optional[str] = None,
        limit_files: int = 100,
        form_name: Optional[str] = None,
        include_extensions: Optional[List[str]] = None,
        skip_extensions: Optional[List[str]] = None,
    ):
        """Return a list of file metadata dicts from document libraries,
        including files from subfolders.

        Args:
            folder_name: Optional sub-folder path to restrict listing.
            limit_files: Maximum number of files to return.
            form_name: Optional Document Library name filter.
            include_extensions: If provided, only files whose name matches one
                of these extension, filename, or glob-style patterns are returned.
                Accepts values such as ``'pdf'``, ``'.pdf'``, ``'*.pdf'``, or
                ``'report.pdf'``; matched case-insensitively.
            skip_extensions: If provided, files whose name matches any of these
                extension, filename, or glob-style patterns are excluded. Same
                format as *include_extensions*.

        Returns:
            list[dict] on success, :class:`ToolException` on failure.
        """

    @abstractmethod
    def read_file(
        self,
        path: str,
        is_capture_image: bool = False,
        page_number: Optional[int] = None,
        sheet_name: Optional[str] = None,
        excel_by_sheets: bool = False,
    ):
        """Return parsed textual / structured content of the file at *path*.

        Returns:
            str | dict on success, :class:`ToolException` on failure.
        """

    @abstractmethod
    def load_file_content_in_bytes(self, path: str) -> bytes:
        """Return raw bytes of the file at *path*.

        Raises:
            RuntimeError | Exception on failure (do not return ToolException here
            since it is used as an internal helper for the indexer).
        """

    @abstractmethod
    def upload_file(
        self,
        folder_path: str,
        filepath: Optional[str] = None,
        filedata: Optional[str] = None,
        filename: Optional[str] = None,
        replace: bool = True,
    ):
        """Upload a file to a document library folder.

        Returns:
            dict with {id, webUrl, path, size, mime_type} on success.
        Raises:
            :class:`ToolException` on failure.
        """

    @abstractmethod
    def add_attachment_to_list_item(
        self,
        list_title: str,
        item_id: int,
        filepath: Optional[str] = None,
        filedata: Optional[str] = None,
        filename: Optional[str] = None,
        replace: bool = True,
    ):
        """Attach a file to a list item.

        Returns:
            dict with {id, name, size} on success.
        Raises:
            :class:`ToolException` on failure.
        """

    # ------------------------------------------------------------------ #
    #  OneNote  (Graph API only — delegated access required)             #
    # ------------------------------------------------------------------ #

    _ONENOTE_NOT_SUPPORTED = (
        "OneNote operations require Graph API delegated access. "
        "Provide token + scopes to enable OneNote support."
    )

    def onenote_get_notebooks(self, select: Optional[List[str]] = None) -> list:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_get_sections(self, notebook_id: str, select: Optional[List[str]] = None) -> list:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_get_pages(self, section_id: str, limit: int = 100, select: Optional[List[str]] = None) -> list:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_get_page_content(self, page_id: str) -> str:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_list_attachments(self, page_id: str) -> list:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_read_attachment(
        self,
        page_id: str,
        attachment_name: str,
        capture_images: bool = True,
    ) -> str:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_read_page(
        self,
        page_id: str,
        capture_images: bool = True,
        include_attachments: bool = True,
        read_attachment_content: bool = False,
    ) -> str:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_create_notebook(self, display_name: str) -> dict:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_create_section(self, notebook_id: str, display_name: str) -> dict:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_create_page(self, section_id: str, html_content: str) -> dict:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_update_page(self, page_id: str, patch_commands: list) -> str:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_replace_page_content(self, page_id: str, html_content: str) -> str:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    def onenote_delete_page(self, page_id: str) -> str:
        raise ToolException(self._ONENOTE_NOT_SUPPORTED)

    # ------------------------------------------------------------------ #
    #  Sharing Links  (Graph API only — delegated access required)       #
    # ------------------------------------------------------------------ #

    _SHARING_LINK_NOT_SUPPORTED = (
        "Reading files from sharing links requires Graph API delegated access. "
        "Provide token + scopes to enable sharing link support."
    )

    def read_file_from_sharing_link(self, sharing_url: str, is_capture_image: bool = False) -> str:
        """Read a file from a SharePoint/OneDrive sharing link.

        Args:
            sharing_url: Full sharing URL (e.g., https://company-my.sharepoint.com/:x:/...)
            is_capture_image: When True and an LLM is configured, embedded images are transcribed.

        Returns:
            Parsed text content of the file

        Raises:
            ToolException: If sharing links are not supported by this backend
        """
        raise ToolException(self._SHARING_LINK_NOT_SUPPORTED)

