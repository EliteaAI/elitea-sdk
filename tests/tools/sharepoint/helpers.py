"""Doubles shared by the SharePoint change-check tests: a listing backend and a toolkit built on it."""

from elitea_sdk.tools.sharepoint.api_wrapper import SharepointApiWrapper

SITE = "https://tenant.sharepoint.com/sites/x"


def listed_file(name="a.docx", modified="2026-01-01", file_id="f1", link=None):
    return {"Name": name, "Path": f"/sites/x/Lib/{name}", "Created": "c", "Modified": modified,
            "Link": link or f"{SITE}/Lib/{name}", "id": file_id}


class Backend:
    """Backend double: it lists files, and every call that would read file content is recorded.

    The change check must list only, so a test asserts `content_reads` stays empty.
    """

    def __init__(self, files=(), gaps=()):
        self.files = list(files)
        self.gaps = list(gaps)
        self.calls = []
        self.content_reads = []

    def get_files_list(self, folder_name=None, limit_files=100, form_name=None, include_extensions=None,
                       skip_extensions=None, on_file_skipped=None, on_listing_gap=None):
        self.calls.append({"folder": folder_name, "limit": limit_files, "form": form_name,
                           "include": include_extensions, "skip": skip_extensions})
        for reason in self.gaps:
            on_listing_gap(reason)
        return self.files[:limit_files]

    def read_file(self, *args, **kwargs):
        self.content_reads.append(("read_file", args, kwargs))
        return ""

    def read_file_from_url(self, *args, **kwargs):
        self.content_reads.append(("read_file_from_url", args, kwargs))
        return ""

    def read_file_from_sharing_link(self, *args, **kwargs):
        self.content_reads.append(("read_file_from_sharing_link", args, kwargs))
        return ""

    def get_file_content(self, *args, **kwargs):
        self.content_reads.append(("get_file_content", args, kwargs))
        return b""


def toolkit_with(backend, library=None):
    toolkit = SharepointApiWrapper.model_construct(site_url=SITE)
    toolkit._backend = backend
    toolkit._log_tool_event = lambda *a, **k: None
    toolkit._library_join_context = lambda *a, **k: library
    return toolkit


def stored_row(listing_item, **extra):
    """A base row as the indexer stores it for a listed file."""
    return {"metadata": {"id": listing_item.key, "Name": listing_item.name, "Path": listing_item.path,
                         "updated_on": "2026-01-01", **extra}}
