"""#7751: toolkits read downloaded HTML/XML bytes through parse_file_content(file_name=..., file_content=...).

Before the fix the loader was built with file_path=None and every read failed with
"No such file or directory: 'None'".
"""

import os
import tempfile

import pytest

from elitea_sdk.runtime.langchain.document_loaders.EliteAHTMLLoader import EliteAHTMLLoader, EliteAXMLLoader
from elitea_sdk.tools.utils.content_parser import parse_file_content
from elitea_sdk.tools.utils.file_metadata import get_file_metadata

pytest.importorskip("unstructured.partition.html")

HTML = b"<html><body><h1>Title</h1><p>Hello &amp; bye</p></body></html>"
XML = b'<?xml version="1.0"?><root><item>One</item><item>Two</item></root>'


@pytest.mark.parametrize("file_name, content, expected", [
    ("page.html", HTML, "Title\n\nHello & bye"),
    ("page.htm", HTML, "Title\n\nHello & bye"),
    ("data.xml", XML, "One\n\nTwo"),
])
def test_parse_file_content_reads_bytes(file_name, content, expected):
    assert parse_file_content(file_name=file_name, file_content=content) == expected


@pytest.mark.parametrize("file_name, content", [("page.html", HTML), ("data.xml", XML)])
def test_parse_file_content_by_path_still_works(file_name, content):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, file_name)
        with open(path, "wb") as f:
            f.write(content)

        from_path = parse_file_content(file_path=path)

    assert from_path == parse_file_content(file_name=file_name, file_content=content)


@pytest.mark.parametrize("loader_cls, file_name, content", [
    (EliteAHTMLLoader, "page.html", HTML),
    (EliteAXMLLoader, "data.xml", XML),
])
def test_bytes_load_cleans_up_and_reports_file_name(loader_cls, file_name, content):
    loader = loader_cls(file_content=content, file_name=file_name)

    docs = loader.load()

    assert docs[0].metadata == {"source": file_name}
    assert loader.file_path is None  # temp file is not kept between loads
    assert loader.load()[0].page_content == docs[0].page_content


@pytest.mark.parametrize("file_name", ["empty.html", "empty.xml"])
def test_empty_bytes_read_as_empty_text(file_name):
    assert parse_file_content(file_name=file_name, file_content=b"") == ""


@pytest.mark.parametrize("file_name, content", [("page.html", HTML), ("data.xml", XML)])
def test_metadata_line_count_matches_read_content(file_name, content):
    text = parse_file_content(file_name=file_name, file_content=content)

    meta = get_file_metadata(file_name, file_content=content, file_size=len(content))

    assert meta["total_lines"] == len(text.splitlines())
