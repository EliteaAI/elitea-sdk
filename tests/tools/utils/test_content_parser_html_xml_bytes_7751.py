"""#7751: toolkits read downloaded HTML/XML bytes through parse_file_content(file_name=..., file_content=...).

Before the fix the loader was built with file_path=None and every read failed with
"No such file or directory: 'None'".
"""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

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


def test_str_content_is_read_as_utf8():
    assert parse_file_content(file_name="page.html", file_content=HTML.decode()) == "Title\n\nHello & bye"
    assert parse_file_content(file_name="blank.html", file_content="  ") == ""


def test_failed_temp_write_leaves_no_file(monkeypatch, tmp_path):
    class FullDisk:
        def __init__(self, fd, mode):
            os.close(fd)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, data):
            raise OSError("No space left on device")

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(os, "fdopen", FullDisk)

    with pytest.raises(OSError, match="No space left"):
        EliteAHTMLLoader(file_content=HTML, file_name="page.html").load()
    assert list(tmp_path.iterdir()) == []


def test_overlapping_loads_on_one_instance_use_their_own_temp_file(monkeypatch):
    import threading
    import unstructured.partition.html as html_partition

    real_partition, both_inside, seen = html_partition.partition_html, threading.Barrier(2, timeout=5), []

    def partition_html(filename=None, **kwargs):
        seen.append((filename, loader.file_path))
        both_inside.wait()  # both loads are inside partition at the same time
        return real_partition(filename=filename, **kwargs)

    monkeypatch.setattr(html_partition, "partition_html", partition_html)
    loader = EliteAHTMLLoader(file_content=HTML, file_name="page.html")

    with ThreadPoolExecutor(max_workers=2) as pool:
        texts = list(pool.map(lambda _: loader.load()[0].page_content, range(2)))

    assert texts == ["Title\n\nHello & bye"] * 2
    paths = [path for path, _ in seen]
    assert len(set(paths)) == 2 and None not in paths
    assert [shared for _, shared in seen] == [None, None]  # no load writes its temp path to the shared instance
    assert not any(os.path.exists(path) for path in paths)
