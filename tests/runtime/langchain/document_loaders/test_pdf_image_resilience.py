"""One undecodable embedded image must not abort a whole PDF parse.

PyMuPDF hands back native streams — JBIG2, JPEG2000, CCITT fax — that Pillow often
cannot decode. Those are ordinary content in scanned PDFs, so the page text has to
survive them.
"""
import pymupdf
import pytest

from elitea_sdk.runtime.langchain.document_loaders.EliteAPDFLoader import EliteAPDFLoader

UNDECODABLE = {
    "jbig2": b"\x97JB2\r\n\x1a\n" + b"\x00" * 128,
    "jpeg2000": b"\x00\x00\x00\x0cjP  \r\n\x87\n" + b"\x00" * 128,
    "truncated_png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,
    "empty": b"",
}


class StubLLM:
    def invoke(self, messages):
        class Result:
            content = "described"

        return Result()


@pytest.fixture
def pdf_with_image(tmp_path):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "PAGE TEXT THAT MUST SURVIVE")
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 32, 32))
    pixmap.set_rect(pixmap.irect, (200, 30, 30))
    page.insert_image(pymupdf.Rect(72, 200, 172, 300), pixmap=pixmap)
    path = tmp_path / "doc.pdf"
    document.save(path)
    document.close()
    return str(path)


def load_page(pdf_path, monkeypatch, image_bytes):
    loader = EliteAPDFLoader(file_path=pdf_path, llm=StubLLM(), extract_images=True)
    with pymupdf.open(pdf_path) as report:
        monkeypatch.setattr(
            type(report), "extract_image", lambda self, xref: {"image": image_bytes, "ext": "jb2"}
        )
        page = report.load_page(0)
        return loader.read_pdf_page(report, page, 1)


@pytest.mark.parametrize("label", sorted(UNDECODABLE))
def test_undecodable_image_does_not_abort_the_page(pdf_with_image, monkeypatch, label):
    text = load_page(pdf_with_image, monkeypatch, UNDECODABLE[label])

    assert "PAGE TEXT THAT MUST SURVIVE" in text


def test_undecodable_image_is_logged_not_swallowed(pdf_with_image, monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        load_page(pdf_with_image, monkeypatch, UNDECODABLE["jbig2"])

    assert "Skipping image" in caplog.text


def test_decodable_image_still_produces_a_transcript(pdf_with_image, monkeypatch):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 20, 30)).save(buffer, format="PNG")

    text = load_page(pdf_with_image, monkeypatch, buffer.getvalue())

    assert "Image Transcript" in text
    assert "PAGE TEXT THAT MUST SURVIVE" in text
