"""Page numbers and sections in loader chunk metadata (EL-7355).

Every fixture is generated here (PyMuPDF for PDFs, python-docx for Word files); no document is committed.

Run:
  pytest tests/runtime/langchain/document_loaders/test_loader_sections.py -v
"""

import io

import pymupdf
import pytest
from docx import Document as DocxDocument
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from langchain_core.documents import Document

from elitea_sdk.runtime.langchain.document_loaders import sections
from elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader import EliteADocxMammothLoader
from elitea_sdk.runtime.langchain.document_loaders.EliteAPDFLoader import EliteAPDFLoader
from elitea_sdk.runtime.langchain.document_loaders.sections import (
    MAX_HEADINGS_PER_PAGE,
    MAX_SECTION_DEPTH,
    MAX_SECTION_TITLE_CHARS,
    detect_headings,
    follows,
    heading_candidate,
    outline_headings,
    split_heading_text,
)
from elitea_sdk.tools.chunkers.sematic.markdown_chunker import markdown_chunker
from elitea_sdk.tools.chunkers.sematic.text_chunker import text_chunker
from elitea_sdk.tools.utils.content_parser import process_content_by_type

SECTION_KEYS = ("section_number", "section_title")
BODY = ("The team reviews the list of people who can open the shared folders and records the result. " * 3).strip()


# --------------------------------------------------------------------------- fixtures


def _pdf(path, pages, toc=None):
    """Write a PDF whose page ``i`` holds the lines of ``pages[i]``; ``toc`` is PyMuPDF's [level, title, page]."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 60
        for line in lines:
            page.insert_text((50, y), line, fontsize=9)
            y += 14
    if toc:
        doc.set_toc(toc)
    doc.save(str(path))
    doc.close()
    return path


def _load_pdf(path, **kwargs):
    return EliteAPDFLoader(file_path=str(path), **kwargs).load()


def _by_page(docs):
    return {d.metadata["page_number"]: d.metadata for d in docs}


def _wrap(text, width=90):
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


@pytest.fixture
def outline_pdf(tmp_path):
    pages = [
        ["Page 3 of 10", "Handbook", *_wrap(BODY)],
        ["Page 4 of 10", "1 Scope", *_wrap(BODY)],
        ["Page 5 of 10", *_wrap(BODY), "2 Access control", *_wrap(BODY)],
        ["Page 6 of 10", "2.1 Access review", *_wrap(BODY)],
    ]
    toc = [[1, "1 Scope", 2], [1, "2 Access control", 3], [2, "2.1 Access review", 4]]
    return _pdf(tmp_path / "outline.pdf", pages, toc)


@pytest.fixture
def numbered_pdf(tmp_path):
    contents = ["Contents"] + [f"{n} Topic number {n}" for n in range(1, MAX_HEADINGS_PER_PAGE + 3)]
    pages = [
        contents,
        ["Document title", "1 Introduction", *_wrap(BODY), "1. ensure that every request is logged",
         "12.03.2024 Approved by the board"],
        [*_wrap(BODY * 2), "2 Roles and duties", *_wrap(BODY)],
        ["2.1 Owner duties", *_wrap(BODY), "7 Reasons to stop", *_wrap(BODY)],
        ["Chapter 4: Operations", *_wrap(BODY)],
    ]
    return _pdf(tmp_path / "numbered.pdf", pages)


# --------------------------------------------------------------------------- PDF


class TestPdfPages:
    def test_page_number_is_one_based_and_page_is_unchanged(self, outline_pdf):
        docs = _load_pdf(outline_pdf)
        assert [d.metadata["page"] for d in docs] == [0, 1, 2, 3]
        assert [d.metadata["page_number"] for d in docs] == [1, 2, 3, 4]
        assert all(d.metadata["page_number_from"] == d.metadata["page_number_to"] == d.metadata["page_number"]
                   for d in docs)

    def test_printed_page_labels_do_not_change_page_number(self, outline_pdf):
        docs = _load_pdf(outline_pdf)
        assert "Page 5 of 10" in docs[2].page_content
        assert docs[2].metadata["page_number"] == 3


class TestPdfOutline:
    def test_section_follows_the_bookmark_above_the_chunk(self, outline_pdf):
        meta = _by_page(_load_pdf(outline_pdf))
        assert not any(k in meta[1] for k in SECTION_KEYS)  # before the first bookmark
        assert (meta[2]["section_number"], meta[2]["section_title"]) == ("1", "Scope")
        # page 3 starts with the tail of section 1; its bookmark is further down the page
        assert (meta[3]["section_number"], meta[3]["section_title"]) == ("1", "Scope")
        assert (meta[4]["section_number"], meta[4]["section_title"]) == ("2.1", "Access review")

    def test_split_chunks_get_the_section_of_their_own_text(self, outline_pdf):
        docs = [d for d in _load_pdf(outline_pdf, max_tokens=40) if d.metadata["page_number"] == 3]
        assert len(docs) > 2
        assert docs[0].metadata["section_title"] == "Scope"
        after = [d for d in docs if d.page_content.lstrip().startswith("2 Access control")
                 or "2 Access control" in d.page_content[:150]]
        assert after and after[-1].metadata["section_title"] == "Access control"
        assert docs[-1].metadata["section_number"] == "2"
        assert all("start_index" not in d.metadata for d in docs)

    def test_outline_wins_over_text_headings(self, tmp_path):
        path = _pdf(tmp_path / "both.pdf", [["1 Text heading", *_wrap(BODY)]], toc=[[1, "Bookmark title", 1]])
        meta = _load_pdf(path)[0].metadata
        assert meta["section_title"] == "Bookmark title"
        assert "section_number" not in meta


class TestPdfHeadings:
    def test_numbered_headings_without_outline(self, numbered_pdf):
        meta = _by_page(_load_pdf(numbered_pdf))
        assert not any(k in meta[1] for k in SECTION_KEYS)  # a contents page is not a section
        assert (meta[2]["section_number"], meta[2]["section_title"]) == ("1", "Introduction")
        assert (meta[3]["section_number"], meta[3]["section_title"]) == ("1", "Introduction")
        # "7 Reasons to stop" does not continue 2.1, so it is not a heading
        assert (meta[4]["section_number"], meta[4]["section_title"]) == ("2.1", "Owner duties")
        assert (meta[5]["section_number"], meta[5]["section_title"]) == ("Chapter 4", "Operations")

    def test_split_chunk_after_heading_gets_that_heading(self, numbered_pdf):
        docs = [d for d in _load_pdf(numbered_pdf, max_tokens=40) if d.metadata["page_number"] == 3]
        assert docs[0].metadata["section_title"] == "Introduction"
        assert docs[-1].metadata["section_title"] == "Roles and duties"

    def test_no_headings_means_no_section_fields(self, tmp_path):
        path = _pdf(tmp_path / "plain.pdf", [_wrap(BODY), _wrap(BODY)])
        docs = _load_pdf(path)
        assert [d.metadata["page_number"] for d in docs] == [1, 2]
        assert all(k not in d.metadata for d in docs for k in SECTION_KEYS)

    def test_production_path_carries_the_fields(self, outline_pdf):
        docs = list(process_content_by_type(outline_pdf.read_bytes(), "handbook.pdf"))
        assert [d.metadata["page_number"] for d in docs] == [1, 2, 3, 4]
        assert docs[3].metadata["section_title"] == "Access review"


# --------------------------------------------------------------------------- heuristic and bounds


class TestHeuristic:
    @pytest.mark.parametrize("line", [
        "1. ensure that every request is logged",     # list item, lower case
        "12.03.2024 Approved by the board",           # date
        "4.2 Access review ........ 7",                # contents line
        "4.2 Access review 7",                        # contents line, no leader
        "3 The team reviews the list of people who can open the shared folders and records it.",
        "2020 Annual report",
        "Section 4 describes the",
        "4.2",
    ])
    def test_not_a_heading(self, line):
        assert heading_candidate(line, 0, 0) is None

    @pytest.mark.parametrize("line,number,title", [
        ("4.2 Access review", "4.2", "Access review"),
        ("4.2. Hozzáférés felülvizsgálata", "4.2", "Hozzáférés felülvizsgálata"),
        ("3.1.2 Logs", "3.1.2", "Logs"),
        ("Chapter 4", "Chapter 4", None),
        ("Appendix B - Forms", "Appendix B", "Forms"),
    ])
    def test_heading(self, line, number, title):
        heading = heading_candidate(line, 0, 0)
        assert heading is not None and (heading.number, heading.title) == (number, title)

    @pytest.mark.parametrize("previous,number,expected", [
        (None, (1,), True), (None, (1, 1), True), (None, (5, 2), False),
        ((4, 1), (4, 2), True), ((4, 1), (4, 1, 1), True), ((4, 1), (5,), True), ((4, 1), (5, 1), True),
        ((4, 1), (4, 1), False), ((4, 1), (9,), False), ((4, 1), (3,), False), ((4, 1), (4, 3, 2), False),
    ])
    def test_numbering_continuity(self, previous, number, expected):
        assert follows(previous, number) is expected

    def test_two_agreeing_headings_restart_the_numbering(self):
        pages = ["5 Late start\nbody", "6 Next part\nbody", "1 Restart\nbody\n1.1 Child\nbody"]
        found = [(h.page, h.number) for h in detect_headings(pages)]
        assert found == [(0, "5"), (1, "6"), (2, "1"), (2, "1.1")]

    def test_page_with_too_many_headings_is_ignored(self):
        page = "\n".join(f"{n} Item {n}" for n in range(1, MAX_HEADINGS_PER_PAGE + 2))
        assert detect_headings([page]) == []

    def test_outline_depth_and_title_are_bounded(self):
        toc = [[MAX_SECTION_DEPTH + 1, "Too deep", 1], [1, "x" * 500, 1], [1, "Missing page", 9]]
        headings = outline_headings(toc, ["text"])
        assert len(headings) == 1 and len(headings[0].title) == MAX_SECTION_TITLE_CHARS

    def test_outline_entry_count_is_bounded(self, monkeypatch):
        monkeypatch.setattr(sections, "MAX_OUTLINE_ENTRIES", 3)
        assert len(outline_headings([[1, f"Entry {i}", 1] for i in range(10)], ["text"])) == 3

    def test_split_heading_text(self):
        assert split_heading_text("4.2 Access review") == ("4.2", "Access review")
        assert split_heading_text("Overview") == (None, "Overview")
        assert split_heading_text("  ") == (None, None)


# --------------------------------------------------------------------------- DOCX


_NUMBERING = (
    '<w:abstractNum {ns} w:abstractNumId="90">'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:pStyle w:val="Heading1"/>'
    '<w:lvlText w:val="%1."/></w:lvl>'
    '<w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:pStyle w:val="Heading2"/>'
    '<w:lvlText w:val="%1.%2"/></w:lvl>'
    '<w:lvl w:ilvl="2"><w:start w:val="1"/><w:numFmt w:val="lowerLetter"/><w:pStyle w:val="Heading3"/>'
    '<w:lvlText w:val="%3)"/></w:lvl>'
    '</w:abstractNum>'
)


def _docx(path, headings, auto_numbered=False):
    doc = DocxDocument()
    if auto_numbered:
        numbering = doc.part.numbering_part.element
        numbering.insert(0, parse_xml(_NUMBERING.format(ns=nsdecls("w"))))
        numbering.append(parse_xml(f'<w:num {nsdecls("w")} w:numId="90"><w:abstractNumId w:val="90"/></w:num>'))
        for level in (1, 2, 3):
            p_pr = doc.styles[f"Heading {level}"].element.get_or_add_pPr()
            p_pr.append(parse_xml(f'<w:numPr {nsdecls("w")}><w:numId w:val="90"/></w:numPr>'))
    for level, text in headings:
        if level:
            doc.add_heading(text, level=level)
        else:
            doc.add_paragraph(text)
    doc.save(str(path))
    return path


def _sections_of(docs):
    return [(d.metadata.get("section_number"), d.metadata.get("section_title")) for d in docs]


class TestDocx:
    def test_nested_headings_with_written_numbers(self, tmp_path):
        path = _docx(tmp_path / "nested.docx", [
            (0, "Preface text before any heading. " * 5),
            (1, "4 Access control"), (0, BODY),
            (2, "4.2 Access review"), (0, BODY),
            (3, "4.2.1 Review records"), (0, BODY),
            (2, "Exceptions"), (0, BODY),
        ])
        docs = EliteADocxMammothLoader(file_path=str(path), max_tokens=512).load()
        assert _sections_of(docs) == [
            (None, None), ("4", "Access control"), ("4.2", "Access review"), ("4.2.1", "Review records"),
            (None, "Exceptions"),
        ]
        assert all("page_number" not in d.metadata for d in docs)  # Word has no fixed pages

    def test_automatic_heading_numbers(self, tmp_path):
        path = _docx(tmp_path / "auto.docx", [
            (1, "Scope"), (0, BODY),
            (1, "Access control"), (0, BODY),
            (2, "Access review"), (0, BODY),
            (2, "Exceptions"), (0, BODY),
            (3, "Lettered level"), (0, BODY),
        ], auto_numbered=True)
        docs = EliteADocxMammothLoader(file_path=str(path), max_tokens=512).load()
        assert _sections_of(docs) == [
            ("1", "Scope"), ("2", "Access control"), ("2.1", "Access review"), ("2.2", "Exceptions"),
            (None, "Lettered level"),  # a lettered list level is not rendered
        ]
        assert "2.1 Access review" in docs[2].page_content

    def test_section_split_by_token_budget_keeps_the_section(self, tmp_path):
        path = _docx(tmp_path / "long.docx", [(1, "5 Operations"), (0, BODY * 10)])
        docs = EliteADocxMammothLoader(file_path=str(path), max_tokens=40).load()
        assert len(docs) > 2
        assert set(_sections_of(docs)) == {("5", "Operations")}


# --------------------------------------------------------------------------- chunkers


class TestChunkersPropagate:
    META = {"page": 6, "page_number": 7, "page_number_from": 7, "page_number_to": 7,
            "section_number": "4.2", "section_title": "Access review"}

    def _doc(self):
        return Document(page_content="# Heading\n\n" + BODY * 20, metadata=dict(self.META))

    @pytest.mark.parametrize("chunker", [text_chunker, markdown_chunker])
    def test_every_chunk_keeps_page_and_section(self, chunker):
        chunks = list(chunker(iter([self._doc()]), {"max_tokens": 50}))
        assert len(chunks) > 2
        for chunk in chunks:
            assert {k: chunk.metadata[k] for k in self.META} == self.META

    def test_unknown_fields_stay_absent(self):
        chunks = list(text_chunker(iter([Document(page_content=BODY * 5, metadata={"page": 0})]), {"max_tokens": 50}))
        assert all(k not in c.metadata for c in chunks for k in (*SECTION_KEYS, "page_number"))


def test_docx_auto_number_skips_overridden_lists(tmp_path):
    from elitea_sdk.runtime.langchain.document_loaders.docx_headings import number_headings
    path = _docx(tmp_path / "override.docx", [(1, "Scope")], auto_numbered=True)
    doc = DocxDocument(str(path))
    num = doc.part.numbering_part.element.findall(qn("w:num"))[-1]
    num.append(parse_xml(f'<w:lvlOverride {nsdecls("w")} w:ilvl="0"><w:startOverride w:val="5"/></w:lvlOverride>'))
    assert number_headings(doc) == 0
    assert doc.paragraphs[0].text == "Scope"
    buffer = io.BytesIO()
    doc.save(buffer)
