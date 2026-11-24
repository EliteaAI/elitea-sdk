"""Page numbers and sections in loader chunk metadata (EL-7355).

Every fixture is generated here (PyMuPDF for PDFs, python-docx for Word files); no document is committed.

Run:
  pytest tests/runtime/langchain/document_loaders/test_loader_sections.py -v
"""

import io

import pymupdf
import pytest
import tiktoken
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
    page_metadata,
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
    """Metadata of the first chunk that starts on each page (a page cut at a top section has several)."""
    found = {}
    for doc in docs:
        found.setdefault(doc.metadata["page_number"], doc.metadata)
    return found


def _spans(docs):
    return [(d.metadata["page_number_from"], d.metadata["page_number_to"]) for d in docs]


@pytest.fixture
def text_pages(monkeypatch, tmp_path):
    """Load page texts through the PDF loader without building a PDF (the outline is then empty)."""
    def load(pages, **kwargs):
        class FakePages:
            def __init__(self, **_kwargs):
                pass

            def load(self):
                return [Document(page_content=text, metadata={"page": i, "source": "fake.pdf"})
                        for i, text in enumerate(pages)]

        monkeypatch.setattr("elitea_sdk.runtime.langchain.document_loaders.EliteAPDFLoader.PyPDFium2Loader",
                            FakePages)
        return EliteAPDFLoader(file_path=str(tmp_path / "missing.pdf"), **kwargs).load()
    return load


def _two_sections_pdf(tmp_path, toc=None):
    """One page: section 1 at the top, the top-level section 2 in the middle."""
    return _pdf(tmp_path / "two.pdf", [["1 Scope", *_wrap(BODY), "2 Methods", *_wrap(BODY)]], toc)


def _join_pdf(tmp_path):
    """Six pages: page 5 ends with a short part of section 3 and page 6 continues it."""
    pages = [["1 Scope", *_wrap(BODY)], _wrap(BODY), ["2 Methods", *_wrap(BODY)], _wrap(BODY),
             [*_wrap(BODY), "3 Results", "Short tail text."], _wrap(BODY)]
    return _pdf(tmp_path / "join.pdf", pages)


TWO_TOC = [[1, "1 Scope", 1], [1, "2 Methods", 1]]


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
        ["Chapter 5: Review", *_wrap(BODY), "Article 5 Data rules apply to every request", *_wrap(BODY)],
    ]
    return _pdf(tmp_path / "numbered.pdf", pages)


# --------------------------------------------------------------------------- PDF


class TestPdfPages:
    def test_page_number_is_one_based_and_page_is_unchanged(self, outline_pdf):
        docs = _load_pdf(outline_pdf)
        # page 3 is cut at "2 Access control"; the running header of page 2 stays on page 2
        assert [d.metadata["page"] for d in docs] == [0, 1, 2, 2, 3]
        assert [d.metadata["page_number"] for d in docs] == [1, 2, 3, 3, 4]
        assert _spans(docs) == [(1, 1), (2, 2), (3, 3), (3, 3), (4, 4)]

    def test_fifth_viewer_page_has_page_number_five(self, tmp_path):
        path = _pdf(tmp_path / "six.pdf", [_wrap(BODY)] * 6)
        by_page = {d.metadata["page"]: d.metadata for d in _load_pdf(path)}
        assert sorted(by_page) == [0, 1, 2, 3, 4, 5]
        assert by_page[4]["page_number"] == 5
        assert by_page[4]["page_number_from"] == 5
        assert by_page[4]["page_number_to"] == 5

    def test_split_chunks_keep_the_range_of_their_page(self, tmp_path):
        path = _pdf(tmp_path / "split.pdf", [_wrap(BODY * 2)] * 2)
        docs = _load_pdf(path, max_tokens=40)
        assert len(docs) > 2
        assert {d.metadata["page_number"] for d in docs} == {1, 2}
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
        index = next(i for i, d in enumerate(docs) if "2 Access control" in d.page_content)
        # the chunk that contains the heading starts in section 1; the chunks after it are in section 2
        if not docs[index].page_content.lstrip().startswith("2 Access control"):
            assert docs[index].metadata["section_title"] == "Scope"
        assert all(d.metadata["section_title"] == "Access control" for d in docs[index + 1:])
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
        # two agreeing chapters are needed before a chapter chain starts
        assert (meta[5]["section_number"], meta[5]["section_title"]) == ("Chapter 4", "Operations")
        # a lone "Article 5 ..." line in body text is not a section
        assert (meta[6]["section_number"], meta[6]["section_title"]) == ("Chapter 5", "Review")

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
        assert [d.metadata["chunk_id"] for d in docs] == [0, 1, 2, 3, 4]
        assert [d.metadata["page_number"] for d in docs] == [1, 2, 3, 3, 4]
        assert [d.metadata["page_number_from"] for d in docs] == [1, 2, 3, 3, 4]
        assert [d.metadata["page_number_to"] for d in docs] == [1, 2, 3, 3, 4]
        assert [d.metadata["section_title"] for d in docs[2:]] == ["Scope", "Access control", "Access review"]


class TestPdfSectionBounds:
    def test_level_one_bookmark_mid_page_splits_the_page(self, tmp_path):
        docs = _load_pdf(_two_sections_pdf(tmp_path, TWO_TOC))
        assert len(docs) == 2
        assert "2 Methods" not in docs[0].page_content and docs[1].page_content.startswith("2 Methods")

    def test_level_one_numbered_heading_mid_page_splits_the_page(self, tmp_path):
        docs = _load_pdf(_two_sections_pdf(tmp_path))
        assert len(docs) == 2
        assert "2 Methods" not in docs[0].page_content and docs[1].page_content.startswith("2 Methods")

    def test_boundary_inside_one_page_gives_two_chunks_with_the_same_page(self, tmp_path):
        docs = _load_pdf(_two_sections_pdf(tmp_path))
        assert [d.metadata["chunk_id"] for d in docs] == [0, 1]
        assert all(d.metadata["page"] == 0 and d.metadata["page_number"] == 1 for d in docs)
        assert _spans(docs) == [(1, 1), (1, 1)]
        assert _sections_of(docs) == [("1", "Scope"), ("2", "Methods")]

    def test_token_chunks_never_hold_text_of_two_top_sections(self, tmp_path):
        page = ["1 Scope", *_wrap(BODY), "closing zetamark", "2 Methods", "omegamark opens", *_wrap(BODY)]
        docs = _load_pdf(_pdf(tmp_path / "marks.pdf", [page]), max_tokens=40)
        assert len(docs) > 3
        assert not any("zetamark" in d.page_content and "omegamark" in d.page_content for d in docs)
        assert _sections_of([next(d for d in docs if "omegamark" in d.page_content)]) == [("2", "Methods")]

    def test_level_two_heading_does_not_split(self, tmp_path):
        path = _pdf(tmp_path / "l2.pdf", [["1 Scope", *_wrap(BODY), "1.1 Detail", *_wrap(BODY)]])
        assert len(_load_pdf(path)) == 1

    def test_text_before_a_top_of_page_heading_joins_the_previous_page(self, tmp_path):
        pages = [["1 Scope", *_wrap(BODY)], ["Closing words of one", "and the last ones", "2 Methods", *_wrap(BODY)]]
        docs = _load_pdf(_pdf(tmp_path / "header.pdf", pages))
        assert len(docs) == 2
        assert "Closing words" in docs[0].page_content and "Closing words" not in docs[1].page_content
        assert docs[1].page_content.startswith("2 Methods")
        assert _sections_of(docs) == [("1", "Scope"), ("2", "Methods")]
        assert _spans(docs) == [(1, 2), (2, 2)]

    @pytest.mark.parametrize("first_page", [["1 Scope", *_wrap(BODY)], ["Corporate Policy v2", "1 Scope", *_wrap(BODY)]])
    def test_running_header_line_stays_with_its_chapter(self, tmp_path, first_page):
        pages = [first_page, ["Page 2 of 2", "2 Methods", *_wrap(BODY)]]
        docs = _load_pdf(_pdf(tmp_path / "running.pdf", pages))
        assert [d.page_content.splitlines()[0] for d in docs] == [first_page[0], "Page 2 of 2"]
        assert _spans(docs) == [(1, 1), (2, 2)]
        assert _sections_of(docs) == [("1", "Scope"), ("2", "Methods")]

    def test_level_two_heading_at_the_top_of_a_cut_page_gets_its_own_section(self, text_pages):
        for top in ("Page 2 of 2\n1.1 Access review", " 1.1 Access review"):
            docs = text_pages([f"1 Scope\n{BODY}", f"{top}\n{BODY}\n2 Methods\n{BODY}"])
            assert _sections_of(docs) == [("1", "Scope"), ("1.1", "Access review"), ("2", "Methods")]

    def test_outline_cuts_are_bounded_per_page(self, tmp_path):
        lines = [line for n in range(1, MAX_HEADINGS_PER_PAGE + 6) for line in (f"{n} Topic", "Short text")]
        toc = [[1, f"{n} Topic", 1] for n in range(1, MAX_HEADINGS_PER_PAGE + 6)]
        assert len(_load_pdf(_pdf(tmp_path / "many.pdf", [lines], toc))) == MAX_HEADINGS_PER_PAGE

    def test_no_headings_keeps_one_chunk_per_page(self, tmp_path):
        docs = _load_pdf(_pdf(tmp_path / "short.pdf", [["Short page one."], ["Short page two."]]))
        assert [d.metadata["chunk_id"] for d in docs] == [0, 1]
        assert all(d.metadata[key] == value for i, d in enumerate(docs) for key, value in page_metadata(i).items())

    def test_lone_part_before_a_top_of_page_heading_keeps_the_previous_section(self, tmp_path):
        docs = _load_pdf(_pdf(tmp_path / "lone.pdf", [["Intro line", "and one more", "1 Scope", *_wrap(BODY)]]))
        assert len(docs) == 2
        assert docs[0].page_content.startswith("Intro line")
        assert not any(key in docs[0].metadata for key in SECTION_KEYS)
        assert _sections_of(docs)[1] == ("1", "Scope")

    @pytest.mark.parametrize("lead", ["  ", "\n", " \n  "])
    def test_cut_page_that_opens_with_an_indented_heading_gets_that_heading(self, text_pages, lead):
        docs = text_pages([f"1 Scope\n{BODY}", f"{lead}2 Methods\nshort text\n3 Results\n{BODY}"])
        assert [d.page_content.strip().splitlines()[0] for d in docs] == ["1 Scope", "2 Methods", "3 Results"]
        assert _sections_of(docs) == [("1", "Scope"), ("2", "Methods"), ("3", "Results")]


class TestPageRange:
    def test_range_of_text_over_pages_five_and_six(self):
        assert page_metadata(4, 5) == {"page_number": 5, "page_number_from": 5, "page_number_to": 6}
        assert page_metadata(4) == {"page_number": 5, "page_number_from": 5, "page_number_to": 5}


class TestPdfPageJoin:
    def test_short_section_tail_joins_the_next_page(self, tmp_path):
        docs = list(process_content_by_type(_join_pdf(tmp_path).read_bytes(), "doc.pdf"))
        assert len(docs) == 6
        assert _spans(docs) == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (5, 6)]
        meta = docs[5].metadata
        assert (meta["page"], meta["page_number"]) == (4, 5)
        assert (meta["section_number"], meta["section_title"]) == ("3", "Results")
        assert "Short tail text." in docs[5].page_content and BODY[:40] in docs[5].page_content

    def test_short_part_at_page_start_joins_the_previous_page(self, tmp_path):
        pages = [["1 Scope", *_wrap(BODY)], ["end of one", "more of one", "2 Methods", *_wrap(BODY)]]
        path = _pdf(tmp_path / "start.pdf", pages)
        docs = _load_pdf(path)
        assert _spans(docs) == [(1, 2), (2, 2)]
        assert [d.metadata["page_number"] for d in docs] == [1, 2]

    def test_no_join_across_a_top_section_start(self, tmp_path):
        pages = [["1 Scope", *_wrap(BODY), "2 Methods", "Short tail text."], ["3 Results", *_wrap(BODY)]]
        assert _spans(_load_pdf(_pdf(tmp_path / "top.pdf", pages))) == [(1, 1), (1, 1), (2, 2)]

    def test_long_parts_do_not_join(self, tmp_path):
        pages = [["1 Scope", *_wrap(BODY)], [*_wrap(BODY), "2 Methods", *_wrap(BODY)]]
        assert _spans(_load_pdf(_pdf(tmp_path / "long.pdf", pages))) == [(1, 1), (2, 2), (2, 2)]

    def test_forward_join_wins_when_two_short_parts_share_a_page(self, tmp_path):
        pages = [["1 Scope", *_wrap(BODY), "2 Methods", "Short tail text."], _wrap(BODY),
                 ["Header line", "second line", "3 Results", *_wrap(BODY)]]
        docs = _load_pdf(_pdf(tmp_path / "forward.pdf", pages))
        assert _spans(docs) == [(1, 1), (1, 2), (3, 3), (3, 3)]
        assert docs[2].page_content.startswith("Header line")
        assert _sections_of(docs) == [("1", "Scope"), ("2", "Methods"), ("2", "Methods"), ("3", "Results")]

    def test_token_chunk_over_a_page_break_gets_both_pages(self, tmp_path):
        pages = [["1 Scope", *_wrap(BODY), "2 Methods", "Short tail text."], _wrap(BODY)]
        docs = _load_pdf(_pdf(tmp_path / "token.pdf", pages), max_tokens=40)
        assert (1, 2) in _spans(docs)
        later = [d for d in docs if d.metadata["page_number_from"] == 2]
        assert later and all(d.metadata["page_number_to"] == 2 for d in later)
        assert set(_sections_of(later)) == {("2", "Methods")}
        assert all("start_index" not in d.metadata for d in docs)

    def test_chunk_that_starts_at_the_added_newline_belongs_to_the_next_page(self, text_pages):
        tail = "2 Methods\nShort tail text"
        size = len(tiktoken.get_encoding("cl100k_base").encode(tail))
        docs = text_pages([f"1 Scope\n{BODY}\n{tail}", BODY], max_tokens=size)
        chunk = next(d for d in docs if d.page_content.startswith("\n"))
        assert (chunk.metadata["page_number_from"], chunk.metadata["page_number_to"]) == (2, 2)
        assert next(d for d in docs if d.page_content == tail).metadata["page_number_to"] == 1

    def test_literal_special_token_text_is_split_like_any_text(self, text_pages):
        docs = text_pages([f"1 Scope\n{BODY} <|endoftext|> {BODY}"], max_tokens=40)
        assert "<|endoftext|>" in "".join(d.page_content for d in docs)

    def test_split_inside_a_multi_byte_character_keeps_page_numbers_in_order(self, text_pages):
        run = "😀" * 12
        docs = text_pages([f"1 Scope\n{BODY}\n2 Methods\n{run}", f"{run} {run}"], max_tokens=4)
        numbers = [d.metadata["page_number_from"] for d in docs]
        assert any("�" in d.page_content for d in docs)
        assert numbers == sorted(numbers) and numbers[-1] == 2


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

    @pytest.mark.parametrize("line,number,title,level", [
        ("Appendix D", "Appendix D", None, 1),
        ("Appendix D.2 Access review", "Appendix D.2", "Access review", 2),
        ("Annex A.1.2 Logs", "Annex A.1.2", "Logs", 3),
        ("Appendix C.1 Forms", "Appendix C.1", "Forms", 2),
        ("Annex I.2 Index", "Annex I.2", "Index", 2),
    ])
    def test_lettered_keyword_numbers_keep_their_dotted_parts(self, line, number, title, level):
        heading = heading_candidate(line, 0, 0)
        assert (heading.number, heading.title, heading.level) == (number, title, level)
        assert split_heading_text(line) == (number, title)
        outline = outline_headings([[level, line, 1]], [f"{line}\nbody"])
        assert (outline[0].number, outline[0].title, outline[0].level) == (number, title, level)

    @pytest.mark.parametrize("page,title,expected", [
        ("See 2 Methods for details.\n2 Methods\nbody", "2 Methods", "2 Methods"),
        ("Our inaccessible door\nAccess\nbody", "access", "Access"),
        ("Preface\n  Terms and\nconditions of use\nbody", "Terms and conditions of use", "Terms and"),
        ("İ intro line\nscope of work\nbody", "Scope of work", "scope of work"),
        ("Preface\nSee 2 Methods for details.", "2 Methods", None),
        ("The word Access is inside.", "Access", None),
    ])
    def test_outline_title_is_found_only_at_the_start_of_a_line(self, page, title, expected):
        (heading,) = outline_headings([[1, title, 1]], [page])
        assert heading.offset == (page.rindex(expected) if expected else 0)

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

    def test_top_sections_never_share_a_chunk(self, tmp_path):
        path = _docx(tmp_path / "tops.docx", [(1, "1 Alpha"), (0, BODY * 3 + " alphaend"),
                                              (1, "2 Beta"), (0, "betastart " + BODY * 3)])
        docs = EliteADocxMammothLoader(file_path=str(path), max_tokens=40).load()
        assert len(docs) > 4
        assert not any("alphaend" in d.page_content and "betastart" in d.page_content for d in docs)
        assert set(_sections_of(docs)) == {("1", "Alpha"), ("2", "Beta")}


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


class TestReviewFixes:
    def test_short_contents_block_is_not_a_section(self):
        pages = ["Contents\n1 Introduction\n2 Scope\n3 Roles\nPreface text follows here.",
                 "1 Introduction\nbody\n2 Scope\nbody"]
        found = [(h.page, h.number) for h in detect_headings(pages)]
        assert found == [(1, "1"), (1, "2")]

    def test_mid_page_chunk_gets_no_slack(self):
        index = sections.SectionIndex([sections.Heading(0, 0, "1", "A", (1,)), sections.Heading(0, 500, "2", "B", (2,))])
        assert index.section_at(0, 0)["section_number"] == "1"
        assert index.section_at(0, 400)["section_number"] == "1"   # heading 100 chars later does not count
        assert index.section_at(0, 500)["section_number"] == "2"

    def test_heading_text_starting_with_digits_still_gets_its_number(self, tmp_path):
        path = _docx(tmp_path / "digits.docx", [(1, "Scope"), (1, "2FA rollout")], auto_numbered=True)
        docs = EliteADocxMammothLoader(file_path=str(path), max_tokens=512).load()
        assert _sections_of(docs) == [("1", "Scope"), ("2", "2FA rollout")]
