import logging
from dataclasses import dataclass
from itertools import pairwise
from typing import Optional

import pymupdf
import fitz
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

from .ImageParser import ImageParser
from .sections import SHORT_PART_CHARS, SectionIndex, detect_headings, outline_headings, page_metadata
from .utils import perform_llm_prediction_for_image_bytes, create_temp_file
from ..tools.utils import encode_image_bytes_for_llm
from langchain_core.tools import ToolException

logger = logging.getLogger(__name__)


@dataclass
class _Part:
    """The whole text of a page, or the stretch of it between two top-level headings."""

    doc: Document                    # the page that holds the first character
    start: int                       # offset of that character in the page text
    text: str
    cut: bool                        # the page was cut at a top-level heading
    opens: bool = False              # starts with a top-level heading (only the first part of a page can)
    tail: Optional['_Part'] = None   # the first part of the next page, joined behind this one
    absorbed: bool = False           # joined behind the part before it

    @property
    def page(self) -> int:
        return self.doc.metadata['page']

    def joined_text(self) -> str:
        return self.text if self.tail is None else f'{self.text}\n{self.tail.text}'

    def locate(self, offset: int) -> tuple['_Part', int]:
        """The piece (this part or the joined one) and the page offset of one character of the joined text."""
        if self.tail is not None and offset > len(self.text):
            return self.tail, self.tail.start + offset - len(self.text) - 1
        return self, self.start + offset


def _cut_pages(docs, sections: SectionIndex) -> list[list[_Part]]:
    """Cut each page where a top-level heading starts after some text, so a part never holds two top sections."""
    pages = []
    for doc in docs:
        text = doc.page_content or ''
        starts = sections.top_starts(doc.metadata['page'])
        cuts = [offset for offset in starts if text[:offset].strip()]
        bounds = [0, *cuts, len(text)]
        parts = [_Part(doc, low, text[low:high], bool(cuts)) for low, high in pairwise(bounds)]
        parts[0].opens = len(cuts) < len(starts)
        pages.append(parts)
    return pages


def _join_short_parts(pages: list[list[_Part]]) -> None:
    """Join the last part of a page with the first part of the next one when a cut made one of them short.

    The first part of a page never starts a top section when it joins, so a join stays in one top section.
    Pages go in order and a part joins once, so the forward join wins when a whole page sits between two
    short parts.
    """
    def short(part):
        return part.cut and len(part.text.strip()) < SHORT_PART_CHARS

    for before, after in pairwise(pages):
        last, first = before[-1], after[0]
        if (not last.absorbed and not first.opens and last.page + 1 == first.page
                and last.text.strip() and first.text.strip() and (short(last) or short(first))):
            last.tail, first.absorbed = first, True


class EliteAPDFLoader:

    @classmethod
    def get_file_metadata(cls, *, filename: str,
                          file_content=None,
                          file_size=None) -> dict:
        """Report total page count and advertise page_number for read_file (PRE-2 #5433).

        page_number is a first-class read_file arg, so it goes under
        first_class_params (like DOCX's is_capture_image).
        """
        total_pages = 0
        if file_content:
            try:
                with pymupdf.open(stream=file_content, filetype="pdf") as report:
                    total_pages = report.page_count
            except Exception as e:  # pylint: disable=broad-except
                # Encrypted/corrupt PDF — report 0 pages rather than fail.
                logger.warning("Failed to read PDF page count for %s: %s",
                               filename, e)

        page_number_desc = (
            "integer (1-indexed) — single page to read. "
            + (f"Valid range 1..{total_pages}. " if total_pages else "")
            + "Omit to read the whole document."
        )
        instruction = {
            "first_class_params": {
                "page_number": page_number_desc,
            },
            "notes": (
                "For large PDFs, pass page_number to read one page at a time "
                "and keep memory and tokens bounded."
            ),
        }
        return {
            "unit": "pages",
            "total_pages": total_pages,
            "instruction_for_readFile": instruction,
        }

    def __init__(self, **kwargs):
        if kwargs.get('file_path'):
            self.file_path = kwargs.get('file_path')
        elif kwargs.get('file_content'):
            self.file_content = kwargs.get('file_content')
        else:
            raise ToolException("'file_path' or 'file_content' parameter should be provided.")
        self.password = kwargs.get('password', None)
        self.page_number = kwargs.get('page_number', None)
        self.extract_images = kwargs.get('extract_images', False)
        self.llm = kwargs.get('llm', None)
        self.prompt = kwargs.get('prompt', "Describe image")
        self.headers = kwargs.get('headers', None)
        self.extraction_mode = kwargs.get('extraction_mode', "plain")
        self.extraction_kwargs = kwargs.get('extraction_kwargs', None)
        self.max_tokens = kwargs.get('max_tokens', None)
        self.image_cache = kwargs.get('image_cache')

    def get_content(self):
        if hasattr(self, 'file_path'):
            with pymupdf.open(filename=self.file_path, filetype="pdf") as report:
                return self.parse_report(report)
        else:
            with pymupdf.open(stream=self.file_content, filetype="pdf") as report:
                return self.parse_report(report)

    def parse_report(self, report):
        text_content = ''
        if self.page_number is not None:
            page = report.load_page(self.page_number - 1)
            text_content += self.read_pdf_page(report, page, self.page_number)
        else:
            for index, page in enumerate(report, start=1):
                text_content += self.read_pdf_page(report, page, index)

        return text_content

    def read_pdf_page(self, report, page, index):
        # Extract text in block format (to more accurately match hyperlinks to text)
        text_blocks = page.get_text("blocks")  # Returns a list of text blocks
        words = page.get_text("words")  # Returns words with their coordinates

        # Extract hyperlinks
        links = page.get_links()

        # Create a list to store the modified text
        modified_text = []

        for block in text_blocks:
            block_rect = fitz.Rect(block[:4])  # Coordinates of the text block
            block_text = block[4]  # The actual text of the block

            # Check if there are hyperlinks intersecting with this text block
            for link in links:
                if "uri" in link:  # Ensure this is a hyperlink
                    link_rect = link["from"]  # Coordinates of the hyperlink area
                    link_uri = link["uri"]  # The URL of the hyperlink

                    # Expand the hyperlink area slightly to account for inaccuracies
                    link_rect = fitz.Rect(
                        link_rect.x0 - 1, link_rect.y0 - 1, link_rect.x1 + 1, link_rect.y1 + 1
                    )

                    # Find words that are inside the hyperlink area
                    link_text = []
                    for word in words:
                        word_rect = fitz.Rect(word[:4])  # Coordinates of the word
                        word_text = word[4]

                        # Check if the word rectangle is fully inside the hyperlink rectangle
                        if link_rect.contains(word_rect):
                            link_text.append(word_text)
                        # If the word partially intersects, check vertical alignment
                        elif link_rect.intersects(word_rect):
                            # Condition: The word must be on the same line as the hyperlink
                            if abs(link_rect.y0 - word_rect.y0) < 2 and abs(link_rect.y1 - word_rect.y1) < 2:
                                link_text.append(word_text)

                    # Format the hyperlink in Markdown
                    full_text = " ".join(link_text) if link_text else "No text"
                    hyperlink = f"[{full_text}]({link_uri})"

                    # Replace the hyperlink text in the block with the formatted hyperlink
                    block_text = block_text.replace(full_text, hyperlink)

            # Add the processed text block to the result
            modified_text.append(block_text)

        # Combine all text blocks into the final text for the page
        text_content = f'Page: {index}\n' + "\n".join(modified_text)

        if self.extract_images:
            images = page.get_images(full=True)
            source = getattr(self, 'file_path', None) or 'pdf'
            for i, img in enumerate(images):
                xref = img[0]
                try:
                    base_image = report.extract_image(xref)
                    img_bytes, image_format = encode_image_bytes_for_llm(base_image["image"])
                    transcript = perform_llm_prediction_for_image_bytes(
                        img_bytes, self.llm, self.prompt,
                        image_format=image_format,
                        cache=self.image_cache,
                        image_name=source,
                    )
                except Exception as image_error:
                    logger.warning(
                        "Skipping image %s on page %s of %s: %s", i, index, source, image_error
                    )
                    continue
                text_content += "\n**Image Transcript:**\n" + transcript + "\n--------------------\n"
        return text_content

    def load(self):
        if not hasattr(self, 'file_path'):
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w+b', delete=True, suffix=".pdf") as temp_file:
                temp_file.write(self.file_content)
                temp_file.flush()
                self.file_path = temp_file.name
                return self._load_docs()
        else:
            return self._load_docs()

    def _load_docs(self):
        """Load PDF pages as Documents, optionally extracting images and splitting by token budget."""
        images_parser = ImageParser(llm=self.llm, prompt=self.prompt) if self.extract_images else None
        docs = PyPDFium2Loader(
                file_path = self.file_path,
                password=self.password,
                headers=self.headers,
                extract_images = self.extract_images,
                images_parser = images_parser,
            ).load()
        sections = self._section_index(docs)
        pages = _cut_pages(docs, sections)
        _join_short_parts(pages)
        splitter = None
        if self.max_tokens and self.max_tokens > 0:
            splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=self.max_tokens, chunk_overlap=0,
                                         add_start_index=True)
        chunks = []
        for part in (part for page in pages for part in page if not part.absorbed):
            text = part.joined_text()
            pieces = [(text, 0)]
            if splitter:
                pieces = [(c.page_content, c.metadata['start_index']) for c in splitter.create_documents([text])]
            cursor = 0
            for chunk_text, start in pieces:
                if start < 0:   # chunk text not found verbatim in the part (token split inside a character)
                    start = cursor
                cursor = start + len(chunk_text)
                first, offset = part.locate(start)
                last, _ = part.locate(max(start, cursor - 1))
                metadata = {**first.doc.metadata, 'chunk_id': len(chunks), **page_metadata(first.page, last.page),
                            **sections.section_at(first.page, offset, top_slack=not first.cut)}
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
        return chunks

    def _section_index(self, docs) -> SectionIndex:
        """Sections from the PDF outline when it has entries, otherwise from numbered headings in the text."""
        page_texts = [''] * (max((d.metadata.get('page', 0) for d in docs), default=-1) + 1)
        for doc in docs:
            page_texts[doc.metadata['page']] = doc.page_content or ''
        toc = []
        try:
            with pymupdf.open(filename=self.file_path, filetype="pdf") as report:
                if report.needs_pass and self.password:
                    report.authenticate(self.password)
                # get_toc() is native and proportional to the file's size; outline_headings bounds the work
                # done with it. (Walking pymupdf.Outline by hand crashes on documents without an outline.)
                toc = report.get_toc(simple=True)
        except Exception as e:  # pylint: disable=broad-except
            # An unreadable outline only costs the outline; the heading heuristic still runs.
            logger.warning("Failed to read the PDF outline of %s: %s", self.file_path, e)
        headings = outline_headings(toc, page_texts) if toc else []
        return SectionIndex(headings or detect_headings(page_texts))
