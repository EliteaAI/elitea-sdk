"""Page and section metadata for document chunks (EL-7355).

Loaders call these helpers to put a human page number and the section a chunk belongs to into chunk
metadata, so that a reference can read ``file.pdf · p. 7 · 4.2 Access review``.

Metadata keys (absent when unknown, never guessed):

- ``page_number``: 1-based page as a PDF viewer shows it (``page`` keeps the 0-based loader value);
  ``page_number_from`` / ``page_number_to`` give the page span of the chunk.
- ``section_number``: e.g. ``4.2`` or ``Chapter 4``; ``section_title``: e.g. ``Access review``.

Sections come from the PDF outline (bookmarks) when there is one, otherwise from a conservative heading
heuristic over the page text: a numbered heading is accepted only when its number continues the numbering
of the previously accepted heading, so list items, table rows and dates are left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

PAGE_NUMBER = 'page_number'
PAGE_NUMBER_FROM = 'page_number_from'
PAGE_NUMBER_TO = 'page_number_to'
SECTION_NUMBER = 'section_number'
SECTION_TITLE = 'section_title'
SECTION_KEYS = (SECTION_NUMBER, SECTION_TITLE)

# Bounds
MAX_OUTLINE_ENTRIES = 5000          # outline entries read from one document
MAX_SECTION_DEPTH = 6               # outline levels and number components ("1.2.3.4.5.6")
MAX_HEADINGS_PER_PAGE = 10          # more candidates on one page = a contents or list page, ignored
MAX_SECTION_TITLE_CHARS = 160       # longer outline titles are cut; longer lines are not headings
MAX_HEADING_LINE_CHARS = 120        # a text line longer than this is never a heading
MAX_HEADING_WORDS = 15
MAX_NUMBER_COMPONENT = 99           # "2024.03" is a date or version, not a section
NUMBER_GAP = 2                      # missed headings tolerated between two accepted numbers
# A heading that starts within this many characters of a chunk's start still names the chunk:
# page chunks usually begin with a running header line above the first heading.
HEADING_TOP_SLACK_CHARS = 200

_NUMBERED_RE = re.compile(r'^(\d{1,2}(?:\.\d{1,2}){0,5})\.?\s+(\S.*)$')
_KEYWORD_RE = re.compile(
    r'^(Chapter|Section|Article|Part|Appendix|Annex)\s+(\d{1,3}(?:\.\d{1,2}){0,5}|[IVXLC]{1,7}|[A-Z])\b\.?'
    r'(?:\s*[:.\-–—]\s*|\s+)?(.*)$')
_DOT_LEADER_RE = re.compile(r'(\.\s?){3,}|…')
_TRAILING_PAGE_RE = re.compile(r'\s\d{1,4}$')
# Only lines that start like a heading reach the Python checks; keeps detection cheap on long documents.
_CANDIDATE_LINE_RE = re.compile(r'^[ \t]*(?:\d|Chapter\b|Section\b|Article\b|Part\b|Appendix\b|Annex\b)[^\r\n]*',
                                re.MULTILINE)
_SPACE_RE = re.compile(r'\s+')


@dataclass(frozen=True)
class Heading:
    """A section start: ``offset`` is the character offset of the heading in its page's text."""

    page: int                       # 0-based page index
    offset: int
    number: Optional[str]
    title: Optional[str]
    numeric: Optional[tuple] = None  # parsed number, for numbered headings only

    def metadata(self) -> dict:
        meta = {}
        if self.number:
            meta[SECTION_NUMBER] = self.number
        if self.title:
            meta[SECTION_TITLE] = self.title
        return meta


def _clean(text: str) -> str:
    return _SPACE_RE.sub(' ', text or '').strip()


def split_heading_text(text: str) -> tuple[Optional[str], Optional[str]]:
    """Split ``"4.2 Access review"`` into ``("4.2", "Access review")``; text without a number is all title."""
    text = _clean(text)[:MAX_SECTION_TITLE_CHARS]
    if not text:
        return None, None
    match = _NUMBERED_RE.match(text)
    if match:
        return match.group(1), _clean(match.group(2)) or None
    match = _KEYWORD_RE.match(text)
    if match:
        return f'{match.group(1)} {match.group(2)}', _clean(match.group(3)) or None
    return None, text


def _plausible_title(title: str) -> bool:
    if not title or len(title) > MAX_HEADING_LINE_CHARS or title[0].islower():
        return False
    if title[-1] in '.,;:' or _DOT_LEADER_RE.search(title) or _TRAILING_PAGE_RE.search(title):
        return False
    if len(title.split()) > MAX_HEADING_WORDS:
        return False
    letters = sum(ch.isalpha() for ch in title)
    return letters >= 2 and letters * 2 >= len(title.replace(' ', ''))


def heading_candidate(line: str, page: int, offset: int) -> Optional[Heading]:
    """Return a heading when one text line looks like ``4.2 Access review`` or ``Chapter 4: Scope``."""
    line = _clean(line)
    if not line or len(line) > MAX_HEADING_LINE_CHARS:
        return None
    match = _NUMBERED_RE.match(line)
    if match:
        numeric = tuple(int(part) for part in match.group(1).split('.'))
        title = _clean(match.group(2))
        if any(part > MAX_NUMBER_COMPONENT for part in numeric) or not _plausible_title(title):
            return None
        return Heading(page, offset, match.group(1), title, numeric)
    match = _KEYWORD_RE.match(line)
    if match:
        title = _clean(match.group(3))
        if title and not _plausible_title(title):
            return None
        return Heading(page, offset, f'{match.group(1)} {match.group(2)}', title or None)
    return None


def _is_nested_start(parts: Sequence[int]) -> bool:
    return all(part in (0, 1) for part in parts)


def follows(previous: Optional[tuple], number: tuple) -> bool:
    """True when ``number`` can be the next heading after ``previous`` (allowing ``NUMBER_GAP`` misses)."""
    if previous is None:
        return number[0] <= 1 and _is_nested_start(number[1:])
    depth = len(previous)
    if len(number) > depth and number[:depth] == previous:
        return len(number) - depth <= 2 and _is_nested_start(number[depth:])
    for level in range(1, min(len(number), depth) + 1):
        if (number[:level - 1] == previous[:level - 1]
                and previous[level - 1] < number[level - 1] <= previous[level - 1] + 1 + NUMBER_GAP
                and _is_nested_start(number[level:])):
            return True
    return False


def detect_headings(page_texts: Sequence[str]) -> list[Heading]:
    """Find headings in page texts (no outline available). Conservative: when in doubt, no heading."""
    accepted: list[Heading] = []
    previous: Optional[tuple] = None
    pending: Optional[Heading] = None
    for page, text in enumerate(page_texts):
        candidates = []
        for line in _CANDIDATE_LINE_RE.finditer(text or ''):
            raw = line.group(0)
            if len(raw) > MAX_HEADING_LINE_CHARS * 2:
                continue
            candidate = heading_candidate(raw, page, line.start() + len(raw) - len(raw.lstrip()))
            if candidate is not None:
                candidates.append(candidate)
                if len(candidates) > MAX_HEADINGS_PER_PAGE:
                    break
        if len(candidates) > MAX_HEADINGS_PER_PAGE:
            continue
        for candidate in candidates:
            if candidate.numeric is None:
                accepted.append(candidate)
                continue
            if follows(previous, candidate.numeric):
                accepted.append(candidate)
                previous, pending = candidate.numeric, None
            elif pending is not None and follows(pending.numeric, candidate.numeric):
                # Two headings that agree with each other restart the numbering (e.g. after a contents
                # page or a heading the text extraction lost).
                accepted.extend((pending, candidate))
                previous, pending = candidate.numeric, None
            else:
                pending = candidate
    return accepted


def outline_headings(toc: Iterable[Sequence], page_texts: Sequence[str]) -> list[Heading]:
    """Map outline entries ``[level, title, page (1-based), ...]`` to headings.

    The entry is placed where its title appears in the page text; when the title is not found the entry
    is taken to start at the top of its page.
    """
    headings = []
    for index, entry in enumerate(toc):
        if index >= MAX_OUTLINE_ENTRIES:
            break
        try:
            level, title, page = int(entry[0]), str(entry[1] or ''), int(entry[2])
        except (TypeError, ValueError, IndexError):
            continue
        if level > MAX_SECTION_DEPTH or not 1 <= page <= len(page_texts):
            continue
        number, clean_title = split_heading_text(title)
        if not number and not clean_title:
            continue
        offset = _find_offset(page_texts[page - 1], title)
        headings.append(Heading(page - 1, offset, number, clean_title))
    headings.sort(key=lambda h: (h.page, h.offset))
    return headings


def _find_offset(page_text: str, title: str) -> int:
    needle = _clean(title)[:60]
    if not needle or not page_text:
        return 0
    position = page_text.find(needle)
    if position < 0:
        position = page_text.casefold().find(needle.casefold())
    return max(position, 0)


class SectionIndex:
    """Answers "which section is the chunk at (page, offset) in?" for one document."""

    def __init__(self, headings: Sequence[Heading]):
        self._headings = sorted(headings, key=lambda h: (h.page, h.offset))
        self._keys = [(h.page, h.offset) for h in self._headings]

    def __bool__(self) -> bool:
        return bool(self._headings)

    def section_at(self, page: int, offset: int = 0) -> dict:
        from bisect import bisect_right
        position = bisect_right(self._keys, (page, offset + HEADING_TOP_SLACK_CHARS))
        if position == 0:
            return {}
        return self._headings[position - 1].metadata()


def page_metadata(page_index: int) -> dict:
    number = page_index + 1
    return {PAGE_NUMBER: number, PAGE_NUMBER_FROM: number, PAGE_NUMBER_TO: number}
