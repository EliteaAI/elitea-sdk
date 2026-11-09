"""Word automatic heading numbers (EL-7355).

Word keeps the "4.2" of a numbered heading in ``numbering.xml``, not in the paragraph text, so a converted
document loses it. ``number_headings`` writes the number Word shows in front of each heading's text, which
gives the chunk text and the section metadata the same number a reader sees.

Conservative: only decimal multi-level numbering ("4", "4.2", "4.2.1.") is rendered. A heading whose list
uses another format, a level override or an unknown definition keeps its text unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from docx.oxml import OxmlElement
from docx.oxml.ns import qn

MAX_PARAGRAPHS = 50000              # paragraphs scanned per document
MAX_STYLE_CHAIN = 10                # basedOn hops followed to find a style's numbering
_LEVELS = 9
_HEADING_STYLE_RE = re.compile(r'^heading ([1-9])$', re.IGNORECASE)
_LABEL_RE = re.compile(r'^\d+(?:\.\d+)*\.?$')
_DECIMAL = {'decimal', 'decimalZero'}


@dataclass(frozen=True)
class _Level:
    start: int
    fmt: str
    text: str
    style_id: Optional[str]


def _val(element, tag: str) -> Optional[str]:
    child = element.find(qn(tag)) if element is not None else None
    return child.get(qn('w:val')) if child is not None else None


def _int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _numbering_definitions(doc) -> dict[str, tuple[str, dict[int, _Level]]]:
    """numId -> (abstractNumId, levels). Definitions used with a level override are left out (not rendered)."""
    try:
        numbering = doc.part.numbering_part.element
    except Exception:  # pylint: disable=broad-except
        return {}
    abstracts = {}
    for abstract in numbering.findall(qn('w:abstractNum')):
        levels = {}
        for lvl in abstract.findall(qn('w:lvl')):
            ilvl = _int(lvl.get(qn('w:ilvl')))
            if ilvl is None or not 0 <= ilvl < _LEVELS:
                continue
            levels[ilvl] = _Level(start=_int(_val(lvl, 'w:start'), 1), fmt=_val(lvl, 'w:numFmt') or '',
                                  text=_val(lvl, 'w:lvlText') or '', style_id=_val(lvl, 'w:pStyle'))
        abstracts[abstract.get(qn('w:abstractNumId'))] = levels
    nums = numbering.findall(qn('w:num'))
    # A level override restarts or reshapes a list; counting it right needs Word's full rules, so every
    # list sharing that definition is left unrendered.
    overridden = {_val(num, 'w:abstractNumId') for num in nums if num.find(qn('w:lvlOverride')) is not None}
    definitions = {}
    for num in nums:
        abstract_id = _val(num, 'w:abstractNumId')
        if abstract_id in abstracts and abstract_id not in overridden:
            definitions[num.get(qn('w:numId'))] = (abstract_id, abstracts[abstract_id])
    return definitions


def _numbering_of(paragraph, definitions) -> Optional[tuple[str, int, dict[int, _Level]]]:
    """(abstractNumId, level, levels) of a numbered paragraph: its own numPr, else its style's."""
    num_pr = paragraph._p.pPr.numPr if paragraph._p.pPr is not None else None
    style_id = None
    if num_pr is None:
        style = paragraph.style
        for _ in range(MAX_STYLE_CHAIN):
            if style is None:
                break
            p_pr = style.element.pPr
            if p_pr is not None and p_pr.numPr is not None:
                num_pr, style_id = p_pr.numPr, style.style_id
                break
            style = style.base_style
    if num_pr is None:
        return None
    num_id = _val(num_pr, 'w:numId')
    if num_id in (None, '0') or num_id not in definitions:
        return None
    abstract_id, levels = definitions[num_id]
    ilvl = _int(_val(num_pr, 'w:ilvl'))
    if ilvl is None:
        ilvl = next((i for i, level in levels.items() if style_id and level.style_id == style_id), 0)
    return (abstract_id, ilvl, levels) if ilvl in levels else None


def _label(levels: dict[int, _Level], counters: list, ilvl: int) -> Optional[str]:
    text = levels[ilvl].text
    for match in re.finditer(r'%([1-9])', text):
        level = int(match.group(1)) - 1
        if level not in levels or levels[level].fmt not in _DECIMAL:
            return None
    label = re.sub(r'%([1-9])',
                   lambda m: str(counters[int(m.group(1)) - 1] if counters[int(m.group(1)) - 1] is not None
                                 else levels[int(m.group(1)) - 1].start), text).strip()
    return label if _LABEL_RE.match(label) else None


def is_heading(paragraph) -> bool:
    style = paragraph.style
    return bool(style is not None and style.name and _HEADING_STYLE_RE.match(style.name.strip()))


def number_headings(doc) -> int:
    """Prefix each automatically numbered heading with the number Word shows. Returns how many were changed."""
    definitions = _numbering_definitions(doc)
    if not definitions:
        return 0
    counters: dict[str, list] = {}
    changed = 0
    for index, paragraph in enumerate(doc.paragraphs):
        if index >= MAX_PARAGRAPHS:
            break
        numbering = _numbering_of(paragraph, definitions)
        if numbering is None:
            continue
        abstract_id, ilvl, levels = numbering
        state = counters.setdefault(abstract_id, [None] * _LEVELS)
        state[ilvl] = levels[ilvl].start if state[ilvl] is None else state[ilvl] + 1
        for deeper in range(ilvl + 1, _LEVELS):
            state[deeper] = None
        if not is_heading(paragraph):
            continue
        label = _label(levels, state, ilvl)
        first_word = (paragraph.text.split(maxsplit=1) or [''])[0]
        if not label or first_word.rstrip('.') == label.rstrip('.'):   # number already typed in the text
            continue
        run = OxmlElement('w:r')
        text = OxmlElement('w:t')
        text.text = f'{label} '
        text.set(qn('xml:space'), 'preserve')
        run.append(text)
        p_pr = paragraph._p.pPr
        if p_pr is not None:
            p_pr.addnext(run)
        else:
            paragraph._p.insert(0, run)
        changed += 1
    return changed
