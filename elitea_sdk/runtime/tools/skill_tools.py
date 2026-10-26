"""Progressive disclosure of agent and project skills.

Without a project catalog the turn works as before: attached skills are listed as
name + description in the cached prefix and load_skill serves one body on demand.
With a project catalog nothing about individual skills reaches the system prompt;
the per-run SkillRegistry is searched with the user's request before the model
runs (top matches ride the uncached suffix), the model can search it itself with
search_skills, and load_skill fetches a body from the platform when the registry
entry carries none. Loaded bodies are ToolMessages that ride the message history,
never the cached prefix.
"""

import logging
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Callable, List, Optional
from xml.sax.saxutils import escape as xml_escape

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from ..exceptions import SkillBodyUnavailable
from ..langchain.constants import (
    INSTRUCTION_REFERENCED_HINT,
    LOAD_SKILL_ALREADY_ACTIVE,
    LOAD_SKILL_LISTED_TOOL_DESCRIPTION,
    LOAD_SKILL_TOOL_DESCRIPTION,
    LOAD_SKILL_UNAVAILABLE,
    LOAD_SKILL_UNAVAILABLE_DENIED,
    LOAD_SKILL_UNAVAILABLE_TRANSIENT,
    LOAD_SKILL_UNKNOWN,
    LOAD_SKILL_UNKNOWN_LISTED,
    LOADED_SKILL_PREFIX_RE,
    LOADED_SKILL_RESULT,
    SEARCH_SKILLS_DESCRIPTION_MAX_CHARS,
    SEARCH_SKILLS_NAME_WEIGHT,
    SEARCH_SKILLS_NO_MATCH,
    SEARCH_SKILLS_RESULT_ENTRY,
    SEARCH_SKILLS_RESULT_LIMIT,
    SEARCH_SKILLS_STOPWORDS,
    SEARCH_SKILLS_TOOL_DESCRIPTION,
    SKILL_SEARCH_REMINDER_SUFFIX,
    SKILL_REGISTRY_DESCRIPTION_FLOOR_CHARS,
    SKILL_REGISTRY_DESCRIPTION_MAX_CHARS,
    SKILL_REGISTRY_ENTRY,
    SKILL_REGISTRY_HEADER,
    SKILL_REGISTRY_TOTAL_DESCRIPTION_CHARS,
    SKILL_REMINDER_SUFFIX,
    SKILL_SEARCH_PROMPT,
    SKILL_SUGGESTIONS_ENTRY,
    SKILL_SUGGESTIONS_FOOTER,
    SKILL_SUGGESTIONS_HEADER,
    SKILL_SUGGESTIONS_LIMIT,
    SKILL_SUGGESTIONS_MIN_TOKEN_CHARS,
    SKILL_SUGGESTIONS_SCORE_FLOOR,
)

logger = logging.getLogger(__name__)

BM25_K1 = 1.5
BM25_B = 0.75
PHRASE_SCORE_MULTIPLIER = 1.5
SKILL_SEARCH_MODE_BM25 = 'bm25'
SKILL_SEARCH_MODE_EMBEDDING = 'embedding'
SOURCE_RANK = {'agent': 0, 'participant': 1, 'project': 2}
QUERY_PART_PATTERN = re.compile(r'"(?P<phrase>[^"]*)"|(?P<term>[^\s"]+)')
WORD_CATEGORY_CLASSES = frozenset("LNM")
FORMAT_CATEGORY = "Cf"
ZERO_WIDTH_SPACE = "​"
DOTTED_SMALL_I = "i̇"
UNSEGMENTED_RUN_PATTERN = re.compile(
    "[฀-໿က-႟ក-៿々-〇぀-ヿ㐀-鿿"
    "가-힯豈-﫿\U00020000-\U0003134f]+"
)


def loaded_skill_names_from_messages(messages) -> set:
    """Names (lowercased) of skills whose loaded bodies are present in the given
    message history. Derived from context rather than stored state: if
    summarization drops a body, its name disappears from this set and a genuine
    re-load becomes possible again.

    Only ToolMessages whose tool_call_id maps back to a load_skill call (via the
    owning AIMessage's tool_calls) are counted, so another tool's output that
    happens to start with the loaded-skill marker cannot suppress a genuine
    load. Unmappable ids fail open (not counted): the worst case is one
    duplicate body, never a permanently unloadable skill."""
    names = set()
    call_tool_names = {}
    for message in messages or []:
        for tool_call in getattr(message, 'tool_calls', None) or []:
            call_id = tool_call.get('id') if isinstance(tool_call, dict) else getattr(tool_call, 'id', None)
            call_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
            if call_id:
                call_tool_names[call_id] = call_name
        content = getattr(message, 'content', None)
        if getattr(message, 'type', '') == 'tool' and isinstance(content, str):
            if call_tool_names.get(getattr(message, 'tool_call_id', None)) != 'load_skill':
                continue
            match = LOADED_SKILL_PREFIX_RE.match(content)
            if match:
                names.add(match.group(1).strip().lower())
    return names


def _normalized_name(skill: dict) -> str:
    return (skill.get('name') or '').strip().lower()


def _skill_key(skill: dict) -> tuple:
    return (skill.get('skill_project_id'), skill.get('skill_id'))


def _skill_version_id(skill: dict):
    return skill.get('skill_version_id', skill.get('version_id'))


def _source_rank(skill: dict, default: str) -> int:
    return SOURCE_RANK.get(skill.get('origin') or default, SOURCE_RANK[default])


class SkillRegistry:
    """Every skill one run may search or load, deduplicated by name and keyed by (skill_project_id, skill_id)."""

    def __init__(self, entries: list, ranker=None):
        self._entries = list(entries)
        self._by_name = {_normalized_name(skill): skill for skill in self._entries}
        self._by_key = {_skill_key(skill): skill for skill in self._entries}
        self._ranker = ranker or BM25Ranker()

    @classmethod
    def from_sources(cls, attached_skills=None, project_skills=None, ranker=None) -> "SkillRegistry":
        """The agent's own skills win a name clash, then chat participants, then project skills; within a source the lowest skill_id wins."""
        ranked = sorted(
            [(_source_rank(skill, 'agent'), skill) for skill in attached_skills or []]
            + [(_source_rank(skill, 'project'), skill) for skill in project_skills or []],
            key=lambda item: (item[0], item[1].get('skill_id') or 0, item[1].get('name') or ''),
        )
        entries, seen_names = [], set()
        for _rank, skill in ranked:
            name = _normalized_name(skill)
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            entries.append(skill)
        return cls(entries, ranker)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list:
        return list(self._entries)

    def resolve(self, name: str) -> Optional[dict]:
        return self._by_name.get((name or '').strip().lower())

    def get(self, skill_project_id, skill_id) -> Optional[dict]:
        return self._by_key.get((skill_project_id, skill_id))

    def search(self, query: str, limit: int = SEARCH_SKILLS_RESULT_LIMIT) -> list:
        return [skill for _score, skill in self.search_scored(query, limit)]

    def search_scored(self, query: str, limit: int = SEARCH_SKILLS_RESULT_LIMIT) -> list:
        return self._ranker.rank_scored(query, self._entries, limit)


class BM25Ranker:
    """BM25 over name (weighted) and description with OR semantics; quoted phrases are required."""

    def rank_scored(self, query: str, skills: list, limit: int = SEARCH_SKILLS_RESULT_LIMIT) -> list:
        return rank_skills_scored(query, skills, limit)


class EmbeddingRanker:
    """Cosine similarity over entry vectors; entries without a vector, or a failed query embedding, fall back to BM25."""

    def __init__(self, query_embedder: Callable[[str], list], fallback=None):
        self._query_embedder = query_embedder
        self._fallback = fallback or BM25Ranker()

    def rank_scored(self, query: str, skills: list, limit: int = SEARCH_SKILLS_RESULT_LIMIT) -> list:
        if not (query or '').strip() or not skills:
            return []
        embedded = [skill for skill in skills if _has_vector(skill)]
        if not embedded:
            return self._fallback.rank_scored(query, skills, limit)
        try:
            query_vector = list(self._query_embedder(query))
        except Exception as e:  # pylint: disable=W0703
            logger.warning("[Skills] query embedding failed, falling back to BM25: %s", e)
            return self._fallback.rank_scored(query, skills, limit)
        scored = sorted(
            ((_cosine(query_vector, skill['embedding']), index) for index, skill in enumerate(embedded)),
            key=lambda item: (-item[0], item[1]),
        )
        ranked = [(score, embedded[index]) for score, index in scored[:limit]]
        unembedded = [skill for skill in skills if not _has_vector(skill)]
        if len(ranked) < limit and unembedded:
            ranked.extend(self._fallback.rank_scored(query, unembedded, limit - len(ranked)))
        return ranked


def _has_vector(skill: dict) -> bool:
    vector = skill.get('embedding')
    return isinstance(vector, (list, tuple)) and len(vector) > 0


def _cosine(left: list, right: list) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def build_skill_ranker(skill_search: Optional[dict]):
    settings = skill_search if isinstance(skill_search, dict) else {}
    mode = settings.get('mode') or SKILL_SEARCH_MODE_BM25
    if mode == SKILL_SEARCH_MODE_EMBEDDING:
        query_embedder = settings.get('query_embedder')
        if callable(query_embedder):
            return EmbeddingRanker(query_embedder, BM25Ranker())
        logger.warning("[Skills] skill_search mode 'embedding' without a query_embedder; using BM25")
    elif mode != SKILL_SEARCH_MODE_BM25:
        logger.warning("[Skills] unknown skill_search mode %r; using BM25", mode)
    return BM25Ranker()


class LoadSkillInput(BaseModel):
    skill: str = Field(description="Exact name of the skill to load, as returned by search_skills.")


class LoadSkillListedInput(BaseModel):
    skill: str = Field(description="Exact name of the skill to load, as listed inside <available_skills> in the system prompt.")


class LoadSkillTool(BaseTool):
    """Serves only skills in the per-turn registry (structurally enforced); the loaded body is returned as a ToolMessage and never injected into the cached system prefix."""

    name: str = "load_skill"
    description: str = LOAD_SKILL_TOOL_DESCRIPTION
    args_schema: type[BaseModel] = LoadSkillInput
    registry: Any = Field(exclude=True)
    invoked_skills: list = Field(default_factory=list, exclude=True)
    body_fetcher: Optional[Any] = Field(default=None, exclude=True)
    listed: bool = Field(default=False, exclude=True)
    metadata: dict = {
        "toolkit_type": "internal",
        "toolkit_name": "skills",
        "display_name": "Skills",
    }
    _already_loaded: set = PrivateAttr(default_factory=set)

    def _load(self, skill: str) -> str:
        matched = self.registry.resolve(skill)
        if not matched:
            logger.info("[Skills] load_skill unknown name %r; %d skills available", skill, len(self.registry))
            if self.listed:
                available = sorted(s.get('name') for s in self.registry.entries if s.get('name'))
                return LOAD_SKILL_UNKNOWN_LISTED.format(name=skill, available=', '.join(available) or '(none)')
            return LOAD_SKILL_UNKNOWN.format(name=skill)
        name = matched.get('name')
        normalized_name = _normalized_name(matched)
        if self._is_invoked(matched):
            logger.info("[Skills] load_skill %r already active via ~name", name)
            return LOAD_SKILL_ALREADY_ACTIVE.format(name=name)
        if normalized_name in self._already_loaded:
            logger.info("[Skills] load_skill %r already loaded in this conversation", name)
            return LOAD_SKILL_ALREADY_ACTIVE.format(name=name)
        instructions, reason = self._instructions(matched)
        if instructions is None:
            logger.warning("[Skills] load_skill %r unavailable: %s", name, reason)
            return LOAD_SKILL_UNAVAILABLE.format(name=name, reason=reason)
        self._already_loaded.add(normalized_name)
        logger.info("[Skills] load_skill served %r", name)
        return LOADED_SKILL_RESULT.format(name=name, instructions=instructions)

    def _is_invoked(self, matched: dict) -> bool:
        invoked_keys = {_skill_key(s) for s in self.invoked_skills if s.get('skill_id') is not None}
        invoked_names = {_normalized_name(s) for s in self.invoked_skills}
        return (matched.get('skill_id') is not None and _skill_key(matched) in invoked_keys) \
            or _normalized_name(matched) in invoked_names

    def _instructions(self, matched: dict) -> tuple:
        inline = matched.get('instructions')
        if isinstance(inline, str) and inline.strip():
            return inline, ''
        if not callable(self.body_fetcher):
            return None, LOAD_SKILL_UNAVAILABLE_TRANSIENT
        try:
            body = self.body_fetcher(
                matched.get('skill_project_id'), matched.get('skill_id'), _skill_version_id(matched),
            )
        except SkillBodyUnavailable as e:
            return None, LOAD_SKILL_UNAVAILABLE_DENIED if e.denied else LOAD_SKILL_UNAVAILABLE_TRANSIENT
        except Exception as e:  # pylint: disable=W0703
            logger.warning("[Skills] body fetch for %r failed: %s", matched.get('name'), e)
            return None, LOAD_SKILL_UNAVAILABLE_TRANSIENT
        if not isinstance(body, str) or not body.strip():
            return None, LOAD_SKILL_UNAVAILABLE_DENIED
        return body, ''

    def mark_already_loaded(self, names) -> None:
        """Seed the already-loaded set (called by the tool loop with names derived
        from the conversation history before each invocation)."""
        self._already_loaded.update((n or '').strip().lower() for n in names or ())

    def _run(self, skill: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._load(skill)

    async def _arun(self, skill: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._load(skill)


class SearchSkillsInput(BaseModel):
    query: str = Field(description='Short description of what the user needs, in plain words. Wrap words in double quotes to require that exact phrase.')


class SearchSkillsTool(BaseTool):
    name: str = "search_skills"
    description: str = SEARCH_SKILLS_TOOL_DESCRIPTION
    args_schema: type[BaseModel] = SearchSkillsInput
    registry: Any = Field(exclude=True)
    metadata: dict = {
        "toolkit_type": "internal",
        "toolkit_name": "skill_search",
        "display_name": "Skill Search",
    }

    def _search(self, query: str) -> str:
        matches = self.registry.search(query)
        logger.info("[Skills] search_skills returned %d of %d skills", len(matches), len(self.registry))
        if not matches:
            return SEARCH_SKILLS_NO_MATCH
        return "\n".join(
            SEARCH_SKILLS_RESULT_ENTRY.format(
                name=skill.get('name'),
                description=_shorten(
                    ' '.join(str(skill.get('description') or '').split()), SEARCH_SKILLS_DESCRIPTION_MAX_CHARS
                ),
            )
            for skill in matches
        )

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._search(query)

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._search(query)


def _tokens(text: str) -> List[str]:
    normalized = unicodedata.normalize('NFKC', str(text or '')).casefold().replace(DOTTED_SMALL_I, 'i')
    return [
        term
        for word in _words(normalized)
        for variant in _format_variants(word)
        for term in _word_terms(variant)
    ]


def _is_word_character(character: str) -> bool:
    category = unicodedata.category(character)
    if category == FORMAT_CATEGORY:
        return character != ZERO_WIDTH_SPACE
    return category[0] in WORD_CATEGORY_CLASSES


def _words(text: str) -> List[str]:
    words = []
    current = []
    for character in text:
        if _is_word_character(character):
            current.append(character)
        elif current:
            words.append(''.join(current))
            current = []
    if current:
        words.append(''.join(current))
    return words


def _format_variants(word: str) -> List[str]:
    parts = []
    current = []
    for character in word:
        if unicodedata.category(character) == FORMAT_CATEGORY:
            parts.append(''.join(current))
            current = []
        else:
            current.append(character)
    parts.append(''.join(current))
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return parts
    return [''.join(parts)] + parts


def _word_terms(word: str) -> List[str]:
    terms = []
    position = 0
    for run in UNSEGMENTED_RUN_PATTERN.finditer(word):
        if run.start() > position:
            terms.append(word[position:run.start()])
        terms.extend(_bigrams(run.group()))
        position = run.end()
    if position < len(word):
        terms.append(word[position:])
    return terms


def _bigrams(run: str) -> List[str]:
    if len(run) == 1:
        return [run]
    return [run[index:index + 2] for index in range(len(run) - 1)]


def parse_skill_query(query: str) -> tuple:
    """(unique scoring terms, required phrases); phrase terms also score, as in tools/openapi/response_selection."""
    terms, phrases = [], []
    for match in QUERY_PART_PATTERN.finditer(str(query or '')):
        phrase = match.group('phrase')
        tokens = _tokens(phrase if phrase is not None else match.group('term'))
        if not tokens:
            continue
        terms.extend(tokens)
        if phrase is not None:
            phrases.append(tuple(tokens))
    return list(dict.fromkeys(terms)), list(dict.fromkeys(phrases))


def _contains_phrase(sequence: List[str], phrase: tuple) -> bool:
    length = len(phrase)
    return any(tuple(sequence[index:index + length]) == phrase for index in range(len(sequence) - length + 1))


def _qualifies(query_terms: List[str], document: Counter, name_terms: set) -> bool:
    content_terms = [term for term in query_terms if term not in SEARCH_SKILLS_STOPWORDS]
    if content_terms:
        return any(document[term] for term in content_terms)
    return any(term in name_terms for term in query_terms)


def rank_skills(query: str, skills: list, limit: int = SEARCH_SKILLS_RESULT_LIMIT) -> list:
    return [skill for _score, skill in rank_skills_scored(query, skills, limit)]


def rank_skills_scored(query: str, skills: list, limit: int = SEARCH_SKILLS_RESULT_LIMIT) -> list:
    """(score, skill) pairs: BM25 with OR semantics over content terms; an all-stopword query matches skill names only; every quoted phrase must appear in name + description; ties keep catalog order."""
    query_terms, phrases = parse_skill_query(query)
    if not query_terms or not skills:
        return []
    name_tokens = [_tokens(skill.get('name')) for skill in skills]
    description_tokens = [_tokens(skill.get('description')) for skill in skills]
    documents = [
        Counter(names * SEARCH_SKILLS_NAME_WEIGHT + description)
        for names, description in zip(name_tokens, description_tokens)
    ]
    lengths = [sum(document.values()) for document in documents]
    average_length = (sum(lengths) / len(lengths)) or 1
    document_frequencies = {
        term: sum(1 for document in documents if term in document) for term in query_terms
    }
    inverse_document_frequency = {
        term: math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequencies.items()
    }
    phrase_boost = sum(
        PHRASE_SCORE_MULTIPLIER * sum(inverse_document_frequency.get(term, 0.0) for term in phrase)
        for phrase in phrases
    )
    ranked = []
    for index, (document, length, names, description) in enumerate(
        zip(documents, lengths, name_tokens, description_tokens)
    ):
        if not all(_contains_phrase(names + description, phrase) for phrase in phrases):
            continue
        if not _qualifies(query_terms, document, set(names)):
            continue
        length_normalization = BM25_K1 * (1 - BM25_B + BM25_B * length / average_length)
        score = sum(
            inverse_document_frequency[term]
            * document[term] * (BM25_K1 + 1) / (document[term] + length_normalization)
            for term in query_terms if document[term]
        ) + phrase_boost
        if score > 0:
            ranked.append((score, index))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [(score, skills[index]) for score, index in ranked[:limit]]


def build_load_skill_tools(
    attached_skills, invoked_skills=None, project_skills=None, skill_search=None, skill_body_fetcher=None,
) -> List[BaseTool]:
    if not attached_skills and not project_skills:
        return []
    registry = SkillRegistry.from_sources(attached_skills, project_skills, build_skill_ranker(skill_search))
    if not project_skills:
        logger.debug("[Skills] Binding load_skill with %d attached skills", len(registry))
        return [LoadSkillTool(
            registry=registry, invoked_skills=invoked_skills or [], body_fetcher=skill_body_fetcher,
            listed=True, description=LOAD_SKILL_LISTED_TOOL_DESCRIPTION, args_schema=LoadSkillListedInput,
        )]
    logger.debug(
        "[Skills] Binding search_skills + load_skill over %d skills (%d attached, %d project)",
        len(registry), len(attached_skills or []), len(project_skills or []),
    )
    return [
        SearchSkillsTool(registry=registry),
        LoadSkillTool(registry=registry, invoked_skills=invoked_skills or [], body_fetcher=skill_body_fetcher),
    ]


def render_skill_registry_index(attached_skills, project_skills=None, project_skill_summary=None) -> str:
    if project_skills:
        return SKILL_SEARCH_PROMPT
    if not attached_skills:
        return ""
    ordered = sorted(
        attached_skills, key=lambda s: (s.get('skill_id') or 0, s.get('name') or '')
    )
    descriptions = [' '.join(str(s.get('description') or '').split()) for s in ordered]
    max_chars = _registry_description_max_chars([len(d) for d in descriptions])
    entries = [
        SKILL_REGISTRY_ENTRY.format(
            name=xml_escape(' '.join(str(s.get('name') or '').split()), {'"': '&quot;'}),
            description=xml_escape(_registry_description(description, max_chars, s)),
        )
        for s, description in zip(ordered, descriptions)
    ]
    logger.info("[Skills] Rendered registry index with %d skills into cached prefix", len(ordered))
    return (
        "<available_skills>\n" + SKILL_REGISTRY_HEADER + "\n\n"
        + "\n".join(entries) + "\n</available_skills>"
    )


def _registry_description_max_chars(lengths: List[int]) -> int:
    """Largest common description length that keeps the registry within its total budget.

    Water-filling: descriptions shorter than the result stay whole, longer ones are
    cut to it. When even the floor does not fit, the floor wins and the budget is
    exceeded rather than dropping or blanking a skill.
    """
    def total(limit: int) -> int:
        return sum(min(length, limit) for length in lengths)

    if total(SKILL_REGISTRY_DESCRIPTION_MAX_CHARS) <= SKILL_REGISTRY_TOTAL_DESCRIPTION_CHARS:
        return SKILL_REGISTRY_DESCRIPTION_MAX_CHARS
    if total(SKILL_REGISTRY_DESCRIPTION_FLOOR_CHARS) > SKILL_REGISTRY_TOTAL_DESCRIPTION_CHARS:
        logger.warning(
            "[Skills] %d skills exceed the registry budget even at %d chars each; keeping every skill",
            len(lengths), SKILL_REGISTRY_DESCRIPTION_FLOOR_CHARS,
        )
        return SKILL_REGISTRY_DESCRIPTION_FLOOR_CHARS
    low, high = SKILL_REGISTRY_DESCRIPTION_FLOOR_CHARS, SKILL_REGISTRY_DESCRIPTION_MAX_CHARS
    while low < high:
        middle = (low + high + 1) // 2
        if total(middle) <= SKILL_REGISTRY_TOTAL_DESCRIPTION_CHARS:
            low = middle
        else:
            high = middle - 1
    return low


def _registry_description(description: str, max_chars: int, skill: dict) -> str:
    """The hint is appended after the cut, so a long description never truncates it."""
    hint = INSTRUCTION_REFERENCED_HINT if skill.get('instruction_referenced') else ''
    return ' '.join(part for part in (_shorten(description, max_chars), hint) if part) or '(no description)'


def skill_reminder_suffix(project_skills) -> str:
    return SKILL_SEARCH_REMINDER_SUFFIX if project_skills else SKILL_REMINDER_SUFFIX


def request_text(task) -> str:
    if isinstance(task, str):
        return task
    if isinstance(task, list):
        return ' '.join(
            block.get('text') or '' for block in task
            if isinstance(block, dict) and block.get('type') == 'text'
        )
    return ''


def suggestion_query(text: str) -> str:
    """Request words long enough to carry meaning; digits and two-letter words only pull in unrelated skills."""
    return ' '.join(
        term for term in _tokens(text)
        if len(term) >= SKILL_SUGGESTIONS_MIN_TOKEN_CHARS or not term.isascii()
    )


def render_skill_suggestions(registry: SkillRegistry, task, excluded_names=()) -> str:
    """Per-turn block of the registry's best matches for the user's request, rendered for the uncached suffix."""
    query = suggestion_query(request_text(task))
    if not query or not len(registry):
        return ""
    excluded = {(name or '').strip().lower() for name in excluded_names}
    scored = [
        (score, skill) for score, skill in registry.search_scored(query, SKILL_SUGGESTIONS_LIMIT + len(excluded))
        if _normalized_name(skill) not in excluded
    ]
    floor = scored[0][0] * SKILL_SUGGESTIONS_SCORE_FLOOR if scored else 0
    matches = [skill for score, skill in scored if score >= floor][:SKILL_SUGGESTIONS_LIMIT]
    if not matches:
        return ""
    logger.info("[Skills] Suggesting %d of %d skills for this request", len(matches), len(registry))
    entries = "\n".join(
        SKILL_SUGGESTIONS_ENTRY.format(
            name=skill.get('name'),
            description=_shorten(' '.join(str(skill.get('description') or '').split()), SEARCH_SKILLS_DESCRIPTION_MAX_CHARS),
        )
        for skill in matches
    )
    return f"{SKILL_SUGGESTIONS_HEADER}\n{entries}\n{SKILL_SUGGESTIONS_FOOTER}"


def _shorten(description: str, max_chars: int) -> str:
    if len(description) <= max_chars:
        return description
    cut = description[:max_chars - 1]
    word_end = cut.rfind(' ')
    if word_end > 0:
        cut = cut[:word_end]
    return cut.rstrip() + '…'
