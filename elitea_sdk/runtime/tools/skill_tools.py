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

import difflib
import json
import logging
import math
import posixpath
import re
import unicodedata
from collections import Counter
from typing import Any, Callable, List, Optional
from xml.sax.saxutils import escape as xml_escape

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from ..exceptions import SkillBodyUnavailable
from ...tools.utils.file_metadata import bound_read_result
from ..langchain.constants import (
    INSTRUCTION_REFERENCED_HINT,
    LIST_SKILL_FILES_RESULT,
    LIST_SKILL_FILES_TOOL_DESCRIPTION,
    LOAD_SKILL_ALREADY_ACTIVE,
    LOAD_SKILL_ALREADY_ACTIVE_RE,
    LOAD_SKILL_LISTED_TOOL_DESCRIPTION,
    LOAD_SKILL_TOOL_DESCRIPTION,
    LOAD_SKILL_UNAVAILABLE,
    LOAD_SKILL_UNAVAILABLE_DENIED,
    LOAD_SKILL_UNAVAILABLE_TRANSIENT,
    LOAD_SKILL_USER_ONLY,
    LOAD_SKILL_UNKNOWN,
    LOAD_SKILL_UNKNOWN_LISTED,
    LOADED_SKILL_PREFIX_RE,
    LOADED_SKILL_RESULT,
    READ_SKILL_FILE_ALREADY_READ,
    READ_SKILL_FILE_LINES,
    READ_SKILL_FILE_PREFIX_RE,
    READ_SKILL_FILE_RESULT,
    READ_SKILL_FILE_TOOL_DESCRIPTION,
    READ_SKILL_OUTPUT_TOOL_DESCRIPTION,
    RUN_SKILL_SCRIPT_TOOL_DESCRIPTION,
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
    SKILL_DIR_PLACEHOLDERS,
    SKILL_FILE_BINARY,
    SKILL_FILE_FOLDER_NOT_FOUND,
    SKILL_FILE_NOT_FOUND,
    SKILL_FILE_OUTSIDE_ROOT,
    SKILL_FILE_AFTER_LOAD,
    SKILL_FILE_SKILL_NOT_LOADED,
    SKILL_FILE_SKILL_WITHOUT_FILES,
    SKILL_FILE_SUGGESTION_LIMIT,
    SKILL_FILE_UNAVAILABLE,
    SKILL_FILE_UNAVAILABLE_DENIED,
    SKILL_FILE_UNAVAILABLE_TRANSIENT,
    SKILL_FILE_UNKNOWN_SKILL,
    SKILL_FILES_BLOCK,
    SKILL_FILES_BLOCK_END,
    SKILL_FILES_LISTING_DIRECTORY,
    SKILL_FILES_LISTING_FILE,
    SKILL_FILES_LISTING_LIMIT,
    SKILL_FILES_LISTING_MORE,
    SKILL_FILES_RUN_SCRIPTS_NOTE,
    SKILL_FILES_SCRIPTS_NOTE,
    SKILL_ROOT_URI,
    SKILL_SCRIPT_DIRECTORY,
    SKILL_SCRIPT_EXTENSIONS,
    SKILL_SHELL_UNAVAILABLE,
)
from .sandbox import _read_sandbox_limits_from_env
from .skill_script_runner import INLINE_OUTPUT_CHARS, is_python_script

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
SKILL_URI_SCHEME = 'skill://'
SHELL_FENCE_PATTERN = re.compile(r'^[ \t]*```!.*?^[ \t]*```[ \t]*$', re.MULTILINE | re.DOTALL)
SHELL_LINE_PATTERN = re.compile(r'^[ \t]*!`.*$', re.MULTILINE)
UNSEGMENTED_RUN_PATTERN = re.compile(
    "[฀-໿က-႟ក-៿々-〇぀-ヿ㐀-鿿"
    "가-힯豈-﫿\U00020000-\U0003134f]+"
)


def collect_tool_results(messages, tool_name: str) -> list:
    """Text results in the history of calls to ``tool_name``, matched through the owning AIMessage's tool_calls.

    Only ToolMessages whose tool_call_id maps back to a call of that tool count, so
    another tool's output that happens to start with the same marker is ignored.
    Unmappable ids fail open (not counted).
    """
    results = []
    call_tool_names = {}
    for message in messages or []:
        for tool_call in getattr(message, 'tool_calls', None) or []:
            call_id = tool_call.get('id') if isinstance(tool_call, dict) else getattr(tool_call, 'id', None)
            call_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
            if call_id:
                call_tool_names[call_id] = call_name
        content = getattr(message, 'content', None)
        if getattr(message, 'type', '') == 'tool' and isinstance(content, str) \
                and call_tool_names.get(getattr(message, 'tool_call_id', None)) == tool_name:
            results.append(content)
    return results


def loaded_skill_names_from_messages(messages) -> set:
    """Names (lowercased) of skills whose loaded bodies are present in the given
    message history. Derived from context rather than stored state: if
    summarization drops a body, its name disappears from this set and a genuine
    re-load becomes possible again. The worst case of an unmappable call id is one
    duplicate body, never a permanently unloadable skill."""
    results = [
        content for tool_name in ('load_skill', 'read_skill_file', 'list_skill_files', 'run_skill_script')
        for content in collect_tool_results(messages, tool_name)
    ]
    return {
        match.group(1).strip().lower()
        for match in (LOADED_SKILL_PREFIX_RE.match(content) for content in results)
        if match
    }


def split_auto_loaded(content: str) -> tuple:
    """A file tool's result as the skill it loaded on the way (empty when it loaded none) and its own answer."""
    if not LOADED_SKILL_PREFIX_RE.match(content):
        return '', content
    loaded, _separator, answer = content.partition(SKILL_FILES_BLOCK_END)
    return loaded + SKILL_FILES_BLOCK_END, answer.lstrip('\n')


def read_skill_file_keys_from_messages(messages) -> set:
    """(skill, path, lines) of skill file reads whose contents are still in the history, derived like loaded skills."""
    return {
        (match.group(1).strip().lower(), match.group(2), match.group(3))
        for match in (
            READ_SKILL_FILE_PREFIX_RE.match(split_auto_loaded(content)[1])
            for content in collect_tool_results(messages, 'read_skill_file')
        )
        if match
    }


def select_skill_files(files) -> list:
    """The manifest entries that name a path."""
    return [entry for entry in files or [] if isinstance(entry, dict) and entry.get('path')]


def format_file_size(size) -> str:
    """A file size the way the listing shows it: bytes below 1 KB, then whole KB, then MB."""
    size = size if isinstance(size, int) and size >= 0 else 0
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{math.ceil(size / 1024)} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def render_skill_body(name: str, instructions: str) -> str:
    """The body with skill directory placeholders pointing at the skill root and shell commands disabled."""
    root = SKILL_ROOT_URI.format(name=name)
    for placeholder in SKILL_DIR_PLACEHOLDERS:
        instructions = instructions.replace(placeholder, root)
    instructions = SHELL_FENCE_PATTERN.sub(SKILL_SHELL_UNAVAILABLE, instructions)
    return SHELL_LINE_PATTERN.sub(SKILL_SHELL_UNAVAILABLE, instructions)


def has_scripts(files: list) -> bool:
    """Whether a skill bundles anything the model might try to execute."""
    return any(
        entry['path'].split('/', 1)[0].lower() == SKILL_SCRIPT_DIRECTORY
        or entry['path'].lower().endswith(SKILL_SCRIPT_EXTENSIONS)
        for entry in files
    )


def has_python_scripts(files: list) -> bool:
    """Whether a skill bundles a script the offline sandbox can run."""
    return any(is_python_script(entry['path']) for entry in files)


def render_scripts_note(name: str, files: list, scripts_runnable: bool) -> str:
    """How the model may use the skill's scripts: run them when a runner is bound, read them otherwise."""
    if scripts_runnable and has_python_scripts(files):
        example = min((entry['path'] for entry in files if is_python_script(entry['path'])), key=str.casefold)
        return SKILL_FILES_RUN_SCRIPTS_NOTE.format(name=name, example=example)
    return SKILL_FILES_SCRIPTS_NOTE if has_scripts(files) else ''


def render_skill_files_listing(files: list) -> str:
    """Top-level files with sizes and folders collapsed to a file count, capped at the listing limit."""
    folders = Counter(entry['path'].split('/', 1)[0] for entry in files if '/' in entry['path'])
    entries = sorted(
        [(entry['path'], SKILL_FILES_LISTING_FILE.format(path=entry['path'], size=format_file_size(entry.get('size'))))
         for entry in files if '/' not in entry['path']]
        + [(folder, SKILL_FILES_LISTING_DIRECTORY.format(path=folder, count=count, unit='file' if count == 1 else 'files'))
           for folder, count in folders.items()],
        key=lambda item: item[0].casefold(),
    )
    lines = [line for _key, line in entries[:SKILL_FILES_LISTING_LIMIT]]
    if len(entries) > SKILL_FILES_LISTING_LIMIT:
        lines.append(SKILL_FILES_LISTING_MORE.format(count=len(entries) - SKILL_FILES_LISTING_LIMIT))
    return '\n'.join(lines)


def render_skill_files_block(name: str, files, scripts_runnable: bool = False) -> str:
    """The skill root, how to read its files and their listing; empty for a skill without files."""
    files = select_skill_files(files)
    if not files:
        return ''
    return SKILL_FILES_BLOCK.format(
        name=name,
        example=min((entry['path'] for entry in files), key=lambda path: (path.count('/'), path.casefold())),
        scripts_note=render_scripts_note(name, files, scripts_runnable),
        listing=render_skill_files_listing(files),
    )


def render_loaded_skill(template: str, name: str, instructions: str, files=None, scripts_runnable: bool = False) -> str:
    """A loaded skill as the model reads it: the template around the rendered body, then its files block."""
    rendered = template.format(name=name, instructions=render_skill_body(name, str(instructions)))
    files_block = render_skill_files_block(name, files, scripts_runnable)
    return f"{rendered}\n\n{files_block}" if files_block else rendered


def _normalize_skill_name(skill: dict) -> str:
    return (skill.get('name') or '').strip().lower()


def _build_skill_identity(skill: dict) -> tuple:
    return (skill.get('skill_project_id'), skill.get('skill_id'))


def _rank_by_source(skill: dict, default: str) -> int:
    return SOURCE_RANK.get(skill.get('origin') or default, SOURCE_RANK[default])


def _is_model_invocable(skill: dict) -> bool:
    return not skill.get('disable_model_invocation')


def _disclosable_names(skills) -> list:
    return sorted({s.get('name') for s in skills if s.get('name') and _is_model_invocable(s)})


class SkillRegistry:
    """Every skill the turn knows by name; only the model-invocable ones are disclosed, searched or suggested."""

    def __init__(self, entries: list, ranker=None):
        self._entries = list(entries)
        self._disclosable = [skill for skill in self._entries if _is_model_invocable(skill)]
        self._by_name = {_normalize_skill_name(skill): skill for skill in self._entries}
        self._ranker = ranker or BM25Ranker()

    @classmethod
    def from_sources(cls, attached_skills=None, project_skills=None, ranker=None) -> "SkillRegistry":
        ranked = sorted(
            [(_rank_by_source(skill, 'agent'), skill) for skill in attached_skills or []]
            + [(_rank_by_source(skill, 'project'), skill) for skill in project_skills or []],
            key=lambda item: (item[0], item[1].get('skill_id') or 0, item[1].get('name') or ''),
        )
        entries, seen_names = [], set()
        for _rank, skill in ranked:
            name = _normalize_skill_name(skill)
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

    @property
    def disclosable(self) -> list:
        return list(self._disclosable)

    def resolve(self, name: str) -> Optional[dict]:
        return self._by_name.get((name or '').strip().lower())

    def search(self, query: str, limit: int = SEARCH_SKILLS_RESULT_LIMIT) -> list:
        return [skill for _score, skill in self.search_scored(query, limit)]

    def search_scored(self, query: str, limit: int = SEARCH_SKILLS_RESULT_LIMIT) -> list:
        return self._ranker.rank_scored(query, self._disclosable, limit)


class BM25Ranker:
    def rank_scored(self, query: str, skills: list, limit: int = SEARCH_SKILLS_RESULT_LIMIT) -> list:
        return rank_skills_scored(query, skills, limit)


class EmbeddingRanker:
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
            ((_compute_cosine_similarity(query_vector, skill['embedding']), index) for index, skill in enumerate(embedded)),
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


def _compute_cosine_similarity(left: list, right: list) -> float:
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
    scripts_runnable: bool = Field(default=False, exclude=True)
    metadata: dict = {
        "toolkit_type": "internal",
        "toolkit_name": "skills",
        "display_name": "Skills",
    }
    _already_loaded: set = PrivateAttr(default_factory=set)

    def load(self, skill: str) -> str:
        """The skill's body as load_skill answers it, or the reason the model gets instead."""
        matched = self.registry.resolve(skill)
        if not matched:
            logger.info("[Skills] load_skill unknown name %r; %d skills available", skill, len(self.registry))
            if self.listed:
                available = _disclosable_names(self.registry.entries)
                return LOAD_SKILL_UNKNOWN_LISTED.format(name=skill, available=', '.join(available) or '(none)')
            return LOAD_SKILL_UNKNOWN.format(name=skill)
        name = matched.get('name')
        normalized_name = _normalize_skill_name(matched)
        if self._is_invoked(matched):
            logger.info("[Skills] load_skill %r already active via ~name", name)
            return LOAD_SKILL_ALREADY_ACTIVE.format(name=name)
        if not _is_model_invocable(matched):
            logger.info("[Skills] load_skill %r refused: user-only", name)
            return LOAD_SKILL_USER_ONLY.format(name=name)
        if normalized_name in self._already_loaded:
            logger.info("[Skills] load_skill %r already loaded in this conversation", name)
            return LOAD_SKILL_ALREADY_ACTIVE.format(name=name)
        instructions, reason = self._resolve_instructions(matched)
        if instructions is None:
            logger.warning("[Skills] load_skill %r unavailable: %s", name, reason)
            return LOAD_SKILL_UNAVAILABLE.format(name=name, reason=reason)
        self._already_loaded.add(normalized_name)
        logger.info("[Skills] load_skill served %r", name)
        return render_loaded_skill(LOADED_SKILL_RESULT, name, instructions, matched.get('files'), self.scripts_runnable)

    def _is_invoked(self, matched: dict) -> bool:
        invoked_keys = {_build_skill_identity(s) for s in self.invoked_skills if s.get('skill_id') is not None}
        invoked_names = {_normalize_skill_name(s) for s in self.invoked_skills}
        return (matched.get('skill_id') is not None and _build_skill_identity(matched) in invoked_keys) \
            or _normalize_skill_name(matched) in invoked_names

    def _resolve_instructions(self, matched: dict) -> tuple:
        inline = matched.get('instructions')
        if isinstance(inline, str) and inline.strip():
            return inline, ''
        if not callable(self.body_fetcher):
            return None, LOAD_SKILL_UNAVAILABLE_TRANSIENT
        try:
            body = self.body_fetcher(
                matched.get('skill_project_id'), matched.get('skill_id'),
                matched.get('skill_version_id', matched.get('version_id')),
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
        return self.load(skill)

    async def _arun(self, skill: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.load(skill)


class ReadSkillFileInput(BaseModel):
    skill: str = Field(description="Exact name of a skill that has files.")
    path: str = Field(description="Path of the file relative to the skill root, as listed in <skill_files>, such as \"forms.md\".")
    start_line: Optional[int] = Field(default=None, description="First line to read (1-based, inclusive). Omit to read from the beginning.")
    end_line: Optional[int] = Field(default=None, description="Last line to read (1-based, inclusive). Omit to read to the end.")


class ListSkillFilesInput(BaseModel):
    skill: str = Field(description="Exact name of a skill that has files.")
    dir: Optional[str] = Field(default=None, description="Folder relative to the skill root, such as \"scripts\". Omit to list every file.")


class SkillFileTool(BaseTool):
    """Serves the files of skills in the registry or active through ~name this turn, and nothing else; a skill not
    loaded yet is loaded through ``loader`` by the same call."""

    registry: Any = Field(exclude=True)
    invoked_skills: list = Field(default_factory=list, exclude=True)
    loader: Optional[Any] = Field(default=None, exclude=True)
    metadata: dict = {
        "toolkit_type": "internal",
        "toolkit_name": "skills",
        "display_name": "Skills",
    }
    _already_loaded: set = PrivateAttr(default_factory=set)

    def mark_already_loaded(self, names) -> None:
        """Seed the names of skills whose bodies are in the conversation (called by the tool loop)."""
        self._already_loaded.update((n or '').strip().lower() for n in names or ())

    def _resolve_loaded_skill(self, skill: str) -> tuple:
        """The skill entry, its files and the body this call loaded (empty when already loaded), or the refusal."""
        normalized = (skill or '').strip().lower()
        invoked = {_normalize_skill_name(s): s for s in self.invoked_skills if isinstance(s, dict)}
        matched = self.registry.resolve(skill) or invoked.get(normalized)
        if not matched:
            with_files = [s for s in self.registry.disclosable + list(invoked.values()) if select_skill_files(s.get('files'))]
            available = sorted({s.get('name') for s in with_files if s.get('name')})
            return None, [], SKILL_FILE_UNKNOWN_SKILL.format(name=skill, available=', '.join(available) or '(none)'), ''
        name = matched.get('name')
        files = select_skill_files(matched.get('files'))
        if not files:
            return None, [], SKILL_FILE_SKILL_WITHOUT_FILES.format(name=name), ''
        if normalized not in invoked and not _is_model_invocable(matched):
            logger.info("[Skills] %s refused for %r: user-only", self.name, name)
            return None, [], LOAD_SKILL_USER_ONLY.format(name=name), ''
        if normalized in self._already_loaded or normalized in invoked:
            return matched, files, None, ''
        if self.loader is None:
            logger.info("[Skills] %s refused for %r: not loaded", self.name, name)
            return None, [], SKILL_FILE_SKILL_NOT_LOADED.format(name=name), ''
        loaded = self.loader.load(name)
        if LOAD_SKILL_ALREADY_ACTIVE_RE.match(loaded):
            loaded = ''
        elif not LOADED_SKILL_PREFIX_RE.match(loaded):
            return None, [], loaded, ''
        self._already_loaded.add(normalized)
        logger.info("[Skills] %s loaded %r on the way", self.name, name)
        return matched, files, None, loaded

    @staticmethod
    def _after_load(loaded: str, answer) -> str:
        """The tool's answer, after the skill body this call loaded."""
        return SKILL_FILE_AFTER_LOAD.format(loaded=loaded, result=answer) if loaded else answer


def normalize_skill_path(name: str, path: str) -> Optional[str]:
    """The path relative to the skill root, or None when it points outside the skill."""
    path = (path or '').strip()
    if path.lower().startswith(SKILL_URI_SCHEME):
        root_name, _separator, path = path[len(SKILL_URI_SCHEME):].partition('/')
        if root_name.strip().lower() != (name or '').strip().lower():
            return None
    if path.startswith('/'):
        return None
    normalized = posixpath.normpath(path) if path else ''
    if normalized == '..' or normalized.startswith('../'):
        return None
    return '' if normalized == '.' else normalized


def find_skill_file(files: list, path: str) -> tuple:
    """The manifest entry for a path (exact, then case-insensitive) and, when there is none, the closest manifest paths."""
    exact = [entry for entry in files if entry['path'] == path]
    if exact:
        return exact[0], []
    folded = [entry for entry in files if entry['path'].casefold() == path.casefold()]
    if len(folded) == 1:
        return folded[0], []
    if folded:
        return None, [entry['path'] for entry in folded]
    by_folded = {entry['path'].casefold(): entry['path'] for entry in files}
    closest = difflib.get_close_matches(path.casefold(), list(by_folded), n=SKILL_FILE_SUGGESTION_LIMIT, cutoff=0)
    return None, [by_folded[match] for match in closest]


def describe_line_range(start_line: Optional[int], end_line: Optional[int]) -> str:
    """The line range as it appears in a read result; empty for a whole-file read."""
    if start_line is None and end_line is None:
        return ''
    return READ_SKILL_FILE_LINES.format(start=start_line or 1, end='end' if end_line is None else end_line)


class ReadSkillFileTool(SkillFileTool):
    name: str = "read_skill_file"
    description: str = READ_SKILL_FILE_TOOL_DESCRIPTION
    args_schema: type[BaseModel] = ReadSkillFileInput
    file_fetcher: Optional[Any] = Field(default=None, exclude=True)
    _already_read: set = PrivateAttr(default_factory=set)

    def mark_already_read(self, keys) -> None:
        """Seed the (skill, path, lines) reads whose contents are in the conversation (called by the tool loop)."""
        self._already_read.update(keys or ())

    def _read(self, skill: str, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None):
        """One file of a skill as text, after its body when this call loaded it, or the reason the model gets."""
        matched, files, refusal, loaded = self._resolve_loaded_skill(skill)
        if refusal:
            return refusal
        return self._after_load(loaded, self._read_file(matched, files, path, start_line, end_line))

    def _read_file(self, matched: dict, files: list, path: str, start_line: Optional[int], end_line: Optional[int]):
        """One file of a loaded skill as text, or the reason the model gets instead."""
        name = matched.get('name')
        relative_path = normalize_skill_path(name, path)
        if relative_path is None:
            return SKILL_FILE_OUTSIDE_ROOT.format(name=name, path=path)
        entry, suggestions = find_skill_file(files, relative_path)
        if entry is None:
            return SKILL_FILE_NOT_FOUND.format(
                name=name, path=path, suggestions=', '.join(f'"{s}"' for s in suggestions),
            )
        file_path = entry['path']
        lines = describe_line_range(start_line, end_line)
        read_key = (_normalize_skill_name(matched), file_path, lines)
        if read_key in self._already_read:
            return READ_SKILL_FILE_ALREADY_READ.format(name=name, path=file_path, lines=lines)
        if not entry.get('is_text'):
            return SKILL_FILE_BINARY.format(
                path=file_path, content_type=entry.get('content_type') or 'file', size=entry.get('size') or 0,
            )
        content, reason = self._fetch(matched, file_path)
        if content is None:
            logger.warning("[Skills] read_skill_file %r/%r unavailable: %s", name, file_path, reason)
            return SKILL_FILE_UNAVAILABLE.format(name=name, path=file_path, reason=reason)
        bounded = bound_read_result(content, file_path, start_line=start_line, end_line=end_line)
        if not isinstance(bounded, str):
            return json.dumps(bounded, ensure_ascii=False)
        self._already_read.add(read_key)
        logger.info("[Skills] read_skill_file served %r/%r", name, file_path)
        return READ_SKILL_FILE_RESULT.format(name=name, path=file_path, lines=lines, content=bounded)

    def _fetch(self, matched: dict, path: str) -> tuple:
        """The file's text, or None and the reason it could not be fetched."""
        if not callable(self.file_fetcher):
            return None, SKILL_FILE_UNAVAILABLE_TRANSIENT
        try:
            data = self.file_fetcher(
                matched.get('skill_project_id'), matched.get('skill_id'),
                matched.get('skill_version_id', matched.get('version_id')), path,
            )
        except SkillBodyUnavailable as e:
            return None, SKILL_FILE_UNAVAILABLE_DENIED if e.denied else SKILL_FILE_UNAVAILABLE_TRANSIENT
        except Exception as e:  # pylint: disable=W0703
            logger.warning("[Skills] file fetch for %r/%r failed: %s", matched.get('name'), path, e)
            return None, SKILL_FILE_UNAVAILABLE_TRANSIENT
        if isinstance(data, str):
            return data, ''
        return bytes(data or b'').decode('utf-8', errors='replace'), ''

    def _run(self, skill: str, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None):
        return self._read(skill, path, start_line, end_line)

    async def _arun(self, skill: str, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None,
                    run_manager: Optional[CallbackManagerForToolRun] = None):
        return self._read(skill, path, start_line, end_line)


class ListSkillFilesTool(SkillFileTool):
    name: str = "list_skill_files"
    description: str = LIST_SKILL_FILES_TOOL_DESCRIPTION
    args_schema: type[BaseModel] = ListSkillFilesInput

    def _list(self, skill: str, folder: Optional[str] = None) -> str:
        """The files of a skill, or of one of its folders, after its body when this call loaded it."""
        matched, files, refusal, loaded = self._resolve_loaded_skill(skill)
        if refusal:
            return refusal
        return self._after_load(loaded, self._list_files(matched, files, folder))

    def _list_files(self, matched: dict, files: list, folder: Optional[str]) -> str:
        """The files of a loaded skill, or of one of its folders, with sizes."""
        name = matched.get('name')
        relative_folder = normalize_skill_path(name, folder or '')
        if relative_folder is None:
            return SKILL_FILE_OUTSIDE_ROOT.format(name=name, path=folder)
        prefix = f"{relative_folder.rstrip('/')}/" if relative_folder else ''
        selected = [entry for entry in files if entry['path'].casefold().startswith(prefix.casefold())]
        if not selected:
            folders = sorted({
                posixpath.dirname(entry['path']) for entry in files if '/' in entry['path']
            }, key=str.casefold)
            return SKILL_FILE_FOLDER_NOT_FOUND.format(name=name, path=folder, available=', '.join(folders) or '(none)')
        listing = '\n'.join(
            SKILL_FILES_LISTING_FILE.format(path=entry['path'], size=format_file_size(entry.get('size')))
            for entry in sorted(selected, key=lambda entry: entry['path'].casefold())
        )
        return LIST_SKILL_FILES_RESULT.format(
            name=name, root=f"{SKILL_ROOT_URI.format(name=name)}/{prefix}", listing=listing,
        )

    def _run(self, skill: str, dir: Optional[str] = None,  # pylint: disable=W0622
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._list(skill, dir)

    async def _arun(self, skill: str, dir: Optional[str] = None,  # pylint: disable=W0622
                    run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._list(skill, dir)


class RunSkillScriptInput(BaseModel):
    skill: str = Field(description="Exact name of a loaded skill that has Python scripts.")
    path: str = Field(description="Path of the .py script relative to the skill root, as listed in <skill_files>, such as \"scripts/check.py\".")
    args: List[str] = Field(default_factory=list, description="Command-line arguments for the script, one string each, as sys.argv[1:].")
    files: List[str] = Field(default_factory=list, description="Names of conversation attachments the script must read; each appears under inputs/ in the working directory.")


class RunSkillScriptTool(SkillFileTool):
    name: str = "run_skill_script"
    description: str = RUN_SKILL_SCRIPT_TOOL_DESCRIPTION.format(
        timeout=int(_read_sandbox_limits_from_env()['timeout_seconds']), inline_chars=INLINE_OUTPUT_CHARS,
    )
    args_schema: type[BaseModel] = RunSkillScriptInput
    runner: Any = Field(exclude=True)

    def _resolve_script(self, skill: str, path: str):
        """The loaded skill, its manifest, the script's path and the body this call loaded, or the reason the model
        gets instead."""
        matched, manifest, refusal, loaded = self._resolve_loaded_skill(skill)
        if refusal:
            return None, refusal
        name = matched.get('name')
        relative_path = normalize_skill_path(name, path)
        if relative_path is None:
            return None, self._after_load(loaded, SKILL_FILE_OUTSIDE_ROOT.format(name=name, path=path))
        entry, suggestions = find_skill_file(manifest, relative_path)
        if entry is None:
            return None, self._after_load(loaded, SKILL_FILE_NOT_FOUND.format(
                name=name, path=path, suggestions=', '.join(f'"{s}"' for s in suggestions),
            ))
        if not is_python_script(entry['path']):
            return None, self._after_load(loaded, (
                f'"{entry["path"]}" cannot run here: only Python (.py) scripts run on this platform. '
                'Read it with read_skill_file and follow it yourself.'
            ))
        return (matched, manifest, entry['path'], loaded), None

    def _result(self, loaded: str, outcome):
        """The run's text, after the body this call loaded, carrying the run's facts when the sandbox ran it."""
        text, facts = outcome
        content = self._after_load(loaded, text)
        if facts is None:
            return content
        return ToolMessage(content=content, artifact=facts, name=self.name, tool_call_id='')

    def _run(self, skill: str, path: str, args: Optional[List[str]] = None, files: Optional[List[str]] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None):
        target, refusal = self._resolve_script(skill, path)
        if refusal:
            return refusal
        matched, manifest, script, loaded = target
        return self._result(loaded, self.runner.run(
            matched, manifest, script, args or [], files or [], run_manager.get_child() if run_manager else None,
        ))

    async def _arun(self, skill: str, path: str, args: Optional[List[str]] = None, files: Optional[List[str]] = None,
                    run_manager: Optional[CallbackManagerForToolRun] = None):
        target, refusal = self._resolve_script(skill, path)
        if refusal:
            return refusal
        matched, manifest, script, loaded = target
        return self._result(loaded, await self.runner.arun(
            matched, manifest, script, args or [], files or [], run_manager.get_child() if run_manager else None,
        ))


class ReadSkillOutputInput(BaseModel):
    ref: str = Field(description="The skill-output://<id> reference a run_skill_script result gave.")
    start_line: Optional[int] = Field(default=None, description="First line to read (1-based, inclusive). Omit to read from the beginning.")
    end_line: Optional[int] = Field(default=None, description="Last line to read (1-based, inclusive). Omit to read to the end.")


class ReadSkillOutputTool(BaseTool):
    """Serves the saved output of this conversation's script runs, by line range, and nothing else."""

    name: str = "read_skill_output"
    description: str = READ_SKILL_OUTPUT_TOOL_DESCRIPTION
    args_schema: type[BaseModel] = ReadSkillOutputInput
    store: Any = Field(exclude=True)
    metadata: dict = {
        "toolkit_type": "internal",
        "toolkit_name": "skills",
        "display_name": "Skills",
    }

    def _read(self, ref: str, start_line: Optional[int] = None, end_line: Optional[int] = None):
        """A saved script output as text, or the reason the model gets instead."""
        text = self.store.read_output(ref)
        if text is None:
            return (
                f'"{ref}" is not an available saved script output. Pass the skill-output://<id> reference exactly as '
                "a run_skill_script result gave it; if it has expired with the conversation's script storage, run the "
                "script again."
            )
        bounded = bound_read_result(text, "script-output.txt", start_line=start_line, end_line=end_line)
        if not isinstance(bounded, str):
            return json.dumps(bounded, ensure_ascii=False)
        lines = describe_line_range(start_line, end_line)
        return f"Script output {ref}{lines}:\n<script_output>\n{bounded}\n</script_output>"

    def _run(self, ref: str, start_line: Optional[int] = None, end_line: Optional[int] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None):
        return self._read(ref, start_line, end_line)

    async def _arun(self, ref: str, start_line: Optional[int] = None, end_line: Optional[int] = None,
                    run_manager: Optional[CallbackManagerForToolRun] = None):
        return self._read(ref, start_line, end_line)


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


def _tokenize(text: str) -> List[str]:
    normalized = unicodedata.normalize('NFKC', str(text or '')).casefold().replace(DOTTED_SMALL_I, 'i')
    return [
        term
        for word in _split_words(normalized)
        for variant in _expand_format_variants(word)
        for term in _split_unsegmented_runs(variant)
    ]


def _is_word_character(character: str) -> bool:
    category = unicodedata.category(character)
    if category == FORMAT_CATEGORY:
        return character != ZERO_WIDTH_SPACE
    return category[0] in WORD_CATEGORY_CLASSES


def _split_words(text: str) -> List[str]:
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


def _expand_format_variants(word: str) -> List[str]:
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


def _split_unsegmented_runs(word: str) -> List[str]:
    terms = []
    position = 0
    for run in UNSEGMENTED_RUN_PATTERN.finditer(word):
        if run.start() > position:
            terms.append(word[position:run.start()])
        terms.extend(_build_bigrams(run.group()))
        position = run.end()
    if position < len(word):
        terms.append(word[position:])
    return terms


def _build_bigrams(run: str) -> List[str]:
    if len(run) == 1:
        return [run]
    return [run[index:index + 2] for index in range(len(run) - 1)]


def parse_skill_query(query: str) -> tuple:
    terms, phrases = [], []
    for match in QUERY_PART_PATTERN.finditer(str(query or '')):
        phrase = match.group('phrase')
        tokens = _tokenize(phrase if phrase is not None else match.group('term'))
        if not tokens:
            continue
        terms.extend(tokens)
        if phrase is not None:
            phrases.append(tuple(tokens))
    return list(dict.fromkeys(terms)), list(dict.fromkeys(phrases))


def _contains_phrase(sequence: List[str], phrase: tuple) -> bool:
    length = len(phrase)
    return any(tuple(sequence[index:index + length]) == phrase for index in range(len(sequence) - length + 1))


def _matches_query_terms(query_terms: List[str], document: Counter, name_terms: set) -> bool:
    content_terms = [term for term in query_terms if term not in SEARCH_SKILLS_STOPWORDS]
    if content_terms:
        return any(document[term] for term in content_terms)
    return any(term in name_terms for term in query_terms)


def rank_skills_scored(query: str, skills: list, limit: int = SEARCH_SKILLS_RESULT_LIMIT) -> list:
    query_terms, phrases = parse_skill_query(query)
    if not query_terms or not skills:
        return []
    name_tokens = [_tokenize(skill.get('name')) for skill in skills]
    description_tokens = [_tokenize(skill.get('description')) for skill in skills]
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
        if not _matches_query_terms(query_terms, document, set(names)):
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
    skill_file_fetcher=None, skill_script_runner=None,
) -> List[BaseTool]:
    if not attached_skills and not project_skills:
        return []
    registry = SkillRegistry.from_sources(attached_skills, project_skills, build_skill_ranker(skill_search))
    if not project_skills:
        logger.debug("[Skills] Binding load_skill with %d attached skills", len(registry))
        tools = [LoadSkillTool(
            registry=registry, invoked_skills=invoked_skills or [], body_fetcher=skill_body_fetcher,
            listed=True, description=LOAD_SKILL_LISTED_TOOL_DESCRIPTION, args_schema=LoadSkillListedInput,
        )]
    else:
        logger.debug(
            "[Skills] Binding search_skills + load_skill over %d skills (%d attached, %d project)",
            len(registry), len(attached_skills or []), len(project_skills or []),
        )
        tools = [
            SearchSkillsTool(registry=registry),
            LoadSkillTool(registry=registry, invoked_skills=invoked_skills or [], body_fetcher=skill_body_fetcher),
        ]
    if not any(select_skill_files(skill.get('files')) for skill in registry.entries):
        return tools
    logger.debug("[Skills] Binding read_skill_file + list_skill_files: a registry skill has files")
    loader = tools[-1]
    tools = tools + [
        ReadSkillFileTool(
            registry=registry, invoked_skills=invoked_skills or [], loader=loader, file_fetcher=skill_file_fetcher,
        ),
        ListSkillFilesTool(registry=registry, invoked_skills=invoked_skills or [], loader=loader),
    ]
    if skill_script_runner is None or not any(
        has_python_scripts(select_skill_files(skill.get('files'))) for skill in registry.entries
    ):
        return tools
    logger.debug("[Skills] Binding run_skill_script + read_skill_output: a registry skill has Python scripts")
    loader.scripts_runnable = True
    return tools + [
        RunSkillScriptTool(registry=registry, invoked_skills=invoked_skills or [], loader=loader, runner=skill_script_runner),
        ReadSkillOutputTool(store=skill_script_runner.store),
    ]


def render_skill_registry_index(attached_skills, project_skills=None, project_skill_summary=None) -> str:
    if project_skills:
        return SKILL_SEARCH_PROMPT
    listed = [s for s in attached_skills or [] if _is_model_invocable(s)]
    if not listed:
        return ""
    ordered = sorted(
        listed, key=lambda s: (s.get('skill_id') or 0, s.get('name') or '')
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



def extract_request_text(task) -> str:
    if isinstance(task, str):
        return task
    if isinstance(task, list):
        return ' '.join(
            block.get('text') or '' for block in task
            if isinstance(block, dict) and block.get('type') == 'text'
        )
    return ''


def render_skill_suggestions(registry: SkillRegistry, task, excluded_names=()) -> str:
    query = ' '.join(
        term for term in _tokenize(extract_request_text(task))
        if len(term) >= SKILL_SUGGESTIONS_MIN_TOKEN_CHARS or not term.isascii()
    )
    if not query or not len(registry):
        return ""
    excluded = {(name or '').strip().lower() for name in excluded_names}
    scored = [
        (score, skill) for score, skill in registry.search_scored(query, SKILL_SUGGESTIONS_LIMIT + len(excluded))
        if _normalize_skill_name(skill) not in excluded
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
