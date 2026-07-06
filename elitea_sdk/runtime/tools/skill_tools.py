"""Progressive disclosure of agent skills via the load_skill meta-tool.

Skills attached to an agent version (but not baked into the cached instructions)
are advertised as a name+description registry index in the cached system prefix.
The model loads a single skill body on demand by calling load_skill; the body is
returned as a tool result (ToolMessage) and rides the message history, never the
cached prefix.
"""

import logging
import re
from typing import List, Optional
from xml.sax.saxutils import escape as xml_escape

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from ..langchain.constants import (
    LOAD_SKILL_ALREADY_ACTIVE,
    LOAD_SKILL_TOOL_DESCRIPTION,
    LOAD_SKILL_UNKNOWN,
    LOADED_SKILL_RESULT,
    SKILL_REGISTRY_ENTRY,
    SKILL_REGISTRY_HEADER,
)

logger = logging.getLogger(__name__)


# Anchored on the fixed prefix of LOADED_SKILL_RESULT (constants.py) — keep in sync.
_LOADED_SKILL_PREFIX_RE = re.compile(r'^Skill "([^"]+)" is now active')


def loaded_skill_names_from_messages(messages) -> set:
    """Names (lowercased) of skills whose loaded bodies are present in the given
    message history. Derived from context rather than stored state: if
    summarization drops a body, its name disappears from this set and a genuine
    re-load becomes possible again."""
    names = set()
    for message in messages or []:
        content = getattr(message, 'content', None)
        if getattr(message, 'type', '') == 'tool' and isinstance(content, str):
            match = _LOADED_SKILL_PREFIX_RE.match(content)
            if match:
                names.add(match.group(1).strip().lower())
    return names


class LoadSkillInput(BaseModel):
    skill: str = Field(description="Exact name of the skill to load, as listed inside <available_skills> in the system prompt.")


class LoadSkillTool(BaseTool):
    """Serves only skills in the per-turn attached_skills closure (structurally enforced); the loaded body is returned as a ToolMessage and never injected into the cached system prefix."""

    name: str = "load_skill"
    description: str = LOAD_SKILL_TOOL_DESCRIPTION
    args_schema: type[BaseModel] = LoadSkillInput
    attached_skills: list = Field(exclude=True)
    invoked_skills: list = Field(default_factory=list, exclude=True)
    metadata: dict = {
        "toolkit_type": "internal",
        "toolkit_name": "skills",
        "display_name": "Skills",
    }
    _served: set = PrivateAttr(default_factory=set)

    def _load(self, skill: str) -> str:
        query = (skill or '').strip()
        available_names = sorted(
            (s.get('name') or '') for s in self.attached_skills if s.get('name')
        )
        matched = next(
            (
                s for s in self.attached_skills
                if (s.get('name') or '').strip().lower() == query.lower()
            ),
            None,
        )
        if not matched:
            logger.info("[Skills] load_skill unknown name %r; available=%s", skill, available_names)
            return LOAD_SKILL_UNKNOWN.format(
                name=skill, available=', '.join(available_names) or '(none)'
            )
        name = matched.get('name')
        invoked_ids = {s.get('skill_id') for s in self.invoked_skills}
        invoked_names = {(s.get('name') or '').strip().lower() for s in self.invoked_skills}
        if matched.get('skill_id') in invoked_ids or (name or '').strip().lower() in invoked_names:
            logger.info("[Skills] load_skill %r already active via ~name", name)
            return LOAD_SKILL_ALREADY_ACTIVE.format(name=name)
        if (name or '').strip().lower() in self._served:
            logger.info("[Skills] load_skill %r already loaded in this conversation", name)
            return LOAD_SKILL_ALREADY_ACTIVE.format(name=name)
        self._served.add((name or '').strip().lower())
        logger.info("[Skills] load_skill served %r", name)
        return LOADED_SKILL_RESULT.format(name=name, instructions=matched.get('instructions') or '')

    def mark_already_loaded(self, names) -> None:
        """Seed the already-loaded set (called by the tool loop with names derived
        from the conversation history before each invocation)."""
        self._served.update((n or '').strip().lower() for n in names or ())

    def _run(self, skill: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._load(skill)

    async def _arun(self, skill: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._load(skill)


def build_load_skill_tools(attached_skills, invoked_skills=None) -> List[BaseTool]:
    if not attached_skills:
        return []
    logger.debug("[Skills] Binding load_skill with %d attached skills", len(attached_skills))
    return [LoadSkillTool(attached_skills=attached_skills, invoked_skills=invoked_skills or [])]


def render_skill_registry_index(attached_skills) -> str:
    if not attached_skills:
        return ""
    ordered = sorted(
        attached_skills, key=lambda s: (s.get('skill_id') or 0, s.get('name') or '')
    )
    # Escape XML specials so a description (free text) cannot forge registry
    # structure. Names are platform-validated to [a-z0-9-] so escaping them is
    # a defensive no-op and never diverges from the loadable name.
    entries = [
        SKILL_REGISTRY_ENTRY.format(
            name=xml_escape(' '.join(str(s.get('name') or '').split()), {'"': '&quot;'}),
            description=xml_escape(
                ' '.join(str(s.get('description') or '').split()) or '(no description)'
            ),
        )
        for s in ordered
    ]
    logger.info("[Skills] Rendered registry index with %d skills into cached prefix", len(ordered))
    return (
        "<available_skills>\n" + SKILL_REGISTRY_HEADER + "\n\n"
        + "\n".join(entries) + "\n</available_skills>"
    )
