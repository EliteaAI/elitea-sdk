"""The skill registry stays within its description budget at 50 attached skills (#6930)."""
import hashlib
import logging
import random
import re
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import unescape

from elitea_sdk.runtime.langchain.constants import (
    INSTRUCTION_REFERENCED_HINT,
    SKILL_REGISTRY_DESCRIPTION_FLOOR_CHARS,
    SKILL_REGISTRY_DESCRIPTION_MAX_CHARS,
    SKILL_REGISTRY_TOTAL_DESCRIPTION_CHARS,
)
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.tools.skill_tools import build_load_skill_tools, render_skill_registry_index

MAX_DESCRIPTION_CHARS = 2304
ENTRY_RE = re.compile(r'<skill_option name="([^"]*)">(.*?)</skill_option>')
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _words(seed: int, chars: int) -> str:
    rng = random.Random(seed)
    words = []
    while sum(len(w) + 1 for w in words) < chars:
        words.append(''.join(rng.choice('abcdefghij') for _ in range(rng.randint(2, 11))))
    return ' '.join(words)[:chars]


def _skill(skill_id: int, description: str, **extra) -> dict:
    return {
        'skill_id': skill_id,
        'skill_version_id': 1000 + skill_id,
        'name': f'skill-{skill_id}',
        'description': description,
        'instructions': f'Full body of skill {skill_id}. ' + 'x' * 4000,
        **extra,
    }


def _long_skills(count: int) -> list:
    return [_skill(i, _words(i, MAX_DESCRIPTION_CHARS)) for i in range(1, count + 1)]


def _entries(registry: str) -> dict:
    return {name: unescape(text) for name, text in ENTRY_RE.findall(registry)}


def test_50_max_length_descriptions_fit_the_total_budget():
    entries = _entries(render_skill_registry_index(_long_skills(50)))

    assert len(entries) == 50
    assert sum(len(text) for text in entries.values()) <= SKILL_REGISTRY_TOTAL_DESCRIPTION_CHARS
    assert all(len(text) <= SKILL_REGISTRY_DESCRIPTION_MAX_CHARS for text in entries.values())


def test_a_small_registry_keeps_descriptions_up_to_the_per_skill_cap():
    skills = [_skill(1, _words(1, 400)), _skill(2, _words(2, MAX_DESCRIPTION_CHARS))]

    entries = _entries(render_skill_registry_index(skills))

    assert entries['skill-1'] == skills[0]['description']
    assert len(entries['skill-2']) <= SKILL_REGISTRY_DESCRIPTION_MAX_CHARS
    assert len(entries['skill-2']) > SKILL_REGISTRY_DESCRIPTION_MAX_CHARS - 20


def test_short_descriptions_stay_whole_and_long_ones_share_the_rest():
    short = [_skill(i, f'Use when the user asks about topic {i}.') for i in range(1, 11)]
    long_ = [_skill(i, _words(i, MAX_DESCRIPTION_CHARS)) for i in range(11, 51)]

    entries = _entries(render_skill_registry_index(short + long_))

    for skill in short:
        assert entries[skill['name']] == skill['description']
    long_lengths = [len(entries[skill['name']]) for skill in long_]
    total = sum(len(text) for text in entries.values())
    assert total <= SKILL_REGISTRY_TOTAL_DESCRIPTION_CHARS
    # Water-filling uses the budget instead of cutting every skill to the floor.
    assert total > SKILL_REGISTRY_TOTAL_DESCRIPTION_CHARS - 40 * 12
    assert max(long_lengths) - min(long_lengths) <= 12


def test_cuts_land_on_word_boundaries():
    skills = _long_skills(50)

    entries = _entries(render_skill_registry_index(skills))

    for skill in skills:
        text = entries[skill['name']]
        assert text.endswith('…')
        kept = text[:-1]
        assert skill['description'].startswith(kept)
        assert skill['description'][len(kept)] == ' '


def test_a_cut_never_splits_an_xml_escape():
    description = ('a&b <c> "d" ' * 300).strip()
    registry = render_skill_registry_index([_skill(i, description) for i in range(1, 51)])

    for _name, raw in ENTRY_RE.findall(registry):
        assert re.search(r'&(?!amp;|lt;|gt;)', raw) is None
        assert '<' not in raw and '>' not in raw
        assert description.startswith(unescape(raw)[:-1])


def test_the_registry_is_byte_stable_across_calls_and_input_order():
    skills = _long_skills(50)
    shuffled = list(skills)
    random.Random(7).shuffle(shuffled)

    assert render_skill_registry_index(skills) == render_skill_registry_index(shuffled)


def test_the_registry_is_byte_stable_across_processes():
    script = (
        'import hashlib, random, sys\n'
        f'sys.path.insert(0, {str(Path(__file__).parent)!r})\n'
        'from test_6930_skill_registry_budget import _long_skills\n'
        'from elitea_sdk.runtime.tools.skill_tools import render_skill_registry_index\n'
        'print(hashlib.sha256(render_skill_registry_index(_long_skills(50)).encode()).hexdigest())\n'
    )
    digests = {
        subprocess.run(
            [sys.executable, '-c', script], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
            env={'PYTHONHASHSEED': seed, 'PYTHONPATH': str(PROJECT_ROOT)},
        ).stdout.strip().splitlines()[-1]
        for seed in ('1', '2')
    }

    assert digests == {hashlib.sha256(render_skill_registry_index(_long_skills(50)).encode()).hexdigest()}


def test_the_instruction_reference_hint_survives_the_cut():
    skills = _long_skills(50)
    skills[3]['instruction_referenced'] = True

    entries = _entries(render_skill_registry_index(skills))

    assert entries['skill-4'].endswith(f'… {INSTRUCTION_REFERENCED_HINT}')
    assert sum(INSTRUCTION_REFERENCED_HINT in text for text in entries.values()) == 1


def test_a_referenced_skill_without_description_shows_only_the_hint():
    entries = _entries(render_skill_registry_index([_skill(1, '', instruction_referenced=True)]))

    assert entries['skill-1'] == INSTRUCTION_REFERENCED_HINT


def test_past_the_floor_every_skill_stays_visible(caplog):
    skills = _long_skills(140)

    with caplog.at_level(logging.WARNING):
        entries = _entries(render_skill_registry_index(skills))

    assert len(entries) == 140
    assert all(len(text) <= SKILL_REGISTRY_DESCRIPTION_FLOOR_CHARS for text in entries.values())
    assert all(len(text) > SKILL_REGISTRY_DESCRIPTION_FLOOR_CHARS - 15 for text in entries.values())
    assert 'exceed the registry budget' in caplog.text


def test_load_skill_returns_the_full_body_of_a_shortened_skill():
    skills = _long_skills(50)
    tool = build_load_skill_tools(skills)[0]

    result = tool.invoke({'skill': 'skill-9'})

    assert skills[8]['instructions'] in result


def test_every_skill_core_resolved_for_the_message_is_injected():
    invoked = [{'name': f'skill-{i}', 'instructions': f'Body {i}.'} for i in range(1, 9)]

    section = LLMNode._build_invoked_skills_section(invoked)

    assert section.count('<skill name=') == 8
