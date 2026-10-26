"""Pre-call skill selection: the registry is searched with the user's request before the model runs (#7126)."""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from elitea_sdk.runtime.langchain.constants import (
    LOADED_SKILL_RESULT,
    SKILL_SEARCH_PROMPT,
    SKILL_SEARCH_REMINDER_SUFFIX,
    SKILL_SUGGESTIONS_FOOTER,
    SKILL_SUGGESTIONS_HEADER,
    SKILL_SUGGESTIONS_LIMIT,
)
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.tools.skill_tools import SkillRegistry, render_skill_suggestions, extract_request_text


def _skill(skill_id, name, description, **extra):
    return {'skill_id': skill_id, 'skill_project_id': 7, 'skill_version_id': 100 + skill_id,
            'name': name, 'description': description, 'instructions': f'Body of {name}.', **extra}


PROJECT = [
    _skill(10, 'release-notes', 'Draft release notes or a changelog from merged pull requests.'),
    _skill(11, 'sql-review', 'Review SQL queries for injection risk and performance problems.'),
    _skill(12, 'onboarding-checklist', 'Checklist for onboarding a new engineer to the team.'),
]
RELEASE_REQUEST = 'Please draft release notes for the pull requests we merged this sprint.'


class _RecordingChat(GenericFakeChatModel):
    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _system_blocks(configurable, task=RELEASE_REQUEST, chat_history=None):
    llm = _RecordingChat(messages=iter([AIMessage(content='done')]), seen=[])
    node = LLMNode(client=llm, name='llm', available_tools=[],
                   input_mapping={'system': {'type': 'fixed', 'value': 'You help.'},
                                  'task': {'type': 'fixed', 'value': task},
                                  'chat_history': {'type': 'fixed', 'value': chat_history or []}},
                   input_variables=[], output_variables=['messages'], return_type='dict')
    node.invoke({'messages': []}, {'configurable': configurable})
    system = next(m for m in llm.seen[-1] if isinstance(m, SystemMessage))
    content = system.content if isinstance(system.content, str) else ''.join(
        block.get('text', '') for block in system.content
    )
    end_of_prefix = content.index(SKILL_SEARCH_PROMPT) + len(SKILL_SEARCH_PROMPT)
    return content[:end_of_prefix], content[end_of_prefix:].strip()


def test_the_matching_project_skill_is_suggested_for_the_request():
    prefix, suffix = _system_blocks({'project_skills': PROJECT})

    assert SKILL_SEARCH_PROMPT in prefix
    assert 'release-notes' not in prefix
    assert suffix.startswith(SKILL_SUGGESTIONS_HEADER)
    assert 'release-notes: Draft release notes or a changelog from merged pull requests.' in suffix
    assert 'sql-review' not in suffix
    assert suffix.index(SKILL_SUGGESTIONS_FOOTER) < suffix.index(SKILL_SEARCH_REMINDER_SUFFIX)


def test_the_cached_prefix_is_identical_whatever_the_request_matches():
    prefix_release, _ = _system_blocks({'project_skills': PROJECT})
    prefix_sql, suffix_sql = _system_blocks({'project_skills': PROJECT}, task='is this SQL query safe?')
    prefix_none, suffix_none = _system_blocks({'project_skills': PROJECT}, task='What is 2 + 2?')

    assert prefix_release == prefix_sql == prefix_none
    assert 'sql-review' in suffix_sql
    assert SKILL_SUGGESTIONS_HEADER not in suffix_none
    assert suffix_none.strip() == SKILL_SEARCH_REMINDER_SUFFIX


def test_a_tilde_turn_gets_no_suggestions_at_all():
    invoked = [{**PROJECT[0], 'instructions': 'Body.'}]
    _prefix, suffix = _system_blocks({'project_skills': PROJECT, 'invoked_skills': invoked},
                                     task='~release-notes and also review my sql query for injection')

    assert '<skill name="release-notes">' in suffix
    assert SKILL_SUGGESTIONS_HEADER not in suffix
    assert 'sql-review' not in suffix


def test_digits_and_two_letter_words_do_not_pull_in_skills():
    project = PROJECT + [_skill(13, 'seven', 'Checklist for 7.'), _skill(14, 'hi-bot', 'Says hi.')]
    registry = SkillRegistry.from_sources([], project)

    assert render_skill_suggestions(registry, 'What is 2 + 2? hi') == ''
    assert 'seven' not in render_skill_suggestions(registry, 'draft the release notes for build 7')
    assert 'zh-report' in render_skill_suggestions(SkillRegistry.from_sources([], [_skill(30, 'zh-report', '用于生成报告')]), '生成报告')


def test_weak_matches_below_half_the_best_score_are_dropped():
    project = [
        _skill(20, 'release-notes', 'Draft release notes from merged pull requests; release notes, changelog.'),
        _skill(21, 'notes-archive', 'Archive old notes.'),
    ]
    registry = SkillRegistry.from_sources([], project)

    block = render_skill_suggestions(registry, 'draft release notes for the merged pull requests')

    assert 'release-notes' in block
    assert 'notes-archive' not in block


def test_at_most_three_skills_are_suggested():
    project = [_skill(i, f'report-{i}', 'Builds a sales report quickly.') for i in range(100, 110)]
    registry = SkillRegistry.from_sources([], project)

    assert len(render_skill_suggestions(registry, 'build a sales report').splitlines()[2:-1]) == SKILL_SUGGESTIONS_LIMIT == 3


def test_a_skill_already_loaded_in_the_conversation_is_not_suggested_again():
    history = [
        HumanMessage(content='earlier'),
        AIMessage(content='', tool_calls=[{'id': 'c1', 'name': 'load_skill', 'args': {'skill': 'release-notes'}}]),
        ToolMessage(content=LOADED_SKILL_RESULT.format(name='release-notes', instructions='Body.'), tool_call_id='c1'),
    ]
    _prefix, suffix = _system_blocks({'project_skills': PROJECT}, chat_history=history)

    assert SKILL_SUGGESTIONS_HEADER not in suffix


def test_suggestions_can_be_switched_off_per_run():
    _prefix, suffix = _system_blocks({'project_skills': PROJECT, 'skill_search': {'suggestions': False}})

    assert SKILL_SUGGESTIONS_HEADER not in suffix
    assert SKILL_SEARCH_REMINDER_SUFFIX in suffix


def test_a_multimodal_request_is_searched_by_its_text_blocks():
    task = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}},
            {'type': 'text', 'text': 'write the release notes for this'}]
    _prefix, suffix = _system_blocks({'project_skills': PROJECT}, task=task)

    assert 'release-notes' in suffix


def test_at_most_five_skills_are_suggested_after_exclusions():
    project = [_skill(i, f'report-{i}', 'Builds a report.') for i in range(100, 110)]
    registry = SkillRegistry.from_sources([], project)

    block = render_skill_suggestions(registry, 'build me a report', excluded_names=['report-100', 'report-101'])
    names = [line.split(':', 1)[0] for line in block.splitlines()[2:-1]]
    unexcluded = render_skill_suggestions(registry, 'build me a report', excluded_names=['nope', 'none'])

    assert len(names) == SKILL_SUGGESTIONS_LIMIT
    assert not {'report-100', 'report-101'} & set(names)
    assert len(unexcluded.splitlines()[2:-1]) == SKILL_SUGGESTIONS_LIMIT


def test_suggestions_ride_the_dynamic_suffix_not_the_cached_prefix(monkeypatch):
    captured = {}
    original = LLMNode._anthropic_system_content

    def recording(system_content, client, dynamic_suffix=None):
        captured['prefix'], captured['suffix'] = system_content, dynamic_suffix
        return original(system_content, client, dynamic_suffix)

    monkeypatch.setattr(LLMNode, '_anthropic_system_content', staticmethod(recording))
    _system_blocks({'project_skills': PROJECT})

    assert SKILL_SUGGESTIONS_HEADER not in captured['prefix']
    assert SKILL_SUGGESTIONS_HEADER in captured['suffix']


def test_blank_requests_and_empty_registries_suggest_nothing():
    registry = SkillRegistry.from_sources([], PROJECT)

    assert render_skill_suggestions(registry, '   ') == ''
    assert render_skill_suggestions(registry, None) == ''
    assert render_skill_suggestions(SkillRegistry.from_sources([], []), RELEASE_REQUEST) == ''


def test_request_text_handles_every_task_shape():
    assert extract_request_text('plain') == 'plain'
    assert extract_request_text([{'type': 'text', 'text': 'a'}, 'junk', {'type': 'text', 'text': 'b'}]) == 'a b'
    assert extract_request_text({'not': 'text'}) == ''
