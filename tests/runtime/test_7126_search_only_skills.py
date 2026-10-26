"""Search-only skill disclosure: no skill reaches the prompt, search_skills finds it, load_skill serves it (#7126)."""
import asyncio
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.runtime.exceptions import SkillBodyUnavailable
from elitea_sdk.runtime.langchain.constants import (
    LOAD_SKILL_ALREADY_ACTIVE_PREFIX,
    LOAD_SKILL_UNAVAILABLE_DENIED,
    LOAD_SKILL_UNAVAILABLE_TRANSIENT,
    SEARCH_SKILLS_NO_MATCH,
    SKILL_SEARCH_PROMPT,
    SKILL_SEARCH_REMINDER_SUFFIX,
)
from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.tools.skill_tools import (
    BM25Ranker,
    EmbeddingRanker,
    LoadSkillTool,
    SearchSkillsTool,
    SkillRegistry,
    build_load_skill_tools,
    build_skill_ranker,
    render_skill_registry_index,
)


INLINE_BODY = object()


def _skill(skill_id, name, description, project_id=7, instructions=INLINE_BODY, **extra):
    return {
        'skill_id': skill_id,
        'skill_project_id': project_id,
        'skill_version_id': 100 + skill_id,
        'name': name,
        'description': description,
        'instructions': f'Body of {name}.' if instructions is INLINE_BODY else instructions,
        **extra,
    }


ATTACHED = [
    _skill(3, 'pirate-tone', 'Answer in the voice of a pirate.'),
    _skill(1, 'pdf-report', 'Use when the user asks for a PDF report of the findings.'),
]
PROJECT = [
    _skill(10, 'release-notes', 'Draft release notes from merged pull requests.'),
    _skill(11, 'sql-review', 'Review SQL queries for injection and performance issues.'),
    _skill(12, 'pdf-report', 'A project copy that must never shadow the attached one.', instructions='Shadow body.'),
    _skill(13, 'onboarding-checklist', 'Checklist for onboarding a new engineer to the team.'),
]


def _registry(attached=ATTACHED, project=PROJECT, ranker=None):
    return SkillRegistry.from_sources(attached, project, ranker)


def _search(query, attached=ATTACHED, project=PROJECT, ranker=None):
    return SearchSkillsTool(registry=_registry(attached, project, ranker)).invoke({'query': query})


def _result_names(result):
    return [line.split(':', 1)[0] for line in result.splitlines()]


def _loader(attached=ATTACHED, project=PROJECT, **kwargs):
    return LoadSkillTool(registry=_registry(attached, project), **kwargs)


class _RecordingChat(GenericFakeChatModel):
    seen: list = []
    bound_tools: list = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools.append(list(tools))
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _node(llm, **fields):
    return LLMNode(client=llm, name='llm', available_tools=[],
                   input_mapping={'system': {'type': 'fixed', 'value': 'You help.'},
                                  'task': {'type': 'fixed', 'value': 'Hello'}},
                   input_variables=[], output_variables=['messages'], return_type='dict', **fields)


def _sent(configurable, **fields):
    llm = _RecordingChat(messages=iter([AIMessage(content='done')]), seen=[], bound_tools=[])
    _node(llm, **fields).invoke({'messages': []}, {'configurable': configurable})
    system = next(m for m in llm.seen[-1] if isinstance(m, SystemMessage))
    content = system.content if isinstance(system.content, str) else ''.join(
        block.get('text', '') for block in system.content
    )
    return content, llm.bound_tools


# --- prompt -----------------------------------------------------------------

def test_the_prompt_names_no_skill_and_is_identical_for_any_catalog():
    small = render_skill_registry_index(ATTACHED, PROJECT[:1])
    large = render_skill_registry_index(
        ATTACHED, [_skill(i, f'skill-{i}', f'topic {i}') for i in range(100, 600)], 'summary text',
    )

    assert small == large == SKILL_SEARCH_PROMPT
    for skill in ATTACHED + PROJECT:
        assert skill['name'] not in small
        assert skill['description'] not in small
    assert not any(character.isdigit() for character in small)


def test_no_skills_render_nothing_and_bind_nothing():
    assert render_skill_registry_index([], []) == ''
    assert build_load_skill_tools([], None, []) == []


# --- listed mode: no project catalog means today's behaviour, byte for byte ---

def _main_renderer():
    import importlib.util
    spec = importlib.util.spec_from_file_location('skill_tools_main', MAIN_SKILL_TOOLS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAIN_SKILL_TOOLS = pathlib.Path(__file__).with_name('fixtures_7126_skill_tools_main.py')


def test_without_project_skills_the_listing_is_byte_identical_to_main():
    main = _main_renderer()
    catalogs = [
        ATTACHED,
        [_skill(i, f'skill-{i}', 'word ' * (40 * i), instruction_referenced=(i % 3 == 0)) for i in range(1, 60)],
        [_skill(1, 'a&b', 'uses <tags> & "quotes"')],
    ]
    for catalog in catalogs:
        assert render_skill_registry_index(catalog) == main.render_skill_registry_index(catalog)
        assert render_skill_registry_index(catalog, []) == main.render_skill_registry_index(catalog)
    assert '<available_skills>' in render_skill_registry_index(ATTACHED)


def test_without_project_skills_only_load_skill_is_bound_with_its_listed_description():
    main = _main_renderer()
    tools = build_load_skill_tools(ATTACHED, None, None)

    assert [type(tool) for tool in tools] == [LoadSkillTool]
    assert convert_to_openai_tool(tools[0]) == convert_to_openai_tool(main.LoadSkillTool(attached_skills=ATTACHED))
    assert 'Body of pirate-tone.' in tools[0].invoke({'skill': 'pirate-tone'})
    assert tools[0].invoke({'skill': 'nope'}) == main.LoadSkillTool(attached_skills=ATTACHED).invoke({'skill': 'nope'})


def test_without_project_skills_the_model_receives_the_listing_and_the_old_reminder():
    from elitea_sdk.runtime.langchain.constants import SKILL_REMINDER_SUFFIX
    content, bound = _sent({'attached_skills': ATTACHED})

    assert '<available_skills>' in content
    assert 'pirate-tone' in content
    assert SKILL_SEARCH_PROMPT not in content
    assert '<suggested_skills>' not in content
    assert content.endswith(SKILL_REMINDER_SUFFIX)
    assert [tool.name for tool in bound[-1]] == ['load_skill']


def test_project_skills_alone_bind_both_tools():
    tools = build_load_skill_tools(None, None, PROJECT)

    assert [type(tool) for tool in tools] == [SearchSkillsTool, LoadSkillTool]
    assert tools[0].metadata['toolkit_name'] == 'skill_search'
    assert tools[1].metadata['toolkit_name'] == 'skills'


def test_the_model_receives_the_constant_prompt_reminder_and_both_tools():
    content, bound = _sent({'project_skills': PROJECT})

    assert SKILL_SEARCH_PROMPT in content
    assert content.endswith(SKILL_SEARCH_REMINDER_SUFFIX)
    assert all(skill['name'] not in content for skill in PROJECT)
    assert sorted(tool.name for tool in bound[-1]) == ['load_skill', 'search_skills']


def test_without_skills_the_model_receives_no_skill_text():
    content, _bound = _sent({})

    assert SKILL_SEARCH_PROMPT not in content
    assert SKILL_SEARCH_REMINDER_SUFFIX not in content


# --- registry ---------------------------------------------------------------

def test_registry_precedence_is_agent_then_participant_then_project_then_lowest_id():
    attached = [
        _skill(9, 'dup', 'participant copy', origin='participant'),
        _skill(8, 'dup', 'agent copy', origin='agent'),
        _skill(7, 'dup', 'agent copy two', origin='agent'),
    ]
    project = [_skill(1, 'dup', 'project copy', origin='project')]

    registry = SkillRegistry.from_sources(attached, project)

    assert len(registry) == 1
    assert registry.resolve('DUP')['skill_id'] == 7


def test_registry_keys_entries_by_project_and_skill_id():
    registry = _registry()

    assert registry.get(7, 11)['name'] == 'sql-review'
    assert registry.get(99, 11) is None
    assert registry.resolve('missing') is None


def test_entries_without_a_name_are_dropped():
    assert len(SkillRegistry.from_sources([_skill(1, '', 'nameless')], [])) == 0


# --- search -----------------------------------------------------------------

def test_search_ranks_the_matching_project_skill_first():
    assert _result_names(_search('review my sql query for injection'))[0] == 'sql-review'


def test_search_splits_hyphenated_names_into_terms():
    assert _result_names(_search('release'))[0] == 'release-notes'


def test_a_name_term_outranks_the_same_term_in_a_description():
    attached = [
        _skill(1, 'notes-helper', 'Plans a release, then a release review.'),
        _skill(2, 'release-helper', 'Writes notes.'),
    ]

    assert _result_names(_search('release', attached, []))[0] == 'release-helper'


def test_search_uses_or_semantics_and_drops_non_matching_skills():
    names = _result_names(_search('pirate onboarding'))

    assert sorted(names) == ['onboarding-checklist', 'pirate-tone']


def test_search_returns_at_most_five_results():
    project = [_skill(i, f'report-{i}', 'Builds a report.') for i in range(100, 120)]

    assert len(_search('report', [], project).splitlines()) == 5


def test_equal_scores_keep_attached_first_then_lowest_skill_id():
    attached = [_skill(9, 'beta-report', 'report')]
    project = [_skill(5, 'gamma-report', 'report'), _skill(2, 'alpha-report', 'report')]

    assert _result_names(_search('report', attached, project)) == ['beta-report', 'alpha-report', 'gamma-report']


def test_an_attached_skill_shadows_a_project_skill_of_the_same_name():
    result = _search('pdf report')

    assert result.count('pdf-report') == 1
    assert 'never shadow' not in result
    loaded = _loader().invoke({'skill': 'PDF-Report'})
    assert 'Body of pdf-report.' in loaded
    assert 'Shadow body.' not in loaded


def test_duplicate_project_names_keep_the_lowest_skill_id():
    project = [_skill(8, 'dup', 'second', instructions='Eight.'), _skill(4, 'dup', 'first', instructions='Four.')]

    assert _search('dup', [], project) == 'dup: first'
    assert 'Four.' in _loader([], project).invoke({'skill': 'dup'})


def test_descriptions_are_cut_to_300_chars():
    description = ' '.join(['word'] * 200)

    line = _search('word', [_skill(1, 'long', description)], [])

    assert len(line.split(': ', 1)[1]) <= 300
    assert line.endswith('…')


@pytest.mark.parametrize('query', ['', '   ', '---', 'nothing-matches-zzz', '""'])
def test_an_unmatched_query_says_no_matching_skills(query):
    assert _search(query) == SEARCH_SKILLS_NO_MATCH


def test_search_runs_async():
    tool = SearchSkillsTool(registry=_registry())

    assert _result_names(asyncio.run(tool.ainvoke({'query': 'pirate'}))) == ['pirate-tone']


# --- phrases ----------------------------------------------------------------

PHRASE_SKILLS = [
    _skill(1, 'notes-release', 'Release the notes to the public.'),
    _skill(2, 'release-notes', 'Draft release notes from merged pull requests.'),
    _skill(3, 'merge-helper', 'Merges pull requests.'),
]


def test_a_quoted_phrase_must_appear_in_name_or_description():
    assert _result_names(_search('"release notes"', [], PHRASE_SKILLS)) == ['release-notes']


def test_a_quoted_phrase_may_span_the_name_and_description():
    assert _result_names(_search('"helper merges"', [], PHRASE_SKILLS)) == ['merge-helper']


def test_a_phrase_that_nothing_contains_matches_no_skill():
    assert _search('"notes release" pull', [], PHRASE_SKILLS[1:]) == SEARCH_SKILLS_NO_MATCH


def test_phrase_terms_still_score_and_loose_terms_add_matches_only_when_the_phrase_holds():
    names = _result_names(_search('"pull requests" notes', [], PHRASE_SKILLS))

    assert names == ['release-notes', 'merge-helper']


def test_an_unbalanced_quote_is_read_as_plain_words():
    assert sorted(_result_names(_search('"release notes', [], PHRASE_SKILLS))) == ['notes-release', 'release-notes']


# --- embeddings -------------------------------------------------------------

def _embedded(skill_id, name, vector, instructions=None):
    return _skill(skill_id, name, f'{name} description', embedding=vector, instructions=instructions)


EMBEDDED = [
    _embedded(1, 'alpha', [1.0, 0.0]),
    _embedded(2, 'beta', [0.0, 1.0]),
    _embedded(3, 'gamma', [0.7, 0.7]),
]


def test_embedding_mode_ranks_by_cosine_similarity_not_words():
    ranker = build_skill_ranker({'mode': 'embedding', 'query_embedder': lambda q: [0.0, 1.0]})

    assert _result_names(_search('alpha', [], EMBEDDED, ranker)) == ['beta', 'gamma', 'alpha']


def test_embedding_mode_falls_back_to_bm25_when_no_entry_has_a_vector():
    ranker = build_skill_ranker({'mode': 'embedding', 'query_embedder': lambda q: [0.0, 1.0]})

    assert _result_names(_search('alpha', [], [_skill(1, 'alpha', 'a'), _skill(2, 'beta', 'b')], ranker)) == ['alpha']


def test_embedding_mode_falls_back_to_bm25_when_the_embedder_fails():
    def failing(query):
        raise RuntimeError('embedding service down')

    ranker = build_skill_ranker({'mode': 'embedding', 'query_embedder': failing})

    assert _result_names(_search('alpha', [], EMBEDDED, ranker)) == ['alpha']


def test_entries_without_a_vector_trail_the_embedded_ones_via_bm25():
    ranker = EmbeddingRanker(lambda q: [1.0, 0.0])
    skills = EMBEDDED + [_skill(4, 'delta', 'alpha words only')]

    assert _result_names(_search('alpha', [], skills, ranker)) == ['alpha', 'gamma', 'beta', 'delta']


def test_vectors_of_another_dimension_score_zero():
    ranker = EmbeddingRanker(lambda q: [1.0, 0.0])
    skills = [_embedded(1, 'short', [1.0]), _embedded(2, 'fits', [0.5, 0.5])]

    assert _result_names(_search('anything', [], skills, ranker)) == ['fits', 'short']


@pytest.mark.parametrize('settings', [None, {}, {'mode': 'bm25'}, {'mode': 'embedding'}, {'mode': 'unknown'}])
def test_every_other_setting_yields_bm25(settings):
    assert isinstance(build_skill_ranker(settings), BM25Ranker)


def test_llm_node_passes_skill_search_from_the_configurable():
    llm = _RecordingChat(messages=iter([AIMessage(content='done')]), seen=[], bound_tools=[])
    node = _node(llm)
    configurable = {'project_skills': EMBEDDED,
                    'skill_search': {'mode': 'embedding', 'query_embedder': lambda q: [0.0, 1.0]}}

    tools = node.get_filtered_tools(config={'configurable': configurable})
    search = next(tool for tool in tools if isinstance(tool, SearchSkillsTool))

    assert _result_names(search.invoke({'query': 'alpha'})) == ['beta', 'gamma', 'alpha']


# --- load_skill -------------------------------------------------------------

def test_load_skill_serves_a_project_skill_body():
    assert 'Body of release-notes.' in _loader().invoke({'skill': 'release-notes'})


def test_an_unknown_name_points_to_search_and_lists_no_names():
    result = _loader().invoke({'skill': 'nope'})

    assert result.startswith('No skill named "nope"')
    assert 'search_skills' in result
    assert all(skill['name'] not in result for skill in ATTACHED + PROJECT)


def test_a_skill_invoked_with_tilde_in_another_project_is_not_reported_active():
    invoked = [{'skill_id': 10, 'skill_project_id': 99, 'name': 'other-name'}]

    assert 'Body of release-notes.' in _loader([], PROJECT, invoked_skills=invoked).invoke({'skill': 'release-notes'})


def test_a_skill_invoked_with_tilde_in_the_same_project_is_already_active():
    invoked = [{'skill_id': 10, 'skill_project_id': 7, 'name': 'other-name'}]

    assert _loader([], PROJECT, invoked_skills=invoked).invoke({'skill': 'release-notes'}).startswith(
        LOAD_SKILL_ALREADY_ACTIVE_PREFIX.format(name='release-notes')
    )


def test_a_project_skill_loaded_earlier_is_not_served_twice():
    tool = _loader([], PROJECT)
    tool.mark_already_loaded(['Release-Notes'])

    assert tool.invoke({'skill': 'release-notes'}).startswith(
        LOAD_SKILL_ALREADY_ACTIVE_PREFIX.format(name='release-notes')
    )


# --- lazy bodies ------------------------------------------------------------

LAZY = [_skill(20, 'lazy-skill', 'Fetched on load.', instructions=None, skill_version_id=120)]


def test_a_body_less_entry_is_fetched_by_project_skill_and_version():
    calls = []

    def fetcher(project_id, skill_id, version_id):
        calls.append((project_id, skill_id, version_id))
        return 'Fetched body.'

    result = _loader([], LAZY, body_fetcher=fetcher).invoke({'skill': 'lazy-skill'})

    assert calls == [(7, 20, 120)]
    assert 'Fetched body.' in result
    assert result.startswith('Skill "lazy-skill" is now active')


def test_an_entry_keyed_by_version_id_is_fetched_with_it():
    calls = []
    entry = {'skill_id': 21, 'skill_project_id': 7, 'version_id': 121, 'name': 'keyed', 'description': 'd'}

    _loader([], [entry], body_fetcher=lambda *args: calls.append(args) or 'Body.').invoke({'skill': 'keyed'})

    assert calls == [(7, 21, 121)]


def test_an_inline_body_is_never_fetched():
    fetcher = MagicMock(return_value='Fetched.')

    result = _loader(ATTACHED, [], body_fetcher=fetcher).invoke({'skill': 'pirate-tone'})

    assert 'Body of pirate-tone.' in result
    fetcher.assert_not_called()


@pytest.mark.parametrize('error, reason', [
    (SkillBodyUnavailable('403', denied=True), LOAD_SKILL_UNAVAILABLE_DENIED),
    (SkillBodyUnavailable('500'), LOAD_SKILL_UNAVAILABLE_TRANSIENT),
    (TimeoutError('read timed out'), LOAD_SKILL_UNAVAILABLE_TRANSIENT),
])
def test_a_failed_fetch_gives_a_clear_reply_and_keeps_the_skill_loadable(error, reason):
    attempts = []

    def fetcher(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise error
        return 'Recovered body.'

    tool = _loader([], LAZY, body_fetcher=fetcher)

    first = tool.invoke({'skill': 'lazy-skill'})
    second = tool.invoke({'skill': 'lazy-skill'})

    assert first == f'Skill "lazy-skill" could not be loaded: {reason}. Do not imitate it; proceed without it or search for another skill.'
    assert 'Recovered body.' in second


def test_a_blank_fetched_body_counts_as_unavailable():
    result = _loader([], LAZY, body_fetcher=lambda *args: '   ').invoke({'skill': 'lazy-skill'})

    assert LOAD_SKILL_UNAVAILABLE_DENIED in result


def test_without_a_fetcher_a_body_less_entry_is_reported_unavailable():
    result = _loader([], LAZY).invoke({'skill': 'lazy-skill'})

    assert LOAD_SKILL_UNAVAILABLE_TRANSIENT in result


def test_llm_node_prefers_the_configurable_fetcher_over_the_client():
    client = MagicMock()
    client.get_skill_body.return_value = 'From client.'
    node = _node(_RecordingChat(messages=iter([]), seen=[], bound_tools=[]), elitea_client=client)
    configurable = {'project_skills': LAZY, 'skill_body_fetcher': lambda *args: 'From configurable.'}

    tools = node.get_filtered_tools(config={'configurable': configurable})
    loader = next(tool for tool in tools if isinstance(tool, LoadSkillTool))

    assert 'From configurable.' in loader.invoke({'skill': 'lazy-skill'})
    client.get_skill_body.assert_not_called()


def test_llm_node_falls_back_to_the_run_clients_get_skill_body():
    client = MagicMock()
    client.get_skill_body.return_value = 'From client.'
    node = _node(_RecordingChat(messages=iter([]), seen=[], bound_tools=[]), elitea_client=client)

    tools = node.get_filtered_tools(config={'configurable': {'project_skills': LAZY}})
    loader = next(tool for tool in tools if isinstance(tool, LoadSkillTool))

    assert 'From client.' in loader.invoke({'skill': 'lazy-skill'})
    client.get_skill_body.assert_called_once_with(7, 20, 120)


# --- EliteAClient.get_skill_body ----------------------------------------------

def _client():
    return EliteAClient(base_url='https://platform.example.com', project_id=42, auth_token='tok')


def _response(status_code, payload=None):
    response = MagicMock(status_code=status_code, ok=200 <= status_code < 300, history=[])
    response.json.return_value = payload
    return response


def test_get_skill_body_calls_the_skill_details_endpoint_with_the_run_token():
    with patch('requests.Session.request') as request:
        request.return_value = _response(200, {'version_details': {'instructions': 'The body.'}})

        assert _client().get_skill_body(7, 20, 120) == 'The body.'

    args, kwargs = request.call_args
    assert args == ('get', 'https://platform.example.com/api/v2/elitea_core/skill/prompt_lib/7/20')
    assert kwargs['params'] == {'version_id': 120}
    assert kwargs['headers']['Authorization'] == 'Bearer tok'
    assert kwargs['timeout'] == (5, 30)


def test_get_skill_body_without_a_version_asks_for_the_default():
    with patch('requests.Session.request') as request:
        request.return_value = _response(200, {'version_details': {'instructions': 'Default body.'}})

        _client().get_skill_body(7, 20)

    assert request.call_args[1]['params'] is None


@pytest.mark.parametrize('status, denied', [(403, True), (404, True), (500, False), (502, False)])
def test_get_skill_body_maps_http_failures_to_denied_or_transient(status, denied):
    with patch('requests.Session.request') as request:
        request.return_value = _response(status)
        with pytest.raises(SkillBodyUnavailable) as error:
            _client().get_skill_body(7, 20, 120)

    assert error.value.denied is denied


def test_get_skill_body_treats_a_blank_body_as_denied():
    with patch('requests.Session.request') as request:
        request.return_value = _response(200, {'version_details': {'instructions': ' '}})
        with pytest.raises(SkillBodyUnavailable) as error:
            _client().get_skill_body(7, 20, 120)

    assert error.value.denied is True


def test_the_real_client_serves_a_lazy_entry_through_load_skill():
    with patch('requests.Session.request') as request:
        request.return_value = _response(200, {'version_details': {'instructions': 'Platform body.'}})
        node = _node(_RecordingChat(messages=iter([]), seen=[], bound_tools=[]), elitea_client=_client())
        loader = next(
            tool for tool in node.get_filtered_tools(config={'configurable': {'project_skills': LAZY}})
            if isinstance(tool, LoadSkillTool)
        )

        assert 'Platform body.' in loader.invoke({'skill': 'lazy-skill'})


# --- tokenisation -----------------------------------------------------------

CJK_SKILLS = [
    _skill(1, 'zh-report', '用于生成PDF格式的报告'),
    _skill(2, 'ja-minutes', '会議の議事録を作成する'),
    _skill(3, 'ko-summary', '회의 내용을 요약합니다'),
    _skill(4, 'ja-email', 'メールを書く'),
]


@pytest.mark.parametrize('query,expected', [
    ('生成报告', 'zh-report'),
    ('报告', 'zh-report'),
    ('PDF', 'zh-report'),
    ('議事録', 'ja-minutes'),
    ('会議の議事録を作成して', 'ja-minutes'),
    ('요약', 'ko-summary'),
    ('メール', 'ja-email'),
])
def test_cjk_descriptions_are_found_by_a_partial_query(query, expected):
    assert _result_names(_search(query, CJK_SKILLS, []))[0] == expected


WIDTH_AND_SCRIPT_SKILLS = [
    _skill(1, 'aws-cost', 'ＡＷＳのコストを集計する'),
    _skill(2, 'kana-notation', 'ｶﾀｶﾅ表記に統一する'),
    _skill(3, 'th-minutes', 'สรุปบันทึกการประชุมและรายงาน'),
    _skill(4, 'hi-report', 'हिन्दी रिपोर्ट बनाएँ'),
    _skill(5, 'hi-calendar', 'दिन हिम'),
    _skill(6, 'ja-names', '佐々木さんと𠮷野家の予約'),
    _skill(7, 'ja-diary', '日々の記録'),
    _skill(8, 'ja-contact', '𠮷田さんへ連絡'),
    _skill(9, 'km-minutes', 'សង្ខេបកិច្ចប្រជុំ'),
    _skill(10, 'my-minutes', 'အစည်းအဝေးမှတ်တမ်း'),
]


@pytest.mark.parametrize('query,expected', [
    ('AWS', ['aws-cost']),
    ('カタカナ', ['kana-notation']),
    ('การประชุม', ['th-minutes']),
    ('हिन्दी', ['hi-report']),
    ('佐々木', ['ja-names']),
    ('𠮷野家', ['ja-names']),
    ('ប្រជុំ', ['km-minutes']),
    ('အစည်းအဝေး', ['my-minutes']),
])
def test_width_variants_and_unsegmented_scripts_match_only_their_skill(query, expected):
    assert _result_names(_search(query, WIDTH_AND_SCRIPT_SKILLS, [])) == expected


def test_a_conversational_query_without_skill_terms_matches_nothing():
    assert _search('what is the weather in Paris') == SEARCH_SKILLS_NO_MATCH


def test_function_words_do_not_pull_in_unrelated_skills():
    assert _result_names(_search('talk like a pirate')) == ['pirate-tone']


SUPPORT_SKILLS = [
    _skill(1, 'it-support', 'Help employees with laptops, accounts, network access and software installs.'),
    *[_skill(i, f'{team}-support', f'Questions for {team}.') for i, team in enumerate(
        ['hr', 'customer', 'sales', 'legal', 'finance', 'facilities', 'travel'], start=2,
    )],
    _skill(9, 'to-do-list', 'Keep a running list of tasks for the user and mark them done.'),
    _skill(10, 'shopping-list', 'Build a list.'),
    _skill(11, 'io-performance', 'Diagnose slow disk I/O and network throughput on servers.'),
    _skill(12, 'db-performance', 'Tune performance.'),
    _skill(13, 'about-me', 'Write a short personal bio.'),
    _skill(14, 'm-and-a', 'Draft merger and acquisition due diligence notes.'),
    _skill(15, 'bio-writer', 'Share stories about me and my work.'),
]


@pytest.mark.parametrize('query,expected_first', [
    ('IT support', 'it-support'),
    ('get IT support', 'it-support'),
    ('to-do list', 'to-do-list'),
    ('I/O performance', 'io-performance'),
    ('about me', 'about-me'),
    ('M&A', 'm-and-a'),
])
def test_a_stopword_in_a_skill_name_still_identifies_it(query, expected_first):
    assert _result_names(_search(query, SUPPORT_SKILLS, []))[0] == expected_first


def test_a_query_of_stopwords_matches_only_skill_names():
    assert _result_names(_search('about me', SUPPORT_SKILLS, [])) == ['about-me']


@pytest.mark.parametrize('query', [
    'is it going to rain tomorrow',
    'translate this to french for me',
    'tell me a joke about cats',
])
def test_stopwords_in_skill_names_do_not_match_unrelated_chat(query):
    assert _search(query, SUPPORT_SKILLS, []) == SEARCH_SKILLS_NO_MATCH


FORMAT_AND_CASE_SKILLS = [
    _skill(1, 'fa-request', 'می‌خواهم گزارش بنویسم'),
    _skill(2, 'si-country', 'ශ්‍රී ලංකාව'),
    _skill(3, 'docs', 'Write docu­mentation pages'),
    _skill(4, 'tr-work', 'İş belgesi hazırla'),
    _skill(5, 'de-address', 'Adresse in der Straße prüfen'),
    _skill(6, 'km-zwsp', 'សង្ខេប​កិច្ចប្រជុំ'),
]


@pytest.mark.parametrize('query,expected', [
    ('میخواهم', ['fa-request']),
    ('می خواهم', ['fa-request']),
    ('می‌خواهم', ['fa-request']),
    ('ශ්රී', ['si-country']),
    ('documentation', ['docs']),
    ('iş', ['tr-work']),
    ('strasse', ['de-address']),
])
def test_format_characters_and_case_variants_match(query, expected):
    assert _result_names(_search(query, FORMAT_AND_CASE_SKILLS, [])) == expected


def test_a_zero_width_space_still_separates_words():
    assert _search('បក', FORMAT_AND_CASE_SKILLS, []) == SEARCH_SKILLS_NO_MATCH


def test_space_separated_text_tokenizes_as_before():
    assert _result_names(_search('review sql', ATTACHED, PROJECT)) == ['sql-review']


# --- sub-agents -------------------------------------------------------------

def _child_configurable(parent_configurable, version_details, client=None):
    child_app = MagicMock()
    captured = {}

    def record_invoke(payload, config=None):
        captured['configurable'] = config['configurable']
        return {'output': 'result'}

    child_app.invoke.side_effect = record_invoke
    if client is not None:
        client.application.return_value = child_app
    tool = Application(
        name='Child', description='Child', application=child_app, return_type='str', client=client,
        is_subgraph=True,
        args_runnable={'application_id': 1, 'application_version_id': 1, 'version_details': version_details},
    )
    tool._run(task='do the work', config={'configurable': parent_configurable, 'metadata': {}})
    return captured['configurable']


@pytest.mark.parametrize('client', [MagicMock(), None], ids=['standalone', 'subgraph'])
def test_a_sub_agent_never_inherits_the_callers_project_skills(client):
    parent = {'thread_id': 't', 'attached_skills': ATTACHED, 'project_skills': PROJECT, 'project_skill_summary': 's'}

    child = _child_configurable(parent, {}, client)

    assert child['project_skills'] == []
    assert 'project_skill_summary' not in child
    assert build_load_skill_tools(child['attached_skills'], child['invoked_skills'], child['project_skills']) == []


def test_a_sub_agent_keeps_its_own_project_skills():
    own = [_skill(50, 'child-skill', 'The child version brings its own.')]

    child = _child_configurable({'thread_id': 't', 'project_skills': PROJECT}, {'project_skills': own}, MagicMock())

    assert child['project_skills'] == own
