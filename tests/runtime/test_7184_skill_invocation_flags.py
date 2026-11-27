"""A skill marked ``disable-model-invocation`` is hidden from the model and refused when it asks for it by name (#7184).

elitea_core sends the flag as ``disable_model_invocation: true`` on the skill entry. The user's ``~name``
still injects the body through ``invoked_skills``, which never consults the registry.
"""
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, SystemMessage

from elitea_sdk.runtime.langchain.constants import (
    LOAD_SKILL_ALREADY_ACTIVE_PREFIX,
    LOAD_SKILL_ALREADY_ACTIVE_RE,
    LOAD_SKILL_LISTED_TOOL_DESCRIPTION,
    LOAD_SKILL_UNKNOWN,
    LOAD_SKILL_USER_ONLY,
    LOADED_SKILL_PREFIX_RE,
    SKILL_REMINDER_SUFFIX,
    SKILL_SEARCH_PROMPT,
)
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.tools.skill_tools import (
    ListSkillFilesTool,
    LoadSkillTool,
    ReadSkillFileTool,
    RunSkillScriptTool,
    SearchSkillsTool,
    SkillRegistry,
    build_load_skill_tools,
    render_skill_registry_index,
    render_skill_suggestions,
)
from tests.runtime.test_7126_search_only_skills import _RecordingChat, _node

DEPLOY_FILES = [{'path': 'scripts/deploy.py', 'size': 120, 'content_type': 'text/x-python', 'is_text': True}]


def _skill(skill_id, name, description, instructions=None, **extra):
    return {
        'skill_id': skill_id, 'skill_project_id': 7, 'skill_version_id': 100 + skill_id,
        'name': name, 'description': description, 'instructions': instructions or f'Body of {name}.', **extra,
    }


DEPLOY = _skill(1, 'deploy', 'Deploy the service to an environment.', files=DEPLOY_FILES, disable_model_invocation=True)
STYLE_GUIDE = _skill(2, 'style-guide', 'House style for every written answer.', user_invocable=False)
PLAIN = _skill(3, 'plain', 'Answer plainly.')
ATTACHED = [DEPLOY, STYLE_GUIDE, PLAIN]


def _tool(tool_type, attached=ATTACHED, invoked=None, project=None, **kwargs):
    tools = build_load_skill_tools(list(attached), invoked, project, **kwargs)
    return next(tool for tool in tools if isinstance(tool, tool_type))


def _listing_names(attached):
    return [line.split('"')[1] for line in render_skill_registry_index(attached).splitlines() if line.startswith('<skill_option name="')]


# --- listing, search, suggestions ----------------------------------------------------------------

def test_a_user_only_skill_is_left_out_of_the_listing_and_the_others_stay():
    assert _listing_names(ATTACHED) == ['style-guide', 'plain']


def test_the_listing_bytes_are_unchanged_when_no_skill_carries_the_flag():
    assert render_skill_registry_index([STYLE_GUIDE, PLAIN]) == render_skill_registry_index(
        [{**STYLE_GUIDE, 'user_invocable': False}, {**PLAIN, 'disable_model_invocation': False}]
    )


def test_a_hidden_skill_does_not_use_up_the_description_budget():
    long_description = 'x' * 20000
    with_hidden = render_skill_registry_index([{**DEPLOY, 'description': long_description}, PLAIN])
    assert with_hidden == render_skill_registry_index([PLAIN])


def test_an_agent_whose_only_skill_is_user_only_gets_no_listing_and_no_reminder():
    assert render_skill_registry_index([DEPLOY]) == ""
    llm = _RecordingChat(messages=iter([AIMessage(content='done')]), seen=[], bound_tools=[])
    _node(llm).invoke({'messages': []}, {'configurable': {'attached_skills': [DEPLOY]}})
    system = next(m for m in llm.seen[-1] if isinstance(m, SystemMessage))
    text = system.content if isinstance(system.content, str) else ''.join(b.get('text', '') for b in system.content)
    assert '<available_skills>' not in text and SKILL_REMINDER_SUFFIX not in text
    assert any(isinstance(t, LoadSkillTool) for t in llm.bound_tools[-1])


def test_search_never_returns_a_user_only_skill():
    registry = SkillRegistry.from_sources(ATTACHED, [])
    assert [s['name'] for s in registry.search('deploy the service environment')] == []
    assert len(registry) == 3 and registry.resolve('deploy') is DEPLOY


def test_suggestions_never_name_a_user_only_skill():
    registry = SkillRegistry.from_sources(ATTACHED, [])
    assert render_skill_suggestions(registry, 'deploy the service to the staging environment') == ""


def test_the_unknown_name_answers_never_list_a_user_only_skill():
    loader = _tool(LoadSkillTool)
    assert loader.listed is True
    assert 'deploy' not in loader.load('nope') and 'plain' in loader.load('nope')
    reader = _tool(ReadSkillFileTool, skill_file_fetcher=lambda *a: b'')
    assert 'deploy' not in reader.invoke({'skill': 'nope', 'path': 'x'})


def test_the_model_is_told_to_try_a_user_named_skill_that_is_not_listed():
    """Without this, a model asked for a hidden skill answers that it does not exist and the refusal is never reached."""
    for prompt in (LOAD_SKILL_LISTED_TOOL_DESCRIPTION, SKILL_SEARCH_PROMPT):
        assert 'whether only the user can start it' in prompt
    assert 'Only skills listed in <available_skills> can be loaded' not in LOAD_SKILL_LISTED_TOOL_DESCRIPTION


# --- load_skill ---------------------------------------------------------------------------------

def test_load_skill_refuses_a_user_only_skill_without_fetching_or_marking_it():
    fetches = []
    loader = _tool(LoadSkillTool, skill_body_fetcher=lambda *a: fetches.append(a) or 'body')
    answer = loader.load('deploy')
    assert answer == LOAD_SKILL_USER_ONLY.format(name='deploy')
    assert fetches == [] and 'deploy' not in loader._already_loaded
    assert not LOADED_SKILL_PREFIX_RE.match(answer) and not LOAD_SKILL_ALREADY_ACTIVE_RE.match(answer)
    assert not answer.startswith(LOAD_SKILL_UNKNOWN.split('{name}', 1)[0])


def test_load_skill_refuses_even_when_the_skill_was_loaded_earlier_in_the_conversation():
    loader = _tool(LoadSkillTool)
    loader.mark_already_loaded({'deploy'})
    assert loader.load('deploy') == LOAD_SKILL_USER_ONLY.format(name='deploy')


def test_a_skill_the_user_started_this_turn_answers_already_active():
    loader = _tool(LoadSkillTool, invoked=[DEPLOY])
    assert loader.load('deploy').startswith(LOAD_SKILL_ALREADY_ACTIVE_PREFIX.format(name='deploy'))


def test_a_model_only_skill_loads_as_before():
    assert LOADED_SKILL_PREFIX_RE.match(_tool(LoadSkillTool).load('style-guide'))


# --- file and script tools ----------------------------------------------------------------------

def test_file_and_script_tools_stay_bound_when_the_only_skill_with_files_is_user_only():
    tools = build_load_skill_tools([DEPLOY, PLAIN], None, None, skill_file_fetcher=lambda *a: b'print(1)\n',
                                   skill_script_runner=MagicMock())
    assert {type(t) for t in tools} >= {ReadSkillFileTool, ListSkillFilesTool, RunSkillScriptTool}


def test_the_file_tools_refuse_a_user_only_skill_that_was_not_started_by_the_user():
    fetches = []
    reader = _tool(ReadSkillFileTool, skill_file_fetcher=lambda *a: fetches.append(a) or b'')
    assert reader.invoke({'skill': 'deploy', 'path': 'scripts/deploy.py'}) == LOAD_SKILL_USER_ONLY.format(name='deploy')
    assert _tool(ListSkillFilesTool).invoke({'skill': 'deploy'}) == LOAD_SKILL_USER_ONLY.format(name='deploy')
    assert fetches == []


def test_the_file_tools_refuse_even_when_the_skill_is_already_in_the_loaded_set():
    reader = _tool(ReadSkillFileTool, skill_file_fetcher=lambda *a: b'')
    reader.mark_already_loaded({'deploy'})
    assert reader.invoke({'skill': 'deploy', 'path': 'scripts/deploy.py'}) == LOAD_SKILL_USER_ONLY.format(name='deploy')


def test_the_file_tools_serve_a_user_only_skill_the_user_started_this_turn():
    reader = _tool(ReadSkillFileTool, invoked=[DEPLOY], skill_file_fetcher=lambda *a: b'print("deploy")\n')
    assert 'print("deploy")' in reader.invoke({'skill': 'deploy', 'path': 'scripts/deploy.py'})
