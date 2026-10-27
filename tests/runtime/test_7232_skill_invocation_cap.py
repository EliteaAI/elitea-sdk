"""Issue #7232 - a skill mentioned past the per-message ~skill cap is not applied that turn.

elitea_core injects the first N mentioned skills and hands the rest over as
``capped_skills``. Their names stay in the message, so load_skill must refuse them,
and the per-turn skills section tells the model they were skipped.
"""
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage

from elitea_sdk.runtime.langchain.constants import (
    LOAD_SKILL_ALREADY_ACTIVE_RE,
    LOADED_SKILL_PREFIX_RE,
)
from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.tools.skill_tools import LoadSkillTool, build_load_skill_tools


ATTACHED = [
    {'skill_id': i, 'name': f'skill-{i}', 'description': f'Use for topic {i}.', 'instructions': f'Body {i}.'}
    for i in range(1, 8)
]
INVOKED = [{**s} for s in ATTACHED[:5]]
CAPPED = [{'skill_id': s['skill_id'], 'name': s['name']} for s in ATTACHED[5:]]


def _load_skill_tool(configurable):
    node = LLMNode(client=MagicMock(), name='agent', available_tools=[])
    tools = node.get_filtered_tools({'configurable': configurable})
    return next(tool for tool in tools if tool.name == 'load_skill')


@pytest.mark.parametrize('name', ['skill-6', 'SKILL-7'])
def test_load_skill_refuses_a_capped_skill_and_withholds_its_body(name):
    tool = _load_skill_tool({'attached_skills': ATTACHED, 'invoked_skills': INVOKED, 'capped_skills': CAPPED})

    result = tool.invoke({'skill': name})

    assert 'Body 6.' not in result and 'Body 7.' not in result
    assert 'per-message limit' in result


def test_a_capped_refusal_is_not_counted_as_an_applied_skill():
    tool = build_load_skill_tools(ATTACHED, INVOKED, CAPPED)[0]

    result = tool.invoke({'skill': 'skill-6'})

    assert not LOADED_SKILL_PREFIX_RE.match(result)
    assert not LOAD_SKILL_ALREADY_ACTIVE_RE.match(result)


def test_a_skill_from_another_project_sharing_a_capped_id_still_loads():
    public = {'skill_id': 6, 'skill_project_id': 1, 'name': 'public-six', 'description': 'd', 'instructions': 'Public.'}
    tool = LoadSkillTool(attached_skills=[*ATTACHED, public], capped_skills=CAPPED)

    assert 'Public.' in tool.invoke({'skill': 'public-six'})


def test_an_injected_skill_still_reports_already_active():
    tool = build_load_skill_tools(ATTACHED, INVOKED, CAPPED)[0]

    assert LOAD_SKILL_ALREADY_ACTIVE_RE.match(tool.invoke({'skill': 'skill-1'}))


def test_an_attached_skill_nobody_mentioned_still_loads():
    attached = [*ATTACHED, {'skill_id': 8, 'name': 'skill-8', 'description': 'd', 'instructions': 'Body 8.'}]
    tool = build_load_skill_tools(attached, INVOKED, CAPPED)[0]

    assert 'Body 8.' in tool.invoke({'skill': 'skill-8'})


def test_without_a_capped_list_every_attached_skill_loads():
    tool = _load_skill_tool({'attached_skills': ATTACHED, 'invoked_skills': INVOKED})

    assert 'Body 6.' in tool.invoke({'skill': 'skill-6'})


def test_the_skills_section_names_the_skipped_skills():
    notice = LLMNode._build_capped_skills_notice(CAPPED)

    assert '"skill-6", "skill-7"' in notice
    assert 'NOT applied' in notice


@pytest.mark.parametrize('capped', [None, [], [{'skill_id': 6, 'name': '  '}], 'skill-6'])
def test_no_notice_when_nothing_was_capped(capped):
    assert LLMNode._build_capped_skills_notice(capped) == ''


class _RecordingChat(GenericFakeChatModel):
    seen: list = []

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _system_prompt_sent(configurable):
    llm = _RecordingChat(messages=iter([AIMessage(content='done')]), seen=[])
    node = LLMNode(client=llm, name='llm', available_tools=[],
                   input_mapping={'system': {'type': 'fixed', 'value': 'You help.'},
                                  'task': {'type': 'fixed', 'value': 'skill-1 skill-6'}},
                   input_variables=[], output_variables=['messages'], return_type='dict')
    node.invoke({'messages': []}, {'configurable': configurable})
    return '\n'.join(str(m.content) for m in llm.seen[-1] if isinstance(m, SystemMessage))


def test_the_model_is_told_which_mentioned_skills_were_skipped():
    system = _system_prompt_sent({'invoked_skills': INVOKED, 'capped_skills': CAPPED})

    assert '<skill name="skill-1">' in system
    assert '"skill-6", "skill-7"' in system


def test_a_turn_within_the_cap_carries_no_skipped_notice():
    system = _system_prompt_sent({'invoked_skills': INVOKED, 'capped_skills': []})

    assert '<skill name="skill-1">' in system
    assert 'per-message skill limit' not in system


def test_a_sub_agent_does_not_inherit_the_callers_capped_skills():
    child = MagicMock()
    captured = {}

    def record_invoke(payload, config=None):
        captured['configurable'] = config['configurable']
        return {'output': 'result'}

    child.invoke.side_effect = record_invoke
    client = MagicMock()
    client.application.return_value = child
    tool = Application(
        name='Child', description='Child', application=child, return_type='str', client=client,
        is_subgraph=True, args_runnable={'application_id': 1, 'application_version_id': 1, 'is_subgraph': True},
    )

    tool._run(task='work', config={'configurable': {'thread_id': 't1', 'capped_skills': CAPPED}, 'metadata': {}})

    assert captured['configurable']['capped_skills'] == []
