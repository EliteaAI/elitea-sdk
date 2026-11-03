"""A loaded skill's bundled files reach the model through read_skill_file and list_skill_files (#7185)."""
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from elitea_sdk.runtime.exceptions import SkillFileUnavailable
from elitea_sdk.runtime.langchain.constants import LOADED_SKILL_RESULT, SKILLS_SECTION_ENTRY
from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.tools.skill_tools import (
    ListSkillFilesTool,
    LoadSkillTool,
    ReadSkillFileTool,
    SearchSkillsTool,
    SkillRegistry,
    build_load_skill_tools,
    loaded_skill_names_from_messages,
    read_skill_file_keys_from_messages,
    render_loaded_skill,
)

SCRIPTS = [f'scripts/{name}.py' for name in (
    'check_bounding_boxes', 'check_fillable_fields', 'convert_pdf_to_images', 'create_validation_image',
    'extract_form_field_info', 'fill_fillable_fields', 'fill_pdf_form_with_annotations', 'check_bounding_boxes_test',
)]
PDF_FILES = [
    {'path': 'forms.md', 'size': 12034, 'content_type': 'text/markdown', 'is_text': True},
    {'path': 'reference.md', 'size': 16800, 'content_type': 'text/markdown', 'is_text': True},
    {'path': 'LICENSE.txt', 'size': 1460, 'content_type': 'text/plain', 'is_text': True},
] + [{'path': path, 'size': 3000, 'content_type': 'text/x-python', 'is_text': True} for path in SCRIPTS]
PDF_BODY = 'To fill a form, read FORMS.md and follow it.'


def _skill(name='pdf', files=PDF_FILES, skill_id=5, instructions=PDF_BODY, **extra):
    return {
        'skill_id': skill_id, 'skill_version_id': 50 + skill_id, 'name': name, 'description': f'{name} skill',
        'instructions': instructions, 'files': files, **extra,
    }


PDF = _skill()
PLAIN = _skill('tone', files=[], skill_id=6, instructions='Be kind.')
FILE_TEXT = {'forms.md': 'Step 1: open the form.\nStep 2: fill it.\n', 'scripts/fill_fillable_fields.py': 'print("fill")\n'}


def _fetcher(calls=None):
    def fetch(skill_project_id, skill_id, version_id, path):
        if calls is not None:
            calls.append((skill_project_id, skill_id, version_id, path))
        return FILE_TEXT.get(path, f'text of {path}').encode()
    return fetch


def _tools(attached=(PDF,), invoked=None, project=None, fetcher=None, loaded=('pdf',)):
    tools = build_load_skill_tools(list(attached), invoked, project, skill_file_fetcher=fetcher or _fetcher())
    for tool in tools:
        if isinstance(tool, (ReadSkillFileTool, ListSkillFilesTool)):
            tool.mark_already_loaded(loaded)
    return tools


def _reader(**kwargs):
    return next(tool for tool in _tools(**kwargs) if isinstance(tool, ReadSkillFileTool))


def _lister(**kwargs):
    return next(tool for tool in _tools(**kwargs) if isinstance(tool, ListSkillFilesTool))


def test_a_skill_with_files_binds_both_file_tools_next_to_load_skill():
    tools = build_load_skill_tools([PDF, PLAIN])

    assert [type(tool) for tool in tools] == [LoadSkillTool, ReadSkillFileTool, ListSkillFilesTool]
    for tool in tools[1:]:
        assert (tool.metadata['toolkit_type'], tool.metadata['toolkit_name']) == ('internal', 'skills')


def test_file_less_skills_bind_no_file_tool():
    assert [type(tool) for tool in build_load_skill_tools([PLAIN])] == [LoadSkillTool]
    assert [type(tool) for tool in build_load_skill_tools([PLAIN], None, [_skill('notes', files=[], skill_id=7)])] == [
        SearchSkillsTool, LoadSkillTool,
    ]


def test_a_project_skill_with_files_binds_the_file_tools_in_search_mode():
    tools = build_load_skill_tools(None, None, [_skill(skill_project_id=1)])

    assert [type(tool) for tool in tools] == [SearchSkillsTool, LoadSkillTool, ReadSkillFileTool, ListSkillFilesTool]


class _RecordingChat(GenericFakeChatModel):
    seen: list = []
    bound_tools: list = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools.append(list(tools))
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _node(llm, chat_history=None, **fields):
    return LLMNode(client=llm, name='llm', available_tools=[],
                   input_mapping={'system': {'type': 'fixed', 'value': 'You help.'},
                                  'task': {'type': 'fixed', 'value': 'Fill the PDF form.'},
                                  'chat_history': {'type': 'fixed', 'value': chat_history or []}},
                   input_variables=[], output_variables=['messages'], return_type='dict', **fields)


def _run(configurable, replies, **fields):
    llm = _RecordingChat(messages=iter(replies), seen=[], bound_tools=[])
    result = _node(llm, **fields).invoke({'messages': []}, {'configurable': configurable})
    return llm, result['messages']


def _system_text(llm):
    system = next(m for m in llm.seen[0] if isinstance(m, SystemMessage))
    if isinstance(system.content, str):
        return system.content
    return ''.join(block.get('text', '') for block in system.content)


def _call(name, args, call_id):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': call_id}])


def test_the_model_is_offered_the_file_tools_only_for_a_skill_with_files():
    with_files, _ = _run({'attached_skills': [PDF]}, [AIMessage(content='done')])
    without_files, _ = _run({'attached_skills': [PLAIN]}, [AIMessage(content='done')])

    assert [tool.name for tool in with_files.bound_tools[-1]] == ['load_skill', 'read_skill_file', 'list_skill_files']
    assert [tool.name for tool in without_files.bound_tools[-1]] == ['load_skill']


def test_the_available_skills_listing_is_byte_identical_whether_or_not_skills_have_files():
    with_files, _ = _run({'attached_skills': [PDF, PLAIN]}, [AIMessage(content='done')])
    without_files, _ = _run(
        {'attached_skills': [{**PDF, 'files': []}, PLAIN]}, [AIMessage(content='done')],
    )

    assert '<available_skills>' in _system_text(with_files)
    assert _system_text(with_files) == _system_text(without_files)


def test_load_then_read_in_one_run_serves_the_file_through_the_run_client():
    client = MagicMock()
    client.get_skill_file.side_effect = _fetcher()
    _llm, messages = _run({'attached_skills': [PDF]}, [
        _call('load_skill', {'skill': 'pdf'}, 'c1'),
        _call('read_skill_file', {'skill': 'pdf', 'path': 'FORMS.md'}, 'c2'),
        AIMessage(content='done'),
    ], elitea_client=client)

    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    assert 'Skill root: skill://pdf/' in results['c1']
    assert results['c2'].startswith('Skill "pdf" file "forms.md":\n')
    assert 'Step 1: open the form.' in results['c2']
    client.get_skill_file.assert_called_once_with(None, 5, 55, 'forms.md')


def test_a_read_already_in_the_history_is_not_served_again():
    history = [
        HumanMessage(content='Fill the PDF form.'),
        _call('load_skill', {'skill': 'pdf'}, 'c1'),
        ToolMessage(content=render_loaded_skill(LOADED_SKILL_RESULT, 'pdf', PDF_BODY, PDF_FILES), tool_call_id='c1'),
        _call('read_skill_file', {'skill': 'pdf', 'path': 'forms.md'}, 'c2'),
        ToolMessage(content=_reader().invoke({'skill': 'pdf', 'path': 'forms.md'}), tool_call_id='c2'),
    ]
    client = MagicMock()
    client.get_skill_file.side_effect = _fetcher()
    llm = _RecordingChat(messages=iter([
        _call('read_skill_file', {'skill': 'pdf', 'path': 'forms.md'}, 'c3'), AIMessage(content='done'),
    ]), seen=[], bound_tools=[])

    result = _node(llm, chat_history=history, elitea_client=client).invoke(
        {'messages': []}, {'configurable': {'attached_skills': [PDF]}},
    )

    reply = next(m for m in result['messages'] if isinstance(m, ToolMessage) and m.tool_call_id == 'c3')
    assert reply.content.startswith('Skill "pdf" file "forms.md" is already in this conversation.')
    client.get_skill_file.assert_not_called()


LOADED_PDF = render_loaded_skill(LOADED_SKILL_RESULT, 'pdf', PDF_BODY, PDF_FILES)


def test_a_read_of_a_skill_not_loaded_yet_loads_it_in_the_same_call():
    client = MagicMock()
    client.get_skill_file.side_effect = _fetcher()
    _llm, messages = _run({'attached_skills': [PDF]}, [
        _call('read_skill_file', {'skill': 'pdf', 'path': 'forms.md'}, 'c1'),
        _call('load_skill', {'skill': 'pdf'}, 'c2'),
        _call('read_skill_file', {'skill': 'pdf', 'path': 'scripts/fill_fillable_fields.py'}, 'c3'),
        AIMessage(content='done'),
    ], elitea_client=client)

    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    assert results['c1'] == LOADED_PDF + '\n\n' + (
        'Skill "pdf" file "forms.md":\n<skill_file path="forms.md">\nStep 1: open the form.\nStep 2: fill it.\n\n</skill_file>'
    )
    assert results['c2'].startswith('Skill "pdf" is already loaded')
    assert results['c3'].startswith('Skill "pdf" file "scripts/fill_fillable_fields.py":\n')
    assert client.get_skill_file.call_count == 2


def test_a_skill_loaded_by_load_skill_earlier_in_the_call_is_not_loaded_again_by_a_file_tool():
    load, reader, _lister_tool = _tools(loaded=())
    load.invoke({'skill': 'pdf'})

    assert reader.invoke({'skill': 'pdf', 'path': 'forms.md'}).startswith('Skill "pdf" file "forms.md":\n')


def test_a_skill_whose_body_cannot_be_loaded_answers_the_load_refusal_and_reads_nothing():
    calls = []
    reader = _reader(attached=(_skill(instructions=None),), fetcher=_fetcher(calls), loaded=())

    assert reader.invoke({'skill': 'pdf', 'path': 'forms.md'}).startswith('Skill "pdf" could not be loaded')
    assert calls == []


def test_a_file_tool_without_a_loader_still_refuses_a_skill_that_is_not_loaded():
    reader = ReadSkillFileTool(registry=SkillRegistry.from_sources([PDF]), file_fetcher=_fetcher())

    assert reader.invoke({'skill': 'pdf', 'path': 'forms.md'}).startswith('Skill "pdf" is not loaded in this conversation')


def test_a_body_loaded_by_a_file_tool_counts_as_loaded_and_its_read_as_read_in_the_history():
    combined = _reader(loaded=()).invoke({'skill': 'pdf', 'path': 'forms.md'})
    history = [
        _call('read_skill_file', {'skill': 'pdf', 'path': 'forms.md'}, 'c1'), ToolMessage(content=combined, tool_call_id='c1'),
        _call('list_skill_files', {'skill': 'tone'}, 'c2'),
        ToolMessage(content=render_loaded_skill(LOADED_SKILL_RESULT, 'tone', 'Be kind.', PDF_FILES) + '\n\nFiles',
                    tool_call_id='c2'),
    ]

    assert loaded_skill_names_from_messages(history) == {'pdf', 'tone'}
    assert read_skill_file_keys_from_messages(history) == {('pdf', 'forms.md', '')}


def test_an_unknown_skill_lists_the_skills_with_files():
    assert _reader(attached=(PDF, PLAIN)).invoke({'skill': 'nope', 'path': 'forms.md'}) == (
        'No skill named "nope" with files is available to this agent. Skills with files: pdf. '
        'Pass one of these exact names, or proceed without the file.'
    )


def test_a_loaded_skill_without_files_says_so():
    reader = _reader(attached=(PDF, PLAIN), loaded=('pdf', 'tone'))

    assert reader.invoke({'skill': 'tone', 'path': 'forms.md'}) == (
        'Skill "tone" has no files; its instructions are all it provides.'
    )


def test_a_skill_active_through_tilde_this_turn_is_readable_without_load_skill():
    reader = _reader(invoked=[PDF], loaded=())

    assert 'Step 1: open the form.' in reader.invoke({'skill': 'pdf', 'path': 'forms.md'})


def test_the_skill_name_resolves_case_insensitively():
    assert 'Step 1' in _reader().invoke({'skill': 'PDF', 'path': 'forms.md'})


def test_an_exact_path_is_read_from_the_skills_own_version():
    calls = []
    reader = _reader(attached=(_skill(skill_project_id=1),), fetcher=_fetcher(calls))

    assert reader.invoke({'skill': 'pdf', 'path': 'scripts/fill_fillable_fields.py'}) == (
        'Skill "pdf" file "scripts/fill_fillable_fields.py":\n'
        '<skill_file path="scripts/fill_fillable_fields.py">\nprint("fill")\n\n</skill_file>'
    )
    assert calls == [(1, 5, 55, 'scripts/fill_fillable_fields.py')]


def test_a_path_in_another_case_resolves_to_the_manifest_path():
    assert _reader().invoke({'skill': 'pdf', 'path': 'FORMS.md'}).startswith('Skill "pdf" file "forms.md":\n')


def test_a_skill_uri_and_a_dot_slash_resolve_against_the_root():
    assert _reader().invoke({'skill': 'pdf', 'path': 'skill://pdf/forms.md'}).startswith('Skill "pdf" file "forms.md":')
    assert _reader().invoke({'skill': 'pdf', 'path': './scripts/../forms.md'}).startswith('Skill "pdf" file "forms.md":')


@pytest.mark.parametrize('path', ['../SKILL.md', '/etc/passwd', 'scripts/../../x.md', 'skill://other/forms.md'])
def test_a_path_outside_the_skill_is_refused(path):
    calls = []

    reply = _reader(fetcher=_fetcher(calls)).invoke({'skill': 'pdf', 'path': path})

    assert reply == (
        f'Path "{path}" is outside skill "pdf". Pass a path relative to the skill root, as listed in <skill_files>.'
    )
    assert calls == []


def test_a_missing_path_suggests_the_closest_manifest_paths():
    reader = _reader()

    assert reader.invoke({'skill': 'pdf', 'path': 'form.md'}).startswith(
        'Skill "pdf" has no file "form.md". Did you mean: "forms.md"'
    )
    assert reader.invoke({'skill': 'pdf', 'path': 'docs/forms.md'}).startswith(
        'Skill "pdf" has no file "docs/forms.md". Did you mean: "forms.md"'
    )
    assert 'Did you mean: "' in reader.invoke({'skill': 'pdf', 'path': 'assets/none.md'})


def test_two_paths_differing_only_in_case_are_offered_instead_of_guessed():
    files = [{'path': 'Notes.md', 'size': 1, 'is_text': True}, {'path': 'NOTES.md', 'size': 1, 'is_text': True}]

    reply = _reader(attached=(_skill(files=files),)).invoke({'skill': 'pdf', 'path': 'notes.md'})

    assert reply.startswith('Skill "pdf" has no file "notes.md". Did you mean: "Notes.md", "NOTES.md"?')


def test_a_binary_file_is_described_and_never_fetched():
    calls = []
    files = PDF_FILES + [{'path': 'assets/logo.png', 'size': 2048, 'content_type': 'image/png', 'is_text': False}]

    reply = _reader(attached=(_skill(files=files),), fetcher=_fetcher(calls)).invoke(
        {'skill': 'pdf', 'path': 'assets/logo.png'},
    )

    assert reply == '"assets/logo.png" is a binary image/png of 2048 bytes; it cannot be read as text.'
    assert calls == []


@pytest.mark.parametrize('error, reason', [
    (SkillFileUnavailable('gone', denied=True), 'it is no longer accessible'),
    (SkillFileUnavailable('flaky'), 'it could not be fetched right now'),
    (RuntimeError('boom'), 'it could not be fetched right now'),
])
def test_a_failed_fetch_names_the_reason_and_the_file_stays_readable(error, reason):
    def failing(*_args):
        raise error
    reader = _reader(fetcher=failing)

    assert reader.invoke({'skill': 'pdf', 'path': 'forms.md'}) == (
        f'Skill "pdf" file "forms.md" could not be read: {reason}.'
    )
    reader.file_fetcher = _fetcher()
    assert 'Step 1' in reader.invoke({'skill': 'pdf', 'path': 'forms.md'})


BIG = ''.join(f'line {index} ' + 'x' * 200 + '\n' for index in range(1, 5001))


def _big_reader():
    files = [{'path': 'big.md', 'size': len(BIG), 'content_type': 'text/markdown', 'is_text': True}]
    return _reader(attached=(_skill(files=files),), fetcher=lambda *_args: BIG.encode())


def test_a_file_over_the_read_limit_returns_line_range_guidance():
    reply = json.loads(_big_reader().invoke({'skill': 'pdf', 'path': 'big.md'}))

    assert reply['__result_status__'] == 'content_too_large'
    assert reply['total_lines'] == 5000
    assert set(reply['instruction_for_readFile']['first_class_params']) == {'start_line', 'end_line'}


def test_a_line_range_returns_those_lines():
    reply = _big_reader().invoke({'skill': 'pdf', 'path': 'big.md', 'start_line': 2, 'end_line': 3})

    assert reply.startswith('Skill "pdf" file "big.md" lines 2-3:\n<skill_file path="big.md">\nline 2 ')
    assert 'line 3 ' in reply and 'line 4 ' not in reply


def test_reads_in_the_history_are_keyed_by_skill_path_and_range():
    reader = _reader()
    whole = reader.invoke({'skill': 'pdf', 'path': 'forms.md'})
    part = _reader().invoke({'skill': 'pdf', 'path': 'forms.md', 'start_line': 2})
    history = [
        _call('read_skill_file', {}, 'c1'), ToolMessage(content=whole, tool_call_id='c1'),
        _call('read_skill_file', {}, 'c2'), ToolMessage(content=part, tool_call_id='c2'),
        _call('other_tool', {}, 'c3'), ToolMessage(content=whole.replace('forms', 'reference'), tool_call_id='c3'),
        _call('read_skill_file', {}, 'c4'), ToolMessage(content='Path "x" is outside skill "pdf".', tool_call_id='c4'),
    ]

    assert read_skill_file_keys_from_messages(history) == {('pdf', 'forms.md', ''), ('pdf', 'forms.md', ' lines 2-end')}


def test_a_second_read_of_the_same_range_is_already_in_context_and_another_range_is_served():
    calls = []
    reader = _reader(fetcher=_fetcher(calls))
    reader.mark_already_read({('pdf', 'forms.md', '')})

    assert reader.invoke({'skill': 'pdf', 'path': 'FORMS.md'}) == (
        'Skill "pdf" file "forms.md" is already in this conversation. Use the earlier result instead of reading it again.'
    )
    assert reader.invoke({'skill': 'pdf', 'path': 'forms.md', 'start_line': 1, 'end_line': 1}).startswith(
        'Skill "pdf" file "forms.md" lines 1-1:'
    )
    assert calls == [(None, 5, 55, 'forms.md')]


def test_list_skill_files_lists_every_file_with_sizes():
    reply = _lister().invoke({'skill': 'pdf'})

    assert reply.splitlines()[0] == 'Files of skill "pdf" under skill://pdf/:'
    assert len(reply.splitlines()) == 1 + len(PDF_FILES)
    assert 'forms.md (12 KB)' in reply


def test_list_skill_files_lists_one_folder():
    reply = _lister().invoke({'skill': 'pdf', 'dir': 'scripts/'})

    assert reply.splitlines()[0] == 'Files of skill "pdf" under skill://pdf/scripts/:'
    assert reply.splitlines()[1:] == [f'{path} (3 KB)' for path in sorted(SCRIPTS, key=str.casefold)]


def test_list_skill_files_names_the_folders_when_one_is_missing():
    assert _lister().invoke({'skill': 'pdf', 'dir': 'assets'}) == 'Skill "pdf" has no folder "assets". Its folders: scripts.'


def test_list_skill_files_loads_a_skill_that_is_not_loaded_yet_before_listing():
    reply = _lister(loaded=()).invoke({'skill': 'pdf', 'dir': 'scripts'})

    assert reply.startswith(LOADED_PDF + '\n\nFiles of skill "pdf" under skill://pdf/scripts/:\n')


EXPECTED_PDF_BLOCK = '''Skill root: skill://pdf/
Read this skill's files only with read_skill_file(skill="pdf", path="forms.md") and list them with list_skill_files(skill="pdf", dir="<folder>"), passing paths relative to the skill root; never use other tools for these files and never treat skill:// as a filesystem path.
You cannot run scripts here; read one and follow it yourself.
<skill_files>
forms.md (12 KB)
LICENSE.txt (2 KB)
reference.md (17 KB)
scripts/ (8 files)
</skill_files>'''


def test_the_loaded_skill_result_ends_with_the_root_instructions_and_listing():
    reply = _tools()[0].invoke({'skill': 'pdf'})

    assert reply == LOADED_SKILL_RESULT.format(name='pdf', instructions=PDF_BODY) + '\n\n' + EXPECTED_PDF_BLOCK


def test_a_skill_without_files_loads_exactly_as_before():
    reply = build_load_skill_tools([PLAIN])[0].invoke({'skill': 'tone'})

    assert reply == LOADED_SKILL_RESULT.format(name='tone', instructions='Be kind.')
    assert '<skill_files>' not in reply


def test_a_skill_without_scripts_gets_no_script_sentence():
    block = render_loaded_skill(LOADED_SKILL_RESULT, 'pdf', PDF_BODY, PDF_FILES[:2])

    assert 'You cannot run scripts here' not in block
    assert '<skill_files>\nforms.md (12 KB)\nreference.md (17 KB)\n</skill_files>' in block


@pytest.mark.parametrize('path', ['tools/run.sh', 'build.js', 'scripts/README.md'])
def test_any_script_like_file_adds_the_script_sentence(path):
    files = [{'path': path, 'size': 10, 'is_text': True}]

    assert 'You cannot run scripts here; read one and follow it yourself.' in render_loaded_skill(
        LOADED_SKILL_RESULT, 'pdf', PDF_BODY, files,
    )


def test_the_listing_is_capped_at_thirty_entries():
    files = [{'path': f'file{index:02}.md', 'size': 10, 'is_text': True} for index in range(40)]

    listing = render_loaded_skill(LOADED_SKILL_RESULT, 'many', 'Body.', files).split('<skill_files>\n')[1]

    lines = listing.split('\n</skill_files>')[0].splitlines()
    assert lines[:2] == ['file00.md (10 B)', 'file01.md (10 B)']
    assert len(lines) == 31
    assert lines[-1] == '… and 10 more'


def test_skill_directory_placeholders_point_at_the_skill_root_and_others_stay_literal():
    body = 'Run ${CLAUDE_SKILL_DIR}/scripts/a.py, see ${CLAUDE_PLUGIN_ROOT}/x and keep ${CLAUDE_SESSION_ID}.'

    rendered = render_loaded_skill(SKILLS_SECTION_ENTRY, 'pdf', body)

    assert rendered == (
        '<skill name="pdf">\nRun skill://pdf/scripts/a.py, see skill://pdf/x and keep ${CLAUDE_SESSION_ID}.\n</skill>'
    )


def test_shell_lines_and_fences_are_replaced_and_inline_bangs_kept():
    body = 'Before\n!`git status`\n  !`date`\n```!\nls -la\npwd\n```\nSay !`hi` inline.\n```bash\nls\n```'
    marker = '[shell command execution is not available on this platform]'

    rendered = render_loaded_skill(SKILLS_SECTION_ENTRY, 'pdf', body)

    assert rendered == (
        f'<skill name="pdf">\nBefore\n{marker}\n{marker}\n{marker}\nSay !`hi` inline.\n```bash\nls\n```\n</skill>'
    )


def test_the_tilde_section_carries_the_files_block_only_for_a_skill_with_files():
    section = LLMNode._build_invoked_skills_section([PDF, PLAIN])

    assert f'<skill name="pdf">\n{PDF_BODY}\n</skill>\n\n{EXPECTED_PDF_BLOCK}' in section
    assert section.endswith('<skill name="tone">\nBe kind.\n</skill>')


def test_the_tilde_section_reaches_the_model_and_the_skill_is_readable_that_turn():
    client = MagicMock()
    client.get_skill_file.side_effect = _fetcher()
    llm, messages = _run({'attached_skills': [PDF], 'invoked_skills': [PDF]}, [
        _call('read_skill_file', {'skill': 'pdf', 'path': 'forms.md'}, 'c1'), AIMessage(content='done'),
    ], elitea_client=client)

    assert EXPECTED_PDF_BLOCK in ''.join(
        m.content if isinstance(m.content, str) else ''.join(b.get('text', '') for b in m.content)
        for m in llm.seen[0] if isinstance(m, SystemMessage)
    )
    [reply] = [m for m in messages if isinstance(m, ToolMessage)]
    assert 'Step 1: open the form.' in reply.content


def _child_configurable(parent_configurable, version_details):
    child_app = MagicMock()
    captured = {}

    def record_invoke(payload, config=None):
        captured['configurable'] = config['configurable']
        return {'output': 'result'}

    child_app.invoke.side_effect = record_invoke
    tool = Application(
        name='Child', description='Child', application=child_app, return_type='str', client=None, is_subgraph=True,
        args_runnable={'application_id': 1, 'application_version_id': 1, 'version_details': version_details},
    )
    tool._run(task='do the work', config={'configurable': parent_configurable, 'metadata': {}})
    return captured['configurable']


def test_a_sub_agent_binds_file_tools_only_for_its_own_skills():
    parent = {'thread_id': 't', 'attached_skills': [PDF], 'invoked_skills': [PDF]}

    plain_child = _child_configurable(parent, {'attached_skills': [PLAIN]})
    pdf_child = _child_configurable(parent, {'attached_skills': [_skill(skill_id=9)]})

    assert [type(tool) for tool in build_load_skill_tools(
        plain_child['attached_skills'], plain_child['invoked_skills'], plain_child['project_skills'],
    )] == [LoadSkillTool]
    child_tools = build_load_skill_tools(
        pdf_child['attached_skills'], pdf_child['invoked_skills'], pdf_child['project_skills'],
    )
    assert [type(tool) for tool in child_tools] == [LoadSkillTool, ReadSkillFileTool, ListSkillFilesTool]
    assert child_tools[1].registry.resolve('pdf')['skill_id'] == 9
    assert child_tools[1].invoke({'skill': 'pdf', 'path': 'forms.md'}).startswith('Skill "pdf" is now active')


def test_registry_entries_keep_their_manifest():
    assert SkillRegistry.from_sources([PDF]).resolve('pdf')['files'] == PDF_FILES
