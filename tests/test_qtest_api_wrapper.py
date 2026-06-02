import html

from pydantic import SecretStr

from elitea_sdk.tools.qtest.api_wrapper import QTEST_ID, QtestApiWrapper
from elitea_sdk.tools.utils.content_parser import image_processing_prompt


def _make_wrapper():
    return QtestApiWrapper.model_construct(
        base_url='https://example.invalid',
        qtest_project_id=1,
        qtest_api_token=SecretStr('token'),
        no_of_items_per_page=100,
        page=1,
        no_of_tests_shown_in_dql_search=10,
        llm=None,
        elitea=None,
    )


def test_parse_data_filters_base64_images_from_description_and_precondition():
    wrapper = _make_wrapper()
    parsed_data = []
    base64_content = 'QUJD'
    html_content = (
        f'Before <img src="data:image/png;base64,{base64_content}" '
        'data-filename="proof.png" /> after'
    )

    wrapper._QtestApiWrapper__parse_data(
        {
            'items': [
                {
                    'pid': 'TC-21',
                    'name': 'Embedded image case',
                    'description': html_content,
                    'precondition': html_content,
                    'id': 4626964,
                    'test_steps': [],
                    'properties': [],
                }
            ]
        },
        parsed_data,
        extract_images=False,
        prompt=None,
    )

    parsed_row = parsed_data[0]

    assert parsed_row['Id'] == 'TC-21'
    assert parsed_row[QTEST_ID] == 4626964
    assert base64_content not in parsed_row['Description']
    assert base64_content not in parsed_row['Precondition']
    assert '<img' not in parsed_row['Description']
    assert '<img' not in parsed_row['Precondition']
    assert parsed_row['Description'].replace('  ', ' ').strip() == 'Before after'
    assert parsed_row['Precondition'].replace('  ', ' ').strip() == 'Before after'


def test_parse_data_filters_base64_images_when_data_filename_precedes_src():
    wrapper = _make_wrapper()
    parsed_data = []
    base64_content = 'QUJD'
    html_content = (
        f'Before <img data-filename="proof.png" '
        f'src="data:image/png;base64,{base64_content}" /> after'
    )

    wrapper._QtestApiWrapper__parse_data(
        {
            'items': [
                {
                    'pid': 'TC-70391',
                    'name': 'Embedded image attribute order case',
                    'description': html_content,
                    'precondition': html_content,
                    'id': 4627001,
                    'test_steps': [
                        {
                            'description': html_content,
                            'expected': html_content,
                        }
                    ],
                    'properties': [],
                }
            ]
        },
        parsed_data,
        extract_images=False,
        prompt=None,
    )

    parsed_row = parsed_data[0]
    parsed_step = parsed_row['Steps'][0]

    assert base64_content not in parsed_row['Description']
    assert base64_content not in parsed_row['Precondition']
    assert base64_content not in parsed_step['Test Step Description']
    assert base64_content not in parsed_step['Test Step Expected Result']
    assert '<img' not in parsed_row['Description']
    assert '<img' not in parsed_row['Precondition']
    assert '<img' not in parsed_step['Test Step Description']
    assert '<img' not in parsed_step['Test Step Expected Result']


def test_parse_data_filters_escaped_base64_images_from_all_test_case_fields():
    wrapper = _make_wrapper()
    parsed_data = []
    base64_content = 'QUJD'
    raw_html = (
        f'Before <img src="data:image/png;base64,{base64_content}" '
        'data-filename="image.png" style="width: 460px;"> after'
    )
    escaped_html = html.escape(raw_html)

    wrapper._QtestApiWrapper__parse_data(
        {
            'items': [
                {
                    'pid': 'TC-70392',
                    'name': 'Escaped embedded image case',
                    'description': escaped_html,
                    'precondition': escaped_html,
                    'id': 4627002,
                    'test_steps': [
                        {
                            'description': escaped_html,
                            'expected': escaped_html,
                        }
                    ],
                    'properties': [],
                }
            ]
        },
        parsed_data,
        extract_images=False,
        prompt=None,
    )

    parsed_row = parsed_data[0]
    parsed_step = parsed_row['Steps'][0]

    assert base64_content not in parsed_row['Description']
    assert base64_content not in parsed_row['Precondition']
    assert base64_content not in parsed_step['Test Step Description']
    assert base64_content not in parsed_step['Test Step Expected Result']
    assert '<img' not in parsed_row['Description']
    assert '<img' not in parsed_row['Precondition']
    assert '<img' not in parsed_step['Test Step Description']
    assert '<img' not in parsed_step['Test Step Expected Result']
    assert parsed_row['Description'].replace('  ', ' ').strip() == 'Before after'
    assert parsed_step['Test Step Description'].replace('  ', ' ').strip() == 'Before after'


def test_parse_data_filters_double_escaped_base64_images_from_all_test_case_fields():
    wrapper = _make_wrapper()
    parsed_data = []
    base64_content = 'QUJD'
    raw_html = (
        f'Before <img src="data:image/png;base64,{base64_content}" '
        'data-filename="image.png" style="width: 460px;"> after'
    )
    double_escaped_html = html.escape(html.escape(raw_html))

    wrapper._QtestApiWrapper__parse_data(
        {
            'items': [
                {
                    'pid': 'TC-70395',
                    'name': 'Double escaped embedded image case',
                    'description': double_escaped_html,
                    'precondition': double_escaped_html,
                    'id': 4627005,
                    'test_steps': [
                        {
                            'description': double_escaped_html,
                            'expected': double_escaped_html,
                        }
                    ],
                    'properties': [],
                }
            ]
        },
        parsed_data,
        extract_images=False,
        prompt=None,
    )

    parsed_row = parsed_data[0]
    parsed_step = parsed_row['Steps'][0]

    assert base64_content not in parsed_row['Description']
    assert base64_content not in parsed_row['Precondition']
    assert base64_content not in parsed_step['Test Step Description']
    assert base64_content not in parsed_step['Test Step Expected Result']
    assert '<img' not in parsed_row['Description']
    assert '<img' not in parsed_row['Precondition']
    assert '<img' not in parsed_step['Test Step Description']
    assert '<img' not in parsed_step['Test Step Expected Result']
    assert parsed_row['Description'].replace('  ', ' ').strip() == 'Before after'
    assert parsed_step['Test Step Description'].replace('  ', ' ').strip() == 'Before after'


def test_parse_entity_item_filters_escaped_base64_images_from_test_case_fields():
    wrapper = _make_wrapper()
    base64_content = 'QUJD'
    raw_html = (
        f'Before <img src="data:image/png;base64,{base64_content}" '
        'data-filename="image.png" style="width: 460px;"> after'
    )
    escaped_html = html.escape(raw_html)

    parsed = wrapper._QtestApiWrapper__parse_entity_item(
        'test-cases',
        {
            'pid': 'TC-70393',
            'id': 4627003,
            'name': 'Escaped entity item image case',
            'description': escaped_html,
            'precondition': escaped_html,
            'test_steps': [
                {
                    'description': escaped_html,
                    'expected': escaped_html,
                }
            ],
            'properties': [],
        },
    )

    assert base64_content not in parsed['Description']
    assert base64_content not in parsed['Precondition']
    assert base64_content not in parsed['Steps'][0]['Test Step Description']
    assert base64_content not in parsed['Steps'][0]['Test Step Expected Result']
    assert '<img' not in parsed['Description']
    assert '<img' not in parsed['Steps'][0]['Test Step Description']


def test_parse_data_uses_clean_html_content_for_all_test_case_fields(monkeypatch):
    wrapper = _make_wrapper()
    parsed_data = []
    calls = []

    def fake_clean(content, extract_images=False, image_prompt=None):
        calls.append((content, extract_images, image_prompt))
        return f'clean:{content}'

    monkeypatch.setattr(wrapper, '_clean_html_content', fake_clean)
    monkeypatch.setattr(wrapper, '_process_image', lambda content, extract=False, prompt=None: content)

    wrapper._QtestApiWrapper__parse_data(
        {
            'items': [
                {
                    'pid': 'TC-22',
                    'name': 'Option passthrough case',
                    'description': '<p>description</p>',
                    'precondition': '<p>precondition</p>',
                    'id': 4626965,
                    'test_steps': [
                        {'description': 'step description', 'expected': 'step expected'},
                    ],
                    'properties': [],
                }
            ]
        },
        parsed_data,
        extract_images=True,
        prompt='describe images',
    )

    parsed_row = parsed_data[0]

    assert calls == [
        ('<p>description</p>', True, 'describe images'),
        ('<p>precondition</p>', True, 'describe images'),
        ('step description', True, 'describe images'),
        ('step expected', True, 'describe images'),
    ]
    assert parsed_row['Description'] == 'clean:<p>description</p>'
    assert parsed_row['Precondition'] == 'clean:<p>precondition</p>'
    assert parsed_row['Steps'][0]['Test Step Description'] == 'clean:step description'
    assert parsed_row['Steps'][0]['Test Step Expected Result'] == 'clean:step expected'


def test_parse_data_uses_default_prompt_when_prompt_is_missing_or_empty(monkeypatch):
    wrapper = _make_wrapper()
    base64_content = 'QUJD'
    html_content = (
        f'Before <img src="data:image/png;base64,{base64_content}" '
        'data-filename="image.png" style="width: 460px;"> after'
    )

    for provided_prompt in (None, ''):
        parsed_data = []
        forwarded_prompts = []

        class FakeLoader:
            def __init__(self, prompt):
                self.prompt = prompt

            def get_content(self):
                return f'default-prompt-analysis:{self.prompt[:24]}'

        def fake_prepare_loader(**kwargs):
            forwarded_prompts.append(kwargs['prompt'])
            return FakeLoader(kwargs['prompt'])

        monkeypatch.setattr('elitea_sdk.tools.utils.content_parser.prepare_loader', fake_prepare_loader)

        wrapper._QtestApiWrapper__parse_data(
            {
                'items': [
                    {
                        'pid': 'TC-70394',
                        'name': 'Default prompt image analysis case',
                        'description': html_content,
                        'precondition': '',
                        'id': 4627004,
                        'test_steps': [],
                        'properties': [],
                    }
                ]
            },
            parsed_data,
            extract_images=True,
            prompt=provided_prompt,
        )

        description = parsed_data[0]['Description']

        assert forwarded_prompts == [image_processing_prompt]
        assert base64_content not in description
        assert 'Image Transcript:' in description
        assert 'default-prompt-analysis:' in description
        assert description.index('Before') < description.index('Image Transcript:')
        assert description.index('Image Transcript:') < description.index('after')


def test_parse_data_strips_images_without_decoding_when_extract_disabled(monkeypatch):
    wrapper = _make_wrapper()
    parsed_data = []
    base64_content = 'QUJD'
    html_content = (
        f'Before <img src="data:image/png;base64,{base64_content}" '
        'data-filename="image.png" style="width: 460px;"> after'
    )

    def fail_decode(*args, **kwargs):
        raise AssertionError('base64 decode should not be called when extract_images is disabled')

    monkeypatch.setattr('elitea_sdk.tools.qtest.api_wrapper.base64.b64decode', fail_decode)

    wrapper._QtestApiWrapper__parse_data(
        {
            'items': [
                {
                    'pid': 'TC-70396',
                    'name': 'Disabled image extraction case',
                    'description': html_content,
                    'precondition': html_content,
                    'id': 4627006,
                    'test_steps': [
                        {
                            'description': html_content,
                            'expected': html_content,
                        }
                    ],
                    'properties': [],
                }
            ]
        },
        parsed_data,
        extract_images=False,
        prompt='analyze image',
    )

    parsed_row = parsed_data[0]
    parsed_step = parsed_row['Steps'][0]

    assert base64_content not in parsed_row['Description']
    assert base64_content not in parsed_row['Precondition']
    assert base64_content not in parsed_step['Test Step Description']
    assert base64_content not in parsed_step['Test Step Expected Result']
    assert parsed_row['Description'].replace('  ', ' ').strip() == 'Before after'
    assert parsed_step['Test Step Description'].replace('  ', ' ').strip() == 'Before after'


def test_parse_data_extracts_multiple_images_in_order(monkeypatch):
    wrapper = _make_wrapper()
    parsed_data = []
    base64_content = 'QUJD'
    html_content = (
        'Start '
        f'<img src="data:image/png;base64,{base64_content}" data-filename="one.png"> '
        'Middle '
        f'<img src="data:image/png;base64,{base64_content}" data-filename="two.png"> '
        'End'
    )

    def fake_parse_file_content(**kwargs):
        file_name = kwargs['file_name']
        return {
            'one.png': 'first image transcript',
            'two.png': 'second image transcript',
        }[file_name]

    monkeypatch.setattr('elitea_sdk.tools.qtest.api_wrapper.parse_file_content', fake_parse_file_content)

    wrapper._QtestApiWrapper__parse_data(
        {
            'items': [
                {
                    'pid': 'TC-70397',
                    'name': 'Multiple images in one field',
                    'description': html_content,
                    'precondition': '',
                    'id': 4627007,
                    'test_steps': [],
                    'properties': [],
                }
            ]
        },
        parsed_data,
        extract_images=True,
        prompt='describe images',
    )

    description = parsed_data[0]['Description']

    assert description.count('Image Transcript:') == 2
    assert 'Image Transcript: first image transcript' in description
    assert 'Image Transcript: second image transcript' in description
    assert description.index('Start') < description.index('Image Transcript: first image transcript')
    assert description.index('Image Transcript: first image transcript') < description.index('Middle')
    assert description.index('Middle') < description.index('Image Transcript: second image transcript')
    assert description.index('Image Transcript: second image transcript') < description.index('End')
