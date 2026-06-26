import html

import pytest
import urllib3
from pydantic import SecretStr

from elitea_sdk.tools.qtest.api_wrapper import (
    QTEST_ID,
    QtestApiWrapper,
    _TimeoutApiClient,
    _default_qtest_timeout,
)
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


# ---------------------------------------------------------------------------
# #4953 - optional heavy payload parameters in search_by_dql
# ---------------------------------------------------------------------------

def _patch_search_api(monkeypatch, response):
    """Patch swagger_client.SearchApi with a fake recording search_artifact kwargs."""
    calls = []

    class FakeSearchApi:
        def __init__(self, client):
            pass

        def search_artifact(self, project_id, body, **kwargs):
            calls.append(kwargs)
            return response

    monkeypatch.setattr('swagger_client.SearchApi', FakeSearchApi)
    return calls


def test_perform_search_by_dql_sends_false_for_heavy_params_by_default(monkeypatch):
    wrapper = _make_wrapper()
    wrapper._client = None  # model_construct skips setup_qtest_client; fake SearchApi ignores it
    response = {
        'items': [
            {'pid': 'TC-1', 'name': 'Case', 'description': 'd', 'precondition': 'p', 'id': 101}
        ],
        'links': [],
    }
    calls = _patch_search_api(monkeypatch, response)

    result = wrapper._QtestApiWrapper__perform_search_by_dql("Id = 'TC-1'")

    assert len(calls) == 1
    # Flags are sent explicitly as 'false' so the qTest API excludes heavy data at
    # the source instead of returning it for us to discard during parsing.
    assert calls[0]['append_test_steps'] == 'false'
    assert calls[0]['include_external_properties'] == 'false'
    # Lightweight response (no test_steps/properties keys) parses without error.
    assert result[0]['Id'] == 'TC-1'
    assert result[0]['Steps'] == []


def test_perform_search_by_dql_includes_heavy_params_when_opted_in(monkeypatch):
    wrapper = _make_wrapper()
    wrapper._client = None  # model_construct skips setup_qtest_client; fake SearchApi ignores it
    response = {
        'items': [
            {
                'pid': 'TC-2', 'name': 'Case2', 'description': 'd', 'precondition': 'p', 'id': 102,
                'test_steps': [{'description': 'do something', 'expected': 'all good'}],
                'properties': [],
            }
        ],
        'links': [],
    }
    calls = _patch_search_api(monkeypatch, response)

    result = wrapper._QtestApiWrapper__perform_search_by_dql(
        "Id = 'TC-2'", append_test_steps=True, include_external_properties=True
    )

    assert calls[0]['append_test_steps'] == 'true'
    assert calls[0]['include_external_properties'] == 'true'
    assert 'do something' in result[0]['Steps'][0]['Test Step Description']
    assert 'all good' in result[0]['Steps'][0]['Test Step Expected Result']


def test_parse_data_tolerates_missing_steps_and_properties():
    wrapper = _make_wrapper()
    parsed_data = []

    wrapper._QtestApiWrapper__parse_data(
        {'items': [{'pid': 'TC-9', 'name': 'n', 'description': 'd', 'precondition': 'p', 'id': 9}]},
        parsed_data,
        extract_images=False,
        prompt=None,
    )

    assert parsed_data[0]['Id'] == 'TC-9'
    assert parsed_data[0][QTEST_ID] == 9
    assert parsed_data[0]['Steps'] == []


def test_search_by_dql_forwards_opt_in_flags(monkeypatch):
    wrapper = _make_wrapper()
    captured = {}

    def fake_perform(self, dql, extract_images=False, prompt=None, max_results=None,
                     append_test_steps=False, include_external_properties=False):
        captured['append_test_steps'] = append_test_steps
        captured['include_external_properties'] = include_external_properties
        return []

    monkeypatch.setattr(
        QtestApiWrapper, '_QtestApiWrapper__perform_search_by_dql', fake_perform
    )

    wrapper.search_by_dql("Id = 'TC-1'", append_test_steps=True, include_external_properties=True)

    assert captured == {'append_test_steps': True, 'include_external_properties': True}


def test_search_by_dql_defaults_to_lightweight(monkeypatch):
    wrapper = _make_wrapper()
    captured = {}

    def fake_perform(self, dql, extract_images=False, prompt=None, max_results=None,
                     append_test_steps=False, include_external_properties=False):
        captured['append_test_steps'] = append_test_steps
        captured['include_external_properties'] = include_external_properties
        return []

    monkeypatch.setattr(
        QtestApiWrapper, '_QtestApiWrapper__perform_search_by_dql', fake_perform
    )

    wrapper.search_by_dql("Id = 'TC-1'")

    assert captured == {'append_test_steps': False, 'include_external_properties': False}


# ---------------------------------------------------------------------------
# #4952 - configurable per-request HTTP timeout for the qTest client
# ---------------------------------------------------------------------------

def test_default_qtest_timeout_from_env(monkeypatch):
    monkeypatch.setenv('QTEST_API_TIMEOUT_SECONDS', '42')
    assert _default_qtest_timeout() == 42


def test_default_qtest_timeout_fallback_when_unset(monkeypatch):
    monkeypatch.delenv('QTEST_API_TIMEOUT_SECONDS', raising=False)
    assert _default_qtest_timeout() == 180


def test_default_qtest_timeout_fallback_when_invalid(monkeypatch):
    monkeypatch.setenv('QTEST_API_TIMEOUT_SECONDS', 'not-a-number')
    assert _default_qtest_timeout() == 180


def test_default_qtest_timeout_fallback_when_non_positive(monkeypatch):
    monkeypatch.setenv('QTEST_API_TIMEOUT_SECONDS', '0')
    assert _default_qtest_timeout() == 180


def _make_timeout_client(request_timeout=180):
    import swagger_client
    return _TimeoutApiClient(swagger_client.Configuration(), request_timeout=request_timeout)


def test_timeout_client_injects_default_request_timeout(monkeypatch):
    import swagger_client
    captured = {}

    def fake_call_api(self, resource_path, method, *args, **kwargs):
        captured.update(kwargs)
        return 'ok'

    monkeypatch.setattr(swagger_client.ApiClient, 'call_api', fake_call_api)
    client = _make_timeout_client(180)

    result = client.call_api('/api/v3/projects/1/search', 'POST')

    assert result == 'ok'
    assert captured['_request_timeout'] == 180


def test_timeout_client_respects_explicit_request_timeout(monkeypatch):
    import swagger_client
    captured = {}

    def fake_call_api(self, resource_path, method, *args, **kwargs):
        captured.update(kwargs)
        return 'ok'

    monkeypatch.setattr(swagger_client.ApiClient, 'call_api', fake_call_api)
    client = _make_timeout_client(180)

    client.call_api('/x', 'GET', _request_timeout=5)

    assert captured['_request_timeout'] == 5


def test_timeout_client_translates_read_timeout(monkeypatch):
    import swagger_client

    def boom(self, *args, **kwargs):
        raise urllib3.exceptions.ReadTimeoutError(None, '/x', 'read timed out')

    monkeypatch.setattr(swagger_client.ApiClient, 'call_api', boom)
    client = _make_timeout_client(180)

    with pytest.raises(TimeoutError) as exc_info:
        client.call_api('/api/v3/projects/1/search', 'POST')

    message = str(exc_info.value)
    assert '180' in message
    assert 'timed out' in message.lower()


def test_timeout_client_translates_maxretry_timeout(monkeypatch):
    import swagger_client
    reason = urllib3.exceptions.ConnectTimeoutError('connect timed out')

    def boom(self, *args, **kwargs):
        raise urllib3.exceptions.MaxRetryError(None, '/x', reason=reason)

    monkeypatch.setattr(swagger_client.ApiClient, 'call_api', boom)
    client = _make_timeout_client(180)

    with pytest.raises(TimeoutError):
        client.call_api('/api/v3/projects/1/search', 'POST')


def test_timeout_client_reraises_non_timeout_maxretry(monkeypatch):
    import swagger_client
    reason = urllib3.exceptions.ProtocolError('connection aborted')

    def boom(self, *args, **kwargs):
        raise urllib3.exceptions.MaxRetryError(None, '/x', reason=reason)

    monkeypatch.setattr(swagger_client.ApiClient, 'call_api', boom)
    client = _make_timeout_client(180)

    with pytest.raises(urllib3.exceptions.MaxRetryError):
        client.call_api('/x', 'GET')
