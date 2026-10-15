# Copyright (c) 2026 EPAM Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guards that a loader which swallows a fetch error still reports the item as failed.

Re-broken by any change that lets such an item reach the zero-chunk branch untracked,
where it would be reported as containing no indexable content instead of as a failure.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.tools import ToolException

from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit
from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper
from elitea_sdk.tools.qtest.api_wrapper import QtestApiWrapper
from elitea_sdk.tools.zephyr_enterprise.api_wrapper import ZephyrApiWrapper
from elitea_sdk.tools.zephyr_essential.api_wrapper import ZephyrEssentialApiWrapper
from elitea_sdk.tools.zephyr_scale.api_wrapper import ZephyrScaleApiWrapper


def fresh(wrapper_cls):
    wrapper = wrapper_cls.model_construct()
    wrapper._init_indexing_stats()
    return wrapper


def failed_names(wrapper):
    return wrapper.get_indexing_stats().documents_skipped_error


def raise_tool_exception(*args, **kwargs):
    raise ToolException("HTTP 429 Too Many Requests")


def name_of(document):
    return BaseIndexerToolkit._extract_doc_name(document.metadata)


class TestTheTracker:

    def test_a_document_left_without_content_is_failed(self):
        wrapper = fresh(ZephyrApiWrapper)
        document = Document(page_content="", metadata={"id": "7", "name": "Login works"})

        wrapper._track_document_left_empty_by_error(document)

        assert failed_names(wrapper) == {"Login works"}

    @pytest.mark.parametrize("document", [
        Document(page_content="body", metadata={"id": "7", "name": "Login works"}),
        Document(page_content="", metadata={"id": "7", "name": "Login works",
                                            IndexerKeywords.CONTENT_IN_BYTES.value: b"{}"}),
    ], ids=["text", "bytes"])
    def test_a_document_that_kept_content_is_not_failed(self, document):
        wrapper = fresh(ZephyrApiWrapper)

        wrapper._track_document_left_empty_by_error(document)

        assert failed_names(wrapper) == set()


class TestZephyrEnterprise:

    def test_a_steps_fetch_error_reports_the_test_case_failed(self, monkeypatch):
        wrapper = fresh(ZephyrApiWrapper)
        monkeypatch.setattr(ZephyrApiWrapper, "get_test_case_steps", raise_tool_exception)
        document = Document(page_content="", metadata={"id": "7", "name": "Login works"})

        list(wrapper._extend_data(iter([document])))

        assert failed_names(wrapper) == {"Login works"}


class TestZephyrScale:

    @staticmethod
    def test_case(key="PRJ-T1"):
        return Document(page_content="", metadata={"key": key, "name": key, "type": "TEST_CASE"})

    def test_steps_and_script_both_failing_reports_the_test_case_failed(self, monkeypatch):
        wrapper = fresh(ZephyrScaleApiWrapper)
        monkeypatch.setattr(ZephyrScaleApiWrapper, "get_test_steps", raise_tool_exception)
        monkeypatch.setattr(ZephyrScaleApiWrapper, "get_test_script", raise_tool_exception)

        list(wrapper._extend_data(iter([self.test_case()])))

        assert failed_names(wrapper) == {"PRJ-T1"}

    def test_a_failed_steps_fetch_with_no_script_reports_the_test_case_failed(self, monkeypatch):
        wrapper = fresh(ZephyrScaleApiWrapper)
        monkeypatch.setattr(ZephyrScaleApiWrapper, "get_test_steps", raise_tool_exception)
        monkeypatch.setattr(ZephyrScaleApiWrapper, "get_test_script", lambda self, key: None)

        list(wrapper._extend_data(iter([self.test_case()])))

        assert failed_names(wrapper) == {"PRJ-T1"}

    def test_a_failed_steps_fetch_rescued_by_the_script_is_not_failed(self, monkeypatch):
        wrapper = fresh(ZephyrScaleApiWrapper)
        monkeypatch.setattr(ZephyrScaleApiWrapper, "get_test_steps", raise_tool_exception)
        monkeypatch.setattr(ZephyrScaleApiWrapper, "get_test_script",
                            lambda self, key: {"text": "Given a user"})

        [document] = list(wrapper._extend_data(iter([self.test_case()])))

        assert failed_names(wrapper) == set()
        assert document.metadata["steps_type"] == "script"

    def test_a_test_case_with_no_steps_or_script_is_not_failed(self, monkeypatch):
        wrapper = fresh(ZephyrScaleApiWrapper)
        monkeypatch.setattr(ZephyrScaleApiWrapper, "get_test_steps", lambda self, key: [])
        monkeypatch.setattr(ZephyrScaleApiWrapper, "get_test_script", lambda self, key: None)

        list(wrapper._extend_data(iter([self.test_case()])))

        assert failed_names(wrapper) == set()

    def test_a_step_by_step_case_the_api_reports_has_no_script_is_not_failed(self, monkeypatch):
        wrapper = fresh(ZephyrScaleApiWrapper)

        def no_script(self, key):
            raise ToolException(f"Unable to get test script for test case with key: {key}:\n"
                                "Error 422. Response: b'{}'")

        monkeypatch.setattr(ZephyrScaleApiWrapper, "get_test_steps", lambda self, key: [])
        monkeypatch.setattr(ZephyrScaleApiWrapper, "get_test_script", no_script)

        list(wrapper._extend_data(iter([self.test_case()])))

        assert failed_names(wrapper) == set()


class TestZephyrEssential:

    @staticmethod
    def load(wrapper, monkeypatch):
        monkeypatch.setattr(ZephyrEssentialApiWrapper, "list_test_cases",
                            lambda self: [{"key": "PRJ-T1", "name": "Login works"}])
        return list(wrapper._base_loader())

    def test_steps_and_script_both_failing_reports_the_test_case_failed(self, monkeypatch):
        wrapper = fresh(ZephyrEssentialApiWrapper)
        monkeypatch.setattr(ZephyrEssentialApiWrapper, "get_test_case_test_steps", raise_tool_exception)
        monkeypatch.setattr(ZephyrEssentialApiWrapper, "get_test_case_test_script", raise_tool_exception)

        [document] = self.load(wrapper, monkeypatch)

        assert failed_names(wrapper) == {name_of(document)}

    def test_a_step_case_without_steps_is_not_failed(self, monkeypatch):
        wrapper = fresh(ZephyrEssentialApiWrapper)

        def no_script(self, key):
            raise ToolException(f"Unexpected status code 404 for GET /testcases/{key}/testscript: b'{{}}'")

        monkeypatch.setattr(ZephyrEssentialApiWrapper, "get_test_case_test_steps", lambda self, key: [])
        monkeypatch.setattr(ZephyrEssentialApiWrapper, "get_test_case_test_script", no_script)

        self.load(wrapper, monkeypatch)

        assert failed_names(wrapper) == set()


class TestQtest:

    def test_a_formatting_error_reports_the_test_case_failed(self, monkeypatch):
        wrapper = fresh(QtestApiWrapper)

        def fail_to_format(self, item):
            raise RuntimeError("image description LLM unavailable")

        monkeypatch.setattr(QtestApiWrapper, "_format_test_case_as_markdown", fail_to_format)
        document = Document(page_content="", metadata={"id": "TC-1", "name": "Login works",
                                                       "_raw_item": {"id": 1}})

        list(wrapper._extend_data(iter([document])))

        assert failed_names(wrapper) == {"Login works"}


class TestConfluence:

    @staticmethod
    def page(body):
        return Document(page_content=body, metadata={"id": "2003", "title": "Monitoring One-Pager"})

    def run_with_failing_attachment_listing(self, document):
        wrapper = fresh(ConfluenceAPIWrapper)
        object.__setattr__(wrapper, "_index_include_attachments", True)

        class FailingClient:
            def get_attachments_from_content(self, page_id):
                raise RuntimeError("HTTP 503")

        object.__setattr__(wrapper, "client", FailingClient())
        list(wrapper._process_document(document))
        return wrapper

    def test_an_empty_page_whose_attachments_could_not_be_listed_is_failed(self):
        document = self.page("")

        wrapper = self.run_with_failing_attachment_listing(document)

        assert failed_names(wrapper) == {name_of(document)}

    def test_a_page_with_a_body_is_not_failed_by_an_attachment_listing_error(self):
        wrapper = self.run_with_failing_attachment_listing(self.page("real body"))

        assert failed_names(wrapper) == set()

    @staticmethod
    def run_with_malformed_attachment_history(document, malformed_id):
        wrapper = fresh(ConfluenceAPIWrapper)
        object.__setattr__(wrapper, "_index_include_attachments", True)

        class Response:
            status_code = 200
            content = "attachment text"

        class Client:
            url = "https://confluence.example.com/wiki"

            def get_attachments_from_content(self, page_id):
                return {"results": [
                    {"id": attachment_id, "title": f"{attachment_id}.txt",
                     "metadata": {"mediaType": "text/plain"},
                     "_links": {"download": f"/download/{attachment_id}"}}
                    for attachment_id in ("att-1", "att-2")
                ]}

            def history(self, attachment_id):
                if attachment_id == malformed_id:
                    return ["not", "a", "dict"]
                return {"createdBy": {"displayName": "sam"}, "createdDate": "2026-01-01",
                        "lastUpdated": {"when": "2026-01-02"}}

            def request(self, method, path, advanced_mode=True):
                return Response()

        object.__setattr__(wrapper, "client", Client())
        yielded = list(wrapper._process_document(document))
        return wrapper, yielded

    def test_an_empty_page_that_already_yielded_an_attachment_is_not_failed(self):
        wrapper, yielded = self.run_with_malformed_attachment_history(self.page(""), "att-2")

        assert [document.metadata["name"] for document in yielded] == ["att-1.txt"]
        assert failed_names(wrapper) == set()

    def test_an_empty_page_whose_first_attachment_breaks_is_failed(self):
        document = self.page("")

        wrapper, yielded = self.run_with_malformed_attachment_history(document, "att-1")

        assert yielded == []
        assert failed_names(wrapper) == {name_of(document)}
