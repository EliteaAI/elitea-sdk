"""A tool that reports its failure by RETURNING must not be reported as a success.

Re-breaks if test_toolkit_tool derives success from the absence of an exception again, or if
the payload predicate stops being scoped to index_data.
"""

import pytest

from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.tools.base_indexer_toolkit import (
    INDEX_DATA_TOOL_NAME,
    IndexingStatus,
    read_declared_failure,
)

EMPTY_LOADER_MESSAGE = (
    "Indexing failed: the loader returned no content while the index holds data from "
    "previous runs. Previously indexed data remains available for search."
)


def error_result(message=EMPTY_LOADER_MESSAGE):
    return {
        "status": IndexingStatus.ERROR.value,
        "message": message,
        "report": {"status": "error", "errors": [message], "errors_total": 1},
    }


class TestReadDeclaredFailure:
    def test_an_error_status_with_a_message_is_a_declared_failure(self):
        assert read_declared_failure(INDEX_DATA_TOOL_NAME, error_result()) == EMPTY_LOADER_MESSAGE

    def test_a_successful_run_declares_nothing(self):
        assert (
            read_declared_failure(
                INDEX_DATA_TOOL_NAME, {"status": IndexingStatus.OK.value, "message": "6 files indexed"}
            )
            is None
        )

    def test_a_partly_indexed_run_is_not_a_failure(self):
        assert (
            read_declared_failure(
                INDEX_DATA_TOOL_NAME,
                {"status": IndexingStatus.PARTLY_INDEXED.value, "message": "4 of 6 files indexed"},
            )
            is None
        )

    @pytest.mark.parametrize(
        "tool_name",
        [
            pytest.param("run_pipeline", id="another-toolkit"),
            pytest.param("search_index", id="another-indexer-tool"),
            pytest.param("", id="unnamed"),
        ],
    )
    def test_the_same_payload_from_another_tool_declares_nothing(self, tool_name):
        assert read_declared_failure(tool_name, error_result()) is None

    @pytest.mark.parametrize(
        "result",
        [
            pytest.param("plain string result", id="not-a-dict"),
            pytest.param(["a", "list"], id="list"),
            pytest.param(None, id="none"),
            pytest.param({"issues": []}, id="no-status"),
            pytest.param({"status": "error"}, id="status-without-message"),
            pytest.param({"status": "error", "message": "   "}, id="blank-message"),
            pytest.param({"status": "error", "message": 42}, id="non-string-message"),
            pytest.param({"status": "failed", "message": "boom"}, id="status-outside-the-contract"),
        ],
    )
    def test_payloads_outside_the_indexing_contract_are_left_alone(self, result):
        assert read_declared_failure(INDEX_DATA_TOOL_NAME, result) is None


class _StubTool:
    def __init__(self, name, result):
        self.name = name
        self._result = result

    def invoke(self, params, config=None):
        return self._result


@pytest.fixture
def client(monkeypatch):
    instance = EliteAClient.__new__(EliteAClient)
    monkeypatch.setattr(instance, "_validate_toolkit_config", lambda config: config, raising=False)
    monkeypatch.setattr(instance, "get_llm", lambda model, config: object(), raising=False)
    return instance


def run_tool(client, monkeypatch, result, tool_name="index_data"):
    import elitea_sdk.runtime.utils.toolkit_utils as toolkit_utils

    monkeypatch.setattr(
        toolkit_utils,
        "instantiate_toolkit_with_client",
        lambda *args, **kwargs: [_StubTool(tool_name, result)],
    )
    return client.test_toolkit_tool(
        toolkit_config={"toolkit_name": "artifact", "settings": {}},
        tool_name=tool_name,
        tool_params={"index_name": "demo"},
    )


class TestTestToolkitToolSuccessFlag:
    def test_a_returned_error_is_reported_as_a_failure(self, client, monkeypatch):
        outcome = run_tool(client, monkeypatch, error_result())

        assert outcome["success"] is False
        assert outcome["error"] == EMPTY_LOADER_MESSAGE
        assert outcome["debug_error"] == EMPTY_LOADER_MESSAGE

    def test_the_failed_payload_is_still_returned(self, client, monkeypatch):
        outcome = run_tool(client, monkeypatch, error_result())

        assert outcome["success"] is False
        assert outcome["result"]["report"]["errors_total"] == 1

    def test_a_successful_run_is_still_a_success(self, client, monkeypatch):
        ok = {"status": IndexingStatus.OK.value, "message": "6 files indexed", "report": {"status": "ok"}}

        outcome = run_tool(client, monkeypatch, ok)

        assert outcome["success"] is True
        assert outcome["result"] == ok

    def test_a_partly_indexed_run_is_still_a_success(self, client, monkeypatch):
        partly = {"status": IndexingStatus.PARTLY_INDEXED.value, "message": "4 of 6 files indexed"}

        outcome = run_tool(client, monkeypatch, partly)

        assert outcome["success"] is True

    def test_an_ordinary_tool_result_is_untouched(self, client, monkeypatch):
        other_tool_failure = {"status": "error", "message": "Pipeline run failed on stage build"}

        outcome = run_tool(client, monkeypatch, other_tool_failure, tool_name="run_pipeline")

        assert outcome["success"] is True
        assert outcome["result"] == other_tool_failure
