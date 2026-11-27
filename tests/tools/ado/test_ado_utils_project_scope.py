"""The per-call project scope shared by the ADO wrappers.

The configured project is a default: a tool call may name another one, the call's project
must reach every helper (including worker threads) and must never outlive the call.
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException
from pydantic import BaseModel

from elitea_sdk.tools.ado import utils as ado_utils
from elitea_sdk.tools.ado.utils import (
    NO_PROJECT_MESSAGE,
    AdoProjectScope,
    in_caller_context,
    normalize_project,
)


class _Base:
    def run(self, mode, *args, **kwargs):
        return self._project(), kwargs


class _Wrapper(AdoProjectScope, _Base):
    def __init__(self, project=None):
        self.project = project


class _Args(BaseModel):
    query: str


@pytest.mark.parametrize("value", [None, "", "  ", "null", "None", "undefined", "N/A", "'null'", '""'])
def test_placeholders_mean_not_set(value):
    assert normalize_project(value) is None


def test_project_names_are_trimmed_and_kept():
    assert normalize_project("  'Digital' ") == "Digital"


def test_runtime_project_overrides_the_configured_one():
    assert _Wrapper("Digital").run("get_work_item", project="Other", id=1) == ("Other", {"id": 1})


def test_configured_project_is_the_fallback():
    assert _Wrapper("Digital").run("get_work_item", project="null")[0] == "Digital"


def test_no_project_anywhere_is_an_actionable_error():
    with pytest.raises(ToolException) as excinfo:
        _Wrapper().run("get_work_item")

    assert str(excinfo.value) == NO_PROJECT_MESSAGE


def test_a_call_that_needs_no_project_never_fails_for_lack_of_one():
    wrapper = _Wrapper()

    with wrapper._project_scope(None):
        assert wrapper.project is None


def test_index_data_keeps_the_project_argument_for_the_index_record():
    project, kwargs = _Wrapper("Digital").run("index_data", project="Other", index_name="x")

    assert project == "Other"
    assert kwargs == {"project": "Other", "index_name": "x"}


def test_the_project_does_not_outlive_the_call():
    wrapper = _Wrapper("Digital")
    wrapper.run("get_work_item", project="Other")

    assert wrapper._project() == "Digital"


def test_the_project_is_reset_when_the_call_fails():
    wrapper = _Wrapper("Digital")

    with pytest.raises(RuntimeError):
        with wrapper._project_scope("Other"):
            raise RuntimeError("boom")

    assert wrapper._project() == "Digital"


def test_a_nested_call_without_a_project_keeps_the_outer_one():
    wrapper = _Wrapper("Digital")

    with wrapper._project_scope("Other"), wrapper._project_scope(None):
        assert wrapper._project() == "Other"


def test_a_scope_belongs_to_one_wrapper():
    outer, inner = _Wrapper("Digital"), _Wrapper("Payments")

    with outer._project_scope("Other"):
        assert inner._project() == "Payments"


def test_concurrent_calls_on_one_wrapper_see_only_their_own_project():
    wrapper = _Wrapper("Digital")
    barrier = threading.Barrier(2)

    def call(project):
        with wrapper._project_scope(project):
            barrier.wait()
            return wrapper._project()

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(call, ["A", "B"])) == ["A", "B"]


def test_worker_threads_see_the_call_project_through_in_caller_context():
    wrapper = _Wrapper("Digital")

    with wrapper._project_scope("Other"), ThreadPoolExecutor(max_workers=2) as executor:
        bare = executor.submit(wrapper._project).result()
        wrapped = list(executor.map(in_caller_context(lambda _: wrapper._project()), range(4)))

    assert bare == "Digital"
    assert wrapped == ["Other"] * 4


def test_loaders_keep_their_project_when_resumed_on_another_thread():
    wrapper = _Wrapper("Digital")

    def documents():
        for _ in range(3):
            yield MagicMock(metadata={"seen": wrapper._project()})

    with wrapper._project_scope("Other"):
        loader = wrapper._load_in_project(documents)
    with ThreadPoolExecutor(max_workers=1) as executor:
        loaded = executor.submit(list, loader).result()

    assert [d.metadata for d in loaded] == [{"seen": "Other", "project": "Other"}] * 3


def test_per_document_work_runs_in_the_project_the_document_came_from():
    wrapper = _Wrapper("Digital")
    document = MagicMock(metadata={"project": "Other"})

    seen = list(wrapper._in_document_project(document, lambda: iter([wrapper._project()])))

    assert seen == ["Other"]


def test_the_project_argument_is_the_last_field_and_names_the_default():
    schema = _Wrapper("Digital")._with_project(_Args)

    assert list(schema.model_fields) == ["query", "project"]
    assert "'Digital'" in schema.model_fields["project"].description
    assert schema(query="q", project="null").project is None


def test_without_a_default_the_argument_says_it_is_required():
    description = _Wrapper()._with_project(_Args).model_fields["project"].description

    assert "required when the toolkit configures none" in description


def test_the_configured_project_is_matched_case_insensitively():
    wrapper = _Wrapper("Payments")

    with wrapper._project_scope("payments"):
        assert wrapper._targets_configured_project()
    with wrapper._project_scope("Other"):
        assert not wrapper._targets_configured_project()


def test_index_data_records_the_configured_project_when_the_call_names_none():
    project, kwargs = _Wrapper("Digital").run("index_data", project="null", index_name="x")

    assert (project, kwargs["project"]) == ("Digital", "Digital")


def test_vector_store_tools_take_no_project_argument():
    tools = [{"name": "search_index", "args_schema": _Args}, {"name": "get_wiki", "args_schema": _Args}]

    tools = _Wrapper("Digital")._add_project_argument(tools)

    assert "project" not in tools[0]["args_schema"].model_fields
    assert "project" in tools[1]["args_schema"].model_fields


def test_organization_access_check_lists_one_project():
    with patch.object(ado_utils.requests, "get") as get:
        ado_utils.check_organization_access("https://dev.azure.com/org", "tok")

    assert get.call_args[0][0] == "https://dev.azure.com/org/_apis/projects?$top=1&api-version=7.0"
    assert get.call_args.kwargs["timeout"] == 5
