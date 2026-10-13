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

"""remove_index must name the index it removes (#7023)."""

import pytest
from langchain_core.tools import ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.tools.base.tool import BaseAction
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit, IndexTools
from elitea_sdk.tools.index_params import RemoveIndexParams


class _Toolkit(BaseIndexerToolkit):
    pass


@pytest.fixture
def recorder(monkeypatch):
    calls = {"cleaned": [], "events": []}

    def clean(self, index_name="", including_index_meta=False):
        calls["cleaned"].append(index_name)
        return 0 if index_name == "missing" else 4

    monkeypatch.setattr(VectorStoreWrapperBase, "_clean_collection", clean)
    monkeypatch.setattr(_Toolkit, "_emit_index_data_removed_event",
                        lambda self, index_name: calls["events"].append(index_name))
    monkeypatch.setattr(_Toolkit, "list_indexes", lambda self: ["IdxA", "IdxB"])
    return calls


@pytest.fixture
def toolkit(recorder):
    return _Toolkit.model_construct()


def _remove_index_tool(toolkit):
    entry = next(tool for tool in toolkit.get_available_tools()
                 if tool["name"] == IndexTools.REMOVE_INDEX.value)
    return BaseAction(api_wrapper=toolkit, name=entry["name"],
                      description=entry["description"], args_schema=entry["args_schema"])


def test_the_model_sees_index_name_as_required_and_non_empty(toolkit):
    parameters = convert_to_openai_tool(_remove_index_tool(toolkit))["function"]["parameters"]

    assert parameters["required"] == ["index_name"]
    assert parameters["properties"]["index_name"]["minLength"] == 1


@pytest.mark.parametrize("arguments", [{}, {"index_name": ""}, {"index_name": None}])
def test_the_tool_rejects_a_missing_name_before_deleting_anything(toolkit, recorder, arguments):
    with pytest.raises(ValidationError):
        _remove_index_tool(toolkit).invoke(arguments)

    assert recorder == {"cleaned": [], "events": []}


@pytest.mark.parametrize("index_name", ["", "   ", None])
def test_a_blank_name_is_rejected_before_deleting_anything(toolkit, recorder, index_name):
    with pytest.raises(ToolException) as raised:
        toolkit.remove_index(index_name)

    assert str(raised.value) == (
        "remove_index requires a non-empty 'index_name'. Available indexes: [\"IdxA\", \"IdxB\"]"
    )
    assert recorder == {"cleaned": [], "events": []}


def test_a_named_index_is_still_removed_and_its_schedule_cleanup_announced(toolkit, recorder):
    result = toolkit.remove_index("IdxA")

    assert result.startswith("Index 'IdxA' has been removed from the vector store.")
    assert recorder == {"cleaned": ["IdxA"], "events": ["IdxA"]}


def test_an_unknown_name_is_still_reported_as_not_found(toolkit, recorder):
    with pytest.raises(ToolException) as raised:
        toolkit.remove_index("missing")

    assert str(raised.value).startswith("Index 'missing' not found.")
    assert recorder["events"] == []


def test_the_schema_rejects_an_overlong_name():
    with pytest.raises(ValidationError):
        RemoveIndexParams(index_name="x" * 33)
