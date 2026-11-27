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

"""Excel indexing chunks at LOADER_MAX_TOKENS_DEFAULT unless the user overrides it (#7251)."""

import pytest
from langchain_core.documents import Document

from elitea_sdk.runtime.langchain.constants import LOADER_MAX_TOKENS_DEFAULT
from elitea_sdk.runtime.langchain.document_loaders.constants import loaders_allowed_to_override, loaders_map
from elitea_sdk.tools.utils import content_parser
from tests.runtime.langchain.test_excel_streaming_7047 import (
    full_read,
    index_workbook,
    large_workbook,
    window_tokens,
)


class RecordingLoader:
    calls = []

    def __init__(self, **kwargs):
        # Mirrors EliteAExcelLoader: a missing max_tokens falls back to the loader default.
        RecordingLoader.calls.append(int(kwargs.get("max_tokens", LOADER_MAX_TOKENS_DEFAULT)))

    def load(self):
        return [Document(page_content="row", metadata={})]


@pytest.fixture
def recorded_max_tokens(monkeypatch):
    RecordingLoader.calls = []
    for extension in (".xlsx", ".xls"):
        monkeypatch.setitem(loaders_map, extension, {**loaders_map[extension], "class": RecordingLoader})
    return RecordingLoader.calls


@pytest.mark.parametrize("extension", [".xlsx", ".xls"])
def test_the_indexing_form_default_is_a_bounded_chunk_size(extension):
    assert loaders_allowed_to_override[extension]["max_tokens"] == LOADER_MAX_TOKENS_DEFAULT


@pytest.mark.parametrize("chunking_config", [None, {}, {"__ext__": {}}, {"__ext__": {"max_tokens": LOADER_MAX_TOKENS_DEFAULT}}],
                         ids=["no-config", "empty-config", "empty-extension", "form-default"])
@pytest.mark.parametrize("extension", [".xlsx", ".xls"])
def test_indexing_passes_the_default_chunk_size_to_the_loader(recorded_max_tokens, extension, chunking_config):
    if chunking_config and "__ext__" in chunking_config:
        chunking_config = {extension: chunking_config["__ext__"]}

    list(content_parser.process_content_by_type(b"x", f"book{extension}", chunking_config=chunking_config))

    assert recorded_max_tokens[-1] == LOADER_MAX_TOKENS_DEFAULT


@pytest.mark.parametrize("max_tokens", [-1, 64, 1000])
@pytest.mark.parametrize("extension", [".xlsx", ".xls"])
def test_indexing_keeps_the_chunk_size_the_user_chose(recorded_max_tokens, extension, max_tokens):
    list(content_parser.process_content_by_type(b"x", f"book{extension}", chunking_config={extension: {"max_tokens": max_tokens}}))

    assert recorded_max_tokens[-1] == max_tokens


def test_indexing_a_large_sheet_with_the_default_config_splits_it(tmp_path):
    path = large_workbook(tmp_path, 2_000)

    chunks = index_workbook(path, {".xlsx": dict(loaders_allowed_to_override[".xlsx"])})

    assert len(chunks) > 1
    assert max(window_tokens(chunk.page_content) for chunk in chunks) <= LOADER_MAX_TOKENS_DEFAULT


def test_read_file_still_returns_one_chunk_per_sheet(tmp_path):
    path = large_workbook(tmp_path, 2_000)

    content = content_parser.parse_file_content(file_path=path, extra_params={"skip_size_check": True})

    assert content == {"Large": [chunk for _, chunk in full_read(path, max_tokens=-1)]}
