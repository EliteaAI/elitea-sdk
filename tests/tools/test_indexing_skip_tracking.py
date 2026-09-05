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

"""Tests for what the indexers record about the items they leave out."""

import pytest

from elitea_sdk.runtime.tools.artifact import ArtifactWrapper
from elitea_sdk.tools.non_code_indexer_toolkit import NonCodeIndexerToolkit
from elitea_sdk.tools.sharepoint.file_filters import skip_reporter


class TrackingToolkit(NonCodeIndexerToolkit):
    """Minimal wrapper for exercising the tracking helpers directly."""


@pytest.fixture
def toolkit():
    instance = TrackingToolkit.model_construct()
    instance._init_indexing_stats()
    return instance


class TestAttachmentTracking:
    @pytest.mark.parametrize(
        "reason,attribute",
        [
            ("filtered", "dependent_items_filtered"),
            ("unsupported", "dependent_items_unsupported"),
            ("extension", "dependent_items_unsupported"),
            ("empty", "dependent_items_empty"),
            ("error", "dependent_items_skipped"),
        ],
    )
    def test_reason_selects_the_dependent_set(self, toolkit, reason, attribute):
        toolkit._track_skipped_attachment("diagram.ai", reason=reason)

        assert getattr(toolkit.get_indexing_stats(), attribute) == {"diagram.ai"}

    def test_attachments_never_enter_the_top_level_skip_total(self, toolkit):
        toolkit._track_skipped_attachment("a.png", reason="filtered")
        toolkit._track_skipped_attachment("b.ai", reason="unsupported")
        toolkit._track_skipped_attachment("c.zip", reason="error")
        toolkit._track_skipped_attachment("d.txt", reason="empty")

        assert toolkit.get_indexing_stats().to_dict()["total_skipped"] == 0

    def test_dependent_sets_are_serialized_with_exact_counts(self, toolkit):
        toolkit._track_skipped_attachment("a.png", reason="filtered")

        payload = toolkit.get_indexing_stats().to_dict()

        assert payload["dependent_items_filtered"] == {"count": 1, "items": ["a.png"]}
        assert payload["dependent_items_unsupported"] == {"count": 0, "items": []}


class TestRuntimeSkipTracking:
    def test_configured_filters_are_top_level_skips(self, toolkit):
        toolkit._track_runtime_skipped("draft.md", reason="filtered")

        stats = toolkit.get_indexing_stats()
        assert stats.documents_skipped_filtered == {"draft.md"}
        assert stats.to_dict()["total_skipped"] == 1

    def test_unsupported_extensions_stay_in_their_own_set(self, toolkit):
        toolkit._track_runtime_skipped("scan.raw", reason="extension")

        assert toolkit.get_indexing_stats().runtime_skipped_extension == {"scan.raw"}

    def test_anything_else_is_an_error(self, toolkit):
        toolkit._track_runtime_skipped("broken.md", reason="error")

        assert toolkit.get_indexing_stats().runtime_skipped_error == {"broken.md"}


class TestSkipReporter:
    def test_missing_callback_is_safe_to_call(self):
        skip_reporter(None)("anything.txt")

    def test_registered_callback_receives_the_name(self):
        seen = []

        skip_reporter(seen.append)("report.pdf")

        assert seen == ["report.pdf"]


class FakeArtifactClient:
    def __init__(self, names):
        self.names = names

    def list(self, bucket_name=None, prefix=None, delimiter=None):
        return {"rows": [{"type": "file", "name": name, "key": name} for name in self.names]}


@pytest.fixture
def artifact_toolkit():
    instance = ArtifactWrapper.model_construct(bucket="docs")
    object.__setattr__(instance, "artifact", FakeArtifactClient(
        ["notes.md", "guide.md", "photo.png", "archive.zip"]
    ))
    object.__setattr__(instance, "elitea", None)
    instance._init_indexing_stats()
    return instance


class TestArtifactFilteredFileTracking:
    def test_files_excluded_by_include_patterns_are_reported(self, artifact_toolkit):
        skipped = []

        result = artifact_toolkit.list_files(
            "docs", include=["*.md"], on_file_skipped=skipped.append
        )

        assert [row["name"] for row in result["rows"]] == ["notes.md", "guide.md"]
        assert sorted(skipped) == ["archive.zip", "photo.png"]

    def test_files_excluded_by_skip_patterns_are_reported(self, artifact_toolkit):
        skipped = []

        artifact_toolkit.list_files(
            "docs", skip=["*.zip"], on_file_skipped=skipped.append
        )

        assert skipped == ["archive.zip"]

    def test_listing_without_a_callback_still_filters(self, artifact_toolkit):
        result = artifact_toolkit.list_files("docs", include=["*.md"])

        assert len(result["rows"]) == 2

    def test_loader_records_filtered_files_as_skipped_documents(self, artifact_toolkit):
        list(artifact_toolkit._base_loader(include_extensions=["*.md"]))

        stats = artifact_toolkit.get_indexing_stats()
        assert stats.documents_skipped_filtered == {"photo.png", "archive.zip"}
