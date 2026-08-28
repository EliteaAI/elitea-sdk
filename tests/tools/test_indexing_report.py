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

"""Unit tests for the canonical IndexingReport: build, render and persistence."""

import json
import re

import pytest

from elitea_sdk.tools.base_indexer_toolkit import (
    REPORT_ERRORS_SAMPLE_SIZE,
    REPORT_ERROR_MAX_LENGTH,
    REPORT_ITEMS_SAMPLE_SIZE,
    REPORT_VERSION,
    BaseIndexerToolkit,
    IndexingStats,
    IndexingStatus,
    ReportKind,
    _sample_skipped_payload,
    build_error_report,
    is_up_to_date_run,
    normalize_report_errors,
    render_report_text,
)
from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.runtime.utils.utils import IndexerKeywords

PAGE_LABELS = ("page", "pages")
ATTACHMENT_LABELS = ("attachment", "attachments")

# The regex an older UI applies to every line of the indexing message. Only the
# headline is meant to match it; see test_only_the_headline_matches_legacy_parser.
LEGACY_SUMMARY_REFORMAT = re.compile(r"^(.+?)\s+(\d+\s+\w+)\.?\s*$")


def build_stats(**overrides) -> IndexingStats:
    stats = IndexingStats()
    for attribute, value in overrides.items():
        current = getattr(stats, attribute)
        if isinstance(current, set):
            current.update(value)
        else:
            setattr(stats, attribute, value)
    return stats


def make_report(stats: IndexingStats, indexed_count: int = 0, **overrides):
    kwargs = dict(
        status=IndexingStatus.OK,
        indexed_count=indexed_count,
        item_labels=PAGE_LABELS,
        dependent_labels=ATTACHMENT_LABELS,
    )
    kwargs.update(overrides)
    return stats.build_report(**kwargs)


def category(report, kind: ReportKind):
    return next(item for item in report["categories"] if item["kind"] == kind.value)


def group(report, kind: ReportKind, reason: str, dependent: bool = False):
    return next(
        item for item in category(report, kind)["groups"]
        if item["reason"] == reason and bool(item.get("dependent")) == dependent
    )


class TestReportStructure:
    def test_envelope_carries_version_status_and_labels(self):
        report = make_report(build_stats(), indexed_count=3)

        assert report["version"] == REPORT_VERSION
        assert report["status"] == "ok"
        assert report["item_labels"] == {"singular": "page", "plural": "pages"}
        assert report["dependent_labels"] == {"singular": "attachment", "plural": "attachments"}

    def test_categories_are_the_closed_kind_set_in_order(self):
        report = make_report(build_stats())

        assert [item["kind"] for item in report["categories"]] == [kind.value for kind in ReportKind]

    def test_status_vocabulary_matches_the_tool_result_status(self):
        report = make_report(build_stats(), status=IndexingStatus.PARTLY_INDEXED)

        assert report["status"] == IndexingStatus.PARTLY_INDEXED.value == "partly_indexed"


class TestStatsMapping:
    def test_filter_skips_land_in_skipped(self):
        stats = build_stats(
            documents_skipped_filtered={"a.tmp", "b.tmp"},
            files_skipped_whitelist={"c.png"},
            files_skipped_blacklist={"d.exe"},
            files_skipped_empty={"e.md"},
        )

        report = make_report(stats)

        assert category(report, ReportKind.SKIPPED)["count"] == 5
        assert group(report, ReportKind.SKIPPED, "filtered")["count"] == 2
        assert group(report, ReportKind.SKIPPED, "not_in_whitelist")["count"] == 1
        assert group(report, ReportKind.SKIPPED, "blacklisted")["count"] == 1
        assert group(report, ReportKind.SKIPPED, "empty")["count"] == 1

    def test_unchanged_documents_are_listed_but_never_counted_as_skipped(self):
        """They are in the store, so no consumer should have to subtract them back out."""
        stats = build_stats(documents_already_indexed={"one", "two"})

        report = make_report(stats, indexed_count=5)

        assert group(report, ReportKind.SKIPPED, "unchanged")["count"] == 2
        assert report["totals"]["unchanged"] == 2
        assert report["totals"]["skipped"] == 0
        assert report["totals"]["total"] == 7

    def test_unsupported_extensions_from_both_sets_merge_into_one_group(self):
        stats = build_stats(
            files_unsupported_extension={"design.ai"},
            runtime_skipped_extension={"scan.raw"},
        )

        report = make_report(stats)

        unsupported = group(report, ReportKind.NOT_INDEXED, "unsupported_format")
        assert unsupported["count"] == 2
        assert unsupported["items"] == ["design.ai", "scan.raw"]

    def test_error_skips_are_counted_as_failed(self):
        stats = build_stats(
            files_skipped_read_error={"locked.doc"},
            documents_skipped_error={"broken.pdf"},
            runtime_skipped_error={"timeout.txt"},
        )

        report = make_report(stats)

        assert category(report, ReportKind.FAILED)["count"] == 3
        assert group(report, ReportKind.FAILED, "read_error")["count"] == 1
        assert group(report, ReportKind.FAILED, "processing_error")["count"] == 2

    def test_empty_stats_produce_no_groups(self):
        report = make_report(build_stats(), indexed_count=7)

        assert all(not item["groups"] for item in report["categories"])
        assert report["totals"]["total"] == 7

    def test_item_lists_are_sampled_with_a_more_count(self):
        stats = build_stats(documents_skipped_filtered={f"file-{index:02d}.tmp" for index in range(12)})

        filtered = group(make_report(stats), ReportKind.SKIPPED, "filtered")

        assert filtered["count"] == 12
        assert len(filtered["items"]) == REPORT_ITEMS_SAMPLE_SIZE
        assert filtered["more"] == 12 - REPORT_ITEMS_SAMPLE_SIZE

    def test_short_item_lists_carry_no_more_key(self):
        stats = build_stats(documents_skipped_filtered={"only.tmp"})

        assert "more" not in group(make_report(stats), ReportKind.SKIPPED, "filtered")


class TestCountingInvariant:
    """Where a category has groups, its count is the sum of the counted ones — the rule
    a consumer would otherwise have to infer from `dependent` plus a hardcoded reason.
    `indexed` carries its count directly and has no breakdown."""

    def test_every_category_count_equals_its_counted_groups(self):
        stats = build_stats(
            documents_skipped_filtered={"a.tmp", "b.tmp"},
            files_skipped_empty={"c.md"},
            documents_already_indexed={"d", "e", "f"},
            files_unsupported_extension={"g.ai"},
            documents_skipped_error={"h.pdf"},
            dependent_items_filtered={"i.png"},
            dependent_items_unsupported={"j.raw"},
            dependent_items_empty={"k.txt"},
            dependent_items_skipped={"l.zip"},
        )

        report = make_report(stats, indexed_count=9)

        for item in report["categories"]:
            if not item["groups"]:
                continue
            counted = [group for group in item["groups"] if group.get("counted", True)]
            assert item["count"] == sum(group["count"] for group in counted), item["kind"]

    @pytest.mark.parametrize(
        "attribute,reason",
        [
            ("documents_already_indexed", "unchanged"),
            ("dependent_items_filtered", "filtered"),
            ("dependent_items_unsupported", "unsupported_format"),
            ("dependent_items_empty", "empty"),
            ("dependent_items_skipped", "processing_error"),
        ],
    )
    def test_uncounted_groups_say_so(self, attribute, reason):
        report = make_report(build_stats(**{attribute: {"x"}}), indexed_count=1)

        marked = [
            group
            for item in report["categories"]
            for group in item["groups"]
            if group["reason"] == reason and group.get("counted") is False
        ]
        assert marked, f"{attribute} should be reported as uncounted"

    def test_counted_groups_carry_no_flag(self):
        report = make_report(build_stats(documents_skipped_filtered={"a.tmp"}))

        assert "counted" not in group(report, ReportKind.SKIPPED, "filtered")


class TestDependentItems:
    def test_dependent_group_is_uncounted_and_carries_its_own_labels(self):
        stats = build_stats(dependent_items_skipped={"a.raw", "b.raw", "c.raw", "d.raw"})

        report = make_report(stats, indexed_count=10)

        dependent = group(report, ReportKind.FAILED, "processing_error", dependent=True)
        assert dependent["dependent"] is True
        assert dependent["count"] == 4
        assert dependent["item_labels"] == {"singular": "attachment", "plural": "attachments"}
        assert category(report, ReportKind.FAILED)["count"] == 0
        assert report["totals"]["failed"] == 0
        assert report["totals"]["dependent_not_indexed"] == 4
        assert report["totals"]["total"] == 10

    @pytest.mark.parametrize(
        "attribute,kind,reason",
        [
            ("dependent_items_filtered", ReportKind.SKIPPED, "filtered"),
            ("dependent_items_unsupported", ReportKind.NOT_INDEXED, "unsupported_format"),
            ("dependent_items_skipped", ReportKind.FAILED, "processing_error"),
        ],
    )
    def test_each_dependent_reason_lands_in_its_own_category(self, attribute, kind, reason):
        report = make_report(build_stats(**{attribute: {"x.png"}}), indexed_count=4)

        assert group(report, kind, reason, dependent=True)["count"] == 1
        assert category(report, kind)["count"] == 0
        assert report["totals"]["total"] == 4

    def test_dependent_reasons_share_one_aggregate_total(self):
        stats = build_stats(
            dependent_items_filtered={"a.png", "b.png"},
            dependent_items_unsupported={"c.ai"},
            dependent_items_skipped={"d.raw"},
        )

        assert make_report(stats, indexed_count=9)["totals"]["dependent_not_indexed"] == 4

    def test_a_top_level_group_and_a_dependent_one_coexist_in_a_category(self):
        stats = build_stats(
            documents_skipped_filtered={"page-a", "page-b"},
            dependent_items_filtered={"logo.png"},
        )

        report = make_report(stats, indexed_count=5)

        assert category(report, ReportKind.SKIPPED)["count"] == 2
        assert group(report, ReportKind.SKIPPED, "filtered")["count"] == 2
        assert report["totals"]["skipped"] == 2

    def test_dependent_items_never_inflate_the_total(self):
        with_dependents = make_report(
            build_stats(dependent_items_skipped={"x.raw"}), indexed_count=4
        )
        without_dependents = make_report(build_stats(), indexed_count=4)

        assert with_dependents["totals"]["total"] == without_dependents["totals"]["total"]


class TestTotals:
    def test_total_is_the_sum_of_the_four_categories(self):
        stats = build_stats(
            documents_skipped_filtered={"a.tmp"},
            files_unsupported_extension={"b.ai"},
            documents_skipped_error={"c.pdf"},
            documents_already_indexed={"d", "e"},
        )

        totals = make_report(stats, indexed_count=20)["totals"]

        assert totals["total"] == (
            totals["indexed"] + totals["skipped"] + totals["not_indexed"]
            + totals["failed"] + totals["unchanged"]
        )
        assert totals["total"] == 20 + 1 + 1 + 1 + 2

    def test_total_equals_total_fetched_from_the_same_stats(self):
        stats = build_stats(
            items_processed=191,
            documents_skipped_filtered={f"skip-{index}.tmp" for index in range(10)},
            files_skipped_empty={"empty.md", "blank.md"},
            documents_already_indexed={f"unchanged-{index}" for index in range(12)},
            runtime_skipped_extension={"diagram.ai"},
            documents_skipped_error={"corrupted.pdf"},
        )
        stats.total_fetched = stats.items_processed + stats.to_dict()["total_skipped"]

        report = make_report(stats, indexed_count=179)

        assert report["totals"]["total"] == stats.total_fetched

    def test_indexed_excludes_unchanged_documents(self):
        stats = build_stats(documents_already_indexed={"a", "b", "c"})

        totals = make_report(stats, indexed_count=179)["totals"]

        assert totals["indexed"] == 179
        assert totals["unchanged"] == 3


class TestErrorNormalization:
    def test_sql_and_parameter_tails_are_stripped_before_dedup(self):
        errors = [
            "insert failed\n[SQL: INSERT INTO x VALUES (?)]\n[parameters: (1, 2)]",
            "insert failed\n[SQL: INSERT INTO x VALUES (?)]\n[parameters: (3, 4)]",
        ]

        assert normalize_report_errors(errors) == (["insert failed"], 1)

    def test_background_footer_is_stripped(self):
        errors = ["boom (Background on this error at: https://sqlalche.me/e/20/e3q8)"]

        assert normalize_report_errors(errors) == (["boom"], 1)

    def test_long_messages_are_truncated(self):
        sampled, _ = normalize_report_errors(["x" * (REPORT_ERROR_MAX_LENGTH + 200)])

        assert len(sampled[0]) == REPORT_ERROR_MAX_LENGTH + 1
        assert sampled[0].endswith("…")

    def test_distinct_errors_are_capped_but_fully_counted(self):
        errors = [f"failure number {index}" for index in range(12)]

        sampled, total = normalize_report_errors(errors)

        assert len(sampled) == REPORT_ERRORS_SAMPLE_SIZE
        assert total == 12

    def test_report_carries_the_bounded_error_list(self):
        report = make_report(
            build_stats(), errors=[f"distinct failure {index}" for index in range(9)]
        )

        assert len(report["errors"]) == REPORT_ERRORS_SAMPLE_SIZE
        assert report["errors_total"] == 9

    def test_errors_never_enter_the_counted_categories(self):
        report = make_report(build_stats(), indexed_count=5, errors=["boom", "bang"])

        assert report["totals"]["failed"] == 0
        assert report["totals"]["total"] == 5

    def test_blank_and_missing_errors_are_dropped(self):
        assert normalize_report_errors(["", "   ", None]) == ([], 0)


class TestUpToDateRule:
    def test_run_with_only_unchanged_items_is_up_to_date(self):
        assert is_up_to_date_run({"indexed": 0, "failed": 0, "unchanged": 196})

    def test_run_that_indexed_something_is_not_up_to_date(self):
        assert not is_up_to_date_run({"indexed": 1, "failed": 0, "unchanged": 196})

    def test_run_with_failures_is_not_up_to_date(self):
        assert not is_up_to_date_run({"indexed": 0, "failed": 2, "unchanged": 196})

    def test_empty_run_is_not_up_to_date(self):
        assert not is_up_to_date_run({"indexed": 0, "failed": 0, "unchanged": 0})


class TestRenderReportText:
    def test_successful_run_renders_categories_under_the_headline(self):
        stats = build_stats(
            documents_skipped_filtered={"a.tmp", "b.tmp"},
            documents_skipped_error={"broken.pdf"},
        )

        text = render_report_text(make_report(stats, indexed_count=179))

        assert text.splitlines()[0] == "Successfully indexed 179 pages."
        assert "✓ 179 pages indexed" in text
        assert "⚠ 2 pages skipped" in text
        assert "    - Excluded by configured filters (2): a.tmp, b.tmp" in text
        assert "✕ 1 page failed" in text

    def test_singular_noun_is_used_for_single_items(self):
        text = render_report_text(make_report(build_stats(), indexed_count=1))

        assert text.splitlines()[0] == "Successfully indexed 1 page."

    def test_all_unchanged_run_renders_up_to_date_instead_of_zero_indexed(self):
        stats = build_stats(documents_already_indexed={f"page-{index}" for index in range(196)})

        text = render_report_text(make_report(stats, indexed_count=0))

        assert text == "Up to date — 196 pages unchanged."
        assert "0 pages" not in text
        assert "skipped" not in text

    def test_up_to_date_run_still_reports_genuine_skips(self):
        stats = build_stats(
            documents_already_indexed={"a", "b"},
            documents_skipped_filtered={"noise.tmp"},
        )

        text = render_report_text(make_report(stats, indexed_count=0))

        assert text.splitlines()[0] == "Up to date — 2 pages unchanged."
        assert "⚠ 1 page skipped" in text
        assert "unchanged)" not in text

    def test_unchanged_items_are_not_reported_as_skipped(self):
        """An incremental reindex touching a few of many documents must not describe the
        rest as skipped — the chip counts them as indexed."""
        stats = build_stats(
            documents_already_indexed={f"page-{index}" for index in range(195)},
            documents_skipped_filtered={"noise.tmp"},
        )

        lines = render_report_text(make_report(stats, indexed_count=5)).splitlines()

        assert lines[0] == "Successfully indexed 5 pages."
        assert "✓ 5 pages indexed" in lines
        assert "⚠ 1 page skipped" in lines
        assert "ℹ 195 pages already indexed (unchanged)" in lines
        assert not any("196 pages skipped" in line for line in lines)

    def test_unchanged_line_is_omitted_when_the_headline_already_says_it(self):
        stats = build_stats(documents_already_indexed={f"page-{index}" for index in range(196)})

        assert render_report_text(make_report(stats)) == "Up to date — 196 pages unchanged."

    def test_empty_run_keeps_the_legacy_nothing_to_do_message(self):
        assert render_report_text(make_report(build_stats())) == "No new documents to index."

    def test_partial_run_headline(self):
        text = render_report_text(
            make_report(build_stats(), indexed_count=40, status=IndexingStatus.PARTLY_INDEXED)
        )

        assert text.splitlines()[0] == "Partially indexed 40 pages."

    def test_failed_run_headline_names_the_item_type(self):
        text = render_report_text(make_report(build_stats(), status=IndexingStatus.ERROR))

        assert text.splitlines()[0] == "Failed to index pages."

    def test_no_chunk_counts_anywhere(self):
        stats = build_stats(documents_skipped_filtered={"a.tmp"})

        text = render_report_text(make_report(stats, indexed_count=9, errors=["boom"]))

        assert "chunk" not in text.lower()

    def test_dependent_only_category_uses_the_dependent_noun(self):
        stats = build_stats(dependent_items_unsupported={"a.raw", "b.raw"})

        text = render_report_text(make_report(stats, indexed_count=10))

        assert "⚠ 2 attachments not indexed" in text
        assert "    - Unsupported format (2 attachments): a.raw, b.raw" in text

    def test_more_counts_are_rendered_with_their_noun(self):
        stats = build_stats(documents_skipped_filtered={f"file-{index:02d}.tmp" for index in range(9)})

        text = render_report_text(make_report(stats, indexed_count=1))

        assert "      ... and 4 more pages" in text

    def test_errors_render_as_a_detail_block_with_a_hidden_count(self):
        report = make_report(
            build_stats(), indexed_count=5, status=IndexingStatus.PARTLY_INDEXED,
            errors=[f"distinct failure {index}" for index in range(8)],
        )

        text = render_report_text(report)

        assert "Errors:" in text
        assert "    ... and 3 more distinct errors" in text

    def test_zero_count_categories_are_hidden(self):
        text = render_report_text(make_report(build_stats(), indexed_count=5))

        assert "skipped" not in text
        assert "not indexed" not in text
        assert "failed" not in text


class TestLegacyUiCompatibility:
    """An older UI parses this message with a regex before the new renderer ships."""

    @pytest.mark.parametrize(
        "indexed_count,expected",
        [(179, "Successfully indexed 179 pages."), (1, "Successfully indexed 1 page.")],
    )
    def test_headline_keeps_the_legacy_grammar(self, indexed_count, expected):
        text = render_report_text(make_report(build_stats(), indexed_count=indexed_count))

        assert text.splitlines()[0] == expected
        assert LEGACY_SUMMARY_REFORMAT.match(text.splitlines()[0])

    def test_only_the_headline_matches_legacy_parser(self):
        stats = build_stats(
            documents_skipped_filtered={f"file-{index:02d}.tmp" for index in range(9)},
            files_skipped_empty={"empty.md"},
            runtime_skipped_extension={"design.ai"},
            documents_skipped_error={"broken.pdf"},
            dependent_items_skipped={"a.raw", "b.raw"},
        )
        report = make_report(
            stats, indexed_count=179, status=IndexingStatus.PARTLY_INDEXED,
            errors=[f"distinct failure {index}" for index in range(8)],
        )

        lines = render_report_text(report).splitlines()

        assert LEGACY_SUMMARY_REFORMAT.match(lines[0])
        mangled = [line for line in lines[1:] if LEGACY_SUMMARY_REFORMAT.match(line)]
        assert mangled == []

    def test_up_to_date_headline_passes_through_the_legacy_parser(self):
        stats = build_stats(documents_already_indexed={f"page-{index}" for index in range(196)})

        headline = render_report_text(make_report(stats)).splitlines()[0]

        assert not LEGACY_SUMMARY_REFORMAT.match(headline)


class TestErrorReport:
    def test_error_report_carries_the_exception_and_zero_totals(self):
        report = build_error_report(
            "connection refused",
            item_labels=PAGE_LABELS,
            dependent_labels=ATTACHMENT_LABELS,
        )

        assert report["status"] == IndexingStatus.ERROR.value
        assert report["errors"] == ["connection refused"]
        assert report["totals"]["total"] == 0

    def test_error_report_keeps_whatever_was_tracked_before_the_failure(self):
        stats = build_stats(documents_skipped_filtered={"a.tmp"})

        report = build_error_report(
            "boom",
            item_labels=PAGE_LABELS, dependent_labels=ATTACHMENT_LABELS, stats=stats,
        )

        assert group(report, ReportKind.SKIPPED, "filtered")["count"] == 1


class TestSkippedPayloadSampling:
    def test_name_lists_are_sampled_while_counts_stay_exact(self):
        stats = build_stats(documents_skipped_filtered={f"file-{index:03d}.tmp" for index in range(500)})

        sampled = _sample_skipped_payload(stats.to_dict())

        assert sampled["documents_skipped"]["filtered_count"] == 500
        assert len(sampled["documents_skipped"]["filtered"]) == REPORT_ITEMS_SAMPLE_SIZE

    def test_scalar_fields_survive_untouched(self):
        stats = build_stats(items_processed=42)
        stats.total_fetched = 50

        sampled = _sample_skipped_payload(stats.to_dict())

        assert sampled["items_processed"] == 42
        assert sampled["total_fetched"] == 50

    def test_sampling_does_not_mutate_the_source_payload(self):
        stats = build_stats(documents_skipped_filtered={f"file-{index}.tmp" for index in range(20)})
        payload = stats.to_dict()

        _sample_skipped_payload(payload)

        assert len(payload["documents_skipped"]["filtered"]) == 20


class RecordingToolkit(BaseIndexerToolkit):
    """Exercises index_meta_update / index_meta_init against an in-memory meta row."""


@pytest.fixture
def toolkit(monkeypatch):
    instance = RecordingToolkit.model_construct()
    object.__setattr__(instance, "_stored_meta", None)
    written = []
    object.__setattr__(instance, "written", written)

    def fake_add_documents(vectorstore=None, documents=None, ids=None):
        metadata = dict(documents[0].metadata)
        written.append(metadata)
        object.__setattr__(
            instance, "_stored_meta", {"id": "meta-1", "content": "index_meta_x", "metadata": metadata}
        )

    monkeypatch.setattr(
        "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents", fake_add_documents
    )
    # index_meta reads go through super(), so the stubs have to sit on the base class.
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_index_meta", lambda self, name: self._stored_meta)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_indexed_count", lambda self, name: 965)
    monkeypatch.setattr(RecordingToolkit, "_is_scheduled_run", lambda self: False)
    return instance


def seed_completed_run(toolkit, **metadata_overrides):
    metadata = {
        "collection": "x",
        "state": IndexerKeywords.INDEX_META_COMPLETED.value,
        "indexed": 191,
        "total": 205,
        "report": json.dumps({"status": "ok", "totals": {"indexed": 179}}),
        "skipped": json.dumps({"items_processed": 191}),
        "error": None,
        "history": json.dumps([{"state": "created"}, {"state": "completed"}]),
    }
    metadata.update(metadata_overrides)
    object.__setattr__(toolkit, "_stored_meta", {"id": "meta-1", "content": "c", "metadata": metadata})


class TestIndexMetaPersistence:
    def test_successful_report_drives_total_and_unchanged_inclusive_indexed(self, toolkit):
        seed_completed_run(toolkit)
        report = make_report(
            build_stats(documents_already_indexed={"a", "b", "c"}), indexed_count=179
        )

        toolkit.index_meta_update("x", IndexerKeywords.INDEX_META_COMPLETED.value, 900, report=report)

        stored = toolkit.written[-1]
        assert stored["total"] == report["totals"]["total"]
        assert stored["indexed"] == 179 + 3
        assert json.loads(stored["report"])["totals"]["indexed"] == 179

    def test_report_totals_win_over_the_skipped_heuristic(self, toolkit):
        seed_completed_run(toolkit)
        stats = build_stats(items_processed=50, documents_skipped_filtered={"a.tmp"})
        report = make_report(stats, indexed_count=49)

        toolkit.index_meta_update(
            "x", IndexerKeywords.INDEX_META_COMPLETED.value, 900,
            skipped=stats.to_dict(), docs_count=49, report=report,
        )

        assert toolkit.written[-1]["total"] == 50

    def test_error_report_preserves_the_previous_runs_total(self, toolkit):
        seed_completed_run(toolkit)
        error_report = build_error_report(
            "auth failed",
            item_labels=PAGE_LABELS, dependent_labels=ATTACHMENT_LABELS,
        )

        toolkit.index_meta_update(
            "x", IndexerKeywords.INDEX_META_FAILED.value, 0, error="auth failed", report=error_report
        )

        stored = toolkit.written[-1]
        # Both counts stay as the last good run left them: the store still holds and
        # serves those documents, and they are the only figures still true.
        assert stored["total"] == 205
        assert stored["indexed"] == 191
        assert json.loads(stored["report"])["status"] == "error"

    def test_skipped_blob_is_sampled_before_persisting(self, toolkit):
        seed_completed_run(toolkit)
        stats = build_stats(documents_skipped_filtered={f"file-{index:03d}.tmp" for index in range(300)})

        toolkit.index_meta_update(
            "x", IndexerKeywords.INDEX_META_COMPLETED.value, 900,
            skipped=stats.to_dict(), report=make_report(stats),
        )

        persisted = json.loads(toolkit.written[-1]["skipped"])
        assert persisted["documents_skipped"]["filtered_count"] == 300
        assert len(persisted["documents_skipped"]["filtered"]) == REPORT_ITEMS_SAMPLE_SIZE

    def test_history_entry_mirrors_the_stored_report(self, toolkit):
        seed_completed_run(toolkit)
        report = make_report(build_stats(), indexed_count=3)

        toolkit.index_meta_update("x", IndexerKeywords.INDEX_META_COMPLETED.value, 3, report=report)

        stored = toolkit.written[-1]
        assert json.loads(stored["history"])[-1]["report"] == stored["report"]

    def test_reindex_init_clears_the_previous_runs_report(self, toolkit):
        seed_completed_run(toolkit)

        toolkit.index_meta_init("x", {"index_name": "x"})

        stored = toolkit.written[-1]
        assert stored["report"] is None
        assert stored["skipped"] is None
        assert stored["error"] is None


class TestLoaderStatsStamping:
    def test_yielded_documents_are_counted_when_the_loader_tracks_nothing(self, toolkit):
        object.__setattr__(toolkit, "_indexing_stats", IndexingStats())

        toolkit._stamp_loader_stats(42)

        assert toolkit.get_indexing_stats().items_processed == 42

    def test_a_loader_that_counts_for_itself_wins(self, toolkit):
        object.__setattr__(toolkit, "_indexing_stats", build_stats(items_processed=7))

        toolkit._stamp_loader_stats(120)

        assert toolkit.get_indexing_stats().items_processed == 7

    def test_fetched_total_includes_items_skipped_before_yielding(self, toolkit):
        object.__setattr__(
            toolkit, "_indexing_stats",
            build_stats(documents_skipped_filtered={"a.tmp", "b.tmp", "c.tmp"}),
        )

        toolkit._stamp_loader_stats(10)

        assert toolkit.get_indexing_stats().total_fetched == 13

    def test_a_loader_that_stamps_its_own_fetched_total_wins(self, toolkit):
        stats = build_stats(documents_skipped_filtered={"a.tmp"})
        stats.total_fetched = 999
        object.__setattr__(toolkit, "_indexing_stats", stats)

        toolkit._stamp_loader_stats(10)

        assert toolkit.get_indexing_stats().total_fetched == 999

    def test_stamping_without_stats_is_a_no_op(self, toolkit):
        toolkit._stamp_loader_stats(10)

        assert toolkit.get_indexing_stats() is None


@pytest.fixture
def indexing_toolkit(toolkit, monkeypatch):
    """A toolkit whose index_data runs end to end against stubbed IO."""
    monkeypatch.setattr(RecordingToolkit, "_clean_index", lambda self, name: None)
    monkeypatch.setattr(RecordingToolkit, "_log_tool_event", lambda self, *a, **kw: None)
    monkeypatch.setattr(RecordingToolkit, "_emit_index_event", lambda self, name, error=None, state=None: None)
    monkeypatch.setattr(RecordingToolkit, "_reduce_duplicates", lambda self, docs, name: docs)

    def fake_save(self, base_documents, base_total, chunking_tool, chunking_config, result, index_name=None):
        documents = list(base_documents)
        result["count"] = len(documents)
        result["docs_count"] = len(documents)

    monkeypatch.setattr(RecordingToolkit, "_save_index_generator", fake_save)
    return toolkit


def run_index_data(toolkit, monkeypatch, documents, tracker=None):
    def fake_loader(self, **kwargs):
        if tracker:
            tracker(self)
        return iter(documents)

    monkeypatch.setattr(RecordingToolkit, "_base_loader", fake_loader)
    return toolkit.index_data(index_name="x")


class TestIndexDataResult:
    def test_result_carries_status_message_and_report(self, indexing_toolkit, monkeypatch):
        result = run_index_data(indexing_toolkit, monkeypatch, [object(), object()])

        assert result["status"] == IndexingStatus.OK.value
        assert result["report"]["totals"]["indexed"] == 2
        assert result["message"] == render_report_text(result["report"])

    def test_report_status_matches_the_result_status(self, indexing_toolkit, monkeypatch):
        result = run_index_data(indexing_toolkit, monkeypatch, [object()])

        assert result["report"]["status"] == result["status"]

    def test_message_uses_the_toolkits_own_item_nouns(self, indexing_toolkit, monkeypatch):
        monkeypatch.setattr(RecordingToolkit, "index_item_labels", ("page", "pages"))

        result = run_index_data(indexing_toolkit, monkeypatch, [object(), object()])

        assert result["message"].splitlines()[0] == "Successfully indexed 2 pages."

    def test_stats_do_not_accumulate_across_runs_on_one_instance(self, indexing_toolkit, monkeypatch):
        def track_a_skip(wrapper):
            wrapper.get_indexing_stats().documents_skipped_filtered.add("noise.tmp")

        first = run_index_data(indexing_toolkit, monkeypatch, [object()], tracker=track_a_skip)
        second = run_index_data(indexing_toolkit, monkeypatch, [object()], tracker=track_a_skip)

        assert first["report"]["totals"] == second["report"]["totals"]
        assert second["report"]["totals"]["skipped"] == 1

    def test_persisted_counts_agree_with_the_reported_breakdown(self, indexing_toolkit, monkeypatch):
        def track_a_skip(wrapper):
            wrapper.get_indexing_stats().documents_skipped_filtered.add("noise.tmp")

        result = run_index_data(indexing_toolkit, monkeypatch, [object(), object()], tracker=track_a_skip)

        stored = indexing_toolkit.written[-1]
        assert stored["total"] == result["report"]["totals"]["total"] == 3
        assert stored["indexed"] == 2

    def test_a_document_dropped_after_loading_is_not_also_counted_as_indexed(
        self, indexing_toolkit, monkeypatch
    ):
        """Unsupported formats surface while chunking — after the loader already
        yielded the document."""

        def drop_one_while_chunking(self, base_documents, base_total, chunking_tool,
                                    chunking_config, result, index_name=None):
            documents = list(base_documents)
            self.get_indexing_stats().files_unsupported_extension.add("diagram.ai")
            result["count"] = len(documents) - 1
            result["docs_count"] = len(documents) - 1

        monkeypatch.setattr(RecordingToolkit, "_save_index_generator", drop_one_while_chunking)

        result = run_index_data(indexing_toolkit, monkeypatch, [object() for _ in range(7)])

        totals = result["report"]["totals"]
        assert totals["indexed"] == 6
        assert totals["not_indexed"] == 1
        assert totals["total"] == 7

    def test_a_chunk_yielding_loader_reports_items_not_chunks(self, indexing_toolkit, monkeypatch):
        monkeypatch.setattr(RecordingToolkit, "loader_yields_chunks", True)

        def count_chunks(self, base_documents, base_total, chunking_tool,
                         chunking_config, result, index_name=None):
            documents = list(base_documents)
            self.get_indexing_stats().items_processed = 3
            result["count"] = len(documents)
            result["docs_count"] = len(documents)

        monkeypatch.setattr(RecordingToolkit, "_save_index_generator", count_chunks)

        result = run_index_data(indexing_toolkit, monkeypatch, [object() for _ in range(12)])

        assert result["report"]["totals"]["indexed"] == 3

    def test_failure_persists_an_error_report_and_reraises(self, indexing_toolkit, monkeypatch):
        def exploding_loader(self, **kwargs):
            raise RuntimeError("auth failed")

        monkeypatch.setattr(RecordingToolkit, "_base_loader", exploding_loader)

        with pytest.raises(RuntimeError):
            indexing_toolkit.index_data(index_name="x")

        stored = json.loads(indexing_toolkit.written[-1]["report"])
        assert stored["status"] == IndexingStatus.ERROR.value
        assert stored["errors"] == ["auth failed"]


class TestPreviousRunDetection:
    """Drives the reindex flag on the status event and the scheduled_reindex promotion.
    index_meta_init records the count, so these go through it."""

    def test_a_first_run_has_no_previous_one(self, toolkit):
        toolkit.index_meta_init("x", {"index_name": "x"})

        assert toolkit._has_previous_index_run() is False

    def test_a_platform_pre_created_row_is_still_a_first_run(self, toolkit):
        """The platform creates the row with an in-progress entry before the run starts.
        Counting that as a run announced every first index as a reindex."""
        seed_completed_run(
            toolkit,
            state=IndexerKeywords.INDEX_META_IN_PROGRESS.value,
            history=json.dumps([{"state": IndexerKeywords.INDEX_META_IN_PROGRESS.value}]),
        )

        toolkit.index_meta_init("x", {"index_name": "x"})

        assert toolkit._has_previous_index_run() is False

    @pytest.mark.parametrize(
        "previous_state",
        [
            IndexerKeywords.INDEX_META_COMPLETED.value,
            IndexerKeywords.INDEX_META_PARTLY_OK.value,
            IndexerKeywords.INDEX_META_SCHEDULED_REINDEX.value,
        ],
    )
    def test_any_run_that_built_an_index_counts(self, toolkit, previous_state):
        seed_completed_run(toolkit, history=json.dumps([{"state": previous_state}]))

        toolkit.index_meta_init("x", {"index_name": "x"})

        assert toolkit._has_previous_index_run() is True

    def test_unreadable_history_reads_as_no_previous_run(self, toolkit):
        seed_completed_run(toolkit, history="not json")

        toolkit.index_meta_init("x", {"index_name": "x"})

        assert toolkit._has_previous_index_run() is False

    def test_no_init_means_no_previous_run(self, toolkit):
        assert toolkit._has_previous_index_run() is False
