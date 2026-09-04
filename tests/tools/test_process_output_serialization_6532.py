"""The `process_output` decorator and the tools it wraps must emit valid JSON (#6532).

These toolkits had no coverage at all: reverting their hunks left the suite green.
They matter more than the count suggests, because pandas-backed results routinely
carry NaN for missing values, and a bare NaN token is not valid JSON — the UI parses
before it will render a payload as JSON, so one NaN degrades the whole answer.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from elitea_sdk.tools.aws.delta_lake.api_wrapper import DeltaLakeApiWrapper, process_output
from elitea_sdk.tools.google.bigquery.api_wrapper import process_output as bigquery_process_output
from elitea_sdk.tools.google_places.api_wrapper import GooglePlacesAPIWrapper


class TestProcessOutputDecorator:
    def _decorated(self, result):
        @process_output
        def tool(self):
            return result

        return tool(None)

    def test_records_become_json(self):
        output = self._decorated([{"id": 1, "name": "Bob's row"}])

        assert json.loads(output) == [{"id": 1, "name": "Bob's row"}]
        assert "'id'" not in output

    def test_missing_values_do_not_break_json(self):
        # A pandas frame with a gap produces NaN; json.dumps writes it bare.
        output = self._decorated([{"id": 1, "score": float("nan")}])

        assert json.loads(output) == [{"id": 1, "score": None}]

    def test_non_ascii_is_not_escaped(self):
        output = self._decorated([{"city": "Кириллица"}])

        assert "Кириллица" in output
        assert "\\u" not in output

    def test_plain_text_passes_through(self):
        assert self._decorated("done") == "done"


class TestDeltaLakeQueryTable:
    def test_rows_with_gaps_serialize_as_json(self, monkeypatch):
        pandas = pytest.importorskip("pandas")
        wrapper = DeltaLakeApiWrapper.model_construct()
        frame = pandas.DataFrame([{"id": 1, "score": 0.5}, {"id": 2, "score": None}])
        monkeypatch.setattr(
            DeltaLakeApiWrapper, 'delta_table',
            property(lambda self: SimpleNamespace(to_pandas=lambda: frame)),
            raising=False,
        )

        output = wrapper.query_table()

        # score is NaN in the frame; a bare NaN token would make this unparseable
        assert json.loads(output) == [{"id": 1, "score": 0.5}, {"id": 2, "score": None}]


class TestGooglePlaces:
    def test_find_near_returns_native_results(self):
        wrapper = GooglePlacesAPIWrapper.model_construct()
        places = [{"name": "Cafe *Central*", "place_id": "abc"}]
        wrapper.__dict__['_client'] = MagicMock()
        wrapper._client.geocode.return_value = [{"geometry": {"location": {"lat": 1, "lng": 2}}}]
        wrapper._client.places_nearby.return_value = {"results": places}

        result = wrapper.find_near(current_location_query="here", target="cafe")

        assert result == places
        assert json.loads(json.dumps(result)) == places


class TestBigQueryProcessOutput:
    """The delta_lake sibling was covered; this one had no test at all."""

    def _decorated(self, result):
        @bigquery_process_output
        def tool(self):
            return result

        return tool(None)

    def test_rows_become_json(self):
        rows = [{"id": 1, "name": "Bob's row"}]

        assert json.loads(self._decorated(rows)) == rows

    def test_missing_values_do_not_break_json(self):
        assert json.loads(self._decorated([{"id": 1, "score": float("nan")}])) == [{"id": 1, "score": None}]

    def test_non_ascii_is_not_escaped(self):
        output = self._decorated([{"city": "Кириллица"}])

        assert "Кириллица" in output
        assert "\\u" not in output
