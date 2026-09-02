"""Tests for the current_date default of the Carrier get_ui_reports tool.

The default was written as ``Field(default=datetime.datetime.now()...)``.
Python runs that call one time, when it imports the module. This made two
faults. A worker process gave the date of its start, not the date of the
request. A consumer that keeps a copy of ``model_json_schema()`` got a document
that changed each day.

The default is now a ``default_factory``. These tests hold both properties:
the JSON Schema has no date in it, and each new model gets the current date.
"""

import datetime
from types import SimpleNamespace

import pytest

from elitea_sdk.tools.carrier import ui_reports_tool
from elitea_sdk.tools.carrier.ui_reports_tool import GetUIReportsTool

ARGS_MODEL = GetUIReportsTool.model_fields["args_schema"].default

DAY_ONE = datetime.datetime(2020, 1, 1, 12, 0, 0)
DAY_TWO = datetime.datetime(2020, 1, 2, 12, 0, 0)


class _Clock(datetime.datetime):
    """A datetime class whose now() gives a controlled value.

    The class keeps all of the other datetime behaviour. The module uses
    ``fromisoformat`` and ``strptime`` in other places.
    """

    current = DAY_ONE

    @classmethod
    def now(cls, tz=None):
        return cls.current


@pytest.fixture
def clock(monkeypatch):
    """Replace the datetime that the module reads, for this test only."""
    _Clock.current = DAY_ONE
    monkeypatch.setattr(ui_reports_tool, "datetime", SimpleNamespace(datetime=_Clock))
    return _Clock


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_schema_holds_no_date(mode):
    """The published schema must not contain a date.

    A date in the schema makes the document change each day. A consumer that
    keeps a copy of the document then finds a difference at the next midnight.
    """
    schema = ARGS_MODEL.model_json_schema(mode=mode)
    current_date = schema["properties"]["current_date"]
    assert "default" not in current_date


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_schema_is_the_same_on_two_dates(clock, mode):
    """The schema must be equal on two different dates."""
    clock.current = DAY_ONE
    first = ARGS_MODEL.model_json_schema(mode=mode)
    clock.current = DAY_TWO
    second = ARGS_MODEL.model_json_schema(mode=mode)
    assert first == second


def test_current_date_stays_optional():
    """The caller must not have to supply current_date.

    The field description tells the model that the value is auto-filled. A
    required field would contradict that text.
    """
    schema = ARGS_MODEL.model_json_schema()
    assert "current_date" not in schema.get("required", [])
    assert ARGS_MODEL(report_id="report-1").current_date is not None


def test_each_model_gets_the_date_of_its_own_creation(clock):
    """A model made later must get the later date.

    This is the fault that a long-running worker shows. The module import and
    the request happen on different dates.
    """
    clock.current = DAY_ONE
    early = ARGS_MODEL(report_id="report-1")
    clock.current = DAY_TWO
    late = ARGS_MODEL(report_id="report-2")

    assert early.current_date == "2020-01-01"
    assert late.current_date == "2020-01-02"


def test_a_supplied_value_is_kept(clock):
    """An explicit value from the caller must win over the factory."""
    clock.current = DAY_ONE
    model = ARGS_MODEL(report_id="report-1", current_date="2019-12-31")
    assert model.current_date == "2019-12-31"
