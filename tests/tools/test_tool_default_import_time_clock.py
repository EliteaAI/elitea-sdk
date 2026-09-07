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

"""Tool argument defaults must read the clock at the call, not at the import.

Python evaluates a plain ``Field(default=datetime.datetime.now()...)`` one time,
when it imports the module. A worker process that lives longer than one day then
serves the date of its own start for every later call, and the generated tool
schema differs between two calendar days.

This module holds three checks:

1. A frozen clock moves across a day boundary between two calls, and the two
   calls give two different dates.
2. The generated JSON schema carries no date at all, so the schema snapshot is
   the same on two calendar days.
3. An abstract-syntax-tree audit over the whole ``elitea_sdk`` package proves
   that no other default reads a clock, a random source or the environment at
   import time.
"""

import ast
import datetime
import pathlib
import types

import pytest

from elitea_sdk.tools.carrier import ui_reports_tool


def _args_schema():
    """The generated argument model of the ``get_ui_reports`` tool."""
    return ui_reports_tool.GetUIReportsTool.model_fields["args_schema"].default


class _FrozenClock:
    """Stands in for ``datetime.datetime`` and gives a time the test controls."""

    def __init__(self, moment: datetime.datetime):
        self.moment = moment

    def now(self, tz=None):
        return self.moment


@pytest.fixture
def frozen_clock(monkeypatch):
    clock = _FrozenClock(datetime.datetime(2026, 8, 16, 23, 59, 0))
    monkeypatch.setattr(
        ui_reports_tool,
        "datetime",
        types.SimpleNamespace(datetime=clock),
    )
    return clock


def test_default_follows_the_clock_across_a_day_boundary(frozen_clock):
    """Two calls that straddle midnight give two different dates."""
    model = _args_schema()

    first = model(report_id="report-1")
    assert first.current_date == "2026-08-16"

    # Advance the clock past midnight. The process is the same one; only the
    # clock moved, exactly as it does for a worker that runs for days.
    frozen_clock.moment = datetime.datetime(2026, 8, 17, 0, 1, 0)

    second = model(report_id="report-2")
    assert second.current_date == "2026-08-17"

    assert first.current_date != second.current_date


def test_an_explicit_value_still_wins(frozen_clock):
    """The default must not overwrite a value the caller gives."""
    model = _args_schema()
    given = model(report_id="report-1", current_date="2020-01-01")
    assert given.current_date == "2020-01-01"


def test_generated_schema_holds_no_date(frozen_clock):
    """The schema is the same on two calendar dates, so the snapshot is stable."""
    model = _args_schema()

    first = model.model_json_schema()
    frozen_clock.moment = datetime.datetime(2026, 8, 17, 0, 1, 0)
    second = model.model_json_schema()

    assert first == second

    current_date = first["properties"]["current_date"]
    assert "default" not in current_date, (
        "the schema still carries an evaluated default: " f"{current_date!r}"
    )
    # The field stays optional for the caller.
    assert "current_date" not in first.get("required", [])


# --------------------------------------------------------------------------
# Package-wide audit
# --------------------------------------------------------------------------

# A call to any of these reads a source that changes between two moments of the
# same process. None of them may be evaluated when a module is imported.
_NON_DETERMINISTIC_CALLS = frozenset(
    {
        "now",
        "utcnow",
        "today",
        "time",
        "monotonic",
        "perf_counter",
        "uuid1",
        "uuid4",
        "random",
        "randint",
        "choice",
        "getenv",
        "getcwd",
    }
)

_PACKAGE_ROOT = pathlib.Path(ui_reports_tool.__file__).parents[3] / "elitea_sdk"


def _called_name(call: ast.Call) -> str:
    target = call.func
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return ""


def _non_deterministic_defaults(tree: ast.AST, path: pathlib.Path):
    """Every ``default=`` argument whose value reads a changing source."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            for inner in ast.walk(keyword.value):
                if isinstance(inner, ast.Call) and _called_name(inner) in _NON_DETERMINISTIC_CALLS:
                    yield f"{path}:{inner.lineno}: default={ast.unparse(keyword.value)}"
                    break


def test_no_sdk_default_reads_a_changing_source_at_import_time():
    """No tool in the SDK holds a value captured when its module was imported."""
    offenders = []
    inspected = 0
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        inspected += 1
        offenders.extend(_non_deterministic_defaults(tree, path))

    # Guard the guard: an empty walk would make this test pass for the wrong
    # reason, so prove the audit actually read the package.
    assert inspected > 400, f"the audit only parsed {inspected} modules"
    assert offenders == [], "use default_factory instead:\n" + "\n".join(offenders)
