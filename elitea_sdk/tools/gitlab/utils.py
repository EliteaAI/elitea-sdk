
import re
from typing import Any

def get_diff_w_position(change):
    diff = change["diff"]
    diff_with_ln = {}
    # Regular expression to extract old and new line numbers
    pattern = r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@"
    # GitLab API requires new path and line for added lines, old path and files for removed lines.
    # For unchaged lines it requires both. 
    for index, line in enumerate(diff.split("\n")):
        position = {}
        match = re.match(pattern, line)
        if match:
            old_line = int(match.group(1))
            new_line = int(match.group(3))
        elif line.startswith("+"):
            position["new_line"] = new_line
            position["new_path"] = change["new_path"]
            new_line += 1
        elif line.startswith("-"):
            position["old_line"] = old_line
            position["old_path"] = change["old_path"]
            old_line += 1
        elif line.startswith(" "):
            position["old_line"] = old_line
            position["old_path"] = change["old_path"]
            position["new_line"] = new_line
            position["new_path"] = change["new_path"]
            new_line += 1
            old_line += 1
        elif line.startswith("\\"):
            # Assign previos position to \\ metadata
            position = diff_with_ln[index - 1][0]
        else:
            # Stop at final empty line
            break

        diff_with_ln[index] = [position, line]

        # Assign next position to @@ metadata
        if index > 0 and diff_with_ln[index - 1][1].startswith("@"):
            diff_with_ln[index - 1][0] = position

    return diff_with_ln



def get_position(line_number, file_path, mr):
    changes = mr.changes()["changes"]
    # Get first change 
    change = next((item for item in changes if item.get("new_path") == file_path), None)
    if change == None:
        change = next((item for item in changes if item.get("old_path") == file_path), None)
    if change == None:
        raise Exception(f"Change for file {file_path} wasn't found in PR")

    position = get_diff_w_position(change=change)[line_number][0]

    position.update({
        "base_sha": mr.diff_refs["base_sha"],
        "head_sha": mr.diff_refs["head_sha"],
        "start_sha": mr.diff_refs["start_sha"],
        'position_type': 'text'
    })

    return position


_UPPER_BOUND = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:(?P<separator>[Tt ])(?P<hour>\d{2})"
    r"(?::(?P<minute>\d{2})(?::(?P<second>\d{2})(?:\.(?P<fraction>\d+))?)?)?"
    r"(?P<offset>[Zz]|[+-]\d{2}(?::?\d{2})?)?)?$"
)


def expand_inclusive_upper_bound(value: Any) -> Any:
    """Widen an upper date bound to the last instant of the precision it states.

    GitLab renders `created_at`/`updated_at` truncated to milliseconds but filters
    at microsecond precision, so a bound copied out of a response names an instant
    below the record it came from and `*_before` drops that very record. Coarser
    bounds are floored to the start of their unit, so `created_before=2026-09-03`
    matches nothing from that day at all.

    Unspecified lower-order fields are filled with their maximum rather than
    computed, because the last instant of a unit never crosses into the next one.
    That keeps this total -- no date arithmetic, no timezone handling, and no
    `datetime.fromisoformat`, which on Python 3.10 rejects both the `Z` suffix and
    any fraction that is not exactly 3 or 6 digits, i.e. the timestamps GitLab
    itself emits.

    Values that are not timestamp strings are returned untouched so GitLab keeps
    reporting its own validation errors rather than this raising a new one.

    A value that already states microsecond precision is returned untouched, and
    that is the deliberate escape hatch: widening applies at whatever precision
    was stated, so a caller chunking a range on a shared boundary would otherwise
    see the boundary second counted in both windows. Stating microseconds opts out
    and restores an exact partition.
    """
    if not isinstance(value, str):
        return value

    match = _UPPER_BOUND.match(value.strip())
    if match is None:
        return value

    fraction = match["fraction"] or ""
    if len(fraction) >= 6:
        return value

    return (
        f'{match["date"]}{match["separator"] or "T"}'
        f'{match["hour"] or "23"}:{match["minute"] or "59"}:{match["second"] or "59"}'
        f'.{fraction.ljust(6, "9")}{match["offset"] or ""}'
    )
