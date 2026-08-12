"""Effect-based tool grouping for toolkit UIs.

Every tool dict gets a `group` stamped at the moment its owning class produces
it, via the `with_tool_groups` decorator on that class's get_available_tools.
A class declares which of its tools belong to which effect group:

    class GitHubClient(BaseModel):
        class ToolGroups:
            read = ["get_issue", "read_file"]
            write = ["create_issue"]
            delete = ["delete_branch"]
            execute = ["trigger_workflow"]

Groups follow the safety rollup: `read` is read-only; `write`, `delete` and
`execute` all change data. A tool listed in no group is deliberately left
unstamped so the UI can surface it instead of trusting a guess.
"""

import functools
from collections.abc import Collection
from types import MappingProxyType

READ = "read"
WRITE = "write"
DELETE = "delete"
EXECUTE = "execute"

GROUPS = (READ, WRITE, DELETE, EXECUTE)


def _validate_declaration(klass, declaration) -> None:
    seen = {}
    for attr, value in vars(declaration).items():
        if attr.startswith("__"):
            continue
        if attr not in GROUPS:
            raise ValueError(
                f"{klass.__qualname__}.ToolGroups.{attr} is not a valid group; expected one of {GROUPS}"
            )
        # Collection (not Iterable) so single-pass generators — which validation
        # would exhaust before resolution reads them — are rejected outright
        if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
            raise ValueError(
                f"{klass.__qualname__}.ToolGroups.{attr} must be a collection of tool names, "
                f"got {type(value).__name__}"
            )
        for tool_name in value:
            if not isinstance(tool_name, str):
                raise ValueError(
                    f"{klass.__qualname__}.ToolGroups.{attr} entries must be tool-name strings, "
                    f"got {type(tool_name).__name__}"
                )
            if tool_name in seen:
                raise ValueError(
                    f"{klass.__qualname__}.ToolGroups lists '{tool_name}' in both "
                    f"'{seen[tool_name]}' and '{attr}' — a tool belongs to one group per "
                    f"declaration; to change an inherited group, re-declare it in the subclass"
                )
            seen[tool_name] = attr


@functools.lru_cache(maxsize=None)
def resolve_declared_groups(cls) -> MappingProxyType:
    """Collect ToolGroups declarations along the class's MRO, subclass wins.

    Cached per class — get_available_tools runs on every tool invocation, so
    validation and the MRO walk happen once. The returned mapping is read-only
    because it is shared between callers.
    """
    declared = {}
    for klass in reversed(cls.__mro__):
        declaration = klass.__dict__.get("ToolGroups")
        if declaration is None:
            continue
        _validate_declaration(klass, declaration)
        for group in GROUPS:
            for tool_name in getattr(declaration, group, ()):
                declared[tool_name] = group
    return MappingProxyType(declared)


def with_tool_groups(method):
    """Stamp a `group` onto each tool dict this method produces.

    Applied per producing class, so composed tool lists (parent toolkits,
    injected tools, sibling client wrappers) arrive pre-stamped and no
    aggregation point needs to know how the list was assembled. An already
    stamped dict is never overwritten.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        tools = method(self, *args, **kwargs)
        declared = resolve_declared_groups(type(self))
        for tool in tools:
            if tool.get("group"):
                continue
            group = declared.get(tool["name"])
            if group:
                tool["group"] = group
        return tools
    return wrapper
