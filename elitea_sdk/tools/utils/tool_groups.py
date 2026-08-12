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

READ = "read"
WRITE = "write"
DELETE = "delete"
EXECUTE = "execute"

GROUPS = (READ, WRITE, DELETE, EXECUTE)


def resolve_declared_groups(cls) -> dict:
    """Collect ToolGroups declarations along the class's MRO, subclass wins."""
    declared = {}
    for klass in reversed(cls.__mro__):
        declaration = klass.__dict__.get("ToolGroups")
        if declaration is None:
            continue
        for group in GROUPS:
            for tool_name in getattr(declaration, group, ()):
                declared[tool_name] = group
    return declared


def with_tool_groups(method):
    """Stamp a `group` onto each tool dict this method produces.

    Applied per producing class, so composed tool lists (parent toolkits,
    injected tools, sibling client wrappers) arrive pre-stamped and no
    aggregation point needs to know how the list was assembled. An already
    stamped dict is never overwritten.
    """
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
