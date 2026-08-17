"""Effect-based tool grouping for toolkit UIs.

A tool method is classified by decorating it where it is implemented:

    @tool_group('read')
    def get_commits_diff(self, ...) -> str:
        ...

The marker travels with the function object, so it covers every place the
method is referenced — inherited, borrowed by attribute assignment, or
injected by a composition decorator. `with_tool_groups` (on each producing
get_available_tools) then stamps the group onto the tool dicts via their
`ref`.

Groups follow the safety rollup: `read` is read-only; `write`, `delete` and
`execute` all change data. An undecorated method is deliberately left
unstamped so the UI can surface the tool instead of trusting a guess.
"""

import functools

READ = "read"
WRITE = "write"
DELETE = "delete"
EXECUTE = "execute"

GROUPS = (READ, WRITE, DELETE, EXECUTE)


def tool_group(group):
    if group not in GROUPS:
        raise ValueError(f"'{group}' is not a valid tool group; expected one of {GROUPS}")

    def decorate(method):
        method._tool_group = group
        return method

    return decorate


def with_tool_groups(method):
    """Stamp a `group` onto each tool dict this method produces.

    An already stamped dict is never overwritten, so composed tool lists keep
    the classification they arrived with.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        tools = method(self, *args, **kwargs)
        for tool in tools:
            if tool.get("group"):
                continue
            group = getattr(tool.get("ref"), "_tool_group", None)
            if group:
                tool["group"] = group
        return tools
    return wrapper
