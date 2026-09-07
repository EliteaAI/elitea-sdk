"""No toolkit tool may hand back a Python repr of a collection (#6532).

Tools return native data; the runtime and the toolkit test panel serialize it.
A prose string with `str(list_of_dicts)` baked in cannot be recovered by either,
which is how GitLab's get_issues came to render as single-quoted Python.

GREEN HERE IS NOT PROOF OF ABSENCE. This is a net for the copy-pasted idiom, and
it recognises a collection only when the name is bound from a literal, a
comprehension, or a call whose name carries one of COLLECTION_CALL_HINTS. It does
NOT see:

* a value bound from ``.get()`` or a subscript -- `jira.get_specific_field_info`
  leaked `['bug', 'ui']` for months behind exactly that shape. Widening to those
  bindings was measured at 40 flagged sites, nearly all scalars (`page_id`,
  `image_name`), so the rule was rejected as noise rather than adopted;
* accumulation into a string (`result += f"...{records}"`), which is not a return;
* anything outside `elitea_sdk/tools` beyond the two runtime rules below;
* percent-formatting -- `return "Extracted: %s" % (names,)` passes all three tests;
* a repr baked into a variable that is later returned, or into indexed content.
"""

import ast
import pathlib

TOOLS_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'elitea_sdk' / 'tools'

# Names that hold an exception, not data: str(e) in a message is fine.
EXCEPTION_NAMES = {
    'e', 'err', 'ex', 'exc', 'error', 've', 'ai_e', 'img_e', 'update_e',
    'conv_error', 'img_error', 'cancel_error', 'api_error',
}

# Helpers whose name looks collection-producing but that return ready text.
TEXT_PRODUCING_HELPERS = {'_parse_validation_error'}

COLLECTION_LITERALS = (ast.List, ast.Dict, ast.ListComp, ast.DictComp, ast.SetComp)
COLLECTION_CALL_HINTS = ('parse', 'get_all', 'to_dict', 'dicts', 'list_')


def _collection_bindings(func):
    """Local names known to hold a list/dict inside this function."""
    names = set()
    for node in ast.walk(func):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        holds_collection = isinstance(value, COLLECTION_LITERALS)
        if isinstance(value, ast.Call):
            callee = getattr(value.func, 'attr', getattr(value.func, 'id', ''))
            holds_collection = (
                callee not in TEXT_PRODUCING_HELPERS
                and any(hint in callee for hint in COLLECTION_CALL_HINTS)
            )
        if not holds_collection:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _is_collection_call(node):
    """`self.list_indexes()` inside an f-string reprs a list just as a bare name does."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != 'self':
        return False
    return any(hint in node.func.attr for hint in COLLECTION_CALL_HINTS)


def _repr_embeds(returned, collection_names, calls_only=False):
    """Yield the repr-producing pieces of a returned or raised expression.

    ``calls_only`` narrows the scan to `self.<collection_call>()`, for the anchors
    where the other rules are all noise: a raise reprs an exception, and a
    conditional is nearly always `str(scalar) if x else ...`.
    """
    for node in ast.walk(returned):
        if calls_only:
            if isinstance(node, ast.FormattedValue) and _is_collection_call(node.value):
                yield ast.unparse(node.value)
        elif isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'str' and node.args:
            argument = node.args[0]
            if isinstance(argument, ast.Name) and argument.id not in EXCEPTION_NAMES:
                yield ast.unparse(node)
        elif isinstance(node, ast.FormattedValue) and _is_collection_call(node.value):
            yield ast.unparse(node.value)
        elif isinstance(node, ast.FormattedValue):
            expression = node.value
            if isinstance(expression, ast.Subscript):
                # Indexing yields one element; only a slice is still a collection.
                if not isinstance(expression.slice, ast.Slice):
                    continue
                base = expression.value
            else:
                base = expression
            if isinstance(base, ast.Name) and base.id in collection_names:
                yield ast.unparse(expression)


def _findings():
    findings = []
    for path in sorted(TOOLS_ROOT.rglob('*.py')):
        tree = ast.parse(path.read_text())
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            collection_names = _collection_bindings(func)
            for node in ast.walk(func):
                if isinstance(node, ast.Return) and node.value is not None:
                    returned, raised = node.value, False
                elif isinstance(node, ast.Raise) and node.exc is not None:
                    returned, raised = node.exc, True
                else:
                    continue
                if not isinstance(returned, (ast.JoinedStr, ast.BinOp, ast.IfExp, ast.Call)):
                    continue
                # Raises and conditionals are scanned for interpolated collections
                # only: their str() calls are overwhelmingly str(exception) or
                # `str(scalar) if x else ...`, which would bury the real findings.
                calls_only = raised or not isinstance(returned, (ast.JoinedStr, ast.BinOp))
                node = type('_Anchor', (), {'lineno': node.lineno, 'value': returned})()
                embeds = _repr_embeds(node.value, collection_names, calls_only)
                for embed in sorted(set(embeds)):
                    if isinstance(node.value, ast.BinOp) and embed.startswith('str('):
                        inner = embed[4:-1]
                        if inner not in collection_names:
                            continue
                    findings.append(
                        f"{path.relative_to(TOOLS_ROOT.parents[1])}:{node.lineno} embeds {embed}"
                    )
    return findings


def test_no_tool_returns_a_repr_of_a_collection():
    findings = _findings()

    assert findings == [], (
        "These returns interpolate a Python repr of a collection. Return the native "
        "list/dict instead, or serialize with "
        "elitea_sdk.tools.utils.serialization.serialize_tool_result:\n  "
        + "\n  ".join(findings)
    )


RUNTIME_ROOT = TOOLS_ROOT.parent / 'runtime' / 'tools'

# str() on a value that came straight out of a tool invocation is the #6532 defect
# itself: the wrapper returns a string, so no later boundary can recover the data.
INVOCATION_CALLS = ('invoke', 'ainvoke', 'run', 'arun')


def _invocation_results(func):
    names = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, (ast.Call, ast.Await)):
            continue
        call = node.value.value if isinstance(node.value, ast.Await) else node.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr not in INVOCATION_CALLS:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _runtime_findings():
    findings = []
    for path in sorted(RUNTIME_ROOT.rglob('*.py')):
        tree = ast.parse(path.read_text())
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            invoked = _invocation_results(func)
            if not invoked:
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                for inner in ast.walk(node.value):
                    if not isinstance(inner, ast.Call) or getattr(inner.func, 'id', '') != 'str':
                        continue
                    if inner.args and isinstance(inner.args[0], ast.Name) and inner.args[0].id in invoked:
                        findings.append(
                            f"{path.relative_to(RUNTIME_ROOT.parents[2])}:{node.lineno} returns {ast.unparse(inner)}"
                        )
    return findings


def test_no_runtime_boundary_stringifies_a_tool_result():
    findings = _runtime_findings()

    assert findings == [], (
        "These runtime boundaries hand a tool result to the model as a Python repr. "
        "Serialize with elitea_sdk.tools.utils.serialization.serialize_tool_result:\n  "
        + "\n  ".join(findings)
    )


def _tool_message_findings():
    """`ToolMessage(content=str(x))` is a boundary reprring a tool result.

    The invocation-tracking scan above cannot see the parallel sub-agent path,
    where the result arrives from asyncio.gather rather than from `.invoke()`.
    """
    findings = []
    for path in sorted(RUNTIME_ROOT.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or getattr(node.func, 'id', '') != 'ToolMessage':
                continue
            for keyword in node.keywords:
                if keyword.arg != 'content':
                    continue
                value = keyword.value
                if isinstance(value, ast.Call) and getattr(value.func, 'id', '') == 'str':
                    findings.append(
                        f"{path.relative_to(RUNTIME_ROOT.parents[2])}:{node.lineno} "
                        f"builds ToolMessage(content={ast.unparse(value)})"
                    )
    return findings


def test_no_tool_message_is_built_from_a_repr():
    findings = _tool_message_findings()

    assert findings == [], (
        "These boundaries hand the model a Python repr. Use "
        "elitea_sdk.tools.utils.serialization.serialize_tool_result:\n  "
        + "\n  ".join(findings)
    )
