"""A tool may not read a parameter its args_schema never declares (#6532).

Found three times by hand during this work, each failing on EVERY model call
because the model cannot supply what the schema does not advertise:

    zephyr_scale.get_links       kwargs['return_only_links'] -> KeyError
    zephyr_scale.get_test_steps  kwargs['return_list']       -> KeyError
    artifact.list_files          return_as_string=True       -> repr by default

Only the crash class is asserted here. A declared-but-unexposed *parameter* is
often legitimate (an internal callback, a caller-only flag), so those are left to
review rather than encoded as a rule that would cry wolf.
"""

import ast
import pathlib

SDK_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'elitea_sdk'


def _schema_fields(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == name for t in node.targets):
            if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', '') == 'create_model':
                return {keyword.arg for keyword in node.value.keywords if keyword.arg}
    return None


def _inline_fields(call):
    if isinstance(call, ast.Call) and getattr(call.func, 'id', '') == 'create_model':
        return {keyword.arg for keyword in call.keywords if keyword.arg}
    return None


def _registrations(tree):
    """method name -> ('name', symbol) | ('inline', fields) for each registered tool."""
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        ref = schema = None
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant):
                continue
            if key.value == 'ref' and isinstance(value, ast.Attribute):
                ref = value.attr
            elif key.value == 'args_schema':
                schema = value
        if ref and schema is not None:
            found[ref] = ('name', schema.id) if isinstance(schema, ast.Name) else ('inline', _inline_fields(schema))
    return found


def _kwargs_reads(func):
    reads = set()
    for node in ast.walk(func):
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name) and node.value.id == 'kwargs'
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            reads.add(node.slice.value)
    return reads


def _findings():
    findings = []
    for path in sorted(SDK_ROOT.rglob('*.py')):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        registered = _registrations(tree)
        if not registered:
            continue
        methods = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for ref, schema in registered.items():
            func = methods.get(ref)
            if func is None:
                continue
            fields = schema[1] if schema[0] == 'inline' else _schema_fields(tree, schema[1])
            if fields is None:
                continue
            missing = sorted(_kwargs_reads(func) - fields)
            if missing:
                findings.append(
                    f"{path.relative_to(SDK_ROOT.parent)}:{func.lineno} {ref} reads kwargs{missing}, "
                    f"which its args_schema does not declare"
                )
    return findings


def test_no_tool_reads_a_kwarg_its_schema_never_declares():
    findings = _findings()

    assert findings == [], (
        "These tools raise KeyError on every model invocation:\n  " + "\n  ".join(findings)
    )
