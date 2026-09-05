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

# Registrations whose args_schema is built in a shape this cannot follow (built
# dynamically per instance, mostly in vectorstore and aha). They are UNCHECKED, so
# the count is a ratchet: lower it when a shape becomes resolvable, never raise it.
# Import and Enum resolution already took it from 113 to 42.
UNRESOLVED_BUDGET = 42


def _model_fields(call, tree, path, seen):
    """Field names of a create_model call, following __base__ into its parents."""
    if not isinstance(call, ast.Call) or getattr(call.func, 'id', '') != 'create_model':
        return None
    fields = set()
    for keyword in call.keywords:
        if keyword.arg == '__base__':
            # zephyr_squad builds Issue -> ProjectIssue -> ProjectIssueStep this way;
            # a parent's fields are declared just as surely as a child's.
            base = _schema_fields(tree, keyword.value.id, path, seen) if isinstance(keyword.value, ast.Name) else None
            fields |= base or set()
        elif keyword.arg:
            fields.add(keyword.arg)
    return fields


def _schema_fields(tree, name, path, seen=None):
    """Fields of an args_schema symbol, resolved across the module it came from."""
    seen = seen or set()
    if (path, name) in seen:
        return None
    seen.add((path, name))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == name for t in node.targets):
            resolved = _model_fields(node.value, tree, path, seen)
            if resolved is not None:
                return resolved
    # Imported rather than defined here: follow the import to its module.
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if name not in {alias.asname or alias.name for alias in node.names}:
            continue
        origin = _resolve_module(path, node)
        if origin is None:
            continue
        try:
            other = ast.parse(origin.read_text())
        except (OSError, SyntaxError):
            continue
        return _schema_fields(other, name, origin, seen)
    return None


def _resolve_module(path, node):
    """Locate the file an `from ... import X` refers to, relative or absolute."""
    base = path.parent
    for _ in range(max(node.level - 1, 0)):
        base = base.parent
    parts = node.module.split('.')
    if node.level == 0:
        if parts[0] != SDK_ROOT.name:
            return None
        base = SDK_ROOT.parent
    candidate = base.joinpath(*parts)
    for option in (candidate.with_suffix('.py'), candidate / '__init__.py'):
        if option.exists():
            return option
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
            if isinstance(schema, ast.Name):
                found[ref] = ('name', schema.id)
            elif _enum_member(schema):
                found[ref] = ('enum', _enum_member(schema))
            else:
                found[ref] = ('inline', schema)
    return found


def _kwargs_reads(func):
    reads = set()
    for node in ast.walk(func):
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name) and node.value.id == 'kwargs'
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            reads.add(node.slice.value)
    return reads


def _enum_member(node):
    """`ArgsSchema.BranchName.value` -> ('ArgsSchema', 'BranchName')."""
    if not (isinstance(node, ast.Attribute) and node.attr == 'value'):
        return None
    inner = node.value
    if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name):
        return (inner.value.id, inner.attr)
    return None


def _enum_fields(tree, holder, member, path):
    """Fields of a create_model assigned to a member of an Enum class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == holder:
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                    getattr(target, 'id', '') == member for target in statement.targets
                ):
                    return _model_fields(statement.value, tree, path, set())
    return None


def _required_params(func):
    """Parameters with no default: a model call omitting one raises TypeError."""
    args = func.args
    positional = args.args[1:]
    required = {arg.arg for arg in positional[:len(positional) - len(args.defaults)]}
    required |= {
        arg.arg for arg, default in zip(args.kwonlyargs, args.kw_defaults) if default is None
    }
    return required - {'self', 'kwargs'}


def _findings():
    findings = []
    unresolved = []
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
            if schema[0] == 'inline':
                fields = _model_fields(schema[1], tree, path, set())
            elif schema[0] == 'enum':
                fields = _enum_fields(tree, schema[1][0], schema[1][1], path)
            else:
                fields = _schema_fields(tree, schema[1], path)
            if fields is None:
                # Report rather than skip: silently passing on an unresolved schema
                # left 21% of registrations unchecked, and a guard that cannot say
                # what it did not look at is worse than no guard.
                unresolved.append(f"{path.relative_to(SDK_ROOT.parent)}:{func.lineno} {ref}")
                continue
            missing = sorted(_kwargs_reads(func) - fields)
            required = sorted(_required_params(func) - fields)
            if missing:
                findings.append(
                    f"{path.relative_to(SDK_ROOT.parent)}:{func.lineno} {ref} reads kwargs{missing}, "
                    f"which its args_schema does not declare"
                )
            if required:
                findings.append(
                    f"{path.relative_to(SDK_ROOT.parent)}:{func.lineno} {ref} requires {required}, "
                    f"which its args_schema does not declare"
                )
    return findings, unresolved


def test_no_tool_needs_a_parameter_its_schema_never_declares():
    findings, _ = _findings()

    assert findings == [], (
        "These tools fail on every model invocation, because the model cannot "
        "supply what the schema does not advertise:\n  " + "\n  ".join(findings)
    )


def test_every_registered_schema_can_be_resolved():
    """A schema this cannot read is a tool it cannot check — say so out loud."""
    _, unresolved = _findings()

    assert len(unresolved) <= UNRESOLVED_BUDGET, (
        f"{len(unresolved)} registrations have an args_schema this test cannot resolve, "
        f"up from a budget of {UNRESOLVED_BUDGET}. Each is unchecked:\n  "
        + "\n  ".join(unresolved[:20])
    )
