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
# Import, Enum and per-call wrapper resolution already took it from 113 to 42.
UNRESOLVED_BUDGET = 42


def _model_fields(call, tree, path, seen, bindings=None):
    """Field names of a create_model call, following __base__ into its parents."""
    if not isinstance(call, ast.Call) or getattr(call.func, 'id', '') != 'create_model':
        return None
    fields = set()
    for keyword in call.keywords:
        if keyword.arg == '__base__':
            # zephyr_squad builds Issue -> ProjectIssue -> ProjectIssueStep this way;
            # a parent's fields are declared just as surely as a child's.
            fields |= _fields_of(keyword.value, tree, path, seen, bindings) or set()
        elif keyword.arg is None:
            # `**self._repo_arg()`: the helper's dict keys are fields too.
            fields |= _spread_fields(keyword.value, tree) or set()
        else:
            fields.add(keyword.arg)
    return fields


def _module_methods(tree):
    """name -> function node, for every function defined in the module."""
    return {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _self_call_target(node, tree):
    """`self._with_repo(x)` -> the function node it calls, or None."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'):
        return None
    return _module_methods(tree).get(node.func.attr)


def _spread_fields(node, tree):
    """Keys a `**self._helper()` spread contributes, from its dict literals."""
    func = _self_call_target(node, tree)
    if func is None:
        return None
    keys = set()
    for inner in ast.walk(func):
        if isinstance(inner, ast.Dict):
            keys |= {
                key.value for key in inner.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    return keys


def _helper_fields(node, tree, path, seen):
    """Fields of `self._with_repo(Schema)`: the wrapped schema plus what it adds.

    ado repos registers every tool through such a wrapper, so without this the
    whole toolkit reads as unresolvable.
    """
    func = _self_call_target(node, tree)
    if func is None or not node.args:
        return None
    parameters = [arg.arg for arg in func.args.args if arg.arg != 'self']
    if not parameters:
        return None
    bindings = {parameters[0]: node.args[0]}
    for inner in ast.walk(func):
        if isinstance(inner, ast.Return) and inner.value is not None:
            fields = _fields_of(inner.value, tree, path, seen, bindings)
            if fields is not None:
                return fields
    return None


def _fields_of(node, tree, path, seen, bindings=None):
    """Fields of any args_schema expression: symbol, enum member, model or wrapper."""
    bindings = bindings or {}
    if isinstance(node, ast.Name):
        if node.id in bindings:
            return _fields_of(bindings[node.id], tree, path, seen)
        return _schema_fields(tree, node.id, path, seen)
    member = _enum_member(node)
    if member:
        return _enum_fields(tree, member[0], member[1], path)
    if isinstance(node, ast.Call):
        fields = _model_fields(node, tree, path, seen, bindings)
        return fields if fields is not None else _helper_fields(node, tree, path, seen)
    return None


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
                fields = _fields_of(schema[1], tree, path, set())
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
