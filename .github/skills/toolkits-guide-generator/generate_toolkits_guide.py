#!/usr/bin/env python3
"""
Generate .github/skills/toolkits-guide-generator/output/toolkits_guide.md by introspecting all registered toolkits.

Two-pass strategy:
  Pass 1 (import): Uses Python imports + Pydantic model introspection — full detail.
  Pass 2 (ast):    Falls back to AST source parsing when optional deps are missing.

Usage:
    python scripts/generate_toolkits_guide.py
    python scripts/generate_toolkits_guide.py --compact   # names + descriptions only
    python scripts/generate_toolkits_guide.py --ast-only  # skip imports, AST only
"""

import ast
import re
import sys
import importlib
import inspect
import logging
import typing
from pathlib import Path
from typing import get_args, get_origin

# Make elitea_sdk importable regardless of cwd
_repo_root = str(Path(__file__).parent.parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.parent  # .github/skills/toolkits-guide-generator -> repo root
TOOLS_ROOT = REPO_ROOT / "elitea_sdk" / "tools"
TOOLS_INIT = REPO_ROOT / "elitea_sdk" / "tools" / "__init__.py"
OUTPUT_PATH = Path(__file__).parent / "output" / "toolkits_guide.md"

COMPACT = "--compact" in sys.argv
AST_ONLY = "--ast-only" in sys.argv

# Toolkits to exclude from the guide (router shims or entries with no own tools)
SKIP_KEYS = {
    "ado",
    # Cloud / infra
    "aws", "azure", "azure_search", "bigquery", "delta_lake", "elastic", "gcp", "k8s",
    # Utility / niche
    "keycloak", "localgit", "ocr", "yagmail",
    # Zephyr legacy (keep zephyr_enterprise, zephyr_essential, zephyr_scale, zephyr_squad)
    "zephyr",
}

# ---------------------------------------------------------------------------
# Helpers — type annotation → readable string
# ---------------------------------------------------------------------------

def _type_str(annotation) -> str:
    if annotation is None or annotation is inspect.Parameter.empty:
        return "any"
    origin = get_origin(annotation)
    if origin is typing.Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _type_str(args[0])
        return " | ".join(_type_str(a) for a in args)
    if origin is typing.Literal:
        vals = get_args(annotation)
        if len(vals) <= 5:
            return "Literal[" + ", ".join(repr(v) for v in vals) + "]"
        return f"Literal ({len(vals)} options)"
    if origin in (list, typing.List):
        args = get_args(annotation)
        return f"List[{_type_str(args[0])}]" if args else "List"
    if origin in (dict, typing.Dict):
        return "dict"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    s = str(annotation)
    for prefix in ("typing.", "pydantic.", "<class '", "'>"):
        s = s.replace(prefix, "")
    return s.strip("'")


def _default_str(field_info) -> str:
    try:
        from pydantic_core import PydanticUndefinedType
        if isinstance(field_info.default, PydanticUndefinedType):
            return "—"
    except ImportError:
        pass
    d = field_info.default
    if d is None:
        return "None"
    if isinstance(d, str) and d == "":
        return '""'
    if isinstance(d, list) and len(d) == 0:
        return "[]"
    return str(d)


def _esc(text: str) -> str:
    """Escape Markdown table special characters."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Registry — read _safe_import_tool() calls from __init__.py without importing
# ---------------------------------------------------------------------------

def read_registry() -> list:
    """
    Parse elitea_sdk/tools/__init__.py with regex to extract all toolkit entries.
    Returns list of dicts: {key, module_path, get_tools_fn, toolkit_class_name}
    """
    source = TOOLS_INIT.read_text(encoding="utf-8")
    pattern = re.compile(
        r"_safe_import_tool\(\s*'([^']+)'\s*,\s*'([^']+)'\s*"
        r"(?:,\s*(?:'([^']*)'|None)\s*)?(?:,\s*(?:'([^']*)'|None)\s*)?\)"
    )
    entries = []
    for m in pattern.finditer(source):
        entries.append({
            "key": m.group(1),
            "module_path": m.group(2),
            "get_tools_fn": m.group(3) or "",
            "toolkit_class_name": m.group(4) or "",
        })
    return entries


# ---------------------------------------------------------------------------
# Import-based extraction (Pass 1)
# ---------------------------------------------------------------------------

def _get_toolkit_class(reg_entry: dict, available_tools: dict):
    """Return the toolkit class from AVAILABLE_TOOLS or by searching its module."""
    entry = available_tools.get(reg_entry["key"], {})
    if "toolkit_class" in entry:
        return entry["toolkit_class"]

    try:
        from langchain_core.tools import BaseToolkit
    except ImportError:
        return None

    for fn_key in ("get_toolkit", "get_tools"):
        fn = entry.get(fn_key)
        if not fn:
            continue
        mod = inspect.getmodule(fn)
        if not mod:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, BaseToolkit)
                    and obj is not BaseToolkit
                    and obj.__module__ == mod.__name__):
                return obj
    return None


def _find_wrapper_class(toolkit_class):
    try:
        from elitea_sdk.tools.elitea_base import BaseToolApiWrapper
    except ImportError:
        return None

    base_module = toolkit_class.__module__
    candidates = [
        f"{base_module}.api_wrapper",
        f"{base_module.rsplit('.', 1)[0]}.api_wrapper",
    ]
    for mod_path in candidates:
        try:
            mod = importlib.import_module(mod_path)
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if (issubclass(obj, BaseToolApiWrapper)
                        and obj is not BaseToolApiWrapper
                        and obj.__module__ == mod.__name__):
                    return obj
        except Exception:
            continue
    return None


def _import_get_available_tools(wrapper_class) -> list:
    try:
        instance = wrapper_class.model_construct()
        tools = instance.get_available_tools()
        return tools if isinstance(tools, list) else []
    except Exception as e:
        logger.debug("get_available_tools() failed on %s: %s", wrapper_class.__name__, e)
        return []


def _import_config_fields(schema_model) -> list:
    if not hasattr(schema_model, "model_fields"):
        return []
    rows = []
    for name, fi in schema_model.model_fields.items():
        if name == "selected_tools":
            continue
        extra = fi.json_schema_extra or {}
        rows.append({
            "name": name,
            "type": _type_str(fi.annotation),
            "required": fi.is_required(),
            "description": fi.description or "",
            "is_secret": bool(extra.get("secret", False)),
            "default": _default_str(fi),
        })
    return rows


def _import_tool_params(args_schema) -> list:
    if not args_schema or not hasattr(args_schema, "model_fields"):
        return []
    rows = []
    for name, fi in args_schema.model_fields.items():
        rows.append({
            "name": name,
            "type": _type_str(fi.annotation),
            "required": fi.is_required(),
            "default": _default_str(fi),
            "description": fi.description or "",
        })
    return rows


def _import_schema_tool_names(schema_model) -> list:
    fi = schema_model.model_fields.get("selected_tools")
    if fi is None:
        return []
    try:
        outer = get_args(fi.annotation)
        if outer:
            return list(get_args(outer[0]))
    except Exception:
        pass
    return []


def extract_via_import(reg_entry: dict, available_tools: dict):
    """Extract full toolkit info via Python imports. Returns None on failure."""
    key = reg_entry["key"]
    if key not in available_tools:
        return None

    toolkit_class = _get_toolkit_class(reg_entry, available_tools)
    if toolkit_class is None:
        return None

    label = key
    categories = []
    extra_categories = []
    config_fields = []
    tool_names_fallback = []
    schema = None

    try:
        schema = toolkit_class.toolkit_config_schema()
        mc = getattr(schema, "model_config", {})
        if isinstance(mc, dict):
            meta = (mc.get("json_schema_extra") or {}).get("metadata", {})
        else:
            meta = {}
        label = meta.get("label", key)
        categories = meta.get("categories", [])
        extra_categories = meta.get("extra_categories", [])
        config_fields = _import_config_fields(schema)
        tool_names_fallback = _import_schema_tool_names(schema)
    except Exception as e:
        logger.debug("toolkit_config_schema() failed for %s: %s", key, e)

    live_tools = []
    wrapper_class = _find_wrapper_class(toolkit_class)
    if wrapper_class:
        for t in _import_get_available_tools(wrapper_class):
            live_tools.append({
                "name": t.get("name", ""),
                "description": (t.get("description") or "").strip(),
                "params": _import_tool_params(t.get("args_schema")),
            })

    if not live_tools and tool_names_fallback:
        args_schemas_dict = {}
        if schema and "selected_tools" in schema.model_fields:
            fi = schema.model_fields["selected_tools"]
            args_schemas_dict = (fi.json_schema_extra or {}).get("args_schemas", {})
        for tname in tool_names_fallback:
            params = []
            if tname in args_schemas_dict:
                props = args_schemas_dict[tname].get("properties", {})
                req_set = set(args_schemas_dict[tname].get("required", []))
                for pname, pschema in props.items():
                    params.append({
                        "name": pname,
                        "type": pschema.get("type", "any"),
                        "required": pname in req_set,
                        "default": str(pschema.get("default", "—")),
                        "description": pschema.get("description", ""),
                    })
            live_tools.append({"name": tname, "description": "", "params": params})

    return {
        "key": key,
        "label": label,
        "categories": categories,
        "extra_categories": extra_categories,
        "config_fields": config_fields,
        "tools": live_tools,
        "source": "import",
    }


# ---------------------------------------------------------------------------
# AST-based extraction (Pass 2 fallback)
# ---------------------------------------------------------------------------

def _ast_const_str(node):
    """Return string value of ast.Constant if it's a str, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _ast_unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def ast_extract_metadata(init_path: Path) -> dict:
    """Parse toolkit __init__.py for label, categories, and config field names."""
    result = {"label": "", "categories": [], "extra_categories": [], "config_fields": []}
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except Exception:
        return result

    # Find 'metadata' dict anywhere in the file
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if _ast_const_str(k) != "metadata" or not isinstance(v, ast.Dict):
                continue
            for mk, mv in zip(v.keys, v.values):
                key_s = _ast_const_str(mk)
                if key_s == "label" and isinstance(mv, ast.Constant):
                    result["label"] = str(mv.value)
                elif key_s == "categories" and isinstance(mv, ast.List):
                    result["categories"] = [
                        e.value for e in mv.elts if isinstance(e, ast.Constant)
                    ]
                elif key_s == "extra_categories" and isinstance(mv, ast.List):
                    result["extra_categories"] = [
                        e.value for e in mv.elts if isinstance(e, ast.Constant)
                    ]
            break

    # Find create_model() kwargs inside toolkit_config_schema function
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "toolkit_config_schema"):
            continue
        for subnode in ast.walk(node):
            if not (isinstance(subnode, ast.Call)
                    and isinstance(subnode.func, ast.Name)
                    and subnode.func.id == "create_model"):
                continue
            for kw in subnode.keywords:
                if not kw.arg or kw.arg in ("__config__", "__validators__", "selected_tools"):
                    continue
                description = ""
                required = True
                type_s = "?"
                if isinstance(kw.value, ast.Tuple) and len(kw.value.elts) >= 2:
                    type_s = _ast_unparse(kw.value.elts[0])
                    if "Optional" in type_s:
                        required = False
                    field_node = kw.value.elts[1]
                    if isinstance(field_node, ast.Call):
                        for fkw in field_node.keywords:
                            if fkw.arg == "description" and isinstance(fkw.value, ast.Constant):
                                description = str(fkw.value.value)
                            elif fkw.arg == "default":
                                required = False
                result["config_fields"].append({
                    "name": kw.arg,
                    "type": type_s,
                    "required": required,
                    "description": description,
                    "is_secret": "SecretStr" in type_s or "secret" in kw.arg.lower(),
                    "default": "—",
                })
        break

    return result


def _collect_string_consts(tree) -> dict:
    """Collect module-level string assignments: VAR_NAME -> value."""
    consts = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            val = node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                consts[node.targets[0].id] = val.value
            elif isinstance(val, (ast.JoinedStr,)):
                # f-strings — skip, too complex
                pass
    return consts


def _parse_list_node(list_node, method_docs: dict, string_consts: dict) -> list:
    """Extract tools from an ast.List node (the return value of get_available_tools)."""
    tools = []
    for elt in list_node.elts:
        if not isinstance(elt, ast.Dict):
            continue
        tool_info = {"name": "", "description": "", "params": []}
        for k, v in zip(elt.keys, elt.values):
            key_s = _ast_const_str(k)
            if key_s == "name":
                tool_info["name"] = _ast_const_str(v) or ""
            elif key_s == "description":
                if isinstance(v, ast.Constant):
                    tool_info["description"] = str(v.value or "")
                elif isinstance(v, ast.Name) and v.id in string_consts:
                    # e.g. description = GET_ISSUE_PROMPT
                    tool_info["description"] = string_consts[v.id]
                elif (isinstance(v, ast.Attribute)
                      and v.attr == "__doc__"
                      and isinstance(v.value, ast.Attribute)):
                    # self.method_name.__doc__
                    tool_info["description"] = method_docs.get(v.value.attr, "")
        if tool_info["name"]:
            tools.append(tool_info)
    return tools


def ast_extract_tools_from_source(source: str) -> list:
    """Extract tools from a single source string."""
    try:
        tree = ast.parse(source)
    except Exception:
        return []

    method_docs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                method_docs[node.name] = doc

    string_consts = _collect_string_consts(tree)

    tools = []
    for node in ast.walk(tree):
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "get_available_tools"):
            continue
        # Collect local variable assignments of the form: var = [...]
        local_lists = {}
        for stmt in ast.walk(node):
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.List)):
                local_lists[stmt.targets[0].id] = stmt.value
        for child in ast.walk(node):
            if not isinstance(child, ast.Return):
                continue
            val = child.value
            list_node = None
            if isinstance(val, ast.List):
                list_node = val
            elif isinstance(val, ast.Name) and val.id in local_lists:
                # e.g. tools = [...]; return tools
                list_node = local_lists[val.id]
            elif isinstance(val, ast.BinOp) and isinstance(val.op, ast.Add):
                # collect all List operands in the chain
                operands = []
                def _collect(n):
                    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                        _collect(n.left)
                        _collect(n.right)
                    elif isinstance(n, ast.List):
                        operands.append(n)
                    elif isinstance(n, ast.Name) and n.id in local_lists:
                        # resolve a named local variable, e.g. basic_tools + indexing_tools
                        operands.append(local_lists[n.id])
                _collect(val)
                for ln in operands:
                    tools.extend(_parse_list_node(ln, method_docs, string_consts))
                list_node = None  # already handled
            if list_node is not None:
                tools.extend(_parse_list_node(list_node, method_docs, string_consts))
        break  # first get_available_tools only
    return tools


def ast_extract_tools_from_all_assignment(source: str) -> list:
    """Extract tool names from module-level __all__ = [{"name": "...", ...}, ...] pattern."""
    try:
        tree = ast.parse(source)
    except Exception:
        return []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
                and isinstance(node.value, ast.List)):
            continue
        tools = []
        for elt in node.value.elts:
            if not isinstance(elt, ast.Dict):
                continue
            name_val = ""
            for k, v in zip(elt.keys, elt.values):
                if _ast_const_str(k) == "name":
                    name_val = _ast_const_str(v) or ""
            if name_val:
                tools.append({"name": name_val, "description": "", "params": []})
        if tools:
            return tools
    return []


def ast_extract_tools(toolkit_dir: Path) -> list:
    """
    Scan all .py files in toolkit_dir for get_available_tools() and extract tools.
    Merges results from api_wrapper.py, *client*.py, etc.
    Falls back to __all__ list in tools.py if no get_available_tools() found.
    """
    all_tools = []
    seen_names = set()

    # Priority: api_wrapper.py first, then other .py files
    py_files = []
    wrapper = toolkit_dir / "api_wrapper.py"
    if wrapper.exists():
        py_files.append(wrapper)
    for f in sorted(toolkit_dir.glob("*.py")):
        if f != wrapper and f.name != "__init__.py":
            py_files.append(f)

    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            for t in ast_extract_tools_from_source(source):
                if t["name"] not in seen_names:
                    seen_names.add(t["name"])
                    all_tools.append(t)
        except Exception as e:
            logger.debug("AST tool scan failed for %s: %s", py_file, e)

    # Fallback: check tools.py for __all__ = [{"name": ..., "tool": ...}] pattern
    if not all_tools:
        tools_py = toolkit_dir / "tools.py"
        if tools_py.exists():
            try:
                source = tools_py.read_text(encoding="utf-8")
                for t in ast_extract_tools_from_all_assignment(source):
                    if t["name"] not in seen_names:
                        seen_names.add(t["name"])
                        all_tools.append(t)
            except Exception as e:
                logger.debug("__all__ extraction failed for %s: %s", toolkit_dir, e)

    return all_tools


def extract_via_ast(reg_entry: dict) -> dict:
    """Extract toolkit info from source files using AST. No imports required."""
    key = reg_entry["key"]
    module_path = reg_entry["module_path"]
    module_dir = TOOLS_ROOT / module_path.replace(".", "/")

    result = {
        "key": key,
        "label": key,
        "categories": [],
        "extra_categories": [],
        "config_fields": [],
        "tools": [],
        "source": "ast",
    }

    for init_path in [module_dir / "__init__.py",
                      module_dir.parent / "__init__.py"]:
        if init_path.exists():
            try:
                meta = ast_extract_metadata(init_path)
                if meta["label"]:
                    result["label"] = meta["label"]
                result["categories"] = meta.get("categories", [])
                result["extra_categories"] = meta.get("extra_categories", [])
                result["config_fields"] = meta.get("config_fields", [])
                break
            except Exception as e:
                logger.debug("AST metadata failed for %s: %s", key, e)

    # Scan toolkit directory for tools (tries module_dir then parent)
    for scan_dir in [module_dir, module_dir.parent]:
        if scan_dir.is_dir() and scan_dir != TOOLS_ROOT:
            try:
                tools = ast_extract_tools(scan_dir)
                if tools:
                    result["tools"] = tools
                    break
            except Exception as e:
                logger.debug("AST tool extraction failed for %s: %s", key, e)

    return result


# ---------------------------------------------------------------------------
# Runtime toolkit entries (not in _safe_import_tool registry)
# ---------------------------------------------------------------------------

RUNTIME_TOOLKIT_ENTRIES = [
    {
        "key": "artifact",
        "default_label": "Artifact",
        "toolkit_file": "elitea_sdk/runtime/toolkits/artifact.py",
        "tool_files": ["elitea_sdk/runtime/tools/artifact.py"],
    },
    {
        "key": "memory",
        "default_label": "Memory",
        "toolkit_file": "elitea_sdk/tools/memory/__init__.py",
        "static_tools": [
            {"name": "manage_memory", "description": "Store information in long-term memory. Use this to remember important facts, user preferences, or any information that should persist across conversations.", "params": []},
            {"name": "search_memory", "description": "Search through stored memories using natural language. Returns memories that are semantically similar to the query.", "params": []},
            {"name": "get_memory", "description": "Retrieve a specific memory by its key.", "params": []},
            {"name": "delete_memory", "description": "Delete a specific memory by its key.", "params": []},
        ],
    },
    {
        "key": "mcp",
        "default_label": "MCP Server",
        "toolkit_file": "elitea_sdk/runtime/toolkits/mcp.py",
        "dynamic_note": True,
    },
]


def extract_runtime_toolkit_via_ast(entry: dict) -> dict:
    """Extract toolkit info for a runtime toolkit not in the _safe_import_tool registry."""
    key = entry["key"]
    toolkit_file = REPO_ROOT / entry["toolkit_file"]

    result = {
        "key": key,
        "label": entry.get("default_label", key),
        "categories": [],
        "extra_categories": [],
        "config_fields": [],
        "tools": [],
        "source": "ast",
    }

    if toolkit_file.exists():
        try:
            meta = ast_extract_metadata(toolkit_file)
            if meta["label"]:
                result["label"] = meta["label"]
            result["categories"] = meta.get("categories", [])
            result["extra_categories"] = meta.get("extra_categories", [])
            result["config_fields"] = meta.get("config_fields", [])
        except Exception as e:
            logger.debug("AST metadata failed for runtime toolkit %s: %s", key, e)

    if "static_tools" in entry:
        result["tools"] = entry["static_tools"]
    elif entry.get("dynamic_note"):
        result["is_dynamic"] = True
    else:
        seen_names: set = set()
        for tool_file_rel in entry.get("tool_files", []):
            tf = REPO_ROOT / tool_file_rel
            if tf.exists():
                try:
                    source = tf.read_text(encoding="utf-8")
                    for t in ast_extract_tools_from_source(source):
                        if t["name"] not in seen_names:
                            seen_names.add(t["name"])
                            result["tools"].append(t)
                except Exception as e:
                    logger.debug("Tool scan failed for %s: %s", tf, e)

    return result


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_params_table(params: list) -> list:
    if not params:
        return []
    lines = [
        "| Parameter | Type | Required | Default | Description |",
        "|-----------|------|----------|---------|-------------|",
    ]
    for p in params:
        req = "Yes" if p["required"] else "No"
        lines.append(
            f"| `{p['name']}` | {_esc(p['type'])} | {req} | {p['default']} | {_esc(p['description'])} |"
        )
    return lines


def _render_config_table(fields: list) -> list:
    if not fields:
        return []
    lines = [
        "### Configuration",
        "",
        "| Field | Type | Required | Description |",
        "|-------|------|----------|-------------|",
    ]
    for f in fields:
        req = "Yes" if f["required"] else "No"
        type_s = _esc(f["type"])
        if f.get("is_secret"):
            type_s += " *(secret)*"
        lines.append(f"| `{f['name']}` | {type_s} | {req} | {_esc(f['description'])} |")
    lines.append("")
    return lines


def render_section(info: dict) -> str:
    lines = []
    label = info.get("label") or info["key"]
    key = info["key"]

    lines.append(f"## {label} (`{key}`)")
    meta_parts = []
    if info.get("categories"):
        meta_parts.append(f"**Categories**: {', '.join(info['categories'])}")
    if info.get("extra_categories"):
        meta_parts.append(f"**Tags**: {', '.join(info['extra_categories'])}")
    if meta_parts:
        lines.append(" | ".join(meta_parts))
    if info.get("source") == "ast":
        lines.append("*\\[Info extracted from source; run with toolkit deps installed for full detail\\]*")
    lines.append("")

    lines.extend(_render_config_table(info.get("config_fields", [])))

    tools = info.get("tools", [])
    if tools:
        lines.append("### Tools")
        lines.append("")
        for tool in tools:
            tname = tool.get("name", "")
            desc = (tool.get("description") or "").strip()
            params = tool.get("params", [])

            lines.append(f"#### `{tname}`")
            if desc:
                first_line = _esc(desc.split("\n")[0].strip())
                lines.append(f"> {first_line}")
            lines.append("")

            if not COMPACT and params:
                lines.extend(_render_params_table(params))
                lines.append("")
    else:
        if key == "openapi":
            lines.append("*Tools are generated dynamically at runtime from the provided OpenAPI spec \u2014 one tool per `operationId`.*")
        elif info.get("is_dynamic"):
            lines.append("*Tools are provided at runtime by the connected MCP server \u2014 the available tool list depends on the server configuration.*")
        else:
            lines.append("*No tool information available.*")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    registry = read_registry()
    if not registry:
        print("ERROR: Could not read toolkit registry from elitea_sdk/tools/__init__.py")
        sys.exit(1)

    available_tools = {}
    failed_imports = {}
    if not AST_ONLY:
        try:
            from elitea_sdk.tools import AVAILABLE_TOOLS, FAILED_IMPORTS
            available_tools = AVAILABLE_TOOLS
            failed_imports = FAILED_IMPORTS
        except ImportError as e:
            logger.warning("Could not import elitea_sdk.tools (%s) — switching to AST-only mode", e)

    sections = {}
    ast_count = 0
    import_count = 0

    for reg_entry in registry:
        key = reg_entry["key"]
        if key in SKIP_KEYS:
            continue
        info = None

        if not AST_ONLY:
            try:
                info = extract_via_import(reg_entry, available_tools)
                if info:
                    import_count += 1
            except Exception as e:
                logger.debug("Import extraction failed for %s: %s", key, e)

        if info is None:
            try:
                info = extract_via_ast(reg_entry)
                ast_count += 1
            except Exception as e:
                logger.warning("AST extraction failed for %s: %s", key, e)
                info = {
                    "key": key, "label": key, "categories": [], "extra_categories": [],
                    "config_fields": [], "tools": [], "source": "error",
                }

        try:
            sections[key] = render_section(info)
        except Exception as e:
            logger.warning("Render failed for %s: %s", key, e)

    # Runtime toolkits (artifact, memory, mcp) — not in _safe_import_tool registry
    for rt_entry in RUNTIME_TOOLKIT_ENTRIES:
        rt_key = rt_entry["key"]
        try:
            rt_info = extract_runtime_toolkit_via_ast(rt_entry)
            ast_count += 1
            sections[rt_key] = render_section(rt_info)
        except Exception as e:
            logger.warning("Runtime toolkit extraction failed for %s: %s", rt_key, e)

    # Assemble document
    doc_lines = [
        "# Alita SDK \u2014 Toolkits Guide",
        "",
        "Auto-generated reference for all available toolkits and their tools.",
        "Use this guide to understand each toolkit's configuration fields and available tools.",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
    ]

    for key in sorted(sections):
        first_line = sections[key].split("\n")[0]
        label_part = first_line.replace("## ", "").split(" (`")[0]
        anchor = key.lower().replace("_", "-")
        doc_lines.append(f"- [{label_part} (`{key}`)](#user-content-{anchor})")

    if failed_imports:
        doc_lines += ["", "---", "", "## Failed Imports", "",
                      "*The following toolkits could not be imported (missing optional deps):*", ""]
        for k, err in sorted(failed_imports.items()):
            short_err = str(err).split("\n")[0][:120]
            doc_lines.append(f"- `{k}`: {short_err}")

    doc_lines += ["", "---", ""]

    for key in sorted(sections):
        doc_lines.append(sections[key])
        doc_lines += ["", "---", ""]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(doc_lines), encoding="utf-8")

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\u2713 Generated {OUTPUT_PATH}")
    print(f"  Toolkits total      : {len(sections)}")
    print(f"  Via imports         : {import_count}")
    print(f"  Via AST (fallback)  : {ast_count}")
    print(f"  Failed imports      : {len(failed_imports)}")
    print(f"  File size           : {size_kb:.1f} KB")
    if failed_imports:
        print(f"\n  Tip: pip install -e '.[tools]' enables full import-based extraction.")


if __name__ == "__main__":
    main()
