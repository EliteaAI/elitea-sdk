"""Code Mode (#7579): the LLM writes one Python program; toolkit calls go through the sandbox bridge."""

import asyncio
import json
import inspect
import keyword
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from ...tools.elitea_base import INDEX_TOOL_NAMES
from ...tools.utils.serialization import serialize_tool_result
from ..utils.toolkit_utils import get_args_json_schema
from .lazy_tools import ToolRegistry, reraise_control_flow, resolve_registry_tool, strip_null_args

logger = logging.getLogger(__name__)

CODE_MODE_META = {"toolkit_type": "internal", "toolkit_name": "code_mode", "display_name": "Code Mode"}

_HINTS = {
    "result_too_large": "Request less data: fewer items, filter arguments, or pagination.",
    "out_of_memory": "Process items page by page and keep only the fields you need; don't hold all results at once.",
    "timeout": ("The whole program, tool calls included, must finish within the time limit. Do less per program "
                "and split the work into several execute_code calls; don't repeat calls that already completed."),
    "call_limit": "Batch the work or split it into several execute_code calls.",
    "python_exception": "Fix the error shown in the message and run the corrected program.",
    "sandbox_crashed": "The sandbox died, most likely out of memory. Split the work into smaller programs.",
    "tool_error": "A tool call failed; check the arguments against get_tool_api.",
}

_JSON_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "array": "list",
               "object": "dict", "null": "None"}


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def read_code_mode_limits() -> Dict[str, int]:
    from .sandbox import _read_sandbox_limits_from_env
    # The platform sandbox timeout is the ceiling for every sandbox run, Code Mode included.
    sandbox_timeout = int(_read_sandbox_limits_from_env()["timeout_seconds"])
    return {
        "timeout_seconds": max(1, min(_env_int("CODE_MODE_TIMEOUT_SECONDS", 120), sandbox_timeout)),
        "max_tool_calls": _env_int("CODE_MODE_MAX_TOOL_CALLS", 150),
        "max_result_bytes": _env_int("CODE_MODE_MAX_RESULT_BYTES", 1024 * 1024),
        "max_output_chars": _env_int("CODE_MODE_MAX_OUTPUT_CHARS", 25000),
        "inline_stubs_max": _env_int("CODE_MODE_INLINE_STUBS_MAX", 15),
    }


# ask_user must reach the user via HITL; the sandbox tools would nest a sandbox in the sandbox.
DIRECT_BIND_INTERNAL_TOOLS = {"ask_user", "pyodide", "data_analysis"}


def is_direct_bind_tool(tool: BaseTool) -> bool:
    meta = getattr(tool, "metadata", None) or {}
    return meta.get("toolkit_name") in DIRECT_BIND_INTERNAL_TOOLS


def code_mode_eligible_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """Indexer tools are long-running jobs, so programs can't call them; direct-bind tools stay outside too."""
    return [t for t in tools if isinstance(t, BaseTool) and t.name not in INDEX_TOOL_NAMES
            and not is_direct_bind_tool(t)]


def _identifier(name: str) -> str:
    ident = re.sub(r"\W", "_", str(name)).strip("_") or "toolkit"
    if ident[0].isdigit() or keyword.iskeyword(ident):
        ident = f"_{ident}"
    return ident


def _hint(prop: Dict[str, Any]) -> str:
    if "anyOf" in prop:
        hints = [_hint(p) for p in prop["anyOf"]]
        hints = [h for h in dict.fromkeys(hints) if h != "None"] + (["None"] if "None" in hints else [])
        return " | ".join(hints) if len(hints) < 4 else "Any"
    kind = prop.get("type")
    if isinstance(kind, list):
        hints = [_JSON_TYPES.get(k, "Any") for k in kind]
        hints = [h for h in dict.fromkeys(hints) if h != "None"] + (["None"] if "None" in hints else [])
        return " | ".join(hints) if hints else "Any"
    return _JSON_TYPES.get(kind, "Any")


def _wrapper_method(tool: BaseTool):
    """The api_wrapper method a toolkit tool dispatches to (`ref` in get_available_tools), if any."""
    wrapper = getattr(tool, "api_wrapper", None)
    if wrapper is None or not hasattr(wrapper, "get_available_tools"):
        return None
    try:
        name = re.sub(r"_\d+$", "", tool.name)
        return next((t.get("ref") for t in wrapper.get_available_tools() if t.get("name") == name), None)
    except Exception:  # pylint: disable=broad-except
        return None


def _annotation_hint(annotation: Any) -> Optional[str]:
    import typing
    if annotation is None or annotation is inspect.Signature.empty:
        return None
    if annotation is type(None):
        return "None"
    origin = typing.get_origin(annotation)
    if origin is typing.Union or (origin is not None and str(origin) == "<class 'types.UnionType'>"):
        parts = [_annotation_hint(a) or "Any" for a in typing.get_args(annotation)]
        return " | ".join(dict.fromkeys(parts))
    base = origin or annotation
    names = {str: "str", int: "int", float: "float", bool: "bool", list: "list", dict: "dict",
             tuple: "list", set: "list"}
    return names.get(base, "Any")


def return_hint(tool: BaseTool) -> Optional[str]:
    """The declared return type of the wrapper method, or None when undeclared or only Any."""
    method = _wrapper_method(tool)
    if method is None:
        return None
    import typing
    try:
        # Toolkit modules often use `from __future__ import annotations`, so resolve string annotations.
        annotation = typing.get_type_hints(method).get("return", inspect.Signature.empty)
    except Exception:  # pylint: disable=broad-except
        annotation = inspect.signature(method).return_annotation
    hint = _annotation_hint(annotation) if not isinstance(annotation, str) else None
    # Many wrappers declare `-> str` yet return lists/dicts; only structured types are trusted.
    return hint if hint and ("list" in hint or "dict" in hint) else None


def returns_doc(tool: BaseTool) -> Optional[str]:
    match = re.search(r"Returns?:\s*(.+?)(?:\n\s*\n|\Z)", tool.description or "", re.S | re.I)
    return _doc_line(match.group(1))[:300] if match else None


# Names the preamble defines; a toolkit class named like one would shadow it.
_RESERVED_NAMES = {"ToolError", "Any", "_call", "_bridge", "_json", "json", "asyncio"}


def _doc_line(text: str) -> str:
    # Stub docstrings are not raw strings, so backslashes in descriptions must not form escapes.
    return " ".join(str(text or "").split()).replace("\\", "\\\\").replace('"""', "'''")


def stubs_for(registry: ToolRegistry) -> "CodeModeStubs":
    """One CodeModeStubs per registry: schema and return-type work is done once, not every turn."""
    stubs = getattr(registry, "_code_mode_stubs", None)
    if stubs is None:
        stubs = CodeModeStubs(registry)
        registry._code_mode_stubs = stubs
    return stubs


class CodeModeStubs:
    """Python stubs per toolkit instance, built from each tool's args_schema; keys are `class.tool` bridge names."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.class_names: Dict[str, str] = {}
        self.method_names: Dict[tuple, str] = {}
        self.bridge_names: Dict[str, tuple] = {}
        for toolkit in registry.get_toolkit_names():
            ident = _identifier(toolkit)
            while ident in self.class_names.values() or ident in _RESERVED_NAMES:
                ident = f"{ident}_"
            self.class_names[toolkit] = ident
            taken = set()
            for tool_name in registry.get_toolkit_tools(toolkit):
                method = _identifier(tool_name)
                while method in taken:
                    method = f"{method}_"
                taken.add(method)
                self.method_names[(toolkit, tool_name)] = method
                self.bridge_names[f"{ident}.{method}"] = (toolkit, tool_name)

    @property
    def tool_count(self) -> int:
        return len(self.bridge_names)

    def resolve_toolkit(self, name: str) -> Optional[str]:
        for toolkit, ident in self.class_names.items():
            if name in (toolkit, ident):
                return toolkit
        return None

    def tool_stub(self, toolkit: str, tool: BaseTool) -> str:
        schema = get_args_json_schema(tool)
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])
        params, arg_docs = [], []
        for name in sorted(props, key=lambda n: n not in required):
            prop = props[name] or {}
            ident = _identifier(name)
            hint = _hint(prop)
            if name in required:
                params.append(f"{ident}: {hint}")
            else:
                params.append(f"{ident}: {hint} = {repr(prop.get('default'))}")
            doc = _doc_line(prop.get("description", ""))
            if prop.get("enum"):
                doc = f"{doc.rstrip('.')}. One of {prop['enum']}.".lstrip(". ")
            if doc:
                arg_docs.append(f"            {ident}: {doc[:300]}")
        signature = ", ".join(["*"] + params) if params else ""
        mapping = ", ".join(f"{json.dumps(n)}: {_identifier(n)}" for n in props)
        doc = _doc_line(tool.description)[:600]
        method = self.method_names[(toolkit, tool.name)]
        bridge_name = f"{self.class_names[toolkit]}.{method}"
        returns = return_hint(tool)
        lines = [
            "    @staticmethod",
            f"    async def {method}({signature}) -> {returns or 'Any'}:",
            f'        """{doc}',
        ]
        if arg_docs:
            lines += ["", "        Args:", *arg_docs]
        returns_text = returns_doc(tool)
        lines += ["", f"        Returns: {returns_text}" if returns_text else
                  "        Returns: shape not declared; print a small sample before relying on fields."]
        lines += ['        """', f"        return await _call({json.dumps(bridge_name)}, {{{mapping}}})"]
        return "\n".join(lines)

    def signature_line(self, toolkit: str, tool: BaseTool) -> str:
        schema = get_args_json_schema(tool)
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])
        params = [f"{_identifier(n)}: {_hint(props[n] or {})}" + ("" if n in required else "=...")
                  for n in sorted(props, key=lambda n: n not in required)]
        summary = _doc_line(tool.description).split(". ")[0][:100]
        returns = return_hint(tool) or "Any"
        return (f"await {self.class_names[toolkit]}.{self.method_names[(toolkit, tool.name)]}({', '.join(params)}) -> {returns}"
                f"  # {summary}")

    def signature_index(self) -> str:
        return "\n".join(self.signature_line(tk, t) for tk in self.class_names
                         for t in self.registry.get_toolkit_tools(tk).values())

    def example_call(self) -> str:
        for toolkit, ident in self.class_names.items():
            for tool in self.registry.get_toolkit_tools(toolkit).values():
                return f"result = await {ident}.{self.method_names[(toolkit, tool.name)]}(...)"
        return "result = await toolkit.tool(...)"

    def strip_toolkit_imports(self, code: str) -> str:
        """Drop `import X` / `from X import ...` of toolkit classes; Pyodide would try to install them as packages."""
        names = "|".join(re.escape(n) for n in self.class_names.values())
        if not names:
            return code
        return re.sub(rf"^[ \t]*(from[ \t]+({names})[ \t]+import[^\n]*|import[ \t]+({names})[ \t]*(as[ \t]+\w+)?)[ \t]*$",
                      "", code, flags=re.M)

    def toolkit_stub(self, toolkit: str, tool_names: Optional[List[str]] = None) -> str:
        tools = self.registry.get_toolkit_tools(toolkit)
        selected = [tools[n] for n in tools if not tool_names or n in tool_names]
        body = [self.tool_stub(toolkit, t) for t in selected] or ["    pass"]
        description = _doc_line(self.registry.get_toolkit_description(toolkit) or toolkit)[:300]
        return f'class {self.class_names[toolkit]}:\n    """{description}"""\n\n' + "\n\n".join(body)

    def all_stubs(self) -> str:
        if getattr(self, "_all_stubs", None) is None:
            self._all_stubs = "\n\n\n".join(self.toolkit_stub(t) for t in self.class_names)
        return self._all_stubs


PREAMBLE = '''import json as _json
from typing import Any
import elitea_bridge as _bridge


class ToolError(Exception):
    """A toolkit call failed; the message says why."""


async def _call(name, args):
    args = {k: v for k, v in args.items() if v is not None}
    try:
        value = await _bridge.call(name, _json.dumps(args, default=str))
    except Exception as exc:
        raise ToolError(str(exc).removeprefix("Error: ")) from None
    return _json.loads(value)


'''


class BridgeDispatcher:
    """Runs a program's bridged calls against the agent's own tools; keeps the call log the error shape reports."""

    def __init__(self, stubs: CodeModeStubs, config: Optional[RunnableConfig], limits: Dict[str, int],
                 sensitive_guard: Any = None):
        self.stubs = stubs
        self.config = config
        self.limits = limits
        self.sensitive_guard = sensitive_guard
        self.call_log: List[Dict[str, Any]] = []
        self.abort: Optional[BaseException] = None

    def _refuse(self, entry: Dict[str, Any], error_class: str, message: str):
        entry.update(ok=False, error_class=error_class, error=message)
        raise RuntimeError(message)

    def _refuse_too_large(self, entry: Dict[str, Any], name: str, size: int, limit: int, at_least: bool = False):
        self._refuse(entry, "result_too_large",
                     f"{name} result was {'over ' if at_least else ''}{_size(size)}, limit {_size(limit)}. "
                     f"{_HINTS['result_too_large']}")

    def _approval_needed(self, tool: BaseTool, args: Dict[str, Any]) -> Optional[str]:
        # Without the guard middleware a direct call runs unreviewed, so the program call does too.
        if self.sensitive_guard is None:
            return None
        return self.sensitive_guard.approval_needed(tool, args)

    async def __call__(self, name: str, args_json: str) -> str:
        entry: Dict[str, Any] = {"tool": name, "ok": False, "ms": 0, "bytes": 0}
        self.call_log.append(entry)
        start = time.monotonic()
        try:
            return await self._dispatch(name, args_json, entry)
        except asyncio.CancelledError:
            # The tool may still finish in its thread or remotely; its effect is unknown.
            entry.update(error_class="indeterminate", error="cancelled while running")
            raise
        finally:
            entry["ms"] = int((time.monotonic() - start) * 1000)

    async def _dispatch(self, name: str, args_json: str, entry: Dict[str, Any]) -> str:
        if self.abort is not None:
            self._refuse(entry, "aborted", "The run is stopping.")
        target = self.stubs.bridge_names.get(name)
        if target is None:
            valid = ", ".join(sorted(self.stubs.bridge_names)[:50])
            self._refuse(entry, "tool_error", f"Unknown tool '{name}'. Available: {valid}")
        toolkit, tool_name = target
        tool, toolkit, error = resolve_registry_tool(self.stubs.registry, toolkit, tool_name)
        if error:
            self._refuse(entry, "tool_error", error)
        try:
            args = strip_null_args(json.loads(args_json or "{}"))
        except ValueError:
            self._refuse(entry, "tool_error", f"Arguments of {name} are not valid JSON.")

        action = self._approval_needed(tool, args)
        if action:
            self._refuse(entry, "approval_required",
                         f"{action} needs human approval, which is not available inside Code Mode programs. "
                         f"Don't call it from execute_code.")

        schema = getattr(tool, "args_schema", None)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                schema.model_validate(args)
            except ValidationError as exc:
                problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors())
                self._refuse(entry, "tool_error", f"Invalid arguments for {name}: {problems}")

        try:
            result = await tool.ainvoke(args, config=self.config)
        except Exception as exc:  # pylint: disable=broad-except
            try:
                reraise_control_flow(exc)
            except BaseException as stop:
                self.abort = stop
                self._refuse(entry, "aborted", "The run is stopping.")
            self._refuse(entry, "tool_error", f"{name} failed: {exc}")

        error_text = returned_error(result)
        if error_text:
            self._refuse(entry, "tool_error", f"{name} failed: {error_text}")
        limit = self.limits["max_result_bytes"]
        # Refuse obvious oversize before building more copies of the result.
        estimate = _size_lower_bound(result)
        if estimate > limit:
            entry["bytes"] = estimate
            self._refuse_too_large(entry, name, estimate, limit, at_least=True)
        result = parse_json_text(result)
        try:
            value = json.dumps(result)
        except (TypeError, ValueError):
            value = json.dumps(serialize_tool_result(result))
        entry["bytes"] = len(value) if value.isascii() else len(value.encode("utf-8"))
        if entry["bytes"] > limit:
            self._refuse_too_large(entry, name, entry["bytes"], limit)
        entry["shape"] = value_shape(result)[:400]
        entry["ok"] = True
        return value


# Toolkit wrappers report failures by returning text like these instead of raising.
_ERROR_TEXT = re.compile(r"^\s*(error\s*[:!-]|failed to\b|unable to\b|tool execution error|exception\s*:)", re.I)


def returned_error(result: Any) -> Optional[str]:
    """The error message when a tool returned a failure instead of raising, else None."""
    if isinstance(result, str) and _ERROR_TEXT.match(result):
        return result.strip()[:2000]
    if isinstance(result, dict) and set(result) <= {"error", "message", "status", "code", "details"}:
        error = result.get("error")
        if error or str(result.get("status", "")).lower() == "error":
            return str(error or result.get("message") or result)[:2000]
    return None


def parse_json_text(result: Any) -> Any:
    """A JSON document returned as a string reaches the program as the list/dict it encodes."""
    if isinstance(result, str) and result.lstrip()[:1] in "[{":
        try:
            return json.loads(result)
        except ValueError:
            return result
    return result


def value_shape(value: Any, depth: int = 0) -> str:
    """Compact structure of a JSON value, e.g. {total:int, rows:[{name:str, size:int}]}."""
    if isinstance(value, dict):
        if depth >= 3:
            return "{...}"
        items = list(value.items())
        inner = ", ".join(f"{k}:{value_shape(v, depth + 1)}" for k, v in items[:12])
        return "{" + inner + (", ..." if len(items) > 12 else "") + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        return f"[{value_shape(value[0], depth + 1)}] x{len(value)}"
    if isinstance(value, str):
        stripped = value.strip()
        if depth == 0 and stripped[:1] in "[{":
            try:
                return "str(JSON " + value_shape(json.loads(stripped), depth) + ")"
            except ValueError:
                pass
        return f"str({len(value)} chars)" if depth == 0 else "str"
    return {bool: "bool", int: "int", float: "float"}.get(type(value), "null" if value is None else type(value).__name__)


def _size_lower_bound(value: Any, depth: int = 0) -> int:
    """A cheap lower bound of the serialized size, so huge results are refused before json.dumps copies them."""
    if isinstance(value, (str, bytes)):
        return len(value)
    if depth >= 2:
        return 0
    if isinstance(value, dict):
        return sum(len(str(k)) + _size_lower_bound(v, depth + 1) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_size_lower_bound(v, depth + 1) for v in value)
    return 0


def _size(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.0f} KB"


def _format_calls(call_log: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for c in call_log:
        status = "ok" if c["ok"] else f"failed ({c.get('error_class', 'error')})"
        shape = f" -> {c['shape']}" if c.get("shape") else ""
        lines.append(f"{c['tool']} {status} {c['ms'] / 1000:.1f}s {_size(c['bytes'])}{shape}")
    return lines


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [output truncated: {len(text)} chars, limit {limit}; print less]"


def build_run_report(result: Any, call_log: List[Dict[str, Any]], max_output_chars: int) -> Dict[str, Any]:
    """The execute_code answer: output and calls on success, the classified error shape on failure."""
    output = _cap(result.stdout or "", max_output_chars)
    if result.status == "success":
        report: Dict[str, Any] = {"status": "success", "output": output, "calls": _format_calls(call_log)}
        if result.result is not None:
            report["result"] = _cap(json.dumps(result.result, default=str), max_output_chars)
        return report

    error_class = result.error_type or "sandbox_crashed"
    message = result.error or ""
    refused = [c for c in call_log if c.get("error_class") == "result_too_large"]
    # An uncaught refusal surfaces as a Python exception; the cause the LLM must fix is the size.
    if error_class == "python_exception" and refused:
        error_class = "result_too_large"
        message = f"{refused[-1]['error']}\n{message}"
    elif error_class == "python_exception" and "MemoryError" in message:
        error_class = "out_of_memory"
    done = [c for c in call_log if c["ok"]]
    unknown = [c for c in call_log if c.get("error_class") == "indeterminate"]
    report_extra = {}
    if unknown:
        report_extra["calls_indeterminate"] = _format_calls(unknown) + [
            "These calls were still running when the program stopped; they may or may not have taken effect. "
            "Check their result before repeating any write."]
    return {
        "status": "error",
        "error_class": error_class,
        "message": _cap(message.strip(), 4000),
        "hint": _HINTS.get(error_class, _HINTS["python_exception"]),
        "calls_completed": _format_calls(done) + (
            ["These calls already ran. Do not repeat the writes among them."] if done else []),
        "output_before_failure": output,
        **report_extra,
    }


EXECUTE_CODE_RULES = """Run one Python program that calls the agent's toolkits. Use it to chain several tool calls, \
loop, filter and aggregate in code instead of calling tools one by one.
Rules:
- The toolkit classes below are already defined in the program's globals. Do not import them; call their \
static methods directly: `{example}`. Top-level await works.
- Arguments are keyword-only. Results are JSON values (dict, list, str); their exact shape is reported in "calls" \
after each run, so if unsure, print a small part first rather than guessing.
- Only what you print() comes back to you, so print just what you need, compactly.
- A failed call raises ToolError; catch it to continue.
- No network or files; only toolkit methods reach the outside world.
- Each program must finish within {timeout_seconds} s, tool calls included (they run one at a time), and make at most \
{max_tool_calls} tool calls; a result over {max_result_kb} KB is refused. For many slow calls, split the work into \
several programs."""


class ExecuteCodeInput(BaseModel):
    code: str = Field(description="The Python program to run. Toolkit classes are already defined.")


class GetToolApiInput(BaseModel):
    toolkit: str = Field(description="Toolkit name as listed in the system prompt.")
    tools: Optional[List[str]] = Field(default=None, description="Tool names; omit for the whole toolkit.")


class ExecuteCodeTool(BaseTool):
    name: str = "execute_code"
    description: str = ""
    args_schema: type[BaseModel] = ExecuteCodeInput
    stubs: Any = Field(exclude=True)
    limits: Dict[str, int] = Field(default_factory=read_code_mode_limits)
    sensitive_guard: Any = Field(default=None, exclude=True)
    metadata: dict = dict(CODE_MODE_META)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.description:
            self.description = EXECUTE_CODE_RULES.format(
                max_result_kb=self.limits["max_result_bytes"] // 1024, example=self.stubs.example_call(), **self.limits)
            if self.stubs.tool_count <= self.limits["inline_stubs_max"]:
                self.description += "\n\nAvailable toolkit API:\n" + self.stubs.all_stubs()
            else:
                self.description += ("\n\nAvailable methods (call get_tool_api only when you need argument docs):\n"
                                     + self.stubs.signature_index())

    def _run(self, code: str, run_manager: Optional[CallbackManagerForToolRun] = None,
             config: Optional[RunnableConfig] = None) -> str:
        # Bridged calls must stay children of this run, so the callbacks travel into the async path.
        coro = self._arun(code, run_manager=run_manager, config=config)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()

    async def _arun(self, code: str, run_manager: Any = None, config: Optional[RunnableConfig] = None) -> str:
        from .sandbox import _read_sandbox_limits_from_env, build_offline_pyodide_sandbox, enforce_sandbox_admission
        sandbox_limits = _read_sandbox_limits_from_env()
        enforce_sandbox_admission(sandbox_limits)
        # Bridged calls run as children of this tool run, so they show up as tool steps and are metered.
        child_config: Dict[str, Any] = dict(config or {})
        if run_manager is not None:
            child_config["callbacks"] = run_manager.get_child()
        dispatcher = BridgeDispatcher(self.stubs, child_config, self.limits, self.sensitive_guard)
        timeout = min(self.limits["timeout_seconds"], sandbox_limits["timeout_seconds"])
        result = await build_offline_pyodide_sandbox().execute_with_bridge(
            PREAMBLE + self.stubs.all_stubs() + "\n\n\n" + self.stubs.strip_toolkit_imports(code),
            dispatcher,
            timeout_seconds=timeout,
            max_tool_calls=self.limits["max_tool_calls"],
            memory_limit_mb=sandbox_limits["wasm_max_mem_mb"],
            max_concurrent_calls=1,
        )
        if dispatcher.abort is not None:
            raise dispatcher.abort
        report = build_run_report(result, dispatcher.call_log, self.limits["max_output_chars"])
        logger.info("[CodeMode] execute_code %s (%s) calls=%d in %.1fs", report["status"],
                    report.get("error_class", "-"), len(dispatcher.call_log), result.execution_time)
        return json.dumps(report, ensure_ascii=False, indent=1)


class GetToolApiTool(BaseTool):
    name: str = "get_tool_api"
    description: str = ("Get argument docs of toolkit methods. Signatures are already in the execute_code description; "
                        "call this only when an argument's meaning is unclear.")
    args_schema: type[BaseModel] = GetToolApiInput
    stubs: Any = Field(exclude=True)
    metadata: dict = dict(CODE_MODE_META)

    def _run(self, toolkit: str, tools: Optional[List[str]] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        resolved = self.stubs.resolve_toolkit(toolkit)
        if resolved is None:
            return f"Toolkit '{toolkit}' not found. Available: {', '.join(self.stubs.class_names.values())}"
        known = self.stubs.registry.get_toolkit_tools(resolved)
        missing = [t for t in tools or [] if t not in known]
        stub = self.stubs.toolkit_stub(resolved, [t for t in tools or [] if t in known] or None)
        if missing:
            stub += f"\n\n# Not found in {toolkit}: {', '.join(missing)}. Available: {', '.join(known)}"
        return stub


def create_code_mode_tools(registry: ToolRegistry, sensitive_guard: Any = None) -> List[BaseTool]:
    stubs = stubs_for(registry)
    execute = ExecuteCodeTool(stubs=stubs, sensitive_guard=sensitive_guard)
    if stubs.tool_count <= execute.limits["inline_stubs_max"]:
        return [execute]
    return [execute, GetToolApiTool(stubs=stubs)]


def generate_code_mode_index(registry: ToolRegistry) -> str:
    """System-prompt index: toolkit classes and tool names only; signatures come from get_tool_api."""
    stubs = stubs_for(registry)
    inline = stubs.tool_count <= read_code_mode_limits()["inline_stubs_max"]
    lines = ["## Code Mode",
             "Your toolkits are Python classes inside the execute_code sandbox. Write one program that does the "
             "whole task (call, loop, filter, aggregate) instead of one tool call at a time.",
             "The toolkit classes are predefined in the program (never import them); their methods are listed in the "
             "execute_code description." + ("" if inline else " get_tool_api gives argument docs when needed."),
             "", "Toolkits:"]
    for toolkit, ident in stubs.class_names.items():
        desc = _doc_line(registry.get_toolkit_description(toolkit) or "")[:120]
        lines.append(f"- {ident}{f' ({desc})' if desc else ''}: {', '.join(registry.get_toolkit_tools(toolkit))}")
    return "\n".join(lines)
