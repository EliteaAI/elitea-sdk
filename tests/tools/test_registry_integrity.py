import ast
import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import elitea_sdk.runtime.langchain.document_loaders as loaders_pkg
from elitea_sdk.tools import FAILED_IMPORTS

PREEXISTING_TOOLS_INTERNAL_CYCLES = {"inventory"}

def test_registry_has_no_unexpected_failed_imports():
    unexpected = set(FAILED_IMPORTS) - PREEXISTING_TOOLS_INTERNAL_CYCLES
    assert not unexpected, (
        f"Toolkits silently dropped from the registry: "
        f"{ {k: v[:120] for k, v in FAILED_IMPORTS.items() if k in unexpected} }"
    )


def module_scope_tools_imports(source, module_package):
    """Module-scope imports resolving under elitea_sdk.tools, absolute or relative.

    Module scope includes bodies of top-level try/if/with blocks — they execute at
    import time just the same — but not function or class bodies.
    """
    offenders = []
    tree = ast.parse(source)

    def resolve_relative(level, module):
        parts = module_package.split(".")
        if level > len(parts):
            return module or ""
        base = parts[: len(parts) - (level - 1)] if level > 1 else parts
        return ".".join(base + ([module] if module else []))

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.ImportFrom):
                target = resolve_relative(child.level, child.module) if child.level else (child.module or "")
                if target == "elitea_sdk.tools" or target.startswith("elitea_sdk.tools."):
                    offenders.append((child.lineno, ast.unparse(child)))
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name == "elitea_sdk.tools" or alias.name.startswith("elitea_sdk.tools."):
                        offenders.append((child.lineno, ast.unparse(child)))
            visit(child)

    visit(tree)
    return offenders


def test_no_loader_imports_tools_at_module_level():
    offenders = []
    for module_info in pkgutil.iter_modules(loaders_pkg.__path__):
        module_name = f"{loaders_pkg.__name__}.{module_info.name}"
        module = importlib.import_module(module_name)
        source_file = getattr(module, "__file__", None)
        if not source_file:
            continue
        with open(source_file) as fh:
            source = fh.read()
        for lineno, stmt in module_scope_tools_imports(source, loaders_pkg.__name__):
            offenders.append(f"{module_info.name}:{lineno}: {stmt}")
    assert not offenders, (
        f"Module-scope elitea_sdk.tools imports in document loaders (circular — import "
        f"lazily at the call site instead): {offenders}"
    )


def test_loaders_imported_first_leave_registry_intact():
    """Deterministically force the dangerous import order in a clean interpreter."""
    loader_names = [m.name for m in pkgutil.iter_modules(loaders_pkg.__path__)]
    assert loader_names, "no document loaders found"
    imports = "; ".join(f"import {loaders_pkg.__name__}.{name}" for name in loader_names)
    script = (
        "import logging, warnings; logging.disable(logging.CRITICAL); warnings.filterwarnings('ignore'); "
        f"{imports}; "
        "from elitea_sdk.tools import FAILED_IMPORTS; "
        f"bad = sorted(set(FAILED_IMPORTS) - {PREEXISTING_TOOLS_INTERNAL_CYCLES!r}); "
        "print('|'.join(bad))"
    )
    repo_root = str(Path(loaders_pkg.__file__).parents[4])
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=180,
    )
    assert result.returncode == 0, f"loader-first interpreter crashed: {result.stderr[-500:]}"
    bad = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    assert not bad, f"Importing loaders before elitea_sdk.tools dropped toolkits: {bad.split('|')}"
