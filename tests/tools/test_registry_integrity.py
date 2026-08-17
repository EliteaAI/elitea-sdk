import ast
import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import elitea_sdk.configurations as configurations_pkg
import elitea_sdk.runtime.langchain.document_loaders as loaders_pkg

PREEXISTING_TOOLS_INTERNAL_CYCLES = {"inventory"}

REPO_ROOT = str(Path(loaders_pkg.__file__).parents[4])

CYCLE_PRONE_PACKAGES = [loaders_pkg, configurations_pkg]


def run_registry_check(first_imports):
    script = (
        "import logging, warnings; logging.disable(logging.CRITICAL); warnings.filterwarnings('ignore'); "
        f"{first_imports}; "
        "from elitea_sdk.tools import FAILED_IMPORTS; "
        f"bad = sorted(set(FAILED_IMPORTS) - {PREEXISTING_TOOLS_INTERNAL_CYCLES!r}); "
        "print('REGISTRY_RESULT:' + '|'.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=180,
    )
    assert result.returncode == 0, f"registry-check interpreter crashed: {result.stderr[-500:]}"
    marker = next(line for line in result.stdout.splitlines() if line.startswith("REGISTRY_RESULT:"))
    return marker.removeprefix("REGISTRY_RESULT:")


def test_tools_import_alone_leaves_registry_intact():
    bad = run_registry_check("pass")
    assert not bad, f"Toolkits silently dropped from the registry on a clean import: {bad.split('|')}"


@pytest.mark.parametrize("package", CYCLE_PRONE_PACKAGES, ids=lambda p: p.__name__)
def test_package_imported_first_leaves_registry_intact(package):
    module_names = [m.name for m in pkgutil.iter_modules(package.__path__)]
    assert module_names, f"no modules found in {package.__name__}"
    imports = "; ".join(f"import {package.__name__}.{name}" for name in module_names)
    bad = run_registry_check(imports)
    assert not bad, f"Importing {package.__name__} before elitea_sdk.tools dropped toolkits: {bad.split('|')}"


def module_scope_tools_imports(source, module_package):
    """Module-scope imports resolving under elitea_sdk.tools, absolute or relative.

    Module scope is everything that executes at import time: top-level statements,
    try/if/with bodies, and class bodies — but not function bodies.
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
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
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


@pytest.mark.parametrize("package", CYCLE_PRONE_PACKAGES, ids=lambda p: p.__name__)
def test_no_module_scope_tools_imports(package):
    offenders = []
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        source_file = getattr(module, "__file__", None)
        if not source_file:
            continue
        with open(source_file) as fh:
            source = fh.read()
        for lineno, stmt in module_scope_tools_imports(source, package.__name__):
            offenders.append(f"{module_info.name}:{lineno}: {stmt}")
    assert not offenders, (
        f"Module-scope elitea_sdk.tools imports in {package.__name__} (circular — import "
        f"lazily at the call site instead): {offenders}"
    )
