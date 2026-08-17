import importlib
import pkgutil

import elitea_sdk.runtime.langchain.document_loaders as loaders_pkg
from elitea_sdk.tools import FAILED_IMPORTS

# 'inventory' has a separate, pre-existing import cycle inside elitea_sdk.tools itself
KNOWN_FAILED_IMPORTS = {"inventory"}


def test_registry_has_no_unexpected_failed_imports():
    unexpected = set(FAILED_IMPORTS) - KNOWN_FAILED_IMPORTS
    assert not unexpected, (
        f"Toolkits silently dropped from the registry: {sorted(unexpected)} — "
        f"{ {k: v[:120] for k, v in FAILED_IMPORTS.items() if k in unexpected} }. "
        "The usual cause is a module-level `from elitea_sdk.tools...` import inside a "
        "document loader, which is circular when the loader is imported before the tools "
        "package: importing any elitea_sdk.tools submodule runs the toolkit registration "
        "cascade, which imports back into the half-initialized loader. Import lazily at "
        "the call site instead (see EliteAImageLoader)."
    )


def test_no_loader_imports_tools_at_module_level():
    offenders = []
    for module_info in pkgutil.iter_modules(loaders_pkg.__path__):
        module = importlib.import_module(f"{loaders_pkg.__name__}.{module_info.name}")
        source_file = getattr(module, "__file__", None)
        if not source_file:
            continue
        with open(source_file) as fh:
            for line_no, line in enumerate(fh, 1):
                stripped = line.strip()
                if not line.startswith((" ", "\t")) and stripped.startswith(
                    ("from elitea_sdk.tools", "import elitea_sdk.tools")
                ):
                    offenders.append(f"{module_info.name}:{line_no}: {stripped}")
    assert not offenders, (
        f"Module-level elitea_sdk.tools imports in document loaders (circular — corrupts "
        f"the toolkit registry when a loader is imported first): {offenders}"
    )
