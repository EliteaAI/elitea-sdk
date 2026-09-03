"""Regression contracts for dependency security alerts."""

from __future__ import annotations

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_critical_and_high_dependency_pins_stay_at_patched_versions():
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    optional_dependencies = project["project"]["optional-dependencies"]
    requirements = {
        canonicalize_name(Requirement(value).name): str(Requirement(value).specifier)
        for group in ("runtime", "tools")
        for value in optional_dependencies[group]
    }

    assert requirements["unstructured"] == "==0.18.18"
    assert requirements["unstructured-inference"] == "==1.0.5"
    assert requirements["pypdf"] == "==6.16.1"
    assert requirements["gitpython"] == "==3.1.59"
    assert requirements["azure-core"] == "==1.38.0"
    assert requirements["lxml"] == "==6.1.0"
