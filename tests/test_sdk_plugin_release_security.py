"""Security contracts for the privileged sdk_plugin release automation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / ".github/scripts/update_sdk_plugin_version.py"
WORKFLOW_PATH = REPOSITORY_ROOT / ".github/workflows/update-sdk-plugin.yml"


def _load_release_script():
    spec = importlib.util.spec_from_file_location("update_sdk_plugin_version", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_privileged_update_workflow_has_no_workflow_run_trigger():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "workflow_run:" not in workflow
    assert "github.event.workflow_run" not in workflow
    assert "Checkout trusted SDK automation" in workflow
    assert "persist-credentials: false" in workflow


def test_update_plugin_changes_only_expected_version_fields(tmp_path: Path):
    release_script = _load_release_script()
    plugin_dir = tmp_path / "sdk_plugin"
    plugin_dir.mkdir()
    requirements = "first==1\nelitea-sdk[all]==0.9.53\nlast==2\n"
    metadata = {"name": "sdk_plugin", "version": "0.9.53", "enabled": True}
    (plugin_dir / "requirements.txt").write_text(requirements, encoding="utf-8")
    (plugin_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    release_script.update_plugin(plugin_dir, "0.9.54")

    assert (plugin_dir / "requirements.txt").read_text(encoding="utf-8") == (
        "first==1\nelitea-sdk[all]==0.9.54\nlast==2\n"
    )
    assert json.loads((plugin_dir / "metadata.json").read_text(encoding="utf-8")) == {
        "name": "sdk_plugin",
        "version": "0.9.54",
        "enabled": True,
    }


@pytest.mark.parametrize(
    "value",
    ["1.2", "1.2.3; echo injected", "1.2.3\nother=value", "v1.2.3"],
)
def test_release_version_validation_rejects_non_numeric_versions(value: str):
    with pytest.raises(ValueError, match="Invalid elitea-sdk release version"):
        _load_release_script().validate_version(value)


@pytest.mark.parametrize(
    "value",
    ["-option", "main\nother=value", "../main", "feature//unsafe", "main.lock"],
)
def test_target_branch_validation_rejects_unsafe_refs(value: str):
    with pytest.raises(ValueError, match="Invalid sdk_plugin target branch"):
        _load_release_script().validate_target_branch(value)


def test_update_plugin_fails_closed_when_sdk_pin_is_ambiguous(tmp_path: Path):
    release_script = _load_release_script()
    plugin_dir = tmp_path / "sdk_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "requirements.txt").write_text(
        "elitea-sdk[all]==0.9.52\nelitea-sdk[all]==0.9.53\n",
        encoding="utf-8",
    )
    (plugin_dir / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected exactly one"):
        release_script.update_plugin(plugin_dir, "0.9.54")
