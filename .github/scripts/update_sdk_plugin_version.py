#!/usr/bin/env python3
"""Validate release inputs and update sdk_plugin's pinned SDK version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


SDK_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
BRANCH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
SDK_PIN_PATTERN = re.compile(r"elitea-sdk\[all\]==[0-9]+\.[0-9]+\.[0-9]+")


def validate_version(version: str) -> str:
    """Accept the numeric release format produced by the SDK publish workflow."""
    if not SDK_VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"Invalid elitea-sdk release version: {version!r}")
    return version


def validate_target_branch(branch: str) -> str:
    """Reject shell metacharacters, output injection, and invalid ref fragments."""
    if (
        not BRANCH_PATTERN.fullmatch(branch)
        or ".." in branch
        or "//" in branch
        or branch.endswith(("/", ".", ".lock"))
    ):
        raise ValueError(f"Invalid sdk_plugin target branch: {branch!r}")
    return branch


def update_plugin(plugin_dir: Path, version: str) -> None:
    """Update exactly one SDK requirement and the plugin metadata version."""
    version = validate_version(version)
    requirements_path = plugin_dir / "requirements.txt"
    metadata_path = plugin_dir / "metadata.json"

    requirements = requirements_path.read_text(encoding="utf-8")
    updated_requirements, replacement_count = SDK_PIN_PATTERN.subn(
        f"elitea-sdk[all]=={version}", requirements
    )
    if replacement_count != 1:
        raise ValueError(
            "Expected exactly one elitea-sdk[all] numeric pin in requirements.txt; "
            f"found {replacement_count}"
        )
    requirements_path.write_text(updated_requirements, encoding="utf-8")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["version"] = version
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--target-branch")
    parser.add_argument("--plugin-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = validate_version(args.version)

    if args.validate_only:
        if not args.target_branch:
            raise ValueError("--target-branch is required with --validate-only")
        branch = validate_target_branch(args.target_branch)
        print(f"sdk_version={version}")
        print(f"target_branch={branch}")
        return

    if args.plugin_dir is None:
        raise ValueError("--plugin-dir is required when updating sdk_plugin")
    update_plugin(args.plugin_dir, version)


if __name__ == "__main__":
    main()
