#!/usr/bin/env python3
"""
CI Cleanup Script - Remove orphaned test data from Elitea platform.

Removes:
  - Pipelines with tag "automation" AND name matching pattern (default: starts with 8 hex chars + underscore)
  - Toolkits whose name matches pattern (default: ends with "-testing-{8 hex chars}")

Supports age-based filtering to only delete entities older than N days.

Usage:
    python ci_cleanup.py [options]

Examples:
    python ci_cleanup.py --dry-run
    python ci_cleanup.py --older-than-days 7 --yes
    python ci_cleanup.py --base-url https://dev.elitea.ai --project-id 123 --dry-run
    python ci_cleanup.py --pipelines-only --yes
    python ci_cleanup.py --pipeline-pattern "^test_" --toolkit-pattern ".*-testing$" --dry-run

Environment Variables:
    DEPLOYMENT_URL or BASE_URL            - Platform base URL
    PROJECT_ID                            - Project ID
    AUTH_TOKEN, ELITEA_TOKEN, or API_KEY  - Bearer token
"""

import argparse
import re
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from utils_common import load_token_from_env, load_base_url_from_env, load_project_id_from_env


# ---------------------------------------------------------------------------
# Generic paginated list + delete helpers
# ---------------------------------------------------------------------------

def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _paginate(url: str, token: str, batch: int = 200, extra_params: dict = None) -> list:
    """Fetch all rows from a paginated Elitea list endpoint."""
    rows, offset = [], 0
    params = {**(extra_params or {}), "limit": batch}
    while True:
        params["offset"] = offset
        resp = requests.get(url, params=params, headers=_auth(token))
        if resp.status_code != 200:
            print(f"  ERROR listing {url} (HTTP {resp.status_code}): {resp.text[:200]}", file=sys.stderr)
            break
        page = resp.json().get("rows", [])
        rows.extend(page)
        if len(page) < batch:
            break
        offset += batch
    return rows


def _run_cleanup(label: str, items: list, delete_url_fn, token: str, dry_run: bool, verbose: bool = False) -> dict:
    """Generic delete loop shared by pipelines and toolkits."""
    deleted = failed = 0
    for item in items:
        iid  = item["id"]
        name = item.get("name", f"<id:{iid}>")
        if dry_run:
            print(f"  [DRY RUN] Would delete {label}: '{name}' (ID: {iid})")
            continue
        ok = requests.delete(delete_url_fn(item), headers=_auth(token)).status_code == 204
        if ok:
            if verbose:
                print(f"  Deleted {label}: '{name}' (ID: {iid})")
            deleted += 1
        else:
            print(f"  FAILED  {label}: '{name}' (ID: {iid})", file=sys.stderr)
            failed += 1
    return {"deleted": deleted, "failed": failed}


# ---------------------------------------------------------------------------
# Age-based filtering
# ---------------------------------------------------------------------------

def is_older_than(entity: dict, days: int) -> bool:
    """Check if entity was created more than N days ago."""
    if days <= 0:
        return True  # No age filter

    created_at = entity.get("created_at")
    if not created_at:
        return True  # No timestamp = assume old, safe to delete

    try:
        # Handle ISO format with or without timezone
        if created_at.endswith('Z'):
            created_at = created_at[:-1] + '+00:00'
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        threshold = datetime.now(timezone.utc) - timedelta(days=days)
        return created < threshold
    except (ValueError, TypeError):
        return True  # Parse error = assume old


# ---------------------------------------------------------------------------
# Pipeline cleanup
# ---------------------------------------------------------------------------

# Default session ID pattern: 8 hex characters (e.g., "a1b2c3d4")
DEFAULT_SESSION_ID_PATTERN = r'[a-f0-9]{8}'

# Default patterns for matching test entities
DEFAULT_PIPELINE_PATTERN = rf'^{DEFAULT_SESSION_ID_PATTERN}_'  # e.g., "a1b2c3d4_My Pipeline"
DEFAULT_TOOLKIT_PATTERN = rf'-testing-{DEFAULT_SESSION_ID_PATTERN}$'  # e.g., "github-testing-a1b2c3d4"


def _has_tag(pipeline: dict, tag: str) -> bool:
    """Check if pipeline has the specified tag."""
    tag_lower = tag.lower()
    for t in pipeline.get("tags", []):
        if isinstance(t, dict) and str(t.get("name", "")).lower() == tag_lower:
            return True
        if isinstance(t, str) and t.lower() == tag_lower:
            return True
    return False


def matches_pipeline(pipeline: dict, tag: str, pattern: str) -> bool:
    """
    Check if pipeline matches cleanup criteria.

    Criteria:
    - If tag is provided, must have that tag (e.g., "automation")
    - Name must match the provided regex pattern
    """
    # If tag is specified, check it
    if tag and not _has_tag(pipeline, tag):
        return False

    # Match by name pattern
    name = pipeline.get("name", "")
    return bool(re.search(pattern, name))


def cleanup_pipelines(base_url, project_id, token, tag, dry_run, verbose=False,
                      older_than_days: int = 0, pattern: str = None) -> dict:
    """Clean up pipelines matching criteria."""
    pattern = pattern or DEFAULT_PIPELINE_PATTERN
    tag_info = f"tag='{tag}'" if tag else "tag=(none)"
    print(f"\n[Pipelines] {tag_info}, pattern='{pattern}', older_than={older_than_days} days")

    all_items = _paginate(
        f"{base_url}/api/v2/elitea_core/applications/prompt_lib/{project_id}",
        token,
        extra_params={"agents_type": "pipeline", "sort_by": "created_at", "sort_order": "desc"},
    )

    # Filter by tag and pattern
    matching = [p for p in all_items if matches_pipeline(p, tag, pattern)]

    # Filter by age
    if older_than_days > 0:
        matching = [p for p in matching if is_older_than(p, older_than_days)]

    print(f"  Total: {len(all_items)}  |  Matching: {len(matching)}")

    return _run_cleanup(
        "pipeline", matching,
        lambda p: f"{base_url}/api/v2/elitea_core/application/prompt_lib/{project_id}/{p['id']}",
        token, dry_run, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Toolkit cleanup
# ---------------------------------------------------------------------------

def matches_toolkit(toolkit: dict, pattern: str) -> bool:
    """
    Check if toolkit matches cleanup criteria.

    Criteria:
    - Name must match the provided regex pattern
    """
    name = toolkit.get("name", "")
    return bool(re.search(pattern, name, re.IGNORECASE))


def cleanup_toolkits(base_url, project_id, token, dry_run, verbose=False,
                     older_than_days: int = 0, pattern: str = None) -> dict:
    """Clean up toolkits matching criteria."""
    pattern = pattern or DEFAULT_TOOLKIT_PATTERN
    print(f"\n[Toolkits] pattern='{pattern}', older_than={older_than_days} days")

    all_items = _paginate(
        f"{base_url}/api/v2/elitea_core/tools/prompt_lib/{project_id}",
        token, batch=500,
    )

    # Filter by pattern
    matching = [t for t in all_items if matches_toolkit(t, pattern)]

    # Filter by age
    if older_than_days > 0:
        matching = [t for t in matching if is_older_than(t, older_than_days)]

    print(f"  Total: {len(all_items)}  |  Matching: {len(matching)}")

    return _run_cleanup(
        "toolkit", matching,
        lambda t: f"{base_url}/api/v2/elitea_core/tool/prompt_lib/{project_id}/{t['id']}",
        token, dry_run, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="CI cleanup: remove orphaned test pipelines and toolkits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run                                    # Use env vars for connection
  %(prog)s --older-than-days 7 --yes                    # Delete entities older than 7 days
  %(prog)s --base-url https://dev.elitea.ai --project-id 123 --dry-run
  %(prog)s --pipeline-pattern "^test_" --dry-run        # Custom pipeline pattern

Default Pattern Matching:
  Pipelines: tag "automation" + name matches "^[a-f0-9]{8}_" (session prefix)
  Toolkits:  name matches "-testing-[a-f0-9]{8}$" (session suffix)

Environment Variables:
  DEPLOYMENT_URL / BASE_URL   - Platform base URL
  PROJECT_ID                  - Project ID
  AUTH_TOKEN / API_KEY        - Bearer token
        """,
    )

    # Connection arguments (use env vars as defaults)
    parser.add_argument("--base-url", default=None,
                        help="Platform base URL (default: from DEPLOYMENT_URL or BASE_URL env var)")
    parser.add_argument("--project-id", default=None, type=int,
                        help="Project ID (default: from PROJECT_ID env var)")

    # Scope control
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--pipelines-only", action="store_true", help="Skip toolkit cleanup")
    scope.add_argument("--toolkits-only",  action="store_true", help="Skip pipeline cleanup")

    # Filtering options
    parser.add_argument("--older-than-days", type=int, default=0, metavar="DAYS",
                        help="Only delete entities older than N days (default: 0 = no age filter)")
    parser.add_argument("--pipeline-tag", default="", metavar="TAG",
                        help="Pipeline tag to match (default: empty = no tag filter)")
    parser.add_argument("--pipeline-pattern", default=None, metavar="REGEX",
                        help=f"Regex pattern for pipeline names (default: {DEFAULT_PIPELINE_PATTERN})")
    parser.add_argument("--toolkit-pattern", default=None, metavar="REGEX",
                        help=f"Regex pattern for toolkit names (default: {DEFAULT_TOOLKIT_PATTERN})")

    # Authentication and execution
    parser.add_argument("--token", default=None, help="Bearer token / API key (or set AUTH_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true", help="List targets without deleting")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation (required for CI)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Resolve connection parameters from args or environment
    base_url = args.base_url or load_base_url_from_env()
    project_id = args.project_id or load_project_id_from_env()
    token = args.token or load_token_from_env()

    if not base_url:
        print("ERROR: No base URL. Set DEPLOYMENT_URL / BASE_URL env var or pass --base-url.", file=sys.stderr)
        return 1

    if not project_id:
        print("ERROR: No project ID. Set PROJECT_ID env var or pass --project-id.", file=sys.stderr)
        return 1

    if not token:
        print("ERROR: No token. Set AUTH_TOKEN / ELITEA_TOKEN / API_KEY env var or pass --token.", file=sys.stderr)
        return 1

    base_url = base_url.rstrip('/')

    # Resolve patterns
    pipeline_pattern = args.pipeline_pattern or DEFAULT_PIPELINE_PATTERN
    toolkit_pattern = args.toolkit_pattern or DEFAULT_TOOLKIT_PATTERN

    print("=" * 60)
    print("  Elitea CI Cleanup - Orphaned Entity Removal")
    print("=" * 60)
    print(f"  URL              : {base_url}")
    print(f"  Project          : {project_id}")
    print(f"  Older than       : {args.older_than_days} days" if args.older_than_days > 0 else "  Older than       : (no age filter)")
    if not args.toolkits_only:
        if args.pipeline_tag:
            print(f"  Pipeline tag     : '{args.pipeline_tag}'")
        else:
            print(f"  Pipeline tag     : (no tag filter)")
        print(f"  Pipeline pattern : '{pipeline_pattern}'")
    if not args.pipelines_only:
        print(f"  Toolkit pattern  : '{toolkit_pattern}'")
    if args.dry_run:
        print("  Mode         : DRY RUN")
    print("=" * 60)

    if not args.dry_run and not args.yes:
        if input("\nProceed with deletion? [y/N]: ").strip().lower() != "y":
            print("Aborted.")
            return 0

    total_deleted = total_failed = 0

    if not args.toolkits_only:
        s = cleanup_pipelines(
            base_url, project_id, token, args.pipeline_tag, args.dry_run,
            verbose=args.verbose,
            older_than_days=args.older_than_days,
            pattern=pipeline_pattern,
        )
        total_deleted += s["deleted"]
        total_failed += s["failed"]

    if not args.pipelines_only:
        s = cleanup_toolkits(
            base_url, project_id, token, args.dry_run,
            verbose=args.verbose,
            older_than_days=args.older_than_days,
            pattern=toolkit_pattern,
        )
        total_deleted += s["deleted"]
        total_failed += s["failed"]

    print("\n" + "=" * 60)
    if args.dry_run:
        print("  DRY RUN complete – no changes made.")
    else:
        print(f"  Done: {total_deleted} deleted, {total_failed} failed.")
    print("=" * 60)

    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

