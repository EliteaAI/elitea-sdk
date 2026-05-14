import os
import subprocess
import sys


def test_ado_import_sets_writable_cache_dir_when_home_is_read_only():
    env = os.environ.copy()
    env["HOME"] = "/var/www"
    env.pop("AZURE_DEVOPS_CACHE_DIR", None)

    script = """
import os
from elitea_sdk import tools

assert os.environ["AZURE_DEVOPS_CACHE_DIR"] == "/tmp/.azure-devops"
assert "ado_repos" in tools.AVAILABLE_TOOLS, tools.FAILED_IMPORTS.get("ado_repos")
assert tools.FAILED_IMPORTS.get("ado_repos") is None
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr