"""
Unit tests for elitea_sdk/runtime/utils/docs/fetch_toolkit_faqs.py

Tests cover:
- TOOLKITS list completeness and expected entries
- fetch_faq(): HTTP 200 with/without FAQ section, HTTP 404, other HTTP errors, network errors
- save_faq(): happy path, I/O failure
- main(): end-to-end with mocked network, verifying saved/not-saved counts
"""

import sys
import types
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, call
import importlib

import pytest

# ---------------------------------------------------------------------------
# Helpers to import the module under test
# ---------------------------------------------------------------------------

_MODULE_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "elitea_sdk"
    / "runtime"
    / "utils"
    / "docs"
    / "fetch_toolkit_faqs.py"
)


def _import_module():
    """Import fetch_toolkit_faqs as a module regardless of package installation."""
    spec = importlib.util.spec_from_file_location("fetch_toolkit_faqs", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ftf = _import_module()


# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------

SAMPLE_MDX_WITH_FAQ = """\
## Introduction
Some content here.

## FAQ
### Q: How do I authenticate?
A: Use an API key.

### Q: What permissions are needed?
A: Read access is required.

## Next Steps
Continue here.
"""

SAMPLE_MDX_WITH_FAQS = """\
## Usage
Some usage text.

## FAQs
### Q: Can I use OAuth?
A: Yes.

## More
More content.
"""

SAMPLE_MDX_WITH_TROUBLESHOOTING = """\
## Overview
Some overview.

## Troubleshooting and Support
Check your credentials first.

## Reference
End.
"""

SAMPLE_MDX_NO_FAQ = """\
## Introduction
No FAQ here.

## Usage
Just usage info.
"""

SAMPLE_MDX_MDX_ICON_FAQ = """\
## Setup
Setup instructions.

##  <Icon icon="circle-question" size={24} /> FAQ
### Q: Icon FAQ question?
A: Icon FAQ answer.

## End
The end.
"""


def _make_http_response(content: str, status: int = 200):
    """Return a context-manager-compatible mock response."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = content.encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: TOOLKITS list
# ---------------------------------------------------------------------------

class TestToolkitsList:
    """Validate the TOOLKITS constant."""

    def test_toolkits_is_list(self):
        assert isinstance(ftf.TOOLKITS, list)

    def test_toolkits_not_empty(self):
        assert len(ftf.TOOLKITS) > 0

    def test_all_entries_are_strings(self):
        assert all(isinstance(t, str) for t in ftf.TOOLKITS)

    def test_no_duplicates(self):
        assert len(ftf.TOOLKITS) == len(set(ftf.TOOLKITS))

    # Toolkits confirmed as "✓ Saved" from the run log
    @pytest.mark.parametrize("toolkit", [
        "artifact", "ado_repos", "ado_wiki", "github", "gitlab", "gitlab_org",
        "bitbucket", "jira", "qtest", "testrail", "xray", "zephyr_scale",
        "zephyr_enterprise", "rally", "reportportal", "confluence", "sharepoint",
        "slack", "servicenow", "salesforce", "figma", "postman", "openapi",
        "sql", "powerpoint",
    ])
    def test_faq_present_toolkits_in_list(self, toolkit):
        assert toolkit in ftf.TOOLKITS

    # Toolkits confirmed as "⊘ No FAQ section found" from the run log
    @pytest.mark.parametrize("toolkit", [
        "ado_work_item", "ado_test_plan", "localgit", "advanced_jira_mining",
        "zephyr", "zephyr_squad", "zephyr_essential", "testio", "gmail",
        "yagmail", "custom_open_api", "aws", "azure_ai", "carrier",
        "keycloak", "elastic", "google", "pandas",
    ])
    def test_no_faq_toolkits_also_in_list(self, toolkit):
        assert toolkit in ftf.TOOLKITS

    def test_total_count_is_43(self):
        """Run log shows 43 total toolkits (after ocr/tesseract removal)."""
        assert len(ftf.TOOLKITS) == 43

    def test_base_url_uses_mintlify_branch(self):
        assert "mintlify" in ftf.BASE_URL

    def test_base_url_points_to_eliteaai(self):
        assert "EliteaAI" in ftf.BASE_URL


# ---------------------------------------------------------------------------
# Tests: fetch_faq()
# ---------------------------------------------------------------------------

class TestFetchFaq:
    """Tests for the fetch_faq() function."""

    def test_returns_faq_content_on_200(self):
        with patch("urllib.request.urlopen", return_value=_make_http_response(SAMPLE_MDX_WITH_FAQ)):
            result = ftf.fetch_faq("github")
        assert result is not None
        assert "How do I authenticate" in result

    def test_returns_faq_content_for_faqs_heading(self):
        with patch("urllib.request.urlopen", return_value=_make_http_response(SAMPLE_MDX_WITH_FAQS)):
            result = ftf.fetch_faq("jira")
        assert result is not None
        assert "OAuth" in result

    def test_returns_faq_content_for_troubleshooting_heading(self):
        with patch("urllib.request.urlopen", return_value=_make_http_response(SAMPLE_MDX_WITH_TROUBLESHOOTING)):
            result = ftf.fetch_faq("confluence")
        assert result is not None
        assert "credentials" in result

    def test_returns_none_when_no_faq_section(self):
        with patch("urllib.request.urlopen", return_value=_make_http_response(SAMPLE_MDX_NO_FAQ)):
            result = ftf.fetch_faq("localgit")
        assert result is None

    def test_returns_none_on_non_200_status(self):
        with patch("urllib.request.urlopen", return_value=_make_http_response("", status=500)):
            result = ftf.fetch_faq("github")
        assert result is None

    def test_returns_none_on_404(self):
        http_err = urllib.error.HTTPError(
            url="http://example.com", code=404, msg="Not Found", hdrs={}, fp=None
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            result = ftf.fetch_faq("nonexistent_toolkit")
        assert result is None

    def test_prints_http_error_code_on_non_404(self, capsys):
        http_err = urllib.error.HTTPError(
            url="http://example.com", code=503, msg="Service Unavailable", hdrs={}, fp=None
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            result = ftf.fetch_faq("github")
        assert result is None
        captured = capsys.readouterr()
        assert "503" in captured.out

    def test_returns_none_on_general_exception(self, capsys):
        with patch("urllib.request.urlopen", side_effect=Exception("connection timeout")):
            result = ftf.fetch_faq("jira")
        assert result is None
        captured = capsys.readouterr()
        assert "connection timeout" in captured.out

    def test_url_construction_uses_toolkit_name(self):
        """Ensure the URL built for a toolkit uses <name>_toolkit.mdx."""
        with patch("urllib.request.urlopen", return_value=_make_http_response(SAMPLE_MDX_NO_FAQ)) as mock_open_url:
            ftf.fetch_faq("testrail")
        called_url = mock_open_url.call_args[0][0]
        assert "testrail_toolkit.mdx" in called_url

    def test_faq_content_stripped_of_whitespace(self):
        content_with_whitespace = "\n## FAQ\n\n   Some answer.   \n\n## Next\nEnd."
        with patch("urllib.request.urlopen", return_value=_make_http_response(content_with_whitespace)):
            result = ftf.fetch_faq("github")
        assert result == "Some answer."

    def test_mdx_icon_faq_heading_parsed(self):
        with patch("urllib.request.urlopen", return_value=_make_http_response(SAMPLE_MDX_MDX_ICON_FAQ)):
            result = ftf.fetch_faq("github")
        assert result is not None
        assert "Icon FAQ" in result

    # Parametrize across toolkits confirmed as "✓ Saved" in the run log
    @pytest.mark.parametrize("toolkit", [
        "artifact", "ado_repos", "ado_wiki", "github", "gitlab", "gitlab_org",
        "bitbucket", "jira", "qtest", "testrail", "xray", "zephyr_scale",
        "zephyr_enterprise", "rally", "reportportal", "confluence", "sharepoint",
        "slack", "servicenow", "salesforce", "figma", "postman", "openapi",
        "sql", "powerpoint",
    ])
    def test_faq_returned_for_toolkits_with_faq(self, toolkit):
        """
        For every toolkit confirmed to have a FAQ section in the run log,
        fetch_faq() must return non-None content when the remote document
        contains a '## FAQ' heading.
        """
        mock_content = (
            "## Intro\nSome intro.\n\n"
            "## FAQ\nSome FAQ content here.\n\n"
            "## End\nDone.\n"
        )
        with patch("urllib.request.urlopen", return_value=_make_http_response(mock_content)):
            result = ftf.fetch_faq(toolkit)
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: save_faq()
# ---------------------------------------------------------------------------

class TestSaveFaq:
    """Tests for the save_faq() function."""

    def test_saves_file_with_correct_name(self, tmp_path):
        ftf.save_faq("github", "FAQ content here", tmp_path)
        assert (tmp_path / "github.md").exists()

    def test_saved_file_contains_title(self, tmp_path):
        ftf.save_faq("github", "FAQ content here", tmp_path)
        content = (tmp_path / "github.md").read_text(encoding="utf-8")
        assert "# Github Toolkit FAQ" in content

    def test_saved_file_contains_official_doc_link(self, tmp_path):
        ftf.save_faq("jira", "FAQ content here", tmp_path)
        content = (tmp_path / "jira.md").read_text(encoding="utf-8")
        assert "jira_toolkit.md" in content
        assert "Official Documentation" in content

    def test_saved_file_contains_faq_content(self, tmp_path):
        faq_body = "### Q: How to connect?\nA: Via API key."
        ftf.save_faq("confluence", faq_body, tmp_path)
        content = (tmp_path / "confluence.md").read_text(encoding="utf-8")
        assert faq_body in content

    def test_saved_file_has_separator(self, tmp_path):
        ftf.save_faq("slack", "Some FAQ", tmp_path)
        content = (tmp_path / "slack.md").read_text(encoding="utf-8")
        assert "---" in content

    def test_display_name_replaces_underscores(self, tmp_path):
        ftf.save_faq("zephyr_scale", "FAQ body", tmp_path)
        content = (tmp_path / "zephyr_scale.md").read_text(encoding="utf-8")
        assert "Zephyr Scale" in content

    def test_returns_true_on_success(self, tmp_path):
        result = ftf.save_faq("github", "FAQ content", tmp_path)
        assert result is True

    def test_returns_false_on_io_error(self, tmp_path, capsys):
        """Simulate a write failure."""
        bad_path = tmp_path / "nonexistent_dir"
        # bad_path directory does not exist, so open will fail
        result = ftf.save_faq("github", "FAQ content", bad_path)
        assert result is False
        captured = capsys.readouterr()
        assert "Error saving" in captured.out

    def test_github_doc_url_in_file(self, tmp_path):
        ftf.save_faq("testrail", "Some content", tmp_path)
        content = (tmp_path / "testrail.md").read_text(encoding="utf-8")
        assert "EliteaAI/elitea.github.io" in content

    def test_powerpoint_display_name(self, tmp_path):
        ftf.save_faq("powerpoint", "FAQ body", tmp_path)
        content = (tmp_path / "powerpoint.md").read_text(encoding="utf-8")
        assert "# Powerpoint Toolkit FAQ" in content

    def test_reportportal_display_name(self, tmp_path):
        ftf.save_faq("reportportal", "FAQ body", tmp_path)
        content = (tmp_path / "reportportal.md").read_text(encoding="utf-8")
        assert "# Reportportal Toolkit FAQ" in content


# ---------------------------------------------------------------------------
# Tests: main()
# ---------------------------------------------------------------------------

class TestMain:
    """End-to-end tests for the main() function."""

    def _make_fetch_side_effect(self, toolkits_with_faq, faq_text="FAQ content"):
        """Return a side_effect function for urllib.request.urlopen."""
        def side_effect(url, timeout=10):
            # Derive toolkit name from URL: .../NAME_toolkit.mdx
            name = url.rsplit("/", 1)[-1].replace("_toolkit.mdx", "")
            if name in toolkits_with_faq:
                content = f"## Intro\nText.\n\n## FAQ\n{faq_text}\n\n## End\nDone."
            else:
                content = "## Intro\nNo FAQ here."
            return _make_http_response(content)
        return side_effect

    def test_main_returns_zero_when_no_save_failures(self, tmp_path):
        """main() should return 0 if all saves succeed (failed_count == 0)."""
        with patch("urllib.request.urlopen", side_effect=self._make_fetch_side_effect(set())):
            result = _run_main_with_tmpdir(tmp_path)
        assert result == 0

    def test_main_creates_faq_dir(self, tmp_path):
        faq_dir = _faq_dir_for(tmp_path)
        assert not faq_dir.exists()
        _run_main_with_tmpdir(tmp_path)
        assert faq_dir.exists()

    def test_main_saves_toolkits_with_faq(self, tmp_path):
        toolkits_with_faq = {"github", "jira", "confluence"}
        with patch("urllib.request.urlopen",
                   side_effect=self._make_fetch_side_effect(toolkits_with_faq)):
            _run_main_with_tmpdir(tmp_path)
        faq_dir = _faq_dir_for(tmp_path)
        for tk in toolkits_with_faq:
            assert (faq_dir / f"{tk}.md").exists(), f"{tk}.md should be saved"

    def test_main_does_not_save_toolkits_without_faq(self, tmp_path):
        toolkits_with_faq = {"github"}
        no_faq = {"jira", "confluence"}
        with patch("urllib.request.urlopen",
                   side_effect=self._make_fetch_side_effect(toolkits_with_faq)):
            _run_main_with_tmpdir(tmp_path)
        faq_dir = _faq_dir_for(tmp_path)
        for tk in no_faq:
            assert not (faq_dir / f"{tk}.md").exists(), f"{tk}.md should NOT be saved"

    def test_main_prints_summary(self, tmp_path, capsys):
        with patch("urllib.request.urlopen",
                   side_effect=self._make_fetch_side_effect(set())):
            _run_main_with_tmpdir(tmp_path)
        captured = capsys.readouterr()
        assert "Summary" in captured.out
        assert "Total processed" in captured.out

    def test_main_prints_saved_count(self, tmp_path, capsys):
        toolkits_with_faq = {"github", "jira"}
        with patch("urllib.request.urlopen",
                   side_effect=self._make_fetch_side_effect(toolkits_with_faq)):
            _run_main_with_tmpdir(tmp_path)
        captured = capsys.readouterr()
        assert "Successfully saved" in captured.out

    def test_main_run_log_scenario(self, tmp_path, capsys):
        """
        Simulate the exact run-log scenario: 25 toolkits with FAQ, 19 without.
        Verify summary output reflects the split.
        """
        toolkits_with_faq = {
            "artifact", "ado_repos", "ado_wiki", "github", "gitlab", "gitlab_org",
            "bitbucket", "jira", "qtest", "testrail", "xray", "zephyr_scale",
            "zephyr_enterprise", "rally", "reportportal", "confluence", "sharepoint",
            "slack", "servicenow", "salesforce", "figma", "postman", "openapi",
            "sql", "powerpoint",
        }
        with patch("urllib.request.urlopen",
                   side_effect=self._make_fetch_side_effect(toolkits_with_faq)):
            result = _run_main_with_tmpdir(tmp_path)

        assert result == 0  # no failures
        faq_dir = _faq_dir_for(tmp_path)
        saved_files = list(faq_dir.glob("*.md"))
        assert len(saved_files) == 25

    def test_main_returns_one_on_save_failure(self, tmp_path, capsys):
        """main() returns 1 when at least one save fails."""
        toolkits_with_faq = {"github"}

        def bad_save(toolkit_name, faq_content, output_dir):
            print(f"    Error saving: forced failure")
            return False

        with patch("urllib.request.urlopen",
                   side_effect=self._make_fetch_side_effect(toolkits_with_faq)):
            with patch.object(ftf, "save_faq", side_effect=bad_save):
                result = _run_main_with_tmpdir(tmp_path)

        assert result == 1


# ---------------------------------------------------------------------------
# Helpers: run main() redirected to a temp directory
# ---------------------------------------------------------------------------

def _faq_dir_for(tmp_path: Path) -> Path:
    """
    Compute where main() will write FAQ files given the fake __file__ path.

    main() computes:
        script_dir = Path(__file__).parent           # docs/
        sdk_root   = script_dir.parent.parent.parent  # elitea_sdk/
        faq_dir    = sdk_root / 'docs' / 'faq'

    With fake_file = tmp_path / elitea_sdk/runtime/utils/docs/fetch_toolkit_faqs.py:
        script_dir = tmp_path / elitea_sdk/runtime/utils/docs
        sdk_root   = tmp_path / elitea_sdk
        faq_dir    = tmp_path / elitea_sdk/docs/faq
    """
    return tmp_path / "elitea_sdk" / "docs" / "faq"


def _run_main_with_tmpdir(tmp_path: Path) -> int:
    """
    Execute ftf.main() but redirect its faq_dir to a subdirectory of tmp_path
    by temporarily overriding the module's __file__ attribute so that the
    Path(__file__) resolution inside main() points into tmp_path.

    main() does:
        script_dir = Path(__file__).parent           # .../docs/
        sdk_root   = script_dir.parent.parent.parent  # .../elitea_sdk/
        faq_dir    = sdk_root / 'docs' / 'faq'

    We set __file__ to:
        tmp_path / "elitea_sdk" / "runtime" / "utils" / "docs" / "fetch_toolkit_faqs.py"
    so sdk_root becomes tmp_path / "elitea_sdk" and
    faq_dir becomes tmp_path / "elitea_sdk" / "docs" / "faq".
    """
    fake_file = str(
        tmp_path / "elitea_sdk" / "runtime" / "utils" / "docs" / "fetch_toolkit_faqs.py"
    )
    orig_file_attr = ftf.__file__
    ftf.__file__ = fake_file
    try:
        return ftf.main()
    finally:
        ftf.__file__ = orig_file_attr


