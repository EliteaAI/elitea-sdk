# SharePoint Teams Site Path Test Suite

## Overview

Integration tests for SharePoint `site_path` feature supporting `/teams/` spaces.

**Feature:** [Issue #2584](https://github.com/EliteaAI/elitea_issues/issues/2584) - Add Support for Additional SharePoint Spaces

## Test Cases

| ID | Name | Description | Priority |
|----|------|-------------|----------|
| SPT01 | get_files_list | List files from teams subfolder | Critical |
| SPT02 | read_document | Read document content from teams site | Critical |
| SPT03 | get_files_list_root | List files from Shared Documents root | High |
| SPT04 | read_document_nonexistent | Negative test - non-existent file error handling | High |
| SPT05 | get_files_list_limit | Validate limit_files parameter | High |
| SPT06 | get_files_list_all | List all files (no folder filter) | High |
| SPT07 | get_files_include_extensions | Filter by include_extensions | High |
| SPT08 | get_files_skip_extensions | Filter by skip_extensions | High |
| SPT09 | get_files_form_name | Filter by document library (form_name) | High |
| SPT10 | get_files_combined_filters | Combined include + skip extension filters | Critical |
| SPT11 | get_files_include_exclude_conflict | Edge case: include/skip conflict resolution | High |

## Prerequisites

### Environment Variables

```bash
# Required in .env
SHAREPOINT_CLIENT_SECRET=<your_client_secret>

# Optional (have defaults)
SHAREPOINT_TENANT_URL=https://5clkvm.sharepoint.com
SHAREPOINT_CLIENT_ID=84c01a1e-1ebc-40c0-b55a-8b3fae194e41
SHAREPOINT_TEAMS_SITE_PATH=teams/epam_3
SHAREPOINT_TEAMS_TEST_FOLDER=DO_NOT_DELETE_TeamsTestFiles
SHAREPOINT_TEAMS_TEST_FILE=test-document.txt
SHAREPOINT_TEAMS_DOCUMENT_PATH=/teams/epam_3/Shared Documents/DO_NOT_DELETE_TeamsTestFiles
```

### Test Data Setup

1. Create teams site: `https://<tenant>.sharepoint.com/teams/<name>`
2. Create test folder: `Shared Documents/DO_NOT_DELETE_TeamsTestFiles`
3. Upload test files:
   - `test-document.txt` - Text file for read_document tests
   - `test-data.xlsx` - Excel file for extension filter tests

## Running Tests

```bash
cd .elitea/tests/test_pipelines

# Run single test
./run_test.sh --setup --local -v suites/sharepoint_teams SPT01

# Run multiple tests
./run_test.sh --setup --local -v suites/sharepoint_teams SPT01,SPT02,SPT03

# Run all tests in suite
./run_test.sh --setup --local -v suites/sharepoint_teams

# Run on remote (seed mode)
./run_test.sh --setup --seed -v suites/sharepoint_teams SPT01
```

## Feature Details

The `site_path` field allows specifying SharePoint site path separately from the credential URL:

```yaml
sharepoint_configuration:
  site_url: https://tenant.sharepoint.com  # Tenant URL only
site_path: teams/epam_3                     # Resolves to /teams/epam_3
```

Supported formats:
- `sites/site-name` → `/sites/site-name`
- `teams/team-name` → `/teams/team-name`

## Test Validation

All tests use strict validation:
- Non-empty results required (except negative tests)
- Required fields checked: Name, Path, Created, Modified, Link
- Paths must contain `/teams/` substring
- Extension filtering verified for include/skip tests
