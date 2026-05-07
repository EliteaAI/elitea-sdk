# Pipeline Runner Skill

Quick reference for executing pipelines on Elitea backend and getting raw API responses.

## Prerequisites

**ALWAYS activate the project virtual environment first:**

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (Git Bash)
source .venv/Scripts/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

## Quick Start

```bash
# 1. Activate virtual environment (see Prerequisites above)

# 2. Set up environment variables in .env
TEST_DEPLOYMENT_URL=https://dev.elitea.ai
TEST_API_KEY=your_api_key_here
TEST_PROJECT_ID=123

# 3. Run pipeline (using env vars)
python .github/skills/pipeline-runner/run_pipeline.py 2259

# Or override with CLI arguments
python .github/skills/pipeline-runner/run_pipeline.py 2259 --url https://dev.elitea.ai --project-id 123

# 4. If you see ModuleNotFoundError, install dependencies:
pip install requests python-dotenv
# Then re-run the command
```

## Environment Variables

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `TEST_DEPLOYMENT_URL` | Backend URL | `https://dev.elitea.ai` | Yes |
| `TEST_API_KEY` | Bearer token | `your_api_key` | Yes |
| `TEST_PROJECT_ID` | Project ID | `123` | Yes |

## Script Behavior

**Prerequisites:** Virtual environment must be activated.

1. Parses command-line arguments
2. Loads environment variables from `.env` (CLI args override env vars)
3. Fetches pipeline details (by ID or name)
4. Executes POST request to `/api/v2/elitea_core/predict/prompt_lib/{project_id}/{version_id}`
5. Returns raw API response without processing
6. Reports success or failure

**Note:** If the script fails with `ModuleNotFoundError`, run `pip install requests python-dotenv`, then re-run.

## Command-Line Arguments

All environment variables can be overridden via CLI:

```bash
python .github/skills/pipeline-runner/run_pipeline.py <pipeline_id> [options]

Options:
  pipeline_id           Pipeline ID (positional)
  --name, -n NAME       Pipeline name (alternative to ID)
  --url URL             Backend URL (overrides TEST_DEPLOYMENT_URL)
  --api-key KEY         API Bearer token (overrides TEST_API_KEY)
  --project-id ID       Project ID (overrides TEST_PROJECT_ID)
  --input, -i INPUT     Input message (default: "execute")
  --timeout, -t SEC     Request timeout in seconds (default: 120)
  --json, -j            Output JSON format
  --verbose, -v         Verbose output
  -h, --help            Show help message
```

## Exit Codes

- `0` - Success (HTTP 200/201)
- `1` - Failure (HTTP error, timeout, or pipeline not found)

## API Endpoint

```
POST /api/v2/elitea_core/predict/prompt_lib/{project_id}/{version_id}

Payload:
{
  "chat_history": [],
  "user_input": "<input_message>"
}
```

## Dependencies

```bash
pip install requests python-dotenv
```

## Example Output

```
============================================================
Pipeline: SharePoint Change Tracker (ID: 28)
Status Code: 200
Success: True
Execution Time: 5.42s

Raw Response:
{
  "chat_history": [
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "tool_calls_dict": {...},
  ...
}
============================================================
```

## Troubleshooting

**ModuleNotFoundError: No module named 'requests' (or similar)?**
- Ensure virtual environment is activated (see Prerequisites)
- Run `pip install requests python-dotenv`
- Re-run the command

**Missing environment variables?**
- Create `.env` file in project root
- Or export variables: `export TEST_API_KEY=...`

**Pipeline not found?**
- Check pipeline ID is correct
- Verify `TEST_PROJECT_ID` matches the project containing the pipeline
- Use exact pipeline name (case-sensitive) with `--name`

**401 Unauthorized?**
- Check your `TEST_API_KEY` is valid
- Verify token hasn't expired

**404 Not Found?**
- Verify `TEST_PROJECT_ID` is correct
- Ensure the pipeline exists and has at least one version

**Request timed out?**
- Increase timeout: `--timeout 300`
- Check backend is reachable

## Differences from Test Runner

This simplified runner:
- **Returns raw API response** without processing
- **No test result extraction** (no `test_passed` field)
- **No HITL handling** (returns interrupt response as-is)
- **Exact name matching** (case-sensitive, no fuzzy matching)
- **Uses TEST_* env vars** (matches pipeline-deployer conventions)

For advanced test execution with result validation, see the full test runner at `.elitea/tests/test_pipelines/scripts/run_pipeline.py`.
