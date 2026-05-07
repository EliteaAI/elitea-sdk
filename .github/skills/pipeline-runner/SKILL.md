---
name: pipeline-runner
description: Execute a pipeline predict request on the Elitea backend and return raw API response. Use when user wants to run/test/trigger a pipeline and see the raw output without processing.
---

# Pipeline Runner

**Purpose**: Execute a deployed pipeline on the Elitea backend via predict API and return raw response without processing.

## When to Use This Skill

Use this skill when the user wants to:
- Run or execute a deployed pipeline
- Trigger a pipeline by ID or name and get the raw API response
- Test a pipeline and see the unprocessed output
- Debug pipeline execution by examining the raw response data

## Workflow

1. **Gather Information**
   - Pipeline ID **or** pipeline name (one is required)
   - Optional input message (defaults to `"execute"`)
   - Optional timeout (default: 300s)
   - Environment variables (or prompt user to provide)

2. **Validate Environment Variables**
   Required:
   - `TEST_DEPLOYMENT_URL` - Backend URL (e.g., `https://dev.elitea.ai`)
   - `TEST_API_KEY` - Bearer token for authentication
   - `TEST_PROJECT_ID` - Project ID (numeric)

   > **Note:** The `.env` file at the project root is loaded automatically. Uses TEST_* prefix to match the pipeline-deployer skill.

3. **Activate the Virtual Environment**
   See [Prerequisites](#prerequisites) below.

4. **Execute the Runner Script**
   ```bash
   python .github/skills/pipeline-runner/run_pipeline.py <pipeline_id_or_options>
   ```

5. **Report Results**
   Show the user:
   - Pipeline name and ID
   - HTTP status code
   - Execution success/failure
   - Execution time
   - Raw API response

---

## Prerequisites

**ALWAYS activate the project virtual environment before running any Python scripts:**

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (Git Bash)
source .venv/Scripts/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

> The script imports local modules (`pattern_matcher`, `utils_common`, `logger`) that live in the same `scripts/` directory. Always `cd` into the scripts directory before running.

---

## Script Usage

The runner script is located at:
```
.github/skills/pipeline-runner/run_pipeline.py
```

**Run from anywhere in the project:**

```bash
# Execute by pipeline ID
python .github/skills/pipeline-runner/run_pipeline.py 2259

# Execute by pipeline name (exact match)
python .github/skills/pipeline-runner/run_pipeline.py --name "My Pipeline"

# With custom input message
python .github/skills/pipeline-runner/run_pipeline.py 2259 --input '{"id": 9}'

# With custom timeout (seconds)
python .github/skills/pipeline-runner/run_pipeline.py 2259 --timeout 180

# JSON output (machine-readable)
python .github/skills/pipeline-runner/run_pipeline.py 2259 --json

# Verbose output (shows HTTP requests, debug info)
python .github/skills/pipeline-runner/run_pipeline.py 2259 --verbose

# Override backend URL and project ID from CLI
python .github/skills/pipeline-runner/run_pipeline.py 2259 --url https://dev.elitea.ai --project-id 121
```

**All options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `pipeline_id` | — | — | Pipeline ID (positional, mutually exclusive with `--name`) |
| `--name` | `-n` | — | Pipeline name (exact match) |
| `--url` | — | `TEST_DEPLOYMENT_URL` env | Backend URL |
| `--api-key` | — | `TEST_API_KEY` env | API Bearer token |
| `--project-id` | — | `TEST_PROJECT_ID` env | Project ID |
| `--input` | `-i` | `"execute"` | Input message sent to the pipeline |
| `--timeout` | `-t` | `120` | Request timeout in seconds |
| `--json` | `-j` | off | Output structured JSON |
| `--verbose` | `-v` | off | Enable debug/verbose output |

---

## Environment Variables

The script reads these from the project root `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `TEST_DEPLOYMENT_URL` | Yes | Backend URL (e.g., `https://dev.elitea.ai`) |
| `TEST_API_KEY` | Yes | Bearer token |
| `TEST_PROJECT_ID` | Yes | Project ID (numeric) |

---

## Output Format

### Human-readable (default)
```
============================================================
Pipeline: My Fetch Pipeline (ID: 2259)
Status Code: 200
Success: True
Execution Time: 4.31s

Raw Response:
{
  "chat_history": [...],
  "tool_calls_dict": {...},
  ...
}
============================================================
```

### JSON (with `--json`)
```json
{
  "success": true,
  "status_code": 200,
  "response": {
    "chat_history": [...],
    "tool_calls_dict": {...}
  },
  "execution_time": 4.31,
  "error": null
}
```

**Note:** This script returns the raw API response without any processing or test result extraction.

---

## Example Session

**User**: "Run pipeline 2259 with input `{\"id\": 42}`"

**Agent steps:**
1. Activate venv: `source .venv/bin/activate`
2. Run:
   ```bash
   python .github/skills/pipeline-runner/run_pipeline.py 2259 --input '{"id": 42}' --verbose
   ```
3. Report raw API response to the user.

---

**User**: "Execute the fetch pipeline and show me the result"

**Agent steps:**
1. Activate venv.
2. Determine pipeline — use `TEST_APP` from `.env` if set, otherwise ask user for ID or name.
3. Run:
   ```bash
   python .github/skills/pipeline-runner/run_pipeline.py --name "fetch" --json
   ```
4. Display the raw JSON response.

---

## Notes

- The script executes the **latest version** of the pipeline (first entry in the `versions` list)
- Returns the raw API response without any processing, validation, or test result extraction
- No HITL (Human-in-the-Loop) handling - returns the raw interrupt response if encountered
- Name matching is exact (case-sensitive) - use the exact pipeline name from the platform
- Uses TEST_* environment variables to match the pipeline-deployer skill conventions
- If `ModuleNotFoundError` occurs, install dependencies:
  ```bash
  pip install requests python-dotenv
  ```
