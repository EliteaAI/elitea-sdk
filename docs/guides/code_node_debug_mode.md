# [FEATURE] Code Node Debug Mode — Artifact Capture for Executed Code

## Summary

Add a `debug` boolean flag to pipeline **Code nodes** that, when enabled, automatically saves the fully-assembled Python code (state preamble + user code) to a dedicated `code-debug` artifact bucket before execution. The artifact filename equals the code node's ID, making it trivial to inspect exactly what the sandbox ran.

---

## Problem Statement

When a pipeline Code node (Pyodide sandbox) produces unexpected output, fails silently, or behaves differently than expected, there is currently **no way to inspect the exact code that was sent into the sandbox**. The Pyodide execution preamble — which injects `elitea_state`, `elitea_client`, and other runtime variables — is assembled dynamically at execution time and is never surfaced to the developer.

This makes debugging Code nodes unnecessarily difficult:
- State variable values injected into the preamble are invisible
- Base64-compressed state payloads cannot be read from logs
- There is no audit trail of what code was actually executed per run

---

## Proposed Solution

Introduce a `debug: true | false` flag on Code node definitions in pipeline YAML/JSON. When `debug: true`:

1. The fully-assembled code (preamble + user code) is saved to the **`code-debug`** artifact bucket.
2. The artifact filename is `<node_id>.py`.
3. The bucket is **created automatically** if it does not exist.
4. Saving is **non-blocking** — any upload failure is logged as a warning and never interrupts pipeline execution.
5. The flag defaults to `false`, so there is **zero overhead** for existing pipelines.

---

## Purpose & Benefits

| Benefit | Description |
|---------|-------------|
| **Transparency** | See the exact code the sandbox received, including injected `elitea_state` values |
| **Faster debugging** | Download `<node_id>.py` from artifacts and run it locally to reproduce issues |
| **Audit trail** | Every debug-enabled execution leaves a timestamped artifact snapshot |
| **Zero regression risk** | `debug: false` (default) — no behavioral change for existing pipelines |
| **Per-node granularity** | Enable debug on specific nodes without affecting the rest of the pipeline |

---

## How to Use

### 1. Enable debug on a Code node

Add `debug: true` to the desired code node in your pipeline definition:

```yaml
nodes:
  - id: process_data
    type: code
    debug: true          # ← enable artifact capture
    input:
      - raw_input
    output:
      - processed_result
    code:
      type: fixed
      value: |
        data = elitea_state.get('raw_input', '')
        result = data.strip().upper()
        processed_result = result
```

### 2. Run the pipeline

Execute the pipeline as usual — via the ELITEA Platform, CLI, or API.

### 3. Inspect the saved artifact

After the run, navigate to **Artifacts → `code-debug` bucket** in the ELITEA Platform UI, or retrieve it via the SDK:

```python
artifact = elitea_client.artifact("code-debug")
code = artifact.get("process_data__20260616_143022.py")
print(code)
```

The artifact will contain the **complete**, standalone-executable code in this exact order:

```python
#elitea simplified client
# (1) Full SandboxClient class definition — no SDK import needed)
import logging, re, requests, chardet
from pathlib import Path
...

class SandboxArtifact: ...
class SandboxClient: ...

# (2) Connection details — set your token before running locally
elitea_client = SandboxClient(base_url='https://your-deployment.elitea.ai',
                               project_id=42,
                               auth_token='<YOUR_AUTH_TOKEN>')  # TODO: replace with your token before running

# (3) State preamble — actual runtime values, base64+zlib compressed
#state dict
import json
import base64
import zlib

compressed_state = base64.b64decode('eJyd...')  # real state from the run
state_json = zlib.decompress(compressed_state).decode('utf-8')
elitea_state = json.loads(state_json)
alita_state = elitea_state.copy()
alita_client = elitea_client  # ← valid — defined above

# (4) User code — exactly as authored in the pipeline node
data = elitea_state.get('raw_input', '')
result = data.strip().upper()
processed_result = result
```

### 4. Reproduce locally

1. Download the artifact file from **Artifacts → `code-debug`**
2. Open it in your editor and **replace the auth token placeholder**:
   ```python
   # Change this line:
   auth_token='<YOUR_AUTH_TOKEN>'  # TODO: replace with your token before running
   # To your actual token:
   auth_token='ey...'
   ```
3. Install the two required packages and run:
   ```bash
   pip install requests chardet
   python process_data__20260616_143022.py
   ```

No ELITEA SDK installation needed — the saved file is fully self-contained.

> **Security note:** The `auth_token` is intentionally left as a placeholder (`<YOUR_AUTH_TOKEN>`) in the saved artifact. It is **never written to the artifact automatically**. You must supply your own token locally before running.

---

## Real-World Example

The following is a real artifact saved by a debug-enabled code node (`extract_text`). It was downloaded from `Artifacts → code-debug → extract_text.py` and is fully runnable:

```python
#elitea simplified client
import logging
import re
# ... (full SandboxClient class) ...

elitea_client = SandboxClient(base_url='https://your-deployment.elitea.ai',
                               project_id=42,
                               auth_token='<YOUR_AUTH_TOKEN>')  # TODO: replace with your token before running

#state dict
import json
import base64
import zlib

compressed_state = base64.b64decode(
    'eJydWXtzIscR/yodXTmHLgIBku5Ble1ICNWR6BWB5MqFlGu0O8BYyy7emUWHXf4z3yKfLp8kv...'
)
state_json = zlib.decompress(compressed_state).decode('utf-8')
elitea_state = json.loads(state_json)
# copies for backwards compatibility with old code that references alita_state and alita_client directly
alita_state = elitea_state.copy()
alita_client = elitea_client

# --- user code ---
data = str(alita_state.get("data", None))
match = re.search(r"'text':\s*'(.*)',\s*'type':\s*'text'", data, re.DOTALL)
text_value = data
if match:
    text_value = match.group(1)
    print(text_value)
text_value
```

> **Note:** The `re` module used in the user code above must be imported explicitly when running outside the sandbox. Add `import re` at the top if it is missing — the sandbox environment makes standard library modules available implicitly.

---

## Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `debug` | `bool` | `false` | When `true`, saves assembled code to `code-debug/<node_id>.py` artifact before execution |

### Artifact Details

| Property | Value |
|----------|-------|
| **Bucket** | `code-debug` |
| **Filename** | `<node_id>__<YYYYMMDD_HHMMSS>.py` (UTC timestamp) |
| **Content** | Full assembled Python source (preamble + user code, UTF-8) |
| **Overwrite** | No — each run creates a new timestamped snapshot |
| **Bucket auto-creation** | Yes — bucket is created on first use if it doesn't exist |

---

## Implementation Details

### Files Changed

| File | Change |
|------|--------|
| `elitea_sdk/runtime/tools/function.py` | Added `debug: bool = False` field to `FunctionTool`; added `_save_code_to_artifact()` method; hooked into `invoke()` after code assembly |
| `elitea_sdk/runtime/langchain/langraph_agent.py` | Passes `debug=node.get('debug', False)` to `FunctionTool` when constructing `code` nodes |

### Execution Flow

```
Pipeline YAML (node: debug: true)
    │
    ▼
create_graph() in langraph_agent.py
    │  reads node.get('debug', False)
    ▼
FunctionTool(debug=True, elitea_client=..., name=node['id'])
    │
    ▼
FunctionTool.invoke(state)
    │  _prepare_pyodide_input()  → assembles state preamble + user code
    │  func_args['code'] = state_preamble + "\n" + user_code
    │
    ├─► if debug and elitea_client:
    │       _save_code_to_artifact(code, node_name)
    │           │
    │           ├─► _build_client_preamble()
    │           │       reads sandbox_client.py
    │           │       appends elitea_client = SandboxClient(...)
    │           │
    │           └─► full_code = client_preamble + state_preamble + user_code
    │               elitea_client.artifact("code-debug")
    │                   .create("<node_id>.py", full_code.encode())
    │
    ▼
PyodideSandboxTool.invoke(func_args)   ← execution proceeds normally
    │  _prepare_pyodide_input() also injects client preamble at runtime
    ▼
Deno / Pyodide sandbox executes the code
```

### Error Handling

Artifact save failures are **silently swallowed** and logged at `WARNING` level:

```
[code-debug] Could not save code artifact for node 'process_data': <error>
```

This guarantees the `debug` flag can never cause a pipeline to fail.

---

## Use Cases

### Use Case 1 — Debugging state variable injection

A code node reads `elitea_state.get('customer_id')` but always gets `None`. Enable `debug: true`, run once, download the artifact, and inspect the actual state that was injected to verify the variable name and value.

### Use Case 2 — Reproducing sandbox errors locally

A node fails with a cryptic Pyodide error. Enable `debug: true`, download `<node_id>.py`, and run it locally with a standard Python interpreter to get a clearer traceback.

### Use Case 3 — Auditing code execution

For compliance or review purposes, `code-debug` artifacts provide a timestamped record of the exact code each pipeline node ran.

### Use Case 4 — Comparing runs

Run a pipeline twice (before/after a change). Compare the two `<node_id>.py` artifacts to confirm the correct state values are being injected.

---

## Labels

- `Type:Feature`
- `feat:pipelines`
- `feat:code-node`
- `eng:sdk`

---

## Related Components

- `elitea_sdk/runtime/tools/sandbox.py` — `PyodideSandboxTool`, `SandboxToolkit`
- `elitea_sdk/runtime/clients/sandbox_client.py` — `SandboxArtifact`, `SandboxClient`
- `elitea_sdk/runtime/tools/function.py` — `FunctionTool` (code node executor)
- `elitea_sdk/runtime/langchain/langraph_agent.py` — pipeline graph builder (`create_graph`)

