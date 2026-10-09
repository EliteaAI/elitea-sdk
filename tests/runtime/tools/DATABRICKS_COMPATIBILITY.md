# Databricks tool-result compatibility (#6830)

The ordinary SDK regression tests run without LiteLLM. In particular,
`test_prepare_messages_for_model.py` compares real provider payloads before and
after preparation for Anthropic/Bedrock Messages, OpenAI Chat Completions and
Responses, and the OpenAI-compatible paths used by Gemini, Llama and Claude.
It covers successful/error results, JSON strings, text/image blocks, empty
structured content, parallel calls and repeated iterations. These checks verify
SDK transport behavior rather than live inference by those models.

Run the broader existing runtime suites and the ordinary regressions using the
repository's declared framework/client versions and `pytest-asyncio`:

```sh
python - <<'PY'
from pathlib import Path
import pytest
roots = ('tests/runtime', 'tests/runtime/clients',
         'tests/runtime/tools', 'tests/runtime/langchain')
excluded = {'test_elitea_llm.py', 'test_databricks_tool_message_compatibility.py'}
files = sorted(str(path) for root in roots for path in Path(root).glob('test*.py')
               if path.name not in excluded)
raise SystemExit(pytest.main(files))
PY
```

The legacy `test_elitea_llm.py` exclusion matches PR CI: its implementation was
removed. The MCP rig fixtures need permission to bind local loopback ports.

The separate pinned-adapter compatibility check below supplements this coverage.
The suite executes real SDK tool loops and ChatAnthropic serialization, then
LiteLLM 1.83.14's Anthropic-to-OpenAI and Databricks request transformations.
Only HTTP model responses are scripted. It checks immediate call/result pairing
and preservation of tool data through two iterations, parallel calls and streaming.
Negative controls reproduce dropped results and raw orphaned `tool_use` blocks.

Run from the SDK repository using its existing Python test environment:

```sh
compat_overlay=$(mktemp -d)
python -m pip install --no-deps --target "$compat_overlay" \
  -r tests/runtime/tools/databricks-compatibility-requirements.txt
ELITEA_DATABRICKS_COMPAT_TESTS=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
  PYTHONPATH="$PWD:$compat_overlay${PYTHONPATH:+:$PYTHONPATH}" \
  python -m pytest tests/runtime/tools/test_databricks_tool_message_compatibility.py
```

The PR workflow runs this in a separate step. The regular test run skips this
suite so an unrelated installed LiteLLM version cannot change the regression.
Enabling the suite requires the exact pinned adapter; missing dependencies or a
different LiteLLM version fail the run.

The SDK's rich-content fallback uses the selected model ID (including bound
clients) containing `databricks`, consistent with SDK model-name routing. Native
Anthropic rich blocks remain intact. Blank-result preparation applies to all
providers. Original checkpoint messages, artifacts and status remain unchanged.

Passing establishes the outgoing request contract, not acceptance by a live
Databricks endpoint. Customer attribution still requires the failing payload and
installed SDK version; OpenAPI `"[]"` and `"{}"` strings remain unchanged.
