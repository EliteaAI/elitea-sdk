# Test Inventory

Complete reference for all test files in the Alita SDK `tests/` directory.

---

## `tests/runtime/` — Unit Tests (no network, no credentials)

| File | Classes / Key Topics |
|------|---------------------|
| `tests/runtime/test_alita_llm.py` | `TestMaxRetriesExceededError`, `TestAlitaLLMConstants`, `TestAlitaLLMErrorHandling`, `TestUtilityFunctions`, `TestTypeHints` — LLM error handling, imports, type hints |
| `tests/runtime/test_blocked_tools.py` | `TestBlocklistConfiguration`, `TestFilterBlockedTools`, `TestFinalBlockedToolsFilter`, `TestInvokeToolBlockedGate` — tool/toolkit blocklist logic |
| `tests/runtime/test_logging_utils.py` | Streamlit callback handler, `setup_streamlit_logging`, `dispatch_custom_event`, `evaluate_template` — logging and template utilities |
| `tests/runtime/test_preloaded_model.py` | `TestPreloadedChatModel` — `count_tokens`, `remove_non_system_messages`, `limit_tokens` |
| `tests/runtime/test_prompt_client.py` | `TestAlitaPrompt` — prompt init, `create_pydantic_model`, `predict` |
| `tests/runtime/test_sandbox_sensitive_guard.py` | `TestSandboxToolMatching`, `TestCouldBeSensitive` — sandbox/sensitive tool detection |
| `tests/runtime/test_save_dataframe.py` | DataFrame save utilities |
| `tests/runtime/test_sensitive_tool_masking.py` | Sensitive value masking in tool args |
| `tests/runtime/test_streamlit_utils.py` | Streamlit UI helper functions |
| `tests/runtime/test_utils.py` | General SDK utility functions |
| `tests/runtime/test_utils_constants.py` | Utility constants |

---

## `tests/runtime/langchain/document_loaders/` — Parametrized Loader Tests

All four loaders follow the same structure. See [loader-structure.md](loader-structure.md) for details.

| File | Loader |
|------|--------|
| `test_alita_text_loader.py` | `AlitaTextLoader` |
| `test_alita_csv_loader.py` | `AlitaCSVLoader` |
| `test_alita_json_loader.py` | `AlitaJSONLoader` |
| `test_alita_markdown_loader.py` | `AlitaMarkdownLoader` |

---

## `tests/` — Integration / Analysis Tests (require credentials)

| File | Scope |
|------|-------|
| `tests/test_github_analysis.py` | GitHub toolkit analysis |
| `tests/test_gitlab_analysis.py` | GitLab toolkit analysis |
| `tests/test_ado_analysis.py` | Azure DevOps toolkit analysis |
| `tests/test_jira_analysis.py` | JIRA toolkit analysis |

---

## Key Paths

| Path | Purpose |
|------|---------|
| `tests/runtime/langchain/document_loaders/` | Pytest test modules for document loaders |
| `tests/runtime/langchain/document_loaders/test_data/` | All test assets (inputs, baselines, files) |
| `tests/runtime/langchain/document_loaders/test_data/<LOADER>/input/` | Input JSON definitions (configs + tags) |
| `tests/runtime/langchain/document_loaders/test_data/<LOADER>/output/` | Committed baseline JSON files |
| `tests/runtime/langchain/document_loaders/test_data/<LOADER>/files/` | Actual test data files (.txt, .csv, etc.) |
| `tests/runtime/langchain/document_loaders/test_data/scripts/loader_test_runner.py` | Test execution engine |
| `tests/runtime/langchain/document_loaders/test_data/scripts/loader_test_utils.py` | Serialization & comparison utilities |
| `tests/loader_helpers.py` | `collect_loader_test_params()` and `run_loader_assert()` used by all test modules |
| `tests/conftest.py` | Pytest session setup — sys.path, RP env mapping, marker registration |
| `pyproject.toml` | Pytest config, registered marks |
