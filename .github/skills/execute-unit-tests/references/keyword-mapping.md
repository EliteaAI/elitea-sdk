# Keyword → Test Path Mapping

Use this table to translate user-prompt keywords into pytest targets.

| User says… | Target path / flag |
|---|---|
| "all tests" | `tests/` |
| "all unit tests" | `tests/runtime/` |
| "llm", "alita llm", "max retries", "retry" | `tests/runtime/test_alita_llm.py` |
| "blocked tools", "blocklist", "toolkit blocking" | `tests/runtime/test_blocked_tools.py` |
| "logging", "streamlit logging", "callback handler", "evaluate template" | `tests/runtime/test_logging_utils.py` |
| "preloaded model", "count tokens", "limit tokens", "remove messages" | `tests/runtime/test_preloaded_model.py` |
| "prompt client", "alita prompt", "predict" | `tests/runtime/test_prompt_client.py` |
| "sandbox", "sensitive guard", "could be sensitive" | `tests/runtime/test_sandbox_sensitive_guard.py` |
| "save dataframe", "dataframe" | `tests/runtime/test_save_dataframe.py` |
| "sensitive masking", "tool masking", "mask" | `tests/runtime/test_sensitive_tool_masking.py` |
| "streamlit utils", "streamlit helpers" | `tests/runtime/test_streamlit_utils.py` |
| "utils", "utilities" | `tests/runtime/test_utils.py` |
| "constants", "utils constants" | `tests/runtime/test_utils_constants.py` |
| "text loader", "alita text" | `tests/runtime/langchain/document_loaders/test_alita_text_loader.py` |
| "csv loader", "alita csv" | `tests/runtime/langchain/document_loaders/test_alita_csv_loader.py` |
| "json loader", "alita json" | `tests/runtime/langchain/document_loaders/test_alita_json_loader.py` |
| "markdown loader", "alita markdown" | `tests/runtime/langchain/document_loaders/test_alita_markdown_loader.py` |
| "document loaders", "loaders" | `tests/runtime/langchain/document_loaders/` |
| "github analysis" | `tests/test_github_analysis.py` |
| "gitlab analysis" | `tests/test_gitlab_analysis.py` |
| "ado analysis", "azure devops analysis" | `tests/test_ado_analysis.py` |
| "jira analysis" | `tests/test_jira_analysis.py` |

---

## Interpreting Ambiguous Prompts

When the prompt does not map to a single file or class, apply these heuristics in order:

1. **Exact name match** — if the user mentions a class name (`TestAlitaPrompt`) or function name (`test_predict_with_variables`), use `-k` option or `::` targeting.
2. **Topic keyword** — map the topic (e.g. "masking", "tokens", "sandbox") to the relevant file using the table above.
3. **Multiple topics** — pass multiple file paths to `runTests` or chain them in one pytest command.
4. **Still ambiguous** — ask the user to clarify before running.

---

## `runTests` Tool Examples

| Prompt | `files` param | `testNames` param |
|---|---|---|
| "run blocked tools tests" | `["…/tests/runtime/test_blocked_tools.py"]` | *(omit)* |
| "run TestBlocklistConfiguration" | `["…/tests/runtime/test_blocked_tools.py"]` | `["TestBlocklistConfiguration"]` |
| "run test_default_message in alita llm" | `["…/tests/runtime/test_alita_llm.py"]` | `["test_default_message"]` |
| "run all llm and prompt tests" | `["…/tests/runtime/test_alita_llm.py", "…/tests/runtime/test_prompt_client.py"]` | *(omit)* |
| "run the token-related tests" | `["…/tests/runtime/test_preloaded_model.py"]` | `["test_count_tokens_string", "test_count_tokens_message_list", "test_limit_tokens_no_limit_needed", "test_limit_tokens_with_limiting"]` |
