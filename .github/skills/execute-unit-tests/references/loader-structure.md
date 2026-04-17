# Loader Test Suite Structure

## Repeatable Test Suite Pattern

Every test suite under `tests/runtime/langchain/<component>/` follows this layout:

```
tests/runtime/langchain/<component>/
  test_<subject>.py          # pytest module (parametrized via collect_loader_test_params)
  test_data/
    <SubjectClass>/
      files/                 # actual test data files (inputs to the loader/component)
      input/                 # JSON test definitions: file_path, configs[], tags[]
      output/                # committed baseline JSONs (expected output)
    scripts/
      loader_test_runner.py  # execution engine (LoaderTestInput, run_single_config_test)
      loader_test_utils.py   # serialization + document comparison utilities
```

**Input JSON format** (`input/<name>.json`):
```json
{
  "tags": ["loader:text", "feature:chunking"],
  "file_path": "../files/<name>.txt",
  "configs": [
    {},
    {"max_tokens": 256},
    {"max_tokens": 1024}
  ]
}
```
- `tags` — file-level pytest marks applied to every config in this file (used for `-m` filtering)
- `configs` — list of parameter dicts; each becomes a separate test case

**Shared helpers** (used by all test modules):
- `tests/loader_helpers.py` — `collect_loader_test_params(loader_name)` + `run_loader_assert(...)`
- `tests/conftest.py` — sys.path setup, ReportPortal env mapping, marker registration

---

## Loader Name → Test File Mapping

| Loader | Test file |
|--------|-----------|
| `AlitaTextLoader` | `test_alita_text_loader.py` |
| `AlitaCSVLoader` | `test_alita_csv_loader.py` |
| `AlitaJSONLoader` | `test_alita_json_loader.py` |
| `AlitaMarkdownLoader` | `test_alita_markdown_loader.py` |

---

## Adding a New Test Case

1. Add a test data file to `test_data/<LOADER>/files/`
2. Create an input JSON in `test_data/<LOADER>/input/<name>.json`:
   ```json
   {
     "tags": ["loader:csv", "content:simple"],
     "file_path": "../files/<name>.csv",
     "configs": [
       {},
       {"max_tokens": 256}
     ]
   }
   ```
3. Generate baselines using the bulk script in [baseline-management.md](baseline-management.md)
4. Run the test to confirm it passes:
   ```bash
   python -m pytest tests/runtime/langchain/document_loaders/test_alita_csv_loader.py -v
   ```

---

## Adding a New Test Suite

New test suites under `tests/runtime/langchain/<component>/` mirror the document loader pattern exactly:

1. Create `tests/runtime/langchain/<component>/test_<subject>.py` using `collect_loader_test_params` + `run_loader_assert`
2. Create `test_data/<SubjectClass>/files/`, `input/`, `output/`
3. Write input JSON files with `tags` and `configs`
4. Generate baselines (see [baseline-management.md](baseline-management.md))
5. Register any new marks in `pyproject.toml` under `[tool.pytest.ini_options] markers`
