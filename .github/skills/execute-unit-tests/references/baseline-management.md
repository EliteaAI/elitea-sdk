# Baseline Management

## Regenerating Baselines

When a loader's behavior intentionally changes, update the affected baseline files.

Baseline files live at:
```
tests/runtime/langchain/document_loaders/test_data/<LOADER>/output/<input>_config_<N>.json
```

### Regenerate a single baseline

Run the test with `-s` to see the actual output in the failure message, then copy the actual output over the baseline:

```bash
python -m pytest tests/runtime/langchain/document_loaders/test_alita_text_loader.py::test_loader[text_simple-config0] -v -s
```

The actual output is written to a `tmp_path` directory reported in the failure message. Copy it to the corresponding `output/` baseline file.

### Bulk regenerate all baselines for one loader

Run this snippet from the project root (with venv active):

```python
import sys
sys.path.insert(0, "tests/runtime/langchain/document_loaders/test_data/scripts")
sys.path.insert(0, "tests")

from pathlib import Path
from loader_test_runner import LoaderTestInput, _load_documents_with_production_config
from loader_test_utils import save_documents

LOADER = "AlitaTextLoader"   # change as needed
BASE = Path("tests/runtime/langchain/document_loaders/test_data")
input_dir = BASE / LOADER / "input"
output_dir = BASE / LOADER / "output"

for json_file in sorted(input_dir.glob("*.json")):
    test_input = LoaderTestInput.from_file(json_file)
    file_path = test_input.resolved_file_path(json_file)
    for i, cfg in enumerate(test_input.configs):
        cfg_clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
        docs = _load_documents_with_production_config(file_path, cfg_clean)
        out = output_dir / f"{json_file.stem}_config_{i}.json"
        save_documents(docs, out)
        print(f"Saved {out} ({len(docs)} docs)")
```

---

## Understanding Test Failures

### Count mismatch (`actual=N expected=M`)

The loader produced a different number of documents than the baseline. Common causes:
- Chunking logic changed (check `max_tokens` handling)
- File content was changed
- Loader's split/merge logic was updated

Run with `-s` to see the full diff output from `compare_documents`.

### Metadata mismatch (`source` path)

The `source` field uses `path_suffix` comparison (actual path must *end with* the expected suffix). If this fails, the file path structure changed or the baseline was generated on a different machine with a different root.

### Similarity mismatch (`page_content`)

Page content comparison uses TF-IDF cosine similarity (threshold = 1.0 = exact match after whitespace normalization). If text content changed, regenerate the baseline.

### Baseline not found

```
Baseline not found: tests/runtime/.../output/xxx_config_0.json
```

The baseline file doesn't exist yet. Regenerate it using the bulk script above.
