# Unified chunked-read JSON schema (PRE-1, #5432)

The single response contract returned by **both**:

1. `get_file_metadata` — the *inspect* path (artifact toolkit), and
2. the **over-limit** return path of `read_file` / `read_multiple_files`
   (Phase 4, #5446), built via `build_over_limit_response`.

Both paths emit the **identical shape**. Only the discriminator (and an extra
`context` block on over-limit) differ. This lets an LLM *and* a non-LLM pipeline
node replace one unbounded read with a deterministic sequence of bounded chunk
reads, detecting "this is guidance, not content" **programmatically** —
never by string-matching English prose.

Source of truth: the Pydantic models in
`elitea_sdk/tools/utils/file_metadata.py`. Reference implementation:
`EliteAExcelLoader.get_file_metadata`.

---

## The discriminator

There is exactly **one** machine-detectable field, namespaced with dunders so
it can never collide with a content dict's own keys:

```
__result_status__ : "file_metadata" | "content_too_large" | "error"
```

| Value | Meaning | Consumer action |
|-------|---------|-----------------|
| `file_metadata` | Inspect output from `get_file_metadata`. | Use `instruction_for_readFile` to plan chunk reads. |
| `content_too_large` | A read exceeded the limit; this is guidance, not content. | Loop a bounded chunked read using `instruction_for_readFile`. Do **not** treat as file content. |
| `error` | Inspection/metadata failed. | See `message`. |

**Raw file content carries no `__result_status__` key.** A normal `read_file`
result is a `str` (text) or a per-type `dict` (e.g. an Excel row-range slice) —
neither has the discriminator. That *absence* is how content is told apart from
guidance:

```python
resp = json.loads(tool_output) if isinstance(tool_output, str) else tool_output
status = resp.get("__result_status__") if isinstance(resp, dict) else None
if status == "content_too_large":
    # over-limit: issue the next bounded read
    params = resp["instruction_for_readFile"]["extra_params"]
    ...
elif status in (None,):
    # this IS the file content
    ...
```

---

## Schema

| Field | Type | Notes |
|-------|------|-------|
| `__result_status__` | string enum | The sole discriminator (above). **Always present.** |
| `schema_version` | string | `"1.0"`. Lets future migrations branch. |
| `filename` | string \| null | |
| `type` | string \| null | MIME type. |
| `extension` | string \| null | e.g. `.xlsx`. |
| `filesize` | int \| null | Bytes. |
| `unit` | string \| null | Authoritative natural unit: `rows` \| `lines` \| `pages` \| `sheets` \| null. Names which `total_<unit>` is the total of record. |
| `total_<unit>` | int | **Flat** per-unit total: `total_rows` / `total_lines` / `total_pages` / `total_sheets`. Only the relevant one(s) present. |
| `read_limits` | object | **Required** on `file_metadata` / `content_too_large` (the looping node needs a limit to chunk against); omitted only on `error`. Per-type caps + estimates (below). |
| `instruction_for_readFile` | object | Exact `read_file` params for the next chunk (below). |
| `context` | object | **Over-limit only.** `{limit_chars, actual_chars, requested?}`. |
| `message` | string | **Error only.** |
| *(type extras)* | any | `sheets` (Excel), `image_count` / `image_names` (Docx), etc. Allowed via `extra="allow"`. |

### `read_limits`

Only two keys are **universal and required**:

| Key | Type | Meaning |
|-----|------|---------|
| `max_output_chars` | int | **THE** single bounded-read limit — chars of text returned to context (200000). One limit, no second number. |
| `full_read_allowed` | bool | Whether an unbounded full read is permitted for this file. Always `false` on `content_too_large` (a full read just exceeded the limit). |

The central `get_file_metadata` injects a **universal baseline** for these two
keys for *every* file type, then merges the loader's `read_limits` key-wise on
top. A loader therefore inherits `max_output_chars` / `full_read_allowed`
without restating them, and the documented-required field is genuinely always
present (including for loaders like Docx that supply no `read_limits`).

Everything else is **optional and MUST be type-prefixed**, so no other loader
assumes the key exists. Example Excel extras:

| Key | Unit | Why it differs from `max_output_chars` |
|-----|------|----------------------------------------|
| `excel_full_read_max_bytes` | **bytes** | A cheap raw-bytes *pre-gate*: reject an unbounded full read *before* sampling rows, because a large workbook is presumed too costly to expand. This is bytes-on-disk, an estimate gate — **not** the output-chars cap. Hence the larger raw number. |
| `excel_max_request_rows` | rows | Max rows per request. |
| `excel_max_embedded_images` | count | Image-count cap. |
| `estimated_output_chars`, `estimated_total_rows`, ... | various | Best-effort estimates. |

### `instruction_for_readFile`

Canonical format (superset; matches `EliteADocxMammothLoader`):

```jsonc
{
  "first_class_params": { /* optional: top-level read_file args, e.g. is_capture_image */ },
  "extra_params":       { /* JSON passed as a string in read_file(extra_params=...) */ },
  "notes": "human/LLM guidance, e.g. 'Pass extra_params as a JSON string.'"
}
```

`extra="allow"` on this block lets a loader add type-specific instruction
sub-keys. Each value is a human-readable description of the parameter.

---

## Example — inspect (`file_metadata`)

```jsonc
{
  "__result_status__": "file_metadata",
  "schema_version": "1.0",
  "filename": "report.xlsx",
  "type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "extension": ".xlsx",
  "filesize": 184320,
  "unit": "rows",
  "total_rows": 48213,
  "total_sheets": 3,
  "sheets": [ { "name": "Data", "max_row": 48000, "max_column": 12 }, ... ],
  "read_limits": {
    "max_output_chars": 200000,
    "full_read_allowed": false,
    "excel_full_read_max_bytes": 20971520,
    "excel_max_request_rows": 10000,
    "estimated_output_chars": 91234
  },
  "instruction_for_readFile": {
    "first_class_params": {},
    "extra_params": {
      "sheet_name": "string (optional) — name of the sheet to read ...",
      "start_row":  "integer (1-indexed, inclusive) — first row ...",
      "end_row":    "integer (1-indexed, inclusive) — last row ..."
    },
    "notes": "For large workbooks, call read_file with a small start_row/end_row range. Pass extra_params as a JSON string."
  }
}
```

## Example — over-limit (`content_too_large`)

Identical shape, produced by `build_over_limit_response(metadata, ...)`. Differs
**only** in `__result_status__` and the added `context`:

```jsonc
{
  "__result_status__": "content_too_large",
  "schema_version": "1.0",
  "filename": "report.xlsx",
  "unit": "rows",
  "total_rows": 48213,
  "read_limits": { "max_output_chars": 200000, "full_read_allowed": false, ... },
  "instruction_for_readFile": { /* same as inspect */ },
  "context": { "limit_chars": 200000, "actual_chars": 870123, "requested": "full read" }
}
```

## Example — error

```jsonc
{
  "__result_status__": "error",
  "schema_version": "1.0",
  "filename": "missing.xlsx",
  "message": "Error reading file metadata: ..."
}
```

---

## How a loader conforms

A document loader registered in `loaders_map` adds per-type metadata by defining:

```python
@classmethod
def get_file_metadata(cls, *, filename, file_content=None, file_size=None) -> dict:
    return {
        "unit": "rows",                       # or "lines" / "pages" / "sheets"
        "total_rows": <int>,                  # flat total_<unit>
        "read_limits": {
            "max_output_chars": 200000,       # REQUIRED
            "full_read_allowed": <bool>,      # REQUIRED
            # ... type-prefixed optional extras
        },
        "instruction_for_readFile": {
            "extra_params": { ... },
            "notes": "...",
        },
        # ... any type-specific extras (sheets, image_count, ...)
    }
```

The central `file_metadata.get_file_metadata` merges this over the generic base
(including the universal `read_limits` baseline), stamps `__result_status__` /
`schema_version`, and **validates the merged dict through `ChunkedReadResponse`**
before returning. A loader that returns a malformed shape fails validation — the
central function then returns an `error` response and logs the violation. This is
the single enforcement point every Phase-1 per-type task (PRE-2..12) conforms to.

A loader does **not** need to restate `max_output_chars` / `full_read_allowed`
(the baseline supplies them), but a guidance object that ends up with no
`read_limits` at all is rejected by the model validator.

**All emissions route through the model.** Both the success path
(`get_file_metadata`) and every error path (`build_error_response`) produce dicts
via `ChunkedReadResponse`, so every response — including errors raised in the
artifact tool/client — carries `schema_version` and the discriminator. Never
hand-build a response dict.

```python
from elitea_sdk.tools.utils.file_metadata import (
    validate_chunked_read_response, build_error_response,
)
validate_chunked_read_response(some_dict)        # raises ValidationError on non-conformance
build_error_response("msg", filename="x.dat")    # the only way to emit an error response
```
