# EliteATextLoader Test Suite

## Test Coverage

- **Test files**: 5
- **Total configs**: 15
- **Parameters tested**: `max_tokens` (from `DEFAULT_ALLOWED_BASE`)
- **Loader behavior**: Splits plain text into token-bounded chunks using `text_chunker`

## Test Breakdown

| File | Configs | Tags | Purpose |
|------|---------|------|---------|
| `text_simple.txt` | 2 | `loader:text`, `content:simple`, `feature:chunking` | Basic plain text — single chunk expected |
| `text_empty.txt` | 1 | `loader:text`, `content:empty`, `edge:empty-input` | Empty file — validates empty file handling |
| `text_unicode.txt` | 2 | `loader:text`, `content:unicode`, `edge:encoding`, `feature:chunking` | Multi-byte characters (emoji, CJK, Arabic) |
| `text_large.txt` | 5 | `loader:text`, `content:large`, `feature:chunking`, `performance` | Large file — exercises chunking at multiple token limits |
| `text_markdown.txt` | 5 | `loader:text`, `content:markdown`, `feature:chunking`, `edge:markdown-in-txt` | Markdown content in a .txt file — chunker treats it as plain text |

## Configuration Matrix

### text_simple.txt
- **Config 0** (`{}`): Production defaults (max_tokens=512)
- **Config 1** (`{"max_tokens": 1024}`): Larger token limit

### text_empty.txt
- **Config 0** (`{}`): Production defaults — empty file returns single document with empty string

### text_unicode.txt
- **Config 0** (`{}`): Production defaults with multi-byte content
- **Config 1** (`{"max_tokens": 256}`): Smaller chunks with Unicode content

### text_large.txt
- **Config 0** (`{}`): Production defaults (max_tokens=512)
- **Config 1** (`{"max_tokens": 50}`): Very small chunks — boundary condition
- **Config 2** (`{"max_tokens": 100}`): Small chunks
- **Config 3** (`{"max_tokens": 256}`): Medium chunks
- **Config 4** (`{"max_tokens": 2000}`): Large chunks — minimal splitting

### text_markdown.txt
- **Config 0** (`{}`): Production defaults (max_tokens=512)
- **Config 1** (`{"max_tokens": 50}`): Very small chunks
- **Config 2** (`{"max_tokens": 100}`): Small chunks
- **Config 3** (`{"max_tokens": 256}`): Medium chunks
- **Config 4** (`{"max_tokens": 2000}`): Large chunks — minimal splitting

> **Note**: `text_markdown-config1` and `text_markdown-config2` have known baseline mismatches
> (off-by-one chunk count with max_tokens=50 and max_tokens=100). These are pre-existing
> failures unrelated to recent changes, pending investigation of the text chunker boundary
> behaviour or baseline regeneration.

## Loader Configuration

From `constants.py` — EliteATextLoader uses:
- **kwargs**: `{}` (no production defaults)
- **allowed_to_override**: `DEFAULT_ALLOWED_BASE` = `{'max_tokens': 512}`
- **Behavior**: Uses `text_chunker` for token-bounded splitting; adds metadata `source`, `chunk_id`

## Running Tests

All commands are run from the project root with the virtualenv active.

```bash
# Run all Text loader tests
pytest tests/runtime/langchain/document_loaders/test_elitea_text_loader.py -v

# Run a specific input file
pytest tests/runtime/langchain/document_loaders/test_elitea_text_loader.py -v -k "text_large"

# Run all document loader tests at once
pytest tests/runtime/langchain/document_loaders/ -v
```

### Filtering by Tag

Test files are tagged in `input/*.json`. Use `-m` to select subsets:

```bash
# All Text loader tests
pytest -m "loader_text" -v

# Text tests exercising chunking
pytest -m "loader_text and feature_chunking" -v

# Large-file / performance tests only
pytest -m "performance" -v

# Edge-case tests across all loaders
pytest -m "edge_empty_input or edge_encoding or edge_markdown_in_txt" -v
```

| Tag | pytest mark | Applied to |
|-----|-------------|------------|
| `loader:text` | `loader_text` | All Text configs |
| `content:simple` | `content_simple` | text_simple configs |
| `content:empty` | `content_empty` | text_empty configs |
| `content:unicode` | `content_unicode` | text_unicode configs |
| `content:large` | `content_large` | text_large configs |
| `content:markdown` | `content_markdown` | text_markdown configs |
| `feature:chunking` | `feature_chunking` | text_simple, text_unicode, text_large, text_markdown configs |
| `performance` | `performance` | text_large configs |
| `edge:empty-input` | `edge_empty_input` | text_empty configs |
| `edge:encoding` | `edge_encoding` | text_unicode configs |
| `edge:markdown-in-txt` | `edge_markdown_in_txt` | text_markdown configs |

## Directory Structure

```
EliteATextLoader/
├── files/               # Test data files
│   ├── text_simple.txt
│   ├── text_empty.txt
│   ├── text_unicode.txt
│   ├── text_large.txt
│   └── text_markdown.txt
├── input/               # Input JSON descriptors (one per test file)
│   ├── text_simple.json
│   ├── text_empty.json
│   ├── text_unicode.json
│   ├── text_large.json
│   └── text_markdown.json
└── output/              # Committed baselines (one per config)
    ├── text_simple_config_0.json
    ├── text_simple_config_1.json
    ├── text_empty_config_0.json
    ├── text_unicode_config_0.json
    ├── text_unicode_config_1.json
    ├── text_large_config_0.json
    ├── text_large_config_1.json
    ├── text_large_config_2.json
    ├── text_large_config_3.json
    ├── text_large_config_4.json
    ├── text_markdown_config_0.json
    ├── text_markdown_config_1.json
    ├── text_markdown_config_2.json
    ├── text_markdown_config_3.json
    └── text_markdown_config_4.json
```

## Notes

- All test files use UTF-8 encoding
- Test data is deterministic (no timestamps, random values, or UUIDs)
- Baselines contain all metadata fields set by the loader
- Empty file returns a single document with empty `page_content`
- Config overrides are filtered by `allowed_to_override` as per production behaviour
