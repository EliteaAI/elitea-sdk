# Loader Tags and Filtering

## Tag Reference

Each input JSON has a `tags` field. Tags are converted to pytest marks (`:` and `-` become `_`).

| Mark | Triggers when |
|------|--------------|
| `loader_text` | Any change to `AlitaTextLoader` |
| `loader_csv` | Any change to `AlitaCSVLoader` |
| `loader_json` | Any change to `AlitaJSONLoader` |
| `loader_markdown` | Any change to `AlitaMarkdownLoader` |
| `feature_chunking` | Changes to chunking / `max_tokens` logic |
| `content_empty` | Empty input handling |
| `content_simple` | Baseline simple content handling |
| `content_large` | Large file handling |
| `content_unicode` | Unicode / multibyte encoding |
| `content_special_characters` | Special character parsing |
| `content_nested` | Nested data structure parsing |
| `content_array` | Array/list structure parsing |
| `content_headers` | Header-structured content |
| `content_markdown` | Markdown-formatted content |
| `performance` | Large files, slow tests |
| `edge_empty_input` | Empty file guards |
| `edge_encoding` | Encoding detection / normalization |
| `edge_special_chars` | Special character edge cases |
| `edge_markdown_in_txt` | Markdown content in `.txt` files |

---

## Filtering by Tag (Impact Analysis)

### Filter by loader

```bash
python -m pytest tests/runtime/langchain/document_loaders/ -m "loader_text" -v
python -m pytest tests/runtime/langchain/document_loaders/ -m "loader_csv" -v
python -m pytest tests/runtime/langchain/document_loaders/ -m "loader_json" -v
python -m pytest tests/runtime/langchain/document_loaders/ -m "loader_markdown" -v
```

### Filter by feature or content type

```bash
# All chunking tests (max_tokens logic)
python -m pytest tests/runtime/langchain/document_loaders/ -m "feature_chunking" -v

# Large file / performance tests
python -m pytest tests/runtime/langchain/document_loaders/ -m "performance" -v

# All unicode/encoding edge cases
python -m pytest tests/runtime/langchain/document_loaders/ -m "edge_encoding" -v

# Empty input edge cases
python -m pytest tests/runtime/langchain/document_loaders/ -m "content_empty" -v
```

### Combine tags (AND / OR / NOT)

```bash
# CSV + chunking only
python -m pytest tests/runtime/langchain/document_loaders/ -m "loader_csv and feature_chunking" -v

# All loaders, skip slow tests (fast CI run)
python -m pytest tests/runtime/langchain/document_loaders/ -m "not performance" -v

# Markdown loader, large content tests only
python -m pytest tests/runtime/langchain/document_loaders/ -m "loader_markdown and content_large" -v
```
