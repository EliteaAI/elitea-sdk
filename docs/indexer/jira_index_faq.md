# Jira Indexing — FAQ

How the Jira indexer extracts content and what gets stored, with focus on the `process_images` flag.

## `include_attachments` vs `process_images` — two unrelated flows

The names sound similar but they touch different APIs and produce different output. Orthogonal flags — enable either, both, or neither.

### `include_attachments` — file-as-document
**Default:** `False`. Path: `_process_document`.

```
client.issue(key, fields="attachment")        # 1 REST call/issue
  └─ for each attachment in fields.attachment:
      └─ client.get_attachment_content(id)    # 1 download/attachment
         → yield Document(page_content='',
                          metadata={CONTENT_IN_BYTES: <bytes>,
                                    type='attachment',
                                    parent=<issue>, ...})
```

- Indexes **every** file attached to the issue (PDFs, .docx, images, anything) as its **own** dependent document.
- The chunker parses the bytes by extension/MIME (e.g. PDF text extraction).
- **No LLM involved.** Pure file ingest.
- Skips extensions listed in `skip_attachment_extensions`.

### `process_images` — markup-to-text-substitution
**Default:** `False`. Paths: `_extend_data` (description), `get_processed_comments_list_with_image_description` (comments).

```
issue.description (or comment.body)
  └─ regex r'!([^!|]+)(?:\|[^!]*)?!' finds inline image refs
      └─ AttachmentResolver(issue_key)        # 3+ REST calls/issue (one-time)
      └─ for each !ref!:
          ├─ resolver.find_attachment(ref)
          ├─ _download_attachment(content_url)
          ├─ _collect_context_for_image       # ±500 chars
          └─ _process_image_with_llm          # vision-LLM call
              → "[Image <name> Description: <text>]"
                 → substituted IN-PLACE into the description/comment text
```

Activation requires **all** of:
1. `process_images=True` is passed to `index_data`,
2. an LLM is configured on the toolkit,
3. the text actually contains Jira image markup.

When inactive, the markup stays in the text as-is — chunks are still created, just without LLM-generated descriptions in place of the markers.

### Side-by-side

| | `include_attachments` | `process_images` |
|---|---|---|
| Triggered by | every file in `fields.attachment` | inline `!ref!` markup in description/comments |
| Output shape | new `type='attachment'` dependent Document | text substituted into existing base/comment doc |
| What lands in the index | file bytes (parsed by chunker) | LLM-generated English description |
| Catches non-image attachments? | yes (PDFs, docs, anything) | no — only `!ref!` markup, useful for images |
| Catches images uploaded but not referenced? | yes | no |
| LLM cost | none | 1 vision call per reference |
| Jira REST cost per issue | 1 + N downloads | 3 + N downloads (resolver overhead) |

Use `process_images=True` when you want searchable image content embedded in the issue's text context. Use `include_attachments=True` when you want files (including non-image attachments) ingested as standalone searchable docs.

## Extraction flow (per issue)

### 1. Description path — `_extend_data` (`elitea_sdk/tools/jira/api_wrapper.py`)

```
issue.description
  └─ regex r'!([^!|]+)(?:\|[^!]*)?!' finds image refs
      └─ AttachmentResolver(issue_key)            # 3+ Jira REST calls
      │   ├─ get_attachments_ids_from_issue
      │   ├─ get_attachment(id)  per attachment   # metadata
      │   └─ issue(key, fields="attachment")
      └─ for each match: process_image_match
          ├─ resolver.find_attachment(ref)        # by id / filename / thumbnail
          ├─ _download_attachment(content_url)    # bytes
          ├─ _collect_context_for_image           # ±500 chars around the ref
          └─ _process_image_with_llm              # vision-LLM call
              └─ returns "[Image <name> Description: <llm text>]"
                 → substituted in place of the !ref! markup
```

Output: description text (with image markers replaced when `process_images` is active), encoded into `CONTENT_IN_BYTES` for the chunker.

### 2. Comments path — `_process_document`

```
client.issue_get_comments(issue_key)              # 1 REST call per issue
  └─ for each comment.body:
      └─ same regex + AttachmentResolver + LLM image description
         (via get_processed_comments_list_with_image_description)
  → yields one Document per processed comment
    (page_content empty, CONTENT_IN_BYTES = processed body,
     type='comment', parent=issue)
```

## What is indexed

| Field | Source | Indexed as |
|---|---|---|
| Issue summary + description (with `[Image … Description: …]` substitutions if `process_images`) | `_base_loader` → `_process_issue_for_indexing` → `_extend_data` | base doc, chunked by `chunking_tool` |
| Selected `fields_to_index` content | `_process_issue_for_indexing` | appended to base doc content |
| Comment body (with image substitutions if `process_images`) | `_process_document` (gated by `include_comments`) | dependent doc, `type='comment'` |
| Attachment binary content | `_process_document` (gated by `include_attachments`) | dependent doc, `type='attachment'` |

## Flag matrix

| Flag | Default | Effect |
|---|---|---|
| `process_images` | `False` | When `True` (and LLM configured): runs `AttachmentResolver` + vision-LLM substitution in descriptions and comments. When `False`: skip resolver and LLM calls; image markup stays as-is. |
| `include_comments` | `False` | When `True`: fetch comments per issue and index each as a dependent doc (`type='comment'`); per-comment image work runs only if `process_images` and LLM also active. When `False` (default): comments path skipped entirely (no `issue_get_comments`, no per-comment image work). |
| `include_attachments` | `False` | When `True`: creates dependent attachment documents (binary content). Independent of `process_images`. |
| `max_total_issues` | `1000` | Hard cap on issues fetched from JQL. Silently truncates if JQL would return more. |
| `skip_attachment_extensions` | `[]` | Filename extensions to exclude from attachment dependent docs. |
| `chunking_tool` | `"markdown"` | Chunker applied to base + dependent doc content. |

## Cost notes

- Per-issue cost when `process_images=True` and the description has N image refs:
  - 3 + (number of attachments) Jira REST calls (resolver init)
  - N attachment downloads
  - N vision-LLM calls
- Same multipliers apply per-comment when `include_comments=True` and a comment contains image markup.
- For 1000+ issue datasets, leave `process_images=False` unless image content is essential.

## Recommended settings for large datasets (>500 issues)

```yaml
process_images: false
include_comments: false        # turn on only if comment text is needed
include_attachments: false
max_total_issues: <expected JQL count + buffer>
chunking_tool: markdown
```

If image descriptions are needed for a smaller subset, run a separate index over that subset with `process_images: true`.
