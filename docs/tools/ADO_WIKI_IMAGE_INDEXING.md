# ADO Wiki — Image & Attachment Indexing

Status: implemented on branch `fix/ado-workitem-links-wiql`
Files touched: `elitea_sdk/tools/ado/wiki/ado_wrapper.py`

## Problem

The ADO wiki toolkit already knew how to describe wiki-page images at *tool-call
time* — `get_wiki_page*` calls `_process_images`, which fetches each markdown
image (both `/.attachments/…` items and external URLs), asks the toolkit's LLM to
describe it, and rewrites the markdown so the description replaces the image
link.

The **indexing path was blind to this**. `_base_loader` fetched raw markdown,
hashed it, and yielded it as a `Document`. Any image content in a wiki page was
completely absent from the vector store: the retriever could only match on alt
text and the URL string itself. Architecture diagrams, screenshots, and embedded
PDFs referenced under `.attachments/` were effectively invisible to search over
the indexed wiki.

Confluence had already solved the equivalent problem: it exposes
`include_attachments` + extension filters in `_index_tool_params` and overrides
`_process_document` to yield each attachment as a dependent document. This
change ports that pattern to ADO wiki, plus adds an option to inline image
descriptions in the parent page.

## What was implemented

Two independent, opt-in capabilities. Both default to `False` so existing
indexing behavior is unchanged.

### A. Inline image descriptions (`process_images`)

When enabled, `_base_loader` runs the existing `_process_images` over each page
before yielding, so `![alt](url)` markdown is replaced with the LLM's textual
description of the image. The description becomes part of the page's
`page_content` and is chunked/embedded like the surrounding prose.

Dedup safety: the `updated_on` hash is computed against the **raw** markdown
returned by ADO, not against the LLM-augmented text. LLM output is not
deterministic, so hashing the augmented content would force every page to be
re-indexed on every run. Hashing the source keeps incremental indexing stable.

If image processing throws for a page, the loader logs a warning and falls back
to the raw content — one bad image does not fail the page.

### B. Attachments as dependent documents (`include_attachments`)

When enabled, each unique `/.attachments/…` reference in a page's raw markdown
is picked up — from **both** image syntax `![alt](/.attachments/foo.png)` and
plain link syntax `[label](/.attachments/foo.md)`. The latter is important
because non-image attachments (PDFs, `.md`, `.docx`, anything a user drops onto
a wiki page) render as ordinary links, not as `![…]`, and would be missed by an
image-only regex. Once matched, the attachment is:

1. Downloaded as bytes from the wiki's backing git repo (`wikiMaster` branch),
   via `ReposApiWrapper.download_file` — the same code path the tool-side
   attachment handling already uses.
2. Filtered against `include_extensions` / `skip_extensions` (glob patterns,
   Confluence semantics).
3. Yielded as a **dependent** `Document` from `_process_document`, with
   `IndexerKeywords.CONTENT_IN_BYTES` and `IndexerKeywords.CONTENT_FILE_NAME`
   set so the indexer pipeline hands the bytes to `parse_file_content` — which
   dispatches to the right loader (LLM description for images/PDFs, text for
   supported formats).
4. Keyed by `f"{parent_page_id}::{attachment_path}"` and dedup'd by
   `sha256(attachment_bytes)` in `updated_on`, so unchanged attachments do not
   re-embed across indexing runs.

External image URLs (`http…`) are **not** indexed as attachments — they are not
stored in the wiki repo, and pulling arbitrary internet URLs on every reindex is
not something the indexer should do. If you want their descriptions in the
search corpus, use option A.

### Cross-cutting refactor

`_process_images` used to construct a `ReposApiWrapper` per invocation. Because
the indexer calls it once per page, that construction was extracted into
`_get_repos_wrapper(wiki_identified)` and cached per wiki on the instance, so a
run over N pages builds the wrapper once.

## New indexing parameters

Added to `AzureDevOpsApiWrapper._index_tool_params`:

| Param | Type | Default | Purpose |
|---|---|---|---|
| `process_images` | `bool` | `False` | Inline LLM image descriptions in each page's content (Option A). |
| `image_description_prompt` | `str \| None` | `None` | Custom prompt for image descriptions. Applied to both A and B. |
| `include_attachments` | `bool` | `False` | Also index `/.attachments/` files as dependent docs (Option B). |
| `include_extensions` | `list[str]` | `[]` | Glob patterns to allow-list attachments (empty = all). |
| `skip_extensions` | `list[str]` | `[]` | Glob patterns to skip; evaluated first. |

Pre-existing params (`chunking_tool`, `wiki_identifier`, `path_contains`) are
unchanged.

## How the pieces fit

```
_base_loader(wiki_identifier, path_contains, process_images, include_attachments, …)
  ├─ for each wiki page:
  │    raw_content = ADO GET page (include_content=True)
  │    updated_on  = sha256(raw_content)                       ← deterministic
  │    if process_images:  content = _process_images(raw_content)   (Option A)
  │    if include_attachments: metadata["_ado_wiki_attachments"]
  │                              = [/.attachments/… URLs from raw_content]
  │    yield Document(page_content=content, metadata=…)
  │
_process_document(doc)                                          (Option B)
  └─ if include_attachments and doc has attachments:
       for each /.attachments/ URL:
         apply include/skip extension filters
         download bytes from wikiMaster branch  (cached ReposApiWrapper)
         yield Document(loader_content=bytes, loader_content_type=".ext",
                        updated_on=sha256(bytes), id=f"{page_id}::{path}")

_remove_metadata_keys()
  └─ strips the transient "_ado_wiki_attachments" list before vector-store write
```

## Choosing which option

- **Only A** — cheapest way to make image content searchable. Descriptions appear
  in-context inside each page's embedding, so a query like "the diagram of the
  auth flow" retrieves the surrounding page. No separate rows.
- **Only B** — matches the Confluence pattern. Each attachment is its own
  document, retrievable directly, and the parent page's own text is unchanged.
  Good when consumers want to point at the raw file.
- **Both** — image content searchable both inline in the parent page and as
  standalone rows. Doubles the LLM cost per image but each surface exists for a
  different query shape.
- **Neither (default)** — original behavior.

## Deduplication & re-index behavior

- Page `updated_on` = sha256 of raw ADO markdown → stable even with option A on.
  Only pages whose source changed re-embed.
- If any attachment for a page fails, the page's `updated_on` is suffixed with
  a deterministic partial marker so it re-processes next run — see
  [Failure recovery](#failure-recovery).
- Attachment `updated_on` = sha256 of the downloaded blob → unchanged files stay
  cached across runs.
- Dependent docs are keyed as `parent_id::attachment_path`, so the same
  `.attachments/foo.png` referenced by two pages is indexed once per page (with
  each parent's context). If you'd rather share a single row across pages, key
  by blob SHA — a small follow-up.

## Cost considerations

- Option A: one LLM call per referenced image per page whose source changed.
  Large image-heavy wikis will be slow on first run; incremental runs are cheap.
- Option B: one LLM call per unchanged/new attachment on first run;
  `sha256(bytes)` dedup skips the LLM afterwards.
- External images downloaded via `requests.get(url)` in option A: those still
  fetch on every run because there's no place to cache the remote bytes. If
  external images are common, prefer copying them into `/.attachments/` and
  turning on option B.

## Skipped-item accounting

Uses the tracking helpers already on `NonCodeIndexerToolkit`:

- Attachment filtered by extension → `_track_runtime_skipped(name, "extension")`
- Attachment download failed (after retries) → `_track_dependent_item_skipped(name)`
- Attachment downloaded but empty → `_track_skipped_file_empty(name)`
- Repo wrapper couldn't be built (e.g., wiki lookup failed) → every attachment
  on those pages is marked `_track_dependent_item_skipped`.

All of these surface in the standard `get_indexing_stats_summary()` output.

## Failure recovery

Attachment downloads and dedup semantics were designed together so a transient
error on one attachment does not permanently strand it. Two mechanisms:

**Download retry.** `_download_attachment_with_retry` wraps
`ReposApiWrapper.download_file` in a bounded exponential backoff (3 total
attempts, 0.5s → 1.0s → 2.0s), retrying only on transient exception classes
(`requests.RequestException`, `ConnectionError`, `TimeoutError`,
`AzureDevOpsServiceError`). Non-transient errors propagate on the first
attempt. The wrapper is instance-cached, so retries reuse the same connection.

**Parent-hash perturbation on partial failure.** If any attachment for a page
fails (download exhausted retries, downloaded blob was empty, or the wiki's
repo wrapper couldn't be constructed at all), the parent page's `updated_on`
gets a deterministic suffix: `{raw_hash}::partial::{sha256(sorted failed
names)[:12]}`. Because `_reduce_duplicates` compares `updated_on` verbatim,
next run:

- If the failure resolves — new hash is clean `raw_hash`, dedup mismatches
  the stored partial-marked hash → parent + all attachments re-processed →
  clean hash written → subsequent runs see no change and skip.
- If the failure persists with the same failure set — same partial-marked
  hash written back, dedup sees stored != incoming `raw_hash` on every run
  and re-processes. This intentionally trades embedding cost for surfacing
  a persistent problem (a permanently 404'd attachment path, a stale
  reference, or a hard permission issue) instead of hiding it.

Extension-filter mismatches do **not** perturb the parent — filtering is an
explicit user choice, not a failure, so filtered attachments won't trigger
re-processing next run.

## Backwards compatibility

- Existing agents/pipelines calling `index_data` without the new params behave
  exactly as before (all new flags default off).
- No changes to public tool schemas (`get_wiki_page*` still accept
  `process_images` + `image_description_prompt` at call time).
- `AdoConfiguration`/toolkit schema unchanged — nothing to migrate in the UI.

## Follow-ups (not in this change)

- Test suite: add a wiki page with an inline image + a `.attachments/` PDF to
  `.elitea/tests/test_pipelines/suites/ado` and assert the image description
  turns up in a query when the new flags are on.
- Share attachment rows across pages by keying dependent docs on blob SHA
  instead of `parent_id::path`.
- Vision model override: today image descriptions use `self.llm`. If teams want a
  separate vision-capable model for indexing, expose a `bins_with_llm`-style
  param like the Confluence toolkit has.
