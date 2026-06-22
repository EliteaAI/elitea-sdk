# Figma Indexing Flow

## Overview

The Figma indexing flow transforms Figma design files into searchable documents optimized for RAG (Retrieval-Augmented Generation). It combines structural data extraction with LLM-powered visual analysis to create rich, semantically meaningful documents.

### Key Features

- **TOON Format**: Token-Optimized Object Notation for compact, LLM-friendly output
- **Parallel Processing**: Multi-threaded page and frame processing
- **Resilient Fetching**: Bisection strategy for handling large files and API limits
- **Retry Mechanisms**: Automatic retry with backoff for transient errors
- **Streaming Output**: Documents yielded as they complete for memory efficiency
- **Frame Type Support**: Image frames (LLM-analyzed) and text-only frames (no LLM cost)
- **Eager Split Optimization**: Skip timeout-prone bulk requests for large batches

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              _base_loader()                                  │
│  Entry point: parses URLs/file keys, orchestrates per-file processing       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    _process_file_toon_streaming()                           │
│  Per-file orchestration: loads pages, spawns page processing threads        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  _load_pages()      │   │  _load_pages()      │   │  _load_pages()      │
│  Bisection strategy │   │  (parallel pages)   │   │  (parallel pages)   │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
              │                       │                       │
              ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      process_page_streaming()                               │
│  Per-page: fetch full content, collect frames, process with LLM             │
│  Runs in parallel threads (default: 5 threads)                              │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ├──► _collect_frames_for_analysis()  ──► Frame extraction + image URLs
              │
              ├──► process_page_to_toon_data()     ──► TOON serialization
              │
              └──► process_leaf_frame() [ThreadPool]
                            │
                    ┌───────┴───────┐
                    ▼               ▼
           ┌──────────────┐  ┌──────────────┐
           │ Image Frame  │  │ Text-Only    │
           │ (has_image)  │  │ Frame        │
           │              │  │              │
           │ LLM Analysis │  │ TOON Only    │
           │ type='image' │  │ type='text'  │
           └──────────────┘  └──────────────┘
                    │               │
                    └───────┬───────┘
                            ▼
                   ┌─────────────────┐
                   │  Result Queue   │  ◄── Documents streamed as completed
                   └─────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  yield Document │  ◄── Streaming to vectorstore
                   └─────────────────┘
```

---

## Detailed Flow

### 1. Entry Point: `_base_loader()`

**Location**: `api_wrapper.py:366`

Accepts comma-separated Figma URLs or file keys and orchestrates the indexing process.

```python
def _base_loader(
    urls_or_file_keys: str,           # "https://figma.com/file/ABC123,XYZ789"
    node_ids_include: List[str],       # Filter to specific pages
    node_ids_exclude: List[str],       # Exclude specific pages
    number_of_threads: int = 5,        # Parallel processing threads (1-5)
    image_max_dimension: int = 2000,   # Max image size for LLM
    subframe_extract_threshold: int,   # When to extract subframes
    scale_to_limit_threshold: int,     # Image scaling threshold
) -> Generator[Document, None, None]
```

**Process**:
1. Parse URLs/file keys into `(file_key, node_ids)` tuples
2. For each file, call `_process_file_toon_streaming()`
3. Yield documents as they complete (streaming)

---

### 2. Page Loading: `_load_pages()`

**Location**: `api_wrapper.py:498`

Fetches page content from Figma API using the bisection chunking strategy.

**Retry Decorator**: `@retry_on_server_error(max_attempts=3, wait_seconds=(1, 4, 10))`

```python
def _load_pages(file_key: str, page_ids: List[str]) -> List[Dict]
```

**Process**:
1. Create `BisectionChunkingStrategy` instance
2. Define `fetch_pages()` function with server error retry
3. Execute bisection strategy with `is_volume_error_retriable` checker
4. Return successfully loaded pages, track failures

---

### 3. Bisection Chunking Strategy

**Location**: `chunking_strategy.py:33`

Efficiently handles large API requests by binary splitting on failure.

```
Algorithm: Divide and Conquer

Input: [A, B, C, D, E, F, G, H] where C causes timeout

Step 1: Try [A-H] → FAIL (timeout) → split
Step 2: Try [A-D] → FAIL → split    |  Try [E-H] → SUCCESS ✓
Step 3: Try [A-B] → SUCCESS ✓       |  Try [C-D] → FAIL → split
Step 4:                              |  Try [C] → FAIL (min size) ✗
                                     |  Try [D] → SUCCESS ✓

Result: successful={A,B,D,E,F,G,H}, failed=[C]
```

**Key Features**:
- **Parallel Processing**: Left and right halves processed concurrently
- **Single-Item Isolation**: Only individual problematic items marked as failed
- **O(log N) Best Case**: Fewer API calls than sequential for large datasets
- **Eager Split**: Skip first try for batches known to timeout (saves 30-60s per batch)

```python
class BisectionChunkingStrategy:
    def __init__(self, min_chunk_size=1, max_workers=2, eager_split_threshold=None):
        self.min_chunk_size = min_chunk_size        # Stop splitting at this size
        self.max_workers = max_workers              # Parallel workers for halves
        self.eager_split_threshold = eager_split_threshold  # Skip first try if batch > threshold

    def execute(items, fetch_fn, is_retriable_error) -> ChunkResult:
        # Returns: successful dict, failed list, skipped list, stats
```

**Eager Split Thresholds** (based on empirical data):
- **Pages**: >10 pages → eager split (avoids ~100s timeout)
- **Images**: >50 images at scale=1.0 → eager split (avoids ~30s timeout)

---

### 4. Page Processing: `process_page_streaming()`

**Location**: `api_wrapper.py:1483`

Processes a single page and streams documents as LLM analysis completes.

**Retry Decorator**: `@retry_on_volume_error(max_attempts=4, wait_seconds=(1, 4, 10, 30))`

**Process**:
1. Fetch full page content with retry
2. Collect frames via `_collect_frames_for_analysis()`
3. Convert to TOON format via `process_page_to_toon_data()`
4. Find leaf frames (frames without subframes, including text-only)
5. Process leaf frames in parallel with `ThreadPoolExecutor`:
   - **Image frames** (`has_image=True`): Full LLM vision/text analysis
   - **Text-only frames** (`has_image=False`): TOON serialization only, no LLM cost
6. Queue completed documents for streaming

---

### 5. Frame Collection: `_collect_frames_for_analysis()`

**Location**: `api_wrapper.py:935`

Collects frames for analysis with intelligent subframe extraction.

**Process**:
1. Extract top-level frames from pages
2. Apply spatial ordering (top-to-bottom, left-to-right)
3. Detect large frames that need subframe extraction
4. Determine frame strategy for each frame:
   - `normal`: Standard processing with adaptive scaling
   - `scale_to_limit`: Scale down large frames to fit limits
   - `extract_subframes`: Extract child frames for individual processing
   - `text_only`: Text nodes, no image needed (`has_image=False`)
5. Fetch image URLs for frames with `has_image=True` (with scale optimization)
6. Return collected frames (with `has_image` field) and their image URLs

**Frame `has_image` Field**:
The `has_image` field is preserved through TOON processing to track which frames need images:
- `has_image=True`: Image frames requiring LLM analysis
- `has_image=False`: Text-only frames (labels, titles) skipping LLM

---

### 6. Image Fetching: `_get_file_images_with_scale()`

**Location**: `api_wrapper.py:1273`

Fetches image URLs with automatic scale optimization and bisection fallback.

**Retry Decorator**: `@retry_on_server_error(max_attempts=4, wait_seconds=(1, 4, 10, 30))`

**Scale Optimization**:
- Groups frames by optimal scale factor
- Fetches each scale group separately
- Respects `image_max_dimension` limit

**Bisection Fallback for Render Timeouts**:

When Figma returns "render timeout" (400 error) for large batches, the function
automatically applies bisection to recover:

```
Batch [A, B, C, D, E, F] → TIMEOUT (too many frames)
  ├── Batch [A, B, C] → SUCCESS ✓
  └── Batch [D, E, F] → TIMEOUT
        ├── Batch [D] → SUCCESS ✓
        └── Batch [E, F] → SUCCESS ✓
```

- Splits batch in half recursively on render timeout
- Individual frames that still timeout are skipped with warning
- Maximizes image URL recovery for large files

---

### 7. LLM Analysis: `analyze_frame_with_llm()`

**Location**: `toon_tools.py`

Analyzes frames using vision or text-based LLM.

**Retry Decorator**: `@retry_on_llm_error(max_attempts=3, wait_seconds=(1, 4, 10))`

**Process**:
1. Try vision analysis if image URL available
2. Fall back to text-based analysis on failure
3. Return structured `ScreenExplanation` with:
   - `purpose`: Screen's main purpose
   - `user_actions`: Available user actions
   - `visual_focus`: Key visual elements
   - `layout_pattern`: Layout description

---

## Retry Mechanisms

### Shared Retry Utilities

**Location**: `utils/retry.py`

Three categories of retriable errors with dedicated decorators:

| Decorator | Error Types | Use Case |
|-----------|-------------|----------|
| `@retry_on_server_error` | 5xx, 429 (rate limit) | API calls |
| `@retry_on_volume_error` | Timeout, payload too large, network | Large requests |
| `@retry_on_llm_error` | 5xx, 429, timeout | LLM invocations |

### Error Detection Functions

```python
is_server_error_retriable(e)   # 5xx, 429, connection errors
is_volume_error_retriable(e)   # Timeout, size errors, network issues  
is_llm_error_retriable(e)      # LLM-specific transient errors
```

### Retry Configuration

Default wait times: `(1, 4, 10, 30)` seconds between retries.

```python
@retry_on_server_error(
    max_attempts=4,              # Total attempts (1 initial + 3 retries)
    wait_seconds=(1, 4, 10, 30)  # Wait times between retries (uses first N-1)
)
```

---

## Multithreading Architecture

### Thread Hierarchy

```
Main Thread
    │
    ├── Page Processing Threads (5 default, configurable 1-5)
    │       │
    │       └── Frame Processing ThreadPool (per page)
    │               └── LLM calls (parallel per frame)
    │
    └── Bisection Workers (2 parallel for left/right halves)
```

### Synchronization

- **Result Queue**: Thread-safe queue for streaming documents
- **SENTINEL Pattern**: Signal completion of all page processing
- **Stats Lock**: Thread-safe statistics tracking in bisection

### Configuration

```python
number_of_threads: int = 5  # Valid range: 1-5
# Controls parallel page processing
# Each page spawns its own ThreadPoolExecutor for frames
```

---

## TOON Format Output

**Token-Optimized Object Notation** - Compact format for LLM consumption.

### Example Output

```
FILE: Mobile App [key:abc123]
  PAGE: Authentication
    FRAME: 01_Login [0,0 375x812] form/default
      Purpose: User authentication entry point
      Actions: Sign in with email | Forgot password | Create account
      Visual: [focus] Email and password fields | [layout] Centered form
      Headings: Welcome Back
      Buttons: Sign In > auth | Forgot Password? > reset
      Inputs: Email (email) | Password (password, secure)
```

### Document Metadata

Each yielded document includes:
- `file_key`: Figma file identifier
- `page_id`: Page node ID
- `page_name`: Human-readable page name
- `node_id`: Frame node ID
- `node_name`: Frame name
- `image_url`: Rendered frame image URL (empty for text-only frames)
- `image_width`, `image_height`: Frame dimensions
- `type`: Frame type - `'image'` (LLM-analyzed) or `'text'` (text-only, no LLM)
- `llm_issue`: Only present if LLM analysis had issues (omitted on success)

---

## Frame Types

The indexing flow supports two frame types to optimize cost and coverage:

### Image Frames (`type='image'`)

Full UI screens and components that benefit from visual analysis.

**Processing**:
1. Fetch image URL from Figma API
2. Run LLM vision analysis (or text fallback)
3. Generate rich TOON output with Purpose, Actions, Visual sections

**Example Output**:
```
FRAME: Login Screen [0,0 375x812] form/default #123:456
  Purpose: User authentication entry point
  Actions: Sign in with email | Forgot password
  Visual: [focus] Email and password fields | [layout] Centered form
  Headings: Welcome Back
  Buttons: Sign In | Forgot Password?
  Inputs: Email (email) | Password (password, secure)
```

### Text-Only Frames (`type='text'`)

Small text elements (labels, titles, headers) that don't need visual analysis.

**Processing**:
1. Skip image fetching (no `image_url`)
2. Skip LLM analysis (zero LLM cost)
3. Generate TOON output from extracted text content only

**Example Output**:
```
FRAME: Save Cost Simulation [502x65] text/default #66:22481
  Headings: Save Cost Simulation
  Buttons: Save | Cancel
```

**Benefits**:
- **Complete coverage**: All design elements are searchable
- **Cost efficient**: No LLM calls for simple text labels
- **Fast indexing**: Text-only frames process instantly

**Identification**:
Text-only frames are identified during frame collection:
- TEXT nodes in Figma → `strategy='text_only'`, `has_image=False`
- Very small elements (typically labels/titles) extracted from larger frames

---

## Error Handling

### Graceful Degradation

1. **Vision → Text Fallback**: If vision analysis fails, falls back to text-based
2. **Page Isolation**: Failed pages don't block other pages
3. **Frame Isolation**: Failed frames logged, others continue
4. **Bisection Recovery**: Isolates problematic items to single failures
5. **Image Fetch Bisection**: Render timeouts trigger batch splitting to recover URLs

### Tracking

```python
IndexingStats:
    items_processed: int           # Successfully indexed
    total_fetched: int             # Total attempted
    dependent_items_skipped: Set   # Failed sub-items (frames, images)
```

---

## Performance Considerations

### Optimization Strategies

1. **Streaming**: Documents yielded immediately, not batched
2. **Parallel Processing**: Multiple pages/frames processed concurrently
3. **Image Scale Groups**: Batched image fetching by scale factor
4. **Bisection**: O(log N) for bulk failures vs O(N) sequential

### Resource Limits

- `number_of_threads`: 1-5 (prevents API rate limiting)
- `image_max_dimension`: 2000px default (balances quality vs tokens)
- `max_frames_per_page`: Configurable limit per page

---

## Usage Example

```python
from elitea_sdk.tools.figma import FigmaApiWrapper

wrapper = FigmaApiWrapper(
    figma_api_token="...",
    llm=my_llm_instance,
)

# Index a Figma file
for doc in wrapper._base_loader(
    urls_or_file_keys="https://figma.com/file/ABC123",
    number_of_threads=3,
    image_max_dimension=1500,
):
    # Process each document as it streams
    print(f"Indexed: {doc.metadata['node_name']}")
```
