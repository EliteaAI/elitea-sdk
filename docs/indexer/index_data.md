# index_data

The `index_data` method orchestrates the full pipeline for indexing documents into a vector store collection.  
It manages document loading, optional index cleaning, dependency processing, chunking, metadata cleaning, and final storage.  
This method is designed to be extensible and robust for various toolkits and indexing requirements.

---

### 1. Index Cleaning

The method calls `_clean_index` (according to arguments) to remove all existing data from the target collection.

---

### 2. Document Loading

`_base_loader` is called to load the initial high-level data for **base documents** from the data source.  
The toolkit is expected to handle empty sources or errors gracefully.

---

### 3. Duplicate Removal

The method `_reduce_duplicates` is responsible for filtering out already-indexed documents based on previously loaded high-level data (mainly value of key 'updated_on' is used).

---

### 4. Data Extension

This stage is to update or enriche only non-duplicated documents.  
Resource- and time-consuming operations are expected to be performed here via `_extend_data`.  
The type of documents (file extension) and raw content (bytes) of base document should be defined at this stage.

---

### 5. Dependency Collection

Calls `_collect_dependencies` to extract all **dependent documents** on or attached to the base one.  
The type of documents (file extension) and raw content (bytes) of dependent document should be defined at this stage.

---

### 6. Chunking/Parsing

Calls `_apply_loaders_chunkers` to process and split the content of all previously collected **base and dependent documents** using the specified chunking tool and configuration.
At this stage chunking config is used. Settings for particular document types (file extensions) are used to override default one for corresponding loaders/chunkers.

---

### 7. Metadata Cleaning

Calls `_clean_metadata` to remove unnecessary or sensitive metadata fields from each document.  
This prevents leaking sensitive information and reduces index size.

---

### 8. Saving to Index

Calls `_save_index` to persist the processed documents into the vector store, using progress reporting if configured.  
Corner case: Handles empty document lists and ensures atomicity of the save operation.

---

## Common Parameters for `index_data`

- `collection_suffix` (`str`):  
  Suffix for the collection name (max 7 characters) used to separate datasets.

- `progress_step` (`Optional[int]`):  
  Optional step size for progress reporting during indexing (default: 10, range: 0-100).

- `clean_index` (`Optional[bool]`):  
  Optional flag to enforce cleaning the existing index before indexing new data (default: `False`).

- `chunking_config` (`Optional[dict]`):  
  Chunking tool configuration (default: empty dict).   
  **Example:** `{".pdf": {"mode": "page", "chunk_size": 1000}, ".docx": {"mode": "paragraph"}, ".md": {"chunk_size": 333}}`

---

## Toolkit Specific Parameters for `index_data`

Other extra parameters (for particular toolkit needs) can be provided by the implementation of `_index_tool_params()`.

---

## _chunking_config_ Parameter for `index_data`

It is intended to provide unified settings for content parsers handling both base and dependent documents.  
Some parsers support LLM capabilities for advanced content processing (for example, a PDF parser that analyzes images).  
To enable LLM processing for such parsers, set _use_llm_ in chunking_config to _true_.  
The optional _prompt_ parameter can be used to guide the LLM in processing the document content.  
Set _use_prompt_default_ to _true_ to use built-in detailed prompt to process images.  
Note: the complexity of a prompt may affect processing time and token usage.  
**Example:** `{".pdf": {"use_llm": true, "prompt": "Provide only information regarding colors and wrap output with image tag."}}`

### Excel (`.xlsx`, `.xls`) chunk size

Excel sheets are split into chunks of rows of at most _max_tokens_ tokens each (default: `512`), so a large sheet does not overflow the model's context window.
A single row longer than _max_tokens_ becomes a chunk of its own. Set _add_header_to_chunks_ to _true_ to repeat the header row at the top of every chunk.
Set _max_tokens_ to `-1` to index each sheet as one chunk (the pre-R-2.0.9 default).  
**Example:** `{".xlsx": {"max_tokens": 1000, "add_header_to_chunks": true}, ".xls": {"max_tokens": -1}}`

`read_file` is not affected: it still returns each sheet as one chunk.

---

## Page and section fields of PDF and Word chunks

Chunks of PDF and Word files carry these metadata fields. Search results return them as stored.

| Field | Files | Value |
|---|---|---|
| `page` | PDF | Zero-based page index (unchanged). |
| `page_number` | PDF | `page + 1`: the page number a PDF viewer shows. Page labels printed in the text do not change it. |
| `page_number_from`, `page_number_to` | PDF | First and last page of the chunk. The loader makes one chunk per page, and splits a page by _max_tokens_ into parts, so both fields equal `page_number`. |
| `section_number`, `section_title` | PDF, Word | The closest heading above the start of the chunk. |

How the section is found:

- PDF: bookmarks (the outline) first. Without bookmarks, numbered headings in the text.
- Word: heading styles and numbering.
- When several headings are above the chunk, the deepest one is used (for example `2.1`, not `2`).
- A field is absent when it is unknown. A chunk before the first heading has the page fields only. Word files have no page fields.

### Existing indexes

An index run without `clean_index` skips files that did not change (see step 3). Their old chunks keep the fields they had when they were indexed: chunks from before these fields have `page` only. Nothing breaks, because a reader gets a chunk with fewer fields.

To add the new fields to old chunks, run the index with `clean_index` set to _true_ (Clean index in the UI). It removes the old chunks and indexes every file again. A reader of these fields must handle a missing key.

