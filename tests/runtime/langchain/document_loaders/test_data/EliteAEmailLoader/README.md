# EliteAEmailLoader Test Suite

## Test Coverage

- **Test files**: 6 (5 .eml + 1 .msg)
- **Total configs**: 9
- **Parameters tested**: `process_attachments`, `max_tokens`
- **Supported formats**: .eml (RFC 5322), .msg (Outlook)

## Test Breakdown

### EML Format Tests

| File | Configs | Purpose |
|------|---------|---------|
| `email_simple.eml` | 2 | Basic plain text email with sender/recipient/subject |
| `email_html.eml` | 2 | Multipart email with HTML and plain text alternatives |
| `email_unicode.eml` | 1 | Unicode characters (German, Chinese, Japanese, Arabic, emoji) |
| `email_empty.eml` | 1 | Email with empty body - validates edge case handling |
| `email_with_attachment.eml` | 1 | Email with text file attachment - validates attachment metadata |

### MSG Format Tests

| File | Configs | Purpose |
|------|---------|---------|
| `msg_with_attachment.msg` | 2 | Outlook MSG format with TIF attachments - validates MSG parsing |

## Configuration Matrix

### email_simple.eml
- **Config 0** (`{"process_attachments": true, "max_tokens": -1}`): Full processing, no chunking
- **Config 1** (`{"process_attachments": false, "max_tokens": -1}`): Skip attachments

### email_html.eml
- **Config 0** (`{"process_attachments": true, "max_tokens": -1}`): Full multipart processing
- **Config 1** (`{"process_attachments": true, "max_tokens": 256}`): With chunking

### email_unicode.eml
- **Config 0** (`{"process_attachments": true, "max_tokens": -1}`): Unicode content handling

### email_empty.eml
- **Config 0** (`{"process_attachments": true, "max_tokens": -1}`): Empty body edge case

### email_with_attachment.eml
- **Config 0** (`{"process_attachments": true, "max_tokens": -1}`): Attachment metadata extraction

### msg_with_attachment.msg
- **Config 0** (`{"process_attachments": true, "max_tokens": -1}`): Full MSG processing with attachments
- **Config 1** (`{"process_attachments": false, "max_tokens": -1}`): MSG processing without attachment content

## Coverage Metrics

### allowed_to_override Parameters

From `constants.py`:
```python
'.eml': {
    'allowed_to_override': {**DEFAULT_ALLOWED_BASE, 'process_attachments': True}
}
```

**Coverage**: 2/2 parameters (100%)
- ✅ `max_tokens`: Tested with -1 (no limit) and 256
- ✅ `process_attachments`: Tested with true and false

## Expected Behavior

### Email Parsing:
1. **Metadata extraction**: From, To, Cc, Subject extracted to document metadata
2. **Content format**: Markdown-like output with headers and body
3. **Multipart handling**: Both HTML and plain text parts processed
4. **Attachment support**: Nested attachments processed when `process_attachments=true`

### Output Structure:
```json
{
  "page_content": "**Subject:** ...\n**From:** ...\n**To:** ...\n**Date:** ...\n\n---\n\n<body content>\n\n---\n**Attachment: filename.ext**\n\n<attachment content>",
  "metadata": {
    "is_empty_body": false,
    "from": ["sender@example.com"],
    "to": ["recipient@example.com"],
    "subject": "Email Subject",
    "date": "2024-01-01T10:00:00+00:00",
    "has_attachment": true,
    "attachment": ["filename.ext"]
  }
}
```

### Metadata Fields:
- `is_empty_body`: Boolean indicating if email body was empty
- `from`: List of sender addresses
- `to`: List of recipient addresses
- `subject`: Email subject line
- `date`: Email date/time (ISO 8601 format)
- `cc`: List of CC recipients (if present)
- `has_attachment`: Boolean indicating if email has attachments
- `attachment`: List of attachment filenames (only present when `has_attachment` is true)

### Attachment Processing:
When `process_attachments=true`, nested attachments within emails are:
1. Extracted from the email structure
2. Parsed using the appropriate loader based on file extension (same loaders as used by ADO, Jira, etc.)
3. Appended to the document with `---\n**Attachment: filename**\n\n<content>` format

Supported attachment formats include: .txt, .pdf, .docx, .xlsx, .csv, .json, .md, .html, .xml, images, and more (same as `loaders_map` in constants.py).

Note: `source` is NOT included in metadata (handled externally by indexing system)

## Running Tests

```bash
# Run all email loader tests
pytest tests/runtime/langchain/document_loaders/test_elitea_email_loader.py -v

# Run specific input file
pytest tests/runtime/langchain/document_loaders/test_elitea_email_loader.py -v -k "email_simple"

# Run all email tests with mark
pytest -m "loader_email" -v

# Run encoding edge case tests
pytest -m "edge_encoding" -v
```

### Filtering by Tag

| Tag | pytest mark | Applied to |
|-----|-------------|------------|
| `loader:email` | `loader_email` | All email configs |
| `content:simple` | `content_simple` | email_simple |
| `content:html` | `content_html` | email_html |
| `format:eml` | `format_eml` | All .eml files |
| `edge:encoding` | `edge_encoding` | email_unicode |
| `edge:empty-input` | `edge_empty_input` | email_empty |
| `feature:attachment` | `feature_attachment` | email_with_attachment |

## Notes

- Uses `unstructured` library for email parsing (already a dependency)
- Supports both .eml (RFC 5322) and .msg (Outlook) formats
- Lazy import of chunker to avoid circular import issues
- Empty emails return document with empty page_content

## MSG Format Support

MSG attachment extraction uses the `extract_msg` library (already a dependency) to properly 
extract attachments from Outlook MSG files. This is handled by `_extract_attachments_with_content_from_msg()`.

The test `msg_with_attachment.msg` contains 2 TIF attachments which are correctly detected 
and listed in the `attachment` metadata field.
