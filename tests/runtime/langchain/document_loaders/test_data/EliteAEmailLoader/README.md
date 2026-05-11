# EliteAEmailLoader Test Suite

## Test Coverage

- **Test files**: 5 (.eml format)
- **Total configs**: 7
- **Parameters tested**: `process_attachments`, `max_tokens`
- **Supported formats**: .eml (RFC 5322), .msg (Outlook)

## Test Breakdown

| File | Configs | Purpose |
|------|---------|---------|
| `email_simple.eml` | 2 | Basic plain text email with sender/recipient/subject |
| `email_html.eml` | 2 | Multipart email with HTML and plain text alternatives |
| `email_unicode.eml` | 1 | Unicode characters (German, Chinese, Japanese, Arabic, emoji) |
| `email_empty.eml` | 1 | Email with empty body - validates edge case handling |
| `email_with_attachment.eml` | 1 | Email with text file attachment - validates attachment metadata |

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
  "page_content": "**Subject:** ...\n**From:** ...\n**To:** ...\n\n---\n\n<body content>",
  "metadata": {
    "is_empty_body": false,
    "from": ["sender@example.com"],
    "to": ["recipient@example.com"],
    "subject": "Email Subject",
    "has_attachment": false
  }
}
```

### Metadata Fields:
- `is_empty_body`: Boolean indicating if email body was empty
- `from`: List of sender addresses
- `to`: List of recipient addresses
- `subject`: Email subject line
- `cc`: List of CC recipients (if present)
- `has_attachment`: Boolean indicating if email has attachments
- `attachment`: List of attachment filenames (only present when `has_attachment` is true)

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
