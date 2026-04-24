SharePoint Change Tracker Pipeline
 
Purpose: Track changes in SharePoint folders across runs, maintaining version history,
         computing diffs, and managing artifact storage with automatic retention policies.
 
==============================================================================
QUICK START GUIDE (High-Level Overview)
==============================================================================
 
What This Pipeline Does:
-------------------------
1. Monitors a SharePoint folder for file changes (new, modified, removed)
2. Downloads full content of new and changed files
3. Tracks version history (current + previous versions)
4. Computes unified diffs between versions
5. Archives removed files for audit trail
6. Maintains 5-run retention window
 
Input:
------
  {"folder_path": "/sites/MySite/Shared Documents", "bucket_name": "optional-bucket"}
 
What's Tracked from SharePoint:
--------------------------------
  • File metadata: Name, Path, Modified date, Created date, Link, ID
  • File content: Full text content for new and changed files
  • Change types: New files, Modified files, Removed files
 
What's Stored in Artifact Storage:
-----------------------------------
  retrieval_log.txt              → Last 5 run timestamps
  snapshot.json                  → Current state (metadata of all files)
  changes_YYYYMMDD_HHMMSS.json   → Per-run change report (last 5 kept)
  file_[Id]_[name].txt           → Current version (full content)
  file_prev_[Id]_[name].txt      → Previous version (for changed files)
  file_diff_[Id]_[name].txt      → Unified diff (current vs previous)
  file_del_[Id]_[name].txt       → Archived removed files
 
High-Level Flow:
----------------
  ┌─────────────────┐
  │ SharePoint API  │  (get_files_list → read_document)
  └────────┬────────┘
           │ File metadata + content
           ▼
  ┌─────────────────┐
  │ Change Detection│  Compare current vs previous snapshot
  └────────┬────────┘
           │ Classify: new / changed / removed / unchanged
           ▼
  ┌─────────────────┐
  │ Version Control │  Copy file_* → file_prev_* (before update)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Download Content│  Get full text from SharePoint
  └────────┬────────┘
           │ Save as file_[Id]_[name].txt
           ▼
  ┌─────────────────┐
  │ Compute Diffs   │  difflib.unified_diff(prev, current)
  └────────┬────────┘
           │ Save as file_diff_[Id]_[name].txt
           ▼
  ┌─────────────────┐
  │ Cleanup Removed │  file_* → file_del_*, delete prev/diff
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Retention       │  Keep last 5 runs (auto-delete old changes_*.json)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Output Report   │  Summary + artifact filenames
  └─────────────────┘
 
Retention Policy:
-----------------
  • retrieval_log.txt: Last 5 timestamps
  • changes_*.json: Last 5 change reports
  • file_*: Current files only (1 per SharePoint file)
  • file_prev_*: Recent changed files (cleaned up after overwrite)
  • file_diff_*: Recent changed files (cleaned up after overwrite)
  • file_del_*: Removed files (persisted until manual cleanup)
 
==============================================================================
DETAILED SPECIFICATION
==============================================================================
 
Architecture: 21 nodes, 1 toolkit (LFAInternalSharePoint), artifact via alita_client
Storage:      alita_client.artifact(bucket_name) — configured via state variable (default: 'sp-log')
 
Pipeline Flow (21 nodes):
==============================================================================
 
  ParseInput → ValidationRouter
    │                    │
    │ (invalid)          │ (valid)
    ▼                    ▼
  FormatError → END    LoadState
                         │  reads: retrieval_log.txt, snapshot.json
                         ▼
                       GetFilesFromSharepoint  (toolkit: get_files_list)
                         │
                         ▼
                       NormalizeFiles
                         │  serializes datetime objects → ISO strings
                         ▼
                       CompareSnapshots
                         │  detects: new / removed / changed / unchanged
                         ▼
                       SaveState
                         │  writes: retrieval_log.txt (last 5), snapshot.json,
                         │          changes_YYYYMMDD_HHMMSS.json (last 5, auto-cleanup)
                         ▼
                       PrepareVersioning
                         │  builds: files_to_version, files_to_delete, files_to_archive
                         ▼
                 ┌──► CheckVersioningNeeded
                 │      │ (more)          │ (done)
                 │      ▼                 ▼
                 │  VersionFileAndAdvance  CheckRemovalNeeded ──┐
                 │      │  read file_*       │ (more) │ (done)  │
                 │      │  write file_prev_*  ▼        ▼         │
                 └──────┘              CleanupRemovedFileAndAdvance CheckDownloadNeeded ──┐
                                          │  rename file_→file_del_ │ (more) │ (done)    │
                                          │  cleanup file_prev/diff  ▼        ▼           │
                                          └──────────────► CheckRemovalNeeded PrepareDiffComputation
                                                                                 │          │
                                                                                 ▼          ▼
                                                                          PrepareDownload  CheckDiffNeeded ──┐
                                                                                 │       │ (more) │ (done)  │
                                                                                 ▼       ▼        ▼          │
                                                                          DownloadFileContent ComputeDiffAndAdvance FormatEnhancedOutput
                                                                                 │  (toolkit)  │ compute diff │
                                                                                 ▼             │ write file_diff_*│
                                                                          SaveFileAndAdvance  └────────────────┘
                                                                                 │  write file_*         │
                                                                                 │                       ▼
                                                                                 └──────────────────► CheckDownloadNeeded
                                                                                                         │
                                                                                                         ▼
                                                                                                       END
 
Data Sources & Storage:
========================
 
From SharePoint (via toolkit LFAInternalSharePoint):
-----------------------------------------------------
  get_files_list:
    • Retrieves metadata for all files in folder
    • Fields: Name, Path, Modified, Created, Link, id
    • No content downloaded at this stage
    • Returns datetime objects (requires NormalizeFiles for serialization)
  
  read_document:
    • Downloads full text content of individual files
    • Called only for new and changed files (not unchanged)
    • Content stored as file_[Id]_[name].txt
 
Artifact Storage (via alita_client.artifact(bucket_name)):
-----------------------------------------------------------
  READ Operations:
    • retrieval_log.txt     → Load last run timestamp
    • snapshot.json         → Load previous state (file metadata)
    • file_[Id]_[name].txt  → Load current version (for versioning before update)
  
  WRITE Operations:
    • retrieval_log.txt              → Append current timestamp, keep last 5
    • snapshot.json                  → Save current state (metadata only, no content)
    • changes_YYYYMMDD_HHMMSS.json   → Save change report with artifact filenames
    • file_[Id]_[name].txt           → Save current file content from SharePoint
    • file_prev_[Id]_[name].txt      → Version previous content before update
    • file_diff_[Id]_[name].txt      → Save unified diff (prev vs current)
    • file_del_[Id]_[name].txt       → Archive removed file (copy of file_*)
  
  DELETE Operations:
    • changes_*.json (older than 5 runs)  → Retention cleanup
    • file_[Id]_[name].txt (removed files) → After copying to file_del_*
    • file_prev_*/file_diff_* (removed)    → Cleanup obsolete versions
 
State Files Detail:
===================
 
retrieval_log.txt:
------------------
  Format:  One timestamp per line (ISO 8601, seconds precision)
  Example: 2026-04-14T10:00:00
           2026-04-14T11:30:00
           2026-04-14T13:15:00
  Purpose: Track run history, enable retention cleanup logic
  Retention: Last 5 timestamps only
 
snapshot.json:
--------------
  Structure: { "path": { "Name": "...", "Modified": "...", "Created": "...", "Link": "...", "id": "..." } }
  Example:   { "/sites/MySite/doc.txt": { "Name": "doc.txt", "Modified": "2026-04-14T10:00:00", ... } }
  Content:   Metadata only (no file content)
  Purpose:   Enable change detection (compare current vs previous state)
  Note:      Overwritten each run with current state
 
changes_YYYYMMDD_HHMMSS.json:
------------------------------
  Structure: {
               "retrieval_date": "2026-04-14T10:00:00",
               "summary": { "new_files_count": 1, "removed_files_count": 0, "changed_files_count": 2 },
               "new_files": [ { "Name": "...", "Path": "...", "artifact_storage": { "current": "file_..." } } ],
               "removed_files": [ { "Name": "...", "artifact_storage": { "deleted": "file_del_..." } } ],
               "changed_files": [ { "Name": "...", "artifact_storage": { "current": "...", "previous": "...", "diff": "..." } } ]
             }
  Purpose:   Per-run audit trail with artifact storage mapping
  Retention: Last 5 files kept (older files deleted when 6th run completes)
 
File Naming Convention:
=======================
  Current version:  file_[Id]_[sanitized_name].txt
  Previous version: file_prev_[Id]_[sanitized_name].txt
  Diff file:        file_diff_[Id]_[sanitized_name].txt
  Deleted file:     file_del_[Id]_[sanitized_name].txt (renamed from file_*)
 
  Sanitization: re.sub(r'[^\w\-]', '_', name.replace(' ', '-'))  
                (spaces → hyphens, dots/special chars → underscores)
  Example: "java doc.pdf" → file_12345_java-doc_pdf.txt
 
Removed Files Handling:
=======================
  When a file is removed from SharePoint:
  1. Copy file_[Id]_[name].txt → file_del_[Id]_[name].txt (preserve last state)
  2. Delete original file_[Id]_[name].txt
  3. Delete file_prev_[Id]_[name].txt (no longer needed)
  4. Delete file_diff_[Id]_[name].txt (no longer needed)
 
NormalizeFiles Node:
====================
  Purpose:   Serialize datetime objects from SharePoint API to ISO strings
  Why:       SharePoint toolkit returns Python datetime objects which corrupt
             state serialization (falls back to str(state) instead of JSON)
  Method:    Detects if alita_state is a string (poisoned state), uses regex
             to replace datetime.datetime(...) with ISO string literals,
             then ast.literal_eval() to recover the dict
  Fallback:  If state is already a dict, converts any datetime values to
             isoformat() strings
  Output:    Clean all_files list with datetime → ISO string conversion
  Debug:     Includes debug_info with recovery method, conversion counts
 
Diff Computation:
=================
  For each changed file:
  1. Read current version (file_[Id]_[name].txt)
  2. Read previous version (file_prev_[Id]_[name].txt)
  3. Compute unified diff using Python's difflib.unified_diff
  4. Add metadata header (timestamp, file paths, line counts)
  5. Save as file_diff_[Id]_[name].txt in artifact storage
 
Changes Report (changes_*.json):
=================================
  Each file entry includes an 'artifact_storage' object with filenames:
  - New files:     { "artifact_storage": { "current": "file_[Id]_[name].txt" } }
  - Changed files: { "artifact_storage": { "current": "...", "previous": "...", "diff": "..." } }
  - Removed files: { "artifact_storage": { "deleted": "file_del_[Id]_[name].txt" } }
 
Retention Policy:
=================
  To prevent unbounded storage growth:
  - retrieval_log.txt: Keeps only last 5 retrieval dates (format: YYYY-MM-DDTHH:MM:SS, no microseconds)
  - changes_*.json: Keeps only last 5 change reports (format: changes_YYYYMMDD_HHMMSS.json)
  
  Timestamp Consistency:
  - Timestamps are truncated to seconds (microseconds removed) for consistent matching
  - retrieval_log.txt: 2026-04-14T12:43:10 (ISO format, seconds precision)
  - changes_*.json: changes_20260414_124310.json (compact format from same timestamp)
  
  When a retrieval date is removed from the log:
  1. Identify corresponding changes_YYYYMMDD_HHMMSS.json file
  2. Delete the file using store.delete() (true S3 deletion)
  3. Create error_deletion_*.txt if deletion fails for diagnostics
 
Artifact Lifecycle (Run-to-Run):
=================================
 
FIRST RUN (baseline):
---------------------
  Created:
    - retrieval_log.txt              → [2026-04-14T10:00:00]
    - snapshot.json                  → { "file1": {metadata}, "file2": {metadata}, ... }
    - changes_20260414_100000.json   → { new_files: [file1, file2, ...], removed_files: [], changed_files: [] }
    - file_[Id1]_[name1].txt         → content of file1 from SharePoint
    - file_[Id2]_[name2].txt         → content of file2 from SharePoint
    - ... (one file_* per SharePoint file)
 
SECOND RUN (changes detected):
------------------------------
  Scenario: file1 unchanged, file2 changed, file3 newly added, file4 removed
 
  Updated:
    - retrieval_log.txt              → [2026-04-14T10:00:00, 2026-04-14T11:00:00]
    - snapshot.json                  → { "file1": {metadata}, "file2": {new metadata}, "file3": {metadata} }
                                        (file4 removed from snapshot)
 
  Created:
    - changes_20260414_110000.json   → { new_files: [file3], removed_files: [file4], changed_files: [file2] }
    - file_[Id3]_[name3].txt         → content of file3 (new file)
    - file_prev_[Id2]_[name2].txt    → previous content of file2 (copied from file_[Id2]_[name2].txt before update)
    - file_diff_[Id2]_[name2].txt    → unified diff between prev and current versions of file2
    - file_del_[Id4]_[name4].txt     → last known content of file4 (renamed from file_[Id4]_[name4].txt)
 
  Overwritten:
    - file_[Id2]_[name2].txt         → new content of file2 from SharePoint
 
  Deleted (true S3 deletion):
    - file_[Id4]_[name4].txt         → deleted after copying to file_del_*
    - file_prev_[Id4]_[name4].txt    → deleted (no longer needed)
    - file_diff_[Id4]_[name4].txt    → deleted (no longer needed)
 
SIXTH RUN (retention cleanup):
-------------------------------
  Scenario: 6th run triggers cleanup (retention: keep last 5)
 
  Updated:
    - retrieval_log.txt              → [2026-04-14T11:00:00, ..., 2026-04-14T15:00:00] (oldest entry removed)
    - snapshot.json                  → current state of all SharePoint files
 
  Created:
    - changes_20260414_150000.json   → changes detected in this run
    - file_*, file_prev_*, file_diff_* per changes
 
  Deleted (via store.delete()):
    - changes_20260414_100000.json   → oldest change report removed (>5 runs old)
 
  Error Handling (if deletion fails):
    - error_deletion_20260414_100000.txt → diagnostic log created
 
STEADY STATE (run N, N > 5):
-----------------------------
  Each run maintains:
    - retrieval_log.txt: Last 5 timestamps
    - snapshot.json: Current state (1 file)
    - changes_*.json: Last 5 change reports
    - file_[Id]_*.txt: One per current SharePoint file
    - file_prev_[Id]_*.txt: One per recently changed file (versioning)
    - file_diff_[Id]_*.txt: One per recently changed file (diffs)
    - file_del_[Id]_*.txt: One per recently removed file (archive)
 
  Artifact Operations:
    CREATE: New file_* for new SharePoint files, new changes_*.json each run
    UPDATE: retrieval_log.txt, snapshot.json, file_* for changed files
    DELETE: Oldest changes_*.json when exceeding 5 runs
    RENAME: file_* → file_del_* for removed SharePoint files
    VERSION: file_* → file_prev_* before updating changed files
 
Storage Growth Pattern:
=======================
  Fixed artifacts:  2 (retrieval_log.txt, snapshot.json)
  Per-run reports:  5 (changes_*.json, auto-cleanup after 5 runs)
  Per-file current: N (file_[Id]_*.txt, where N = number of files in SharePoint folder)
  Per-change prev:  M (file_prev_[Id]_*.txt, where M = recently changed files not yet overwritten)
  Per-change diff:  M (file_diff_[Id]_*.txt, same as prev)
  Per-removal del:  R (file_del_[Id]_*.txt, where R = recently removed files not yet purged)
 
  Total artifacts ≈ 7 + N + 2M + R (bounded by active SharePoint files + recent changes)
 
==============================================================================