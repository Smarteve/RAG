# NexusRAG

An async multimodal document processing and retrieval-augmented generation (RAG) engine.

## What NexusRAG aims to do

Upload a PDF, ask questions in everyday language, and receive answers supported
by passages from the document, with page citations. Retrieval-augmented
generation (RAG) finds relevant passages and supplies them to an AI model as
evidence before it answers; uploading a document does not retrain the model.

### Example: ask about an employee handbook

Imagine uploading a 40-page employee handbook and asking:

> I joined six months ago. How many days of annual leave can I carry into next year?

If the handbook contains the relevant policy, an illustrative answer would be:

> You can carry over up to five unused days, with approval from your manager.
> They must be used by 31 March. The handbook does not specify a different rule
> for employees who joined six months ago. [Employee Handbook, p. 18]

This is a fictional example of the intended behavior, not a result from the
current application. Answers should identify missing information rather than
invent policies that the document does not support.

The planned flow is:

1. Upload and validate the PDF.
2. Extract text while preserving page numbers.
3. Split the text into passages and index them for search.
4. Retrieve relevant passages when a question arrives.
5. Generate an answer with citations to the supporting pages.

The project currently has upload validation, local PDF storage, page-by-page text
extraction, overlapping chunks with JSON storage and paginated fetching, tests,
Docker, and CI configuration. Retrieval and
AI answers are still to be implemented. The first RAG milestone is answering a
question about one text-based PDF with a supporting page citation.

## Run locally

```bash
uv run uvicorn nexusrag.main:app --reload
```

Then open <http://127.0.0.1:8000/docs>.

`POST /documents` saves accepted PDFs, waits for extraction and chunking in worker
threads, and saves both results as UTF-8 JSON. It returns `201 Created` with
`document_id`, nested `metadata`, `extraction_status`, `page_count`, and
`extraction_error`, plus `chunk_count`. Filesystem paths and full text are not
returned in the upload response.

The same UUID links the original and extracted data:

```text
data/documents/<document_id>.pdf
data/extracted/<document_id>.json
data/chunks/<document_id>.json
```

Extraction JSON contains `document_id`, `status`, `error`, and `pages` (objects
with `page_number` and `text`). Chunk JSON contains `document_id`, `chunk_size`,
`overlap`, and `chunks`. Each chunk keeps its document ID, zero-based chunk index,
one-based page number, exact text, and page-local `[start_char:end_char]` offsets.
The defaults are 1000 Python string characters per window with 200-character
overlap, not byte or token counts. Whitespace-only pages are skipped without
renumbering later pages; chunk text is not stripped.

Storage functions are async wrappers around synchronous worker-thread helpers.
All writes use a uniquely named temporary
file in the destination directory and close it before publishing the final path.

PDFs are published with a hard link, then the temporary name is removed. This
does not copy the bytes again and refuses to overwrite an existing document.
The storage filesystem must support hard links; unsupported filesystems fail
the upload rather than falling back to a partial or overwriting write. JSON
uses replacement instead, allowing an earlier extraction or chunk result to be replaced
only after the new JSON is complete.

Temporary files add directory operations and cleanup logic. Replacing JSON
requires room for both the old and new result temporarily. Ordinary failures
attempt cleanup; a crash or cleanup permission failure can leave hidden `.tmp`
files. Cleanup failures are logged and do not hide an earlier error or turn an
already-published document into a failed upload. There is no automatic stale-file
cleanup yet. This protects final paths from partial writes, but without `fsync`
it does not guarantee durability across power loss.

Known PDF parsing or source-reading failures retain the original PDF and save
a failed extraction record with no pages and a safe error code. The response is
still `201` because the document was stored, but `extraction_status` is `failed`
and `page_count` and `chunk_count` are `null`; no chunk record is created.
A successful extraction with only blank text saves an empty chunk record and
returns `chunk_count: 0`. A scanned page may have empty text without a parsing
error; OCR is not implemented.

If JSON storage fails, the API returns `500` with the saved `document_id` in
`detail`, and retains the PDF. No extraction record is guaranteed if extraction
JSON storage fails. If chunk JSON storage fails, the PDF and extraction JSON
are retained. Internally generated chunk-validation failures are programming
errors, not client validation failures; they are not silently skipped.
There is no automatic retry, search endpoint, or database yet. Upload metadata
is returned but not persisted. A process crash between PDF and JSON storage can
also leave a PDF without a result; the three files are not a single transactional write.

To read an extracted page, use
`GET /documents/{document_id}/pages/{page_number}` with the ID returned by upload.
Page numbers start at 1. A successful response is `200 OK` with this shape:

```json
{
  "document_id": "00000000-0000-4000-8000-000000000001",
  "page": {"page_number": 1, "text": "Extracted page text..."}
}
```

Blank pages return an empty text string. Invalid UUIDs or page numbers return
`422`; unknown documents or missing pages return `404`. Failed extraction or a
saved PDF without an extraction result returns `409`. Corrupt saved JSON, a
mismatched document ID, or a storage error returns `500` with a generic message.
This endpoint returns one whole page; page-list pagination is a later step.

### Fetch saved chunks

Use `GET /documents/{document_id}/chunks?offset=0&limit=10` after uploading.
`offset` is zero-based and must be nonnegative; `limit` defaults to 10 and must
be between 1 and 100. An illustrative response is:

```json
{
  "document_id": "00000000-0000-4000-8000-000000000001",
  "total_chunks": 1,
  "offset": 0,
  "limit": 10,
  "chunks": [
    {
      "document_id": "00000000-0000-4000-8000-000000000001",
      "page_number": 1,
      "chunk_index": 0,
      "start_char": 0,
      "end_char": 12,
      "text": "Annual leave"
    }
  ]
}
```

Indices stay document-wide across response pages. An offset beyond the end or
an empty chunk record returns `200` with `chunks: []`. Invalid UUIDs or query
parameters return `422`, unknown documents return `404`, and saved PDFs without
chunks (including failed extraction and older uploads) return `409`. Corrupt
records or storage errors return a generic `500`.

Re-upload older documents to create chunks; there is no backfill endpoint yet.
Pagination bounds returned chunk count, not storage work: this first version
reads and validates the entire JSON file before slicing. This endpoint fetches
chunks in document order; it does not search or rank them.

To choose other directories, set the environment variables before starting:

```bash
NEXUSRAG_STORAGE_DIR=/tmp/nexusrag-documents \
NEXUSRAG_EXTRACTION_DIR=/tmp/nexusrag-extracted \
NEXUSRAG_CHUNK_DIR=/tmp/nexusrag-chunks \
uv run uvicorn nexusrag.main:app --reload
```

Relative storage paths are resolved from the working directory at startup.
Local `data/` is ignored by Git. In Docker, mount a volume at `/app/data` (or
separate volumes at `/app/data/documents`, `/app/data/extracted`, and
`/app/data/chunks`) to keep originals and results when replacing the container.
All three directories must be
writable by the container user (UID 10001).

## Code responsibilities

- `main.py` defines the HTTP endpoints, translates application errors into HTTP
  responses, and closes request-owned uploads.
- `services/ingestion.py` validates uploads and coordinates PDF storage,
  extraction, chunking, and result storage. It returns an internal `IngestionResult` and
  raises application errors rather than `HTTPException`.
- `services/extraction.py` parses PDFs; `storage.py` handles file persistence.
- `services/chunking.py` splits extracted pages into traceable character windows
  without HTTP or filesystem operations.

## Checks and CI

Run the same checks as GitHub Actions locally:

```bash
uv sync --locked --dev
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
```

The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests,
using Python from `.python-version`. View results in the repository Actions tab.
The `--locked` option fails if `uv.lock` needs updating, keeping CI dependencies
consistent with the committed lockfile.
