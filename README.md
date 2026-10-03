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
extraction with JSON storage, tests, Docker, and CI configuration. Retrieval and
AI answers are still to be implemented. The first RAG milestone is answering a
question about one text-based PDF with a supporting page citation.

## Run locally

```bash
uv run uvicorn nexusrag.main:app --reload
```

Then open <http://127.0.0.1:8000/docs>.

`POST /documents` saves accepted PDFs, waits for extraction in a worker thread,
and saves the extraction result as UTF-8 JSON. It returns `201 Created` with
`document_id`, nested `metadata`, `extraction_status`, `page_count`, and
`extraction_error`. Filesystem paths and full page text are not returned.

The same UUID links the original and extracted data:

```text
data/documents/<document_id>.pdf
data/extracted/<document_id>.json
```

The JSON contains `document_id`, `status`, `error`, and `pages` (objects with
`page_number` and `text`). Both storage functions are async wrappers around
synchronous worker-thread helpers. Both write to a uniquely named temporary
file in the destination directory and close it before publishing the final path.

PDFs are published with a hard link, then the temporary name is removed. This
does not copy the bytes again and refuses to overwrite an existing document.
The storage filesystem must support hard links; unsupported filesystems fail
the upload rather than falling back to a partial or overwriting write. JSON
uses replacement instead, allowing an earlier extraction result to be replaced
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
and `page_count` is `null`. A blank or scanned page may have empty text without
a parsing error; OCR is not implemented.

If JSON storage fails, the API returns `500` with the saved `document_id` in
`detail`, and retains the PDF. No extraction record is guaranteed in that case.
There is no automatic retry, retrieval endpoint, or database yet. Upload metadata
is returned but not persisted. A process crash between PDF and JSON storage can
also leave a PDF without a result; these are not a single transactional write.

To choose other directories, set the environment variables before starting:

```bash
NEXUSRAG_STORAGE_DIR=/tmp/nexusrag-documents \
NEXUSRAG_EXTRACTION_DIR=/tmp/nexusrag-extracted \
uv run uvicorn nexusrag.main:app --reload
```

Relative storage paths are resolved from the working directory at startup.
Local `data/` is ignored by Git. In Docker, mount a volume at `/app/data` (or
separate volumes at `/app/data/documents` and `/app/data/extracted`) to keep both
originals and results when replacing the container. Both directories must be
writable by the container user (UID 10001).

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
