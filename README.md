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

The project currently has upload validation, local PDF storage, tests, Docker,
and CI configuration. Text extraction, retrieval, and AI answers are still to
be implemented. The first RAG milestone is answering a
question about one text-based PDF with a supporting page citation.

## Run locally

```bash
uv run uvicorn nexusrag.main:app --reload
```

Then open <http://127.0.0.1:8000/docs>.

`POST /documents` saves accepted PDFs and returns `201 Created` with
`document_id` and nested `metadata`. Files are saved as `<document_id>.pdf` in
`data/documents` by default. Metadata is returned but is not yet persisted.

To choose another directory, set the environment variable before starting:

```bash
NEXUSRAG_STORAGE_DIR=/tmp/nexusrag-documents uv run uvicorn nexusrag.main:app --reload
```

Relative storage paths are resolved from the working directory at startup.
Local `data/` is ignored by Git. In Docker, mount a volume at `/app/data/documents`
to keep uploads when replacing the container; the directory must be writable by
the container user (UID 10001).

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
