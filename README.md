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
extraction, overlapping chunks with JSON storage and paginated fetching,
single-document keyword and semantic search, explicit semantic-index building,
and a question-answer endpoint with citations through optional Groq generation.
Tests, Docker, and CI configuration are included. The answer pipeline is tested
offline with fake providers; real-model answer quality still needs manual evaluation.
The first RAG milestone is answering a question about one text-based PDF with a
supporting page citation, not a guarantee of factual correctness.

## Run locally

Supported setup: Linux with Python 3.15.0 and uv 0.13.0. The lockfile resolves
Linux dependencies only; macOS and Windows are not supported by this setup.
`uv sync` installs the pinned interpreter if needed and creates `.venv`.
CI and Docker use the same Python version.

### Start Redis, the worker, and the API

Run these commands from the project root. Redis runs in Docker; the Python API
and RQ worker run locally in separate terminals.

Before the first run, install the optional processing/generation dependencies:

```bash
uv sync --locked --extra embeddings --extra generation
```

Cache MiniLM using the instructions under [Try local MiniLM embeddings](#try-local-minilm-embeddings).
For `/ask`, put `GROQ_API_KEY` in your local `.env` as described under
[Ask a question and receive an answer](#ask-a-question-and-receive-an-answer).
Document processing itself does not call Groq.

1. Start the Redis container in the background:

   ```bash
   docker compose up -d redis
   docker compose ps
   docker compose exec redis redis-cli ping
   ```

   Expect `PONG`. If Redis is already healthy, leave it running.

2. Start the worker in terminal 1 and leave it running:

   ```bash
   uv run --no-sync python -m nexusrag.run_worker
   ```

   This creates an RQ worker, initializes SQLite, and registers our exception
   and killed-workhorse hooks. It listens for tasks on the configured queue;
   you do not restart it for each upload. Use this command instead of a plain
   `rq worker` command so the application hooks are registered.

3. Start the API in terminal 2 and leave it running:

   ```bash
   uv run --env-file .env --no-sync uvicorn nexusrag.main:app --reload
   ```

   If you only want to upload/process documents and do not have a `.env`, omit
   `--env-file .env`. `--no-sync` preserves the extras installed above.

4. Open <http://127.0.0.1:8000/docs> and try the flow:

   - Execute `POST /documents` with `examples/employee-handbook.pdf`.
   - Copy `job_id` and `document_id` from **Server response → Response body**,
     not the static **Example Value**.
   - Execute `GET /jobs/{job_id}` repeatedly until `succeeded` or `failed`.
     A successful job moves through `queued → running → succeeded`.
   - Once it succeeds, execute `POST /documents/{document_id}/ask` with a question.
     Use the document ID for `/ask`, and the job ID for status checks.
   - If processing fails, inspect the worker terminal. Asking while queued,
     running, or failed returns `409` with the job ID and status.

The worker and API can start in either order after Redis; both must be running
for the complete upload-to-answer flow. If the worker is stopped, accepted jobs
remain queued until a worker consumes them.

Stop the API and worker with Ctrl+C in their terminals. Stop Redis with:

```bash
docker compose down
```

The named `redis_data` volume retains Redis persistence files. Avoid
`docker compose down -v` unless you intend to delete those files. Redis uses AOF
with once-per-second syncing; a sudden crash can lose recent writes.

### Queue behavior and storage

`POST /documents` validates and saves the PDF, records a queued job in SQLite,
and submits its ID to RQ through Redis. It returns `202 Accepted` with
`document_id`, `job_id`, and `status: "queued"`. Acceptance does not mean the
PDF has been extracted or indexed. The worker executes `nexusrag.tasks.process_document`.

The API and worker must share the configured database, source directories, and
cached local embedding model. Defaults use `data/jobs.sqlite3`, queue `ingestion`,
and Redis at `redis://localhost:6379/0`. Environment overrides are
`NEXUSRAG_JOB_DATABASE_PATH`, `NEXUSRAG_INGESTION_QUEUE_NAME`, and
`NEXUSRAG_REDIS_URL`. The worker startup command reads these settings too.

PDF or job persistence failure returns `500`; unconfirmed queue acceptance
returns `503`. Source PDFs and queued records are retained after enqueue errors
because delivery may have succeeded before a connection failed. There is no
automatic reconciliation or submission deduplication yet; blindly retrying an
upload may create another document.

Use `GET /jobs/{job_id}` (also available in `/docs`) with the ID returned by
upload to check progress. It returns the job/document IDs, status (`queued`,
`running`, `succeeded`, or `failed`), creation/update timestamps, and an optional
safe error. Unknown IDs return `404`, invalid UUIDs return `422`, and corrupt
records or database failures return a generic `500`. Ask questions after the
job reaches `succeeded`.

Worker failure hooks are registered by `python -m nexusrag.run_worker`. They
record safe failures for Python exceptions and unexpected workhorse termination,
only when SQLite still says running. Queued and terminal records remain unchanged.
Rejected claims do not fail a different execution. SQLite recovery errors are
logged without exception details and do not block RQ's normal failure handling.
A plain `rq worker` command does not register these application hooks. Whole-worker
or machine loss needs the reconciliation pass described below; periodic scheduling
is not implemented.
Automatic retries remain disabled; execution ownership for retries must be
specified before enabling them.

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

Worker processing failures retain the source PDF and mark the job `failed`
with a safe public error. Partial extraction/chunk artifacts may remain, but
success is recorded only after indexing finishes. Each index JSON is published
atomically. Automatic retries and whole-worker interruption recovery are not implemented.
Upload metadata is validated but is not persisted or returned. OCR is not
implemented. `/ask` checks the document processing job before retrieval. Queued, running,
and failed jobs return `409` with a safe message, job ID, and status. Successful
jobs continue through the existing artifact/index checks. Documents created
before job tracking retain those artifact checks. Database errors fail closed
with `500`.

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

### Search a document

Upload a text-based PDF, copy its document ID, and use the search operation in
`/docs`, or request:

```text
GET /documents/{document_id}/search?query=annual%20leave&limit=5
```

The query is required, must be 1–500 characters, and must contain a searchable
word. `limit` defaults to 5 and accepts 1–20. An illustrative response is:

```json
{
  "document_id": "00000000-0000-4000-8000-000000000001",
  "query": "annual leave",
  "hits": [
    {
      "chunk": {
        "document_id": "00000000-0000-4000-8000-000000000001",
        "page_number": 1,
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 12,
        "text": "Annual leave"
      },
      "score": 2
    }
  ]
}
```

This is a deterministic keyword baseline, not semantic search or an AI answer.
Queries and chunk text are case-normalized and split into words; each score is
the number of distinct query words present in that chunk, not a probability.
Repeated words do not increase the score. Zero-score chunks are omitted; higher
scores come first, with lower chunk indices breaking ties. Heap selection returns
the highest-ranked hits. Synonyms, stemming, stop-word filtering, and
language-specific tokenization are not implemented; overlapping chunks can both
appear in the results.

A valid query without matches returns `200` with `hits: []`, including for a
successfully processed blank document. Invalid UUIDs, queries, or limits return
`422`; unknown documents return `404`; saved PDFs without chunks return `409`.
Corrupt storage and internal ranking errors are logged and return generic `500`
responses without exposing internal details. Invalid queries are checked before
storage access so they are consistently reported as input errors.

Loading and ranking run in worker threads. Search still reads the full chunk
record and scores every chunk; `limit` bounds returned hits, not processing work.
It searches only the requested document. Try `annual leave` against the included
sample handbook to retrieve supporting passages with page references.

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

## Try local MiniLM embeddings

Embeddings are an optional extra. Ordinary API startup, uploads, keyword search,
and normal CI do not load a model. The Docker image does not install the extra.
Install it locally and explicitly download/cache the model using the demo:

```bash
uv sync --locked --extra embeddings --dev
uv run --no-sync python examples/minilm_smoke.py
```

The script loads `sentence-transformers/all-MiniLM-L6-v2` on CPU at a pinned
model commit, encodes a vacation query and two passages, and compares their
vectors with our cosine similarity helper. It prints dimensions, token counts,
and two scores. Change the sentences to experiment; scores are not probabilities
and an unrelated query still receives a similarity score.

The first demo run downloads public model files from Hugging Face into ignored
`data/models/`. Override that location with `NEXUSRAG_MODEL_CACHE_DIR`, set before
running the demo or server. Relative paths resolve from the working directory;
run both commands from the project root to use the same default cache.
Text encoding runs locally, not through a hosted inference API.
Dependencies also require downloads and additional disk/RAM beyond model weights.
Linux uses PyTorch's explicit CPU wheel index to avoid CUDA dependencies.
See the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

MiniLM outputs 384 values per vector and has a default 256-token input limit.
The demo checks the tokenizer's count, including special tokens, and rejects
oversized inputs before encoding instead of accepting silent truncation. Our
1000-character chunk size does not guarantee that every chunk fits. MiniLM does
not need BGE/E5 query prefixes. See the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

After the first successful download, you can check that the model works offline:

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python examples/minilm_smoke.py
```

Run the demo with `--no-sync` after installing the extra: a plain `uv sync` or
`uv run` without `--extra embeddings` may remove the optional packages. Normal
tests only exercise the token-budget helper with fakes; they do not import the
embedding library or download a model. The demo now uses `MiniEmbedder`, which
wraps the already-loaded model and its configured identity. `embed_chunks()`
encodes validated chunk records in batches of eight and returns a
`DocumentEmbeddings` record with model identity, dimensions, source digest, and
aligned chunk vectors. It does not load models or write files.

`storage.save_embeddings()` and `storage.load_embeddings()` save/load
`data/embeddings/<document_id>.json` using the same atomic JSON publication and
worker-thread pattern as chunk storage. Override the directory with
`NEXUSRAG_EMBEDDING_DIR`; storage functions still receive the directory explicitly.
Loading validates the schema and document ID. The semantic retrieval caller must
also check model compatibility and the source digest against current chunks.
Failed writes preserve any previous complete record; no database is required.
The Dockerfile prepares this directory for the non-root app user, but does not
install the optional embedding dependencies or load a model.

### Build a document's semantic index

After installing the extra and running the demo, start the server without
removing the optional packages:

```bash
uv run --no-sync uvicorn nexusrag.main:app --reload
```

Upload a PDF, copy its document ID, and use this operation in `/docs`:

```text
POST /documents/{document_id}/semantic-index
```

No request body or page number is needed. The operation waits for completion,
saves `data/embeddings/<document_id>.json`, and returns `200 OK` with only
`document_id`, `model_id`, `model_revision`, `dimensions`, and `embedding_count`.
It can create or rebuild an index; vectors, paths, and document text are not
included in the response. Blank documents produce an empty index with count zero.

The API loads the pinned CPU model lazily from the local cache only, using
[`local_files_only=True`](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).
Requests do not initiate model downloads. One model is reused per application
process, with loading and inference offloaded and serialized. Multiple server
processes each have their own model and lock; document ingestion uses a separate RQ worker.

Invalid UUIDs return `422`, unknown documents `404`, and PDFs without available
chunks `409`. Blank or over-token-budget chunk text also returns `409`; adjust
chunking rather than silently truncating. Unavailable optional packages or cached
model files return a safe `503`. Corrupt records, invalid model output, or storage
failures return a generic `500` with internal details logged.

Rebuild failures retain the source PDF/chunks and any previous complete index.
Atomic replacement protects this one JSON file, not a transaction across source
and index files; search must verify the stored source digest against current chunks.
Upload and keyword search remain unchanged and do not build embeddings
automatically.

### Search using embeddings

Build the index first, then use this operation in `/docs`:

```text
GET /documents/{document_id}/semantic-search?query=vacation%20rollover%20rules&limit=5
```

The query is required, must be nonblank and 1–500 characters, and must fit the
model's token budget. `limit` defaults to 5 and accepts 1–20. The endpoint loads
the saved chunks and embeddings, encodes only the query in a worker thread, and
returns `document_id`, the original `query`, and `hits`. Each hit contains the
original `chunk` (text, page number, index, and offsets) plus a finite float
`similarity` between -1 and 1. No vectors or filesystem paths are returned.

Higher similarities rank first, with earlier chunk indices breaking ties.
Zero and negative similarities are retained: these are nearest passages, not
guaranteed answers. Empty documents return `hits: []`; unrelated questions can
still return hits. Keyword `/search` remains a separate baseline.

The index must match the current chunk digest, count/index alignment, configured
model ID/revision, and loaded model dimensions. A missing, stale, or incompatible
index returns `409` and must be built/rebuilt explicitly with the POST operation;
search never rebuilds it. Unknown documents return `404`, bad request input or
over-budget queries `422`, an unavailable model `503`, and corrupt storage or
internal failures a safe `500`. Model loading and query inference use the same
serialized runtime as index building. Searches do not change saved records.

### Ask a question and receive an answer

Use semantic retrieval only for answers; keyword search remains a separate
baseline. Install both optional extras and cache MiniLM as above. Put your Groq
key in a local project-root `.env` file; paste its value after the equals sign:

```dotenv
GROQ_API_KEY=
```

Start the server from the project root with explicit environment-file loading:

```bash
uv sync --locked --extra embeddings --extra generation
uv run --env-file .env --no-sync uvicorn nexusrag.main:app --reload
```

`uv --env-file` loads the file into the launched process's environment, so Python
still reads the key through `os.environ`; no dotenv dependency is needed in the
application. `.env` and `.env.*` are excluded from Git and Docker build context
(except a future nonsecret `.env.example`). Do not force-add credentials to Git.
The prepared `.env` starts with an empty key; fill it locally before asking.
Existing exported environment variables take precedence: if an older key is
still exported, run `unset GROQ_API_KEY` before starting to use the file's value.
Restart the server after changing credentials. See the
[uv environment-file option](https://docs.astral.sh/uv/reference/cli/#uv-run--env-file).
To run the connection test with the same file, use
`uv run --env-file .env --no-sync python examples/groq_smoke.py`.
Stay on Groq's free tier if you want no usage charges. Quotas are limited and
account-specific; upgrading billing changes the cost. See
[Groq's limits](https://console.groq.com/docs/rate-limits).

1. Upload a text-based PDF using POST `/documents`.
2. Build its index using POST `/documents/{document_id}/semantic-index`.
3. In `/docs`, use POST `/documents/{document_id}/ask` with:

```json
{"question": "How many vacation days do employees get?"}
```

An illustrative response is:

```json
{
  "document_id": "00000000-0000-4000-8000-000000000001",
  "question": "How many vacation days do employees get?",
  "answer": "Employees receive 25 days of annual leave each year.",
  "status": "answered",
  "sources": [
    {
      "source_id": "S1",
      "document_id": "00000000-0000-4000-8000-000000000001",
      "page_number": 1,
      "chunk_index": 0
    }
  ]
}
```

Questions are stripped, must be nonblank, and have a 500-character limit. The
MiniLM token limit also applies. The service retrieves up to five chunks using
the existing compatible index; it never embeds document text again or rebuilds
an index during /ask. The embedding lock is released before hosted generation.
No source/index files are modified, and answers are not persisted.

Context includes whole labelled passages while they fit a source-only budget of
2048 tokens. For this first GPT-OSS integration, counting UTF-8 bytes provides a
conservative bound for ordinary byte-level BPE text, not an exact token count.
This may skip passages an exact tokenizer could fit. Instructions, the question,
schema, provider framing, and up to 1024 output/reasoning tokens have separate
room in the model window. This budget is not a per-minute quota guarantee. See
the [GPT-OSS Harmony encoding](https://developers.openai.com/cookbook/articles/openai-harmony).
Oversized source blocks are skipped without truncation; smaller later hits can
still fit. If no source fits, /ask returns `insufficient_evidence` with an empty
`sources` list, without needing generation credentials or a provider request.

Groq receives the question and selected passage text, not the original PDF or
embedding vectors. Use fictional/public documents for practice, not confidential
material; consult [Groq's data controls](https://console.groq.com/docs/your-data).
The configured model is `openai/gpt-oss-20b`, hosted by Groq. The LLM supplies
answer/status/citation IDs; our application resolves actual document/page/chunk
metadata. Invalid IDs, malformed JSON, and incomplete output are rejected.
Valid citations establish traceability, not proof that every claim is supported.
An unrelated question can still retrieve neighbours; no cosine threshold or
hallucination-free guarantee is claimed. The model is instructed to report
insufficient evidence when the passages do not answer the question.

Invalid input returns `422`, unknown documents `404`, unavailable chunks or a
missing/stale/incompatible index `409`, and unavailable local model or generation
setup/credentials/model access/quota `503`. Generation timeouts return `504`;
provider failures or invalid generated answers return `502`. Corrupt storage and
internal failures return generic `500` responses. Provider details, keys, and
document text are not included in error responses or logged by this route.
The Groq SDK is imported lazily; each provider call owns/closes its client, has
a 30-second request timeout, and disables automatic retries and model fallback.
Other endpoints still work without the generation extra or key.

Keep this learning API bound to localhost. Authentication, application rate
limiting, and production deployment controls are not implemented. The existing
Docker image does not install the optional model/provider extras; /ask is not
ready in that image without a separate container setup update.

## Code responsibilities

- `main.py` defines the HTTP endpoints, translates application errors into HTTP
  responses, and closes request-owned uploads.
- `services/ingestion.py` validates uploads and coordinates PDF storage,
  extraction, chunking, and result storage. It returns an internal `IngestionResult` and
  raises application errors rather than `HTTPException`.
- `services/extraction.py` parses PDFs; `storage.py` handles file persistence.
- `services/chunking.py` splits extracted pages into traceable character windows
  without HTTP or filesystem operations.
- `services/retrieval.py` ranks chunks with pure keyword or cosine-similarity
  functions and checks semantic source alignment. `services/search.py` shares
  model compatibility, query encoding, and semantic ranking between search and /ask.
- `services/answering.py` loads saved records and coordinates retrieval, bounded
  context, generation, and server-resolved source metadata. `services/context.py`
  prepares labelled passages without loading a model or calling a provider.
- `services/generation.py` validates generated answers/citation IDs;
  `services/generation_runtime.py` owns Groq client lifecycle and translates SDK
  failures into safe application errors. HTTP status selection stays in `main.py`.
  Its top-level `lazy import groq` requires Python 3.15 and loads the optional SDK
  only on first use; empty-context answers and API startup need no SDK.

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

CI uses only base/dev dependencies and fake inference adapters: it needs no
Groq key, generation SDK, model download, or external inference request.
After the minimal CI sync above, reinstall both extras before a real /ask demo.


### Inspect and recover interrupted jobs

`inspect_running_jobs(database_path, redis_url)` performs one read-only pass.
Interrupted candidates have `recovery_action="skipped"` in dry runs. Pass
`apply_recovery=True` to mark confirmed interrupted jobs failed. It does not
create retries or schedule repeated sweeps.

Apply mode first acquires a nonblocking document lock, rechecks RQ, then compares
the complete SQLite job snapshot inside a write transaction. Only RQ failed/stopped
jobs with no execution records qualify. Locked documents, stale/missing heartbeats,
missing RQ records, Redis outages, and changed job records are left unchanged.
The result reports `recovery_action` as `none`, `skipped`, or `failed`.

Restart all workers with `uv run --no-sync python -m nexusrag.run_worker` before
using apply mode: older workers do not participate in document locking. Task
entry points hold `.job-locks/<document_id>.lock` beside the configured database
through processing and publication. Recovery uses the same lock. A paused or
orphaned child keeps ownership; process death releases it. Do not delete lock
files while processes may be using them. This exclusion relies on the current
shared local Linux filesystem and is not a distributed lease mechanism.


### Explicit retries

`services/job_retry.py` provides `retry_failed_job(job_id, database_path,
max_attempts)`. It creates a new queued job for the existing PDF and enqueues
its new ID. `max_attempts` counts the initial attempt too. Retry policy currently
allows only `Document processing was interrupted`; generic failures require
investigation. Automatic RQ retries and a retry HTTP endpoint are not enabled.

Retry creation acquires the document lock and validates the latest attempt in
one SQLite transaction. Unique indexes prevent duplicate retry parents, attempt
numbers, and active attempts for a document. Startup adds these indexes to
existing databases; conflicting old records require correction before startup
can complete. Prior attempts remain unchanged. `/ask` follows the latest attempt
of a validated chain, so a queued retry returns `409` until it succeeds.

If queue acceptance cannot be confirmed, `RetrySubmissionError.job_id` identifies
the retained queued attempt. A repeated request for its parent raises
`RetryAlreadyExistsError` with the later job ID; it does not enqueue it again.
Queued-delivery reconciliation remains follow-up work. Transactional recovery
helpers now live in `job_storage.py`; `reconciliation.py` inspects and recovers
running jobs, while `job_retry.py` owns retry policy and submission.
