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

## Run with Docker Compose

Supported setup: Linux with Docker Engine and the Docker Compose plugin
(`docker compose version`). The Docker build uses Python 3.15.0 and uv 0.13.0
to install the locked runtime dependencies, including the embeddings and generation extras.
You do not need a host Python environment to run the application. Local
development and CI use the same Python version; the lockfile resolves Linux
dependencies only.

Run all commands below from the project root. Compose manages five services:

| Service | Purpose |
| --- | --- |
| `api` | FastAPI on <http://127.0.0.1:8000> |
| `worker` | RQ worker with application failure hooks |
| `sweeper` | Guarded recovery, delivery reconciliation, and eligible retries |
| `redis` | Persistent job queue |
| `qdrant` | Persistent vector database |

The API, worker, and sweeper use the same locally built `nexusrag:dev` image with
different commands. `x-app` in `compose.yaml` is shared configuration, not another
service. Containers connect to `redis://redis:6379/0` and `http://qdrant:6333` using
Compose service names; `localhost` inside a container refers to that container.

### First-time setup

1. Find your user and group IDs:

   ```bash
   id -u
   id -g
   ```

   Create or edit the ignored project-root `.env`. Replace `1000` below with the
   respective values above, and supply your Groq key if you want to use `/ask`:

   ```dotenv
   LOCAL_UID=1000
   LOCAL_GID=1000
   GROQ_API_KEY=
   ```

   Compose runs application containers with these IDs so they can write into
   your existing `data` directory. It passes the Groq key only to the API.
   Document processing does not call Groq. Keep credentials out of Git.

2. Prepare shared storage, validate configuration, and build the image:

   ```bash
   mkdir -p data/models
   docker compose config --quiet
   docker compose build api
   ```

3. Cache the pinned MiniLM model. Skip this if the existing `data/models` cache
   already works. This temporary container downloads the model into the shared
   folder and exits; it does not start the other services:

   ```bash
   docker compose run --rm --no-deps -T api /app/.venv/bin/python - <<'PY'
   from sentence_transformers import SentenceTransformer
   from nexusrag import config

   model = SentenceTransformer(
       config.EMBEDDING_MODEL_ID,
       revision=config.EMBEDDING_MODEL_REVISION,
       device="cpu",
       cache_folder=str(config.EMBEDDING_MODEL_CACHE_DIRECTORY),
       trust_remote_code=False,
       model_kwargs={"use_safetensors": True},
   )
   print("Cached model dimensions:", model.get_sentence_embedding_dimension())
   PY
   ```

   Expect 384 dimensions. Network access is required for the initial download;
   normal API requests and worker tasks load only cached model files.

### Migrate from local processes

If you previously ran Uvicorn and the worker in terminals, stop them with Ctrl+C.
If you installed the user systemd sweeper, disable its timer and stop any active
pass before starting the Compose sweeper:

```bash
systemctl --user disable --now nexusrag-sweep.timer
systemctl --user stop nexusrag-sweep.service
```

These systemd commands are unnecessary if you never installed those units.
Use one scheduler for this database. Keep the existing `data` folder and the
Redis/Qdrant named volumes; use the same Compose project name to retain access
to the existing named volumes.

### Start and try the full flow

```bash
docker compose up -d --wait
docker compose ps
```

`-d` runs containers in the background. `--wait` waits for services to be running
or healthy. The worker and sweeper start after the API health check confirms
the API, Redis, and Qdrant are reachable. This check does not validate the model
cache or Groq credentials.

Open <http://127.0.0.1:8000/docs> and try the flow:

1. Execute `POST /documents` with `examples/employee-handbook.pdf`.
2. Copy `job_id` and `document_id` from **Server response → Response body**,
   not the static **Example Value**.
3. Execute `GET /jobs/{job_id}` repeatedly until `succeeded` or `failed`.
   A successful job moves through `queued → running → succeeded`.
4. Once it succeeds, execute `POST /documents/{document_id}/ask` with a question.
   Use the document ID for `/ask`, and the job ID for status checks.
5. If processing fails, inspect `docker compose logs --tail=100 worker`.
   Asking while queued, running, or failed returns `409` with the job ID and status.

The worker stays running and consumes subsequent uploads; no manual worker
startup is needed. If it stops, accepted jobs remain queued until a worker
consumes them. The sweeper starts automatically too: it runs an apply pass
immediately, then waits 60 seconds after each pass before running again, with
a three-attempt limit including the initial attempt. Only failures classified
as interrupted processing qualify for automatic retries.

### Develop with live source code

After the first-time setup above, start the stack with the development override:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up -d --wait
```

`compose.dev.yaml` mounts `./src` read-only at `/app/src` in the API, worker, and
sweeper. It sets `PYTHONPATH=/app/src` so Python imports your current source
instead of the package copy installed in the image. The existing shared data,
dependencies, and service settings are retained. No rebuild is needed for
ordinary Python source edits.

| Change | Action in development mode |
| --- | --- |
| API code under `src/` | Uvicorn reloads automatically |
| Worker/task code, hooks, or shared processing code | Restart the worker |
| Sweep implementation | The next pass starts a new Python process and loads the current source |
| Dependencies (`pyproject.toml`, `uv.lock`) or Dockerfile | Rebuild the application image and recreate services |
| Compose settings or `.env` values referenced by Compose | Run the development startup command again to recreate affected services |

After changing worker code, restart the worker so its parent process and future
task children load the updated modules:

```bash
docker compose -f compose.yaml -f compose.dev.yaml restart worker
```

The worker receives a graceful stop request; an active job may finish before the
restart completes. Restart it before submitting jobs that should use the new
code. Shared modules used by both the API and worker require this worker restart
even though the API reloads automatically. For database initialization changes,
allow the API reload to finish and restart the worker before continuing.

For dependency or Dockerfile changes:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build -d --wait
```

Keep both `-f` options on subsequent development commands. For example:

```bash
docker compose -f compose.yaml -f compose.dev.yaml logs -f api worker sweeper
docker compose -f compose.yaml -f compose.dev.yaml down
```

For other Compose commands elsewhere in this README, add the same two `-f`
options before the subcommand when using development mode. Running `up` with
only `compose.yaml` switches services back to the built-image configuration.
The override follows [Compose's file merging rules](https://docs.docker.com/compose/how-tos/multiple-compose-files/merge/)
and enables [Uvicorn's development reloader](https://uvicorn.dev/settings/#development).

### Logs, updates, and shutdown

Follow application logs (Ctrl+C stops following, leaving containers running):

```bash
docker compose logs -f api worker sweeper
```

With `compose.yaml` alone, source code is copied into the image, so rebuild after
changing Python code or dependencies. For live source edits, use the development
override above instead. This command rebuilds the base configuration and
recreates services when their configuration changes:

```bash
docker compose up --build -d --wait
```

After changing only `GROQ_API_KEY` in `.env`, recreate the API with
`docker compose up -d --wait api`; `docker compose restart api` does not load new
environment values. Compose reads `.env` for interpolation and explicitly passes
the configured variables; it does not inject every `.env` entry into containers.

Stop and remove the containers while preserving stored data:

```bash
docker compose down
```

`./data:/app/data` shares PDFs, extracted text, chunks, embedding records, SQLite,
the model cache, and document locks across application containers. Redis uses
the `redis_data` named volume; Qdrant uses `qdrant_data`. Avoid
`docker compose down -v` unless you intend to delete the named volumes. Redis
uses AOF with once-per-second syncing; a sudden crash can lose recent writes.
These mounts preserve data across container replacement, but are not backups.

### Delete a document

Use `DELETE /documents/{document_id}` in `/docs`, or supply the actual document
ID in this command:

```bash
curl -i -X DELETE "http://127.0.0.1:8000/documents/<document_id>"
```

Deletion removes the original PDF, extraction/chunk/embedding JSON, leftover
temporary files for that document, and all its Qdrant point generations,
including unpublished generations from failed indexing. Other documents, the
shared model cache, and document lock files are preserved. SQLite job history
remains accessible through `GET /jobs/{job_id}`; its status records processing
history, not current document availability. Redis job metadata is not purged.

| Result | Meaning |
| --- | --- |
| `204` | Cleanup completed; already-deleted or unknown IDs also succeed |
| `409` | A job is queued/running, or processing, reindexing, retry, or deletion owns the document lock |
| `422` | The document ID is not a UUID |
| `503` | Qdrant cleanup could not be confirmed; repeat DELETE |
| `500` | Database or file cleanup failed; resolve the problem and repeat DELETE |

The service takes the same document lock as processing/reindexing/retry, checks
for active jobs, then commits a `deleting` record in SQLite before remote or file
cleanup. From that point, document reads, searches, `/ask`, and reindexing return
`404`, and no new job or retry may resurrect the document. Requests already in
flight may finish. Once cleanup succeeds, the record becomes `deleted`.

SQLite, Qdrant, and the filesystem do not share a transaction. A failure after
deletion starts leaves the document hidden, with cleanup safely repeatable using
the same DELETE request. Pending deletions are not automatically resumed by the
job sweeper. Startup adds the deletion table to existing databases; rebuild and
recreate all application services after this update:

```bash
docker compose up --build -d --wait
```

### Queue behavior and storage

`POST /documents` validates and saves the PDF, records a queued job in SQLite,
and submits its ID to RQ through Redis. It returns `202 Accepted` with
`document_id`, `job_id`, and `status: "queued"`. Acceptance does not mean the
PDF has been extracted or indexed. The worker executes `nexusrag.tasks.process_document`.

The API, worker, and sweeper share the configured database, source directories,
document locks, and cached local embedding model through `/app/data` in Compose.
The queue defaults to `ingestion`. Outside Docker, defaults use
`data/jobs.sqlite3` and Redis at `redis://localhost:6379/0`. Environment overrides are
`NEXUSRAG_JOB_DATABASE_PATH`, `NEXUSRAG_INGESTION_QUEUE_NAME`, and
`NEXUSRAG_REDIS_URL`. The worker startup command reads these settings too.

PDF or job persistence failure returns `500`; unconfirmed queue acceptance
returns `503`. Source PDFs and queued records are retained after enqueue errors
because delivery may have succeeded before a connection failed. The periodic sweep can reconcile queued attempts whose delivery is missing,
using atomic RQ uniqueness for the same job ID. Upload requests themselves are
not idempotent; blindly retrying an upload may create another document.

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
or machine loss needs the guarded reconciliation pass described below. Compose
runs it periodically in the `sweeper` service and retries eligible interruptions
up to three total attempts. RQ's own automatic retries remain disabled.

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
The defaults remain a cap of 1000 Python string characters per chunk and a
requested 200-character overlap. New queued uploads also fit each chunk to the
loaded MiniLM model's token limit. The worker passes the model's token counter
into `chunk_document`; it counts each candidate substring with special tokens
and truncation disabled, shortening the character span until it fits. Overlap
is reduced when a fitted chunk is too short for the requested overlap, ensuring
the next window advances. Whitespace-only pages/windows are skipped without
renumbering later pages; emitted chunk text is not stripped.

`chunk_size`, `overlap`, `start_char`, and `end_char` still use character units;
the saved chunk schema is unchanged. Calling `chunk_document` without a token
counter/budget retains the original character-window behavior for the local
baseline and ingestion helper. Existing saved documents are not automatically
rechunked. Upload a fresh copy to use the new chunking; the semantic-index
endpoint rebuilds vectors from saved chunks and does not change their boundaries.

The task child loads the model once, then uses the same tokenizer/model for
splitting and embedding. Tokenization and splitting run in a worker thread.
The embedding boundary still checks complete chunks before encoding, so legacy
or externally supplied oversized chunks fail explicitly rather than being
silently truncated.

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
atomically. The sweeper recovers confirmed interruptions and creates eligible
retries; generic processing failures require investigation.
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

For a host development server, choose other directories by setting environment
variables before starting:

```bash
NEXUSRAG_STORAGE_DIR=/tmp/nexusrag-documents \
NEXUSRAG_EXTRACTION_DIR=/tmp/nexusrag-extracted \
NEXUSRAG_CHUNK_DIR=/tmp/nexusrag-chunks \
uv run --no-sync uvicorn nexusrag.main:app --reload
```

Relative storage paths are resolved from the working directory at startup.
Local `data/` is ignored by Git. Compose already mounts the complete folder at
`/app/data` and overrides the image's default UID 10001 with `LOCAL_UID:LOCAL_GID`.
To customize paths in Docker, update the shared environment and volume mappings
in `compose.yaml` for all application services. Adding a variable to `.env` alone
does not pass it into a container unless Compose references it.

## Try local MiniLM embeddings

The Docker image includes the embeddings extra; see
[First-time setup](#first-time-setup) to cache the model in Docker. The experiment
below is optional and runs in a host Python environment. For local development,
install the extra and explicitly download/cache the model using the demo:

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
oversized inputs before encoding instead of accepting silent truncation. A
1000-character cap alone does not guarantee that every chunk fits; new queued
uploads now additionally fit chunks with this tokenizer. Existing saved chunks
retain their original boundaries. MiniLM does not need BGE/E5 query prefixes.
See the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

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
Failed writes preserve any previous complete record. Qdrant stores the searchable
vectors; the local JSON record identifies the published generation. The image
installs embedding dependencies but does not contain model weights; Compose
shares the separately populated `data/models` cache.

### Build a document's semantic index

With the model cache prepared, start the Compose stack:

```bash
docker compose up -d --wait
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
chunks `409`. Blank or over-token-budget saved chunk text also returns `409`.
For oversized legacy chunks, upload a fresh copy to use token-aware chunking;
this endpoint does not rechunk stored text. Unavailable optional packages or cached
model files return a safe `503`. Corrupt records, invalid model output, or storage
failures return a generic `500` with internal details logged.

Rebuild failures retain the source PDF/chunks and any previous complete index.
Atomic replacement protects this one JSON file, not a transaction across source
and index files; search must verify the stored source digest against current chunks.
Queued uploads build embeddings through the worker. The explicit indexing
endpoint rebuilds completed documents; keyword search does not build embeddings.

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

Higher similarities rank first, with earlier chunk indices breaking ties among
the returned hits. Qdrant may select a different subset when equal scores cross
the top-k boundary.
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
baseline. The Docker image includes both required extras; cache MiniLM as above.
Put your Groq key in the existing project-root `.env`, alongside `LOCAL_UID` and
`LOCAL_GID`; paste its value after the equals sign:

```dotenv
GROQ_API_KEY=
```

Start or recreate the API from the project root so it receives the configured key:

```bash
docker compose up -d --wait api
```

Compose passes the key into the API process's environment; Python reads it through
`os.environ`. `.env` and `.env.*` are excluded from Git and Docker build context
(except a nonsecret `.env.example`, if added). Do not force-add credentials to Git.
Exported shell variables take precedence over Compose's `.env`; if an older key
is exported, run `unset GROQ_API_KEY` before recreating the API.

For an optional host development server, install the extras with
`uv sync --locked --extra embeddings --extra generation` and launch with
`uv run --env-file .env --no-sync uvicorn nexusrag.main:app --reload` instead of
the API container. `uv --env-file` explicitly loads credentials for that process.
To run the optional host connection test with the same file, use
`uv run --env-file .env --no-sync python examples/groq_smoke.py`.
Stay on Groq's free tier if you want no usage charges. Quotas are limited and
account-specific; upgrading billing changes the cost. See
[Groq's limits](https://console.groq.com/docs/rate-limits).

1. Upload a text-based PDF using POST `/documents`.
2. Poll `GET /jobs/{job_id}` until `succeeded`; the worker builds its index.
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

Keep this learning API bound to localhost, as configured in Compose.
Authentication, application rate limiting, and production deployment controls
are not implemented. Docker includes the model/provider dependencies; `/ask`
also requires the cached model, an indexed document, and a working Groq key.

## Evaluate retrieval

The evaluator asks fixed questions against an already indexed document using the
same semantic-search service as the API. It measures retrieval only: it does not
call Groq, generate answers, ingest/reindex documents, or change job states.
The default [case set](src/nexusrag/evaluation_cases.py) has seven answerable
handbook questions and one unsupported parking question.

Upload `examples/employee-handbook.pdf`, wait for its job to succeed, and replace
`YOUR_DOCUMENT_ID` with the actual document ID in this command:

```bash
docker compose -f compose.yaml -f compose.dev.yaml exec api \
  /app/.venv/bin/python -m nexusrag.evaluate_retrieval YOUR_DOCUMENT_ID \
  --output /app/data/evaluations/handbook-token-aware.json
```

Development mode loads the new evaluator from the mounted source. With the base
Compose configuration alone, rebuild first and omit the two `-f` options.
`--output` saves the complete report to the host's
`data/evaluations/handbook-token-aware.json` and prints a JSON summary. Omit it to
print the full report instead. Use different filenames to retain separate runs.

- **Hit@5** (`hit_rate_at_k`): the fraction of answerable questions with at least
  one supporting passage among the first five hits.
- **MRR@5** (`mrr_at_k`): the average reciprocal of the first relevant rank. Rank
  1 contributes 1, rank 2 contributes 0.5, and a miss contributes 0.
- Each case records `first_relevant_rank`, the expected passages, and the actual
  returned chunks, pages, character offsets, and similarity scores.

A hit is relevant when it contains an expected phrase on the expected page;
matching ignores case and normalizes whitespace. Multiple expected passages in
a case are alternative acceptable matches. Unsupported cases retain their hits
for inspection but are excluded from both scored denominators. Nearby passages
for an unsupported question do not demonstrate answer support or correct model
abstention. Answer/citation quality still requires separate manual evaluation.

The report also records the dataset ID, PDF SHA-256, source digest, published
generation, model identity/revision, chunk settings/count, top-k, and timestamp.
The PDF must match the case set's fingerprint, and positive labels must occur on
their annotated source pages. Invalid inputs, stale indexes, and operational
failures abort the run instead of silently omitting cases. Exit code 0 means the
run completed, even if retrieval missed every expected passage; it is not a
quality threshold. Failures exit nonzero; input/search failures occur before
report output is written.

The evaluator holds the document lock while reading/searching to keep deletion
and reindexing from changing its snapshot. Queued, running, failed, or deleted
documents cannot be evaluated. Existing legacy documents without job records can
be evaluated when their artifacts form a compatible published index.

Use `--top-k 1` for a stricter first-result check. To compare saved documents from
different chunking versions, evaluate their respective IDs with the same case
set, model, and top-k, saving separate reports. The evaluator never rebuilds the
baseline. Custom labels can be supplied with `--cases /app/data/evaluations/cases.json`,
using the `RetrievalDataset` schema and that PDF's SHA-256.

The [recorded local baseline](docs/retrieval-evaluation.md) compares character
and token-aware chunks with the cached model and private in-memory Qdrant. The
small handbook produces only two chunks, so Hit@5 can include the whole document;
high scores here are a regression baseline, not evidence of general retrieval
quality or improved chunking. Add harder, longer documents before drawing those
conclusions.

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

After updating worker code, rebuild and recreate the stack with
`docker compose up --build -d --wait` before using apply mode: older workers may
not participate in document locking. Task
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
Queued-delivery reconciliation in the sweep can resubmit this same attempt safely. Transactional recovery
helpers now live in `job_storage.py`; `reconciliation.py` inspects and recovers
running jobs, while `job_retry.py` owns retry policy and submission.

### Run one recovery and retry sweep

The Compose sweeper already runs apply passes automatically. To inspect a single
dry run using the running API container's settings and shared storage:

```bash
docker compose exec api /app/.venv/bin/python -m nexusrag.sweep_jobs
```

Apply guarded recovery and enqueue eligible failed attempts:

```bash
docker compose exec api /app/.venv/bin/python -m nexusrag.sweep_jobs --apply --max-attempts 3
```

The command reports JSON lines for inspection, proposed/enqueued retries, skips,
errors, and a summary. It exits nonzero on operational failures. It queries failed
jobs after recovery, so earlier failures recorded by hooks are included. Limits,
retryability, and existing-child checks are revalidated before insertion. Dry runs
do not simulate retries for running jobs that would first require recovery.

Queue submission errors report the retained queued attempt ID. Existing queued
attempts are checked by delivery reconciliation and resubmitted only when RQ
is absent, preserving the same job ID and attempt number.
The Python command performs one pass and exits. Repetition is supplied by the
Compose sweeper's shell loop, which starts immediately, then sleeps 60 seconds
after each completed pass. A nonzero pass exit is reported in logs and the loop
continues. Use `docker compose logs -f sweeper` to follow scheduled results.

To pause scheduled sweeps and resume them later:

```bash
docker compose stop sweeper
docker compose start sweeper
```

For an installation outside Docker, initialize the database by starting the API
or worker first, then use `uv run --no-sync python -m nexusrag.sweep_jobs` with
the same flags and matching storage/Redis settings.

### Alternative: schedule a host sweeper with systemd

This is an alternative for running Python processes outside Docker. It is not
required for the Compose stack. Do not enable this timer alongside the Compose
`sweeper` service for the same database. The full Docker workflow starts its
sweeper with `docker compose up -d --wait` and does not run the installer below.

The tracked service template and timer live in `deploy/systemd/`. The installer
generates a service using this checkout's absolute path and the `uv` executable
on your PATH, then links the timer into `~/.config/systemd/user/`. No personal
paths need to be committed. The service uses default queue and storage settings;
configure matching `NEXUSRAG_*` environment overrides if the API/worker use custom
settings. It does not load shell startup files or `.env` automatically.

After verifying a manual dry run and restarting workers with document locking,
run from the project root:

```bash
uv run --no-sync python scripts/install_sweep.py
systemctl --user enable --now nexusrag-sweep.timer
```

The installer reloads systemd but does not enable or restart the timer. It can
migrate the old service symlink pointing into this checkout, regenerate its own
output, and refuses to overwrite unrelated unit files. To preview generation in
a temporary destination without contacting systemd:

```bash
uv run --no-sync python scripts/install_sweep.py --unit-dir /tmp/nexusrag-sweep-units --no-reload
```

The timer applies recovery and retries with a three-attempt limit. It first runs
about one minute after activation, then one minute after each pass finishes.
The same service is not started again while it is active. Redis and the RQ worker
must be managed separately. This is a user-session timer, not a boot service.
Keep the checkout available while the timer is enabled.

Inspect the schedule and logs, or run one pass immediately:

```bash
systemctl --user list-timers nexusrag-sweep.timer
journalctl --user -u nexusrag-sweep.service -n 50 --no-pager
systemctl --user start nexusrag-sweep.service
```

After editing the service template or moving the checkout, rerun the installer.
After changing the timer schedule, run `systemctl --user daemon-reload` and
`systemctl --user restart nexusrag-sweep.timer`. Git updates do not automatically
regenerate or reload units.
To disable future sweeps:

```bash
systemctl --user disable --now nexusrag-sweep.timer
```

Disabling the timer does not interrupt a pass already running.


### Queued delivery reconciliation

Each sweep now inspects queued attempts after interrupted-job recovery and before
creating retries. Attempts younger than 60 seconds are skipped to give the
original upload/retry submission time to finish. Older queued attempts are
redelivered only if the source PDF exists, SQLite still has the same current
attempt, and RQ has no job record or residual queue entry. Existing queued,
scheduled, or executing RQ tasks are not duplicated. Terminal mismatches, missing
sources, inconsistent records, and Redis outages are reported for attention.

Apply mode takes the document lock, repeats the checks, and submits the existing
ID using RQ's atomic `unique=True` enqueue. All initial/retry submissions now use
that same uniqueness mechanism. A lost acknowledgement leaves SQLite queued;
the next pass observes the existing RQ task instead of creating another attempt.
No attempt counter is incremented by delivery reconciliation. Dry run reports
`would_enqueue`; apply reports `enqueued`, `present`, `skipped`, `uncertain`, or
`delivery_unconfirmed`. Uncertain/unconfirmed delivery produces a nonzero sweep
exit code and a safe diagnostic for operator review.

Restart the API and all workers after updating: workers now wait on document
locks so a task received during the short delivery lock is not rejected. Recovery
and sweeps still acquire locks nonblocking. The Compose sweeper runs this phase
automatically; the alternative host systemd service invokes the same command.

Optional integration tests use generated private queues and clean them afterward:

```bash
NEXUSRAG_TEST_REDIS_URL=redis://localhost:6379/0 uv run --no-sync pytest -q tests/test_delivery_redis.py
```

The default test suite skips these tests and remains offline.


### Queue milestone validation and remaining limits

Manual `POST /documents/{document_id}/semantic-index` uses the same document
lock as task execution and recovery. It rejects queued/running/failed current
attempts with `409` and holds the lock through publication. Existing documents
without jobs retain source/index validation.

RQ's default worker runs each task in a separate child. MiniLM loads once in
that child and is released with it; reuse across tasks is deliberately deferred.
Preloading a threaded ML runtime in a parent before forking needs separate design
and measurement. This is a documented deviation from the original once-per-worker
performance target, not a claim of model reuse.

`AbandonedJobError` from RQ maintenance does not directly fail SQLite through the
exception hook: a surviving orphan child might still hold the document lock.
Guarded reconciliation decides after RQ records abandonment and the lock is free.

A real, isolated crash/restart walkthrough can be run explicitly:

```bash
uv run --no-sync python -m examples.queue_restart_smoke
```

It requires Docker, the locally available `redis:8-alpine` image, embeddings
extras, and the configured cached MiniLM. It creates and removes a private Redis
container/volume, restarts that broker, kills a private RQ parent and child,
waits about two minutes for actual abandonment/registry expiry, and runs recovery/retry to
build real embeddings. It does not restart the application broker or call Groq.
See `docs/queue-validation.md` for the recorded evidence and outstanding manual
answer/CI checks.


### Qdrant indexing publication

Index building requires Qdrant, started with the rest of the Compose stack. The API
supplies a request-owned backend; each RQ task creates and closes its own backend
within the task's event loop. `build_semantic_index` receives that backend,
encodes the saved chunks, assigns a fresh `index_generation`, awaits all Qdrant
writes, and only then atomically saves the local embedding record.

The JSON record still includes vectors for the existing cosine reference and
identifies the published generation. An upload failure preserves the previous
record; a local-save failure leaves the new remote generation unreferenced.
Old generations of retained documents remain for now; generation garbage
collection is a separate step. Deleting a document removes all its generations.
Local chunk text remains the citation source. Semantic search and `/ask` now
use the request-injected Qdrant backend for similarity lookup. The keyword
`/search` endpoint remains independent of Qdrant. The original Python cosine
ranker remains available as a reference for offline comparisons, not a silent
fallback when Qdrant is unavailable.

The queue restart walkthrough uses the real vector adapter with private
in-memory Qdrant clients in its test workhorses, so it needs no Qdrant server and
does not modify application collections. It verifies queue/recovery and local
publication, not Qdrant server persistence.


### Semantic retrieval with Qdrant

`GET /documents/{document_id}/semantic-search` and `/ask` share the same service:
load chunks and the published embedding record, validate source alignment and
model compatibility, encode only the question, release the model lock, then await
Qdrant search. Source alignment is owned by this service and is not repeated in
the adapter. Pydantic validates query/point fields; the adapter checks collection
compatibility and maps verified returned point identities to local source text.

The query filters document ID, published generation, source digest, model ID,
and model revision. Exact cosine search is used for baseline parity. No search
request rebuilds an index or silently switches to local ranking. Legacy embedding
records without `index_generation` return `409`; explicitly rebuild with
`POST /documents/{document_id}/semantic-index` before searching or asking.
Qdrant is required for new indexing and semantic retrieval. Keyword search
continues to work without it. Rebuild and recreate the stack after code updates
with `docker compose up --build -d --wait`. Client errors and malformed results
propagate to the HTTP boundary, which returns safe generic errors rather than
leaking provider details.
