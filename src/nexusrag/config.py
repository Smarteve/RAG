import os
from pathlib import Path

JOB_DATABASE_PATH = (
    Path(os.environ.get("NEXUSRAG_JOB_DATABASE_PATH", "data/jobs.sqlite3"))
    .expanduser()
    .resolve()
)

# Relative paths are resolved from the application's working directory.
STORAGE_DIRECTORY = (
    Path(os.environ.get("NEXUSRAG_STORAGE_DIR", "data/documents"))
    .expanduser()
    .resolve()
)


EXTRACTION_DIRECTORY = (
    Path(os.environ.get("NEXUSRAG_EXTRACTION_DIR", "data/extracted"))
    .expanduser()
    .resolve()
)

CHUNK_DIRECTORY = (
    Path(os.environ.get("NEXUSRAG_CHUNK_DIR", "data/chunks")).expanduser().resolve()
)

EMBEDDING_DIRECTORY = (
    Path(os.environ.get("NEXUSRAG_EMBEDDING_DIR", "data/embeddings"))
    .expanduser()
    .resolve()
)

EMBEDDING_MODEL_CACHE_DIRECTORY = (
    Path(os.environ.get("NEXUSRAG_MODEL_CACHE_DIR", "data/models"))
    .expanduser()
    .resolve()
)


EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# Generation is hosted by Groq; this is separate from the local embedding model.
GENERATION_MODEL_ID = "openai/gpt-oss-20b"
# Source-only budget. Instructions, question, schema, and output have separate room.
GENERATION_CONTEXT_TOKEN_BUDGET = 2048
