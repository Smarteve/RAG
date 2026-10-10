from pathlib import Path
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from nexusrag.schemas.embedding import DocumentEmbeddings
from nexusrag.services.embedding import embed_chunks
from nexusrag.services.embedding_model import MiniEmbedder
from nexusrag.storage import load_chunks, save_embeddings


async def build_semantic_index(
    document_id: UUID,
    embedder: MiniEmbedder,
    chunk_directory: Path,
    embedding_directory: Path,
) -> DocumentEmbeddings:
    """Load chunks, encode in a worker thread, and publish a complete index.

    The caller supplies an already-loaded embedder and controls concurrent
    inference. Storage and encoding errors propagate without HTTP translation.
    Encoding must finish before saving; atomic storage preserves a previous
    complete index if encoding or publishing the replacement fails. Source
    chunks are not modified. The record describes the loaded chunk snapshot,
    so search must still check its digest against the current chunks.
    """
    chunks = await load_chunks(document_id, chunk_directory)
    record = await run_in_threadpool(embed_chunks, chunks, embedder)
    await save_embeddings(record, embedding_directory)
    return record
