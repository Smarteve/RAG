from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from fastapi.concurrency import run_in_threadpool

from nexusrag.schemas.chunk import DocumentChunks
from nexusrag.schemas.embedding import DocumentEmbeddings
from nexusrag.services.embedding import embed_chunks
from nexusrag.services.embedding_model import MiniEmbedder
from nexusrag.storage import load_chunks, save_embeddings


class VectorIndexBackend(Protocol):
    async def index_document(
        self, chunks: DocumentChunks, embeddings: DocumentEmbeddings
    ) -> None: ...


async def build_semantic_index(
    document_id: UUID,
    embedder: MiniEmbedder,
    chunk_directory: Path,
    embedding_directory: Path,
    *,
    backend: VectorIndexBackend,
) -> DocumentEmbeddings:
    """Encode, upload one fresh generation, then atomically publish its record.

    The caller owns the model, processing lock, and backend lifecycle. Qdrant
    writes must complete before local publication; failed uploads preserve the
    previous published record. A local publish failure leaves an unreferenced
    remote generation, without replacing the old one. Errors propagate.
    """
    chunks = await load_chunks(document_id, chunk_directory)
    embeddings = await run_in_threadpool(embed_chunks, chunks, embedder)
    embeddings.index_generation = uuid4()
    await backend.index_document(chunks, embeddings)
    await save_embeddings(embeddings, embedding_directory)
    return embeddings
