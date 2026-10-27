"""Shared semantic-search orchestration, without storage or HTTP handling."""

from typing import Protocol

from fastapi.concurrency import run_in_threadpool

from nexusrag import config
from nexusrag.schemas.chunk import DocumentChunks
from nexusrag.schemas.embedding import DocumentEmbeddings
from nexusrag.schemas.retrieval import SemanticSearchHit
from nexusrag.services import retrieval
from nexusrag.services.embedding_runtime import EmbeddingRuntime


class VectorSearchBackend(Protocol):
    async def search(
        self,
        chunks: DocumentChunks,
        embeddings: DocumentEmbeddings,
        query_vector: list[float],
        limit: int,
    ) -> list[SemanticSearchHit]: ...


async def semantic_search(
    chunks: DocumentChunks,
    embeddings: DocumentEmbeddings,
    query: str,
    runtime: EmbeddingRuntime,
    *,
    backend: VectorSearchBackend,
    limit: int = 5,
) -> list[SemanticSearchHit]:
    """Validate source/model compatibility, encode once, then query the backend.

    This service owns source/index alignment validation before model work.
    The backend owns query shape, remote collection, and response validation.
    The model lock is released before awaiting remote search.
    """
    await run_in_threadpool(retrieval.validate_semantic_index, chunks, embeddings)
    if embeddings.index_generation is None:
        raise retrieval.StaleSemanticIndexError(
            "Qdrant generation is missing; rebuild the semantic index"
        )
    if (
        embeddings.model_id != config.EMBEDDING_MODEL_ID
        or embeddings.model_revision != config.EMBEDDING_MODEL_REVISION
    ):
        raise retrieval.StaleSemanticIndexError("configured model does not match")

    async with runtime.use_embedder() as embedder:
        dimensions = await run_in_threadpool(embedder.get_dimensions)
        if (
            embeddings.model_id != embedder.model_id
            or embeddings.model_revision != embedder.model_revision
            or embeddings.dimensions != dimensions
        ):
            raise retrieval.StaleSemanticIndexError("loaded model does not match")
        query_vector = await run_in_threadpool(embedder.embed_query, query)
    return await backend.search(chunks, embeddings, query_vector, limit)
