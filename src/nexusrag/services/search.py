"""Shared semantic-search orchestration, without storage or HTTP handling."""

from fastapi.concurrency import run_in_threadpool

from nexusrag import config
from nexusrag.schemas.chunk import DocumentChunks
from nexusrag.schemas.embedding import DocumentEmbeddings
from nexusrag.schemas.retrieval import SemanticSearchHit
from nexusrag.services import retrieval
from nexusrag.services.embedding_runtime import EmbeddingRuntime


async def semantic_search(
    chunks: DocumentChunks,
    embeddings: DocumentEmbeddings,
    query: str,
    runtime: EmbeddingRuntime,
    *,
    limit: int = 5,
) -> list[SemanticSearchHit]:
    """Search one saved snapshot; release the model lock before returning hits."""
    await run_in_threadpool(retrieval.validate_semantic_index, chunks, embeddings)
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
        return await run_in_threadpool(
            retrieval.search_semantic_chunks, chunks, embeddings, query_vector, limit
        )
