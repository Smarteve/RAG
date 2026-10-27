from uuid import UUID, uuid5

from qdrant_client import models

from nexusrag.schemas.chunk import DocumentChunks
from nexusrag.schemas.embedding import DocumentEmbeddings
from nexusrag.schemas.retrieval import SemanticSearchHit
from nexusrag.schemas.vector import VectorSearchPoint
from nexusrag.services.retrieval import validate_semantic_index


def build_points(
    chunks: DocumentChunks,
    embeddings: DocumentEmbeddings,
) -> list[models.PointStruct]:
    """Validate the source/index pair once, then map it to Qdrant points."""
    validate_semantic_index(chunks, embeddings)
    points = []
    for chunk, embedding in zip(chunks.chunks, embeddings.embeddings, strict=True):
        point_id = vector_point_id(
            chunks.document_id,
            embeddings.source_digest,
            chunk.chunk_index,
            embeddings.index_generation,
        )
        point = models.PointStruct(
            id=str(point_id),
            vector=embedding.vector,
            payload={
                "document_id": str(chunks.document_id),
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "source_digest": embeddings.source_digest,
                "model_id": embeddings.model_id,
                "model_revision": embeddings.model_revision,
                "index_generation": (
                    str(embeddings.index_generation)
                    if embeddings.index_generation is not None
                    else None
                ),
            },
        )

        points.append(point)

    return points


def vector_point_id(
    document_id: UUID,
    source_digest: str,
    chunk_index: int,
    index_generation: UUID | None = None,
) -> UUID:
    """Use identical deterministic identities for writes and validated reads."""
    identity = f"{source_digest}:{chunk_index}"
    if index_generation is not None:
        identity = f"{index_generation}:{identity}"
    return uuid5(document_id, identity)


def index_identity(embeddings: DocumentEmbeddings) -> dict[str, str]:
    """Fields identifying one Qdrant generation, used for filters and joins."""
    if embeddings.index_generation is None:
        raise ValueError("Qdrant operations require a published index_generation")
    return {
        "document_id": str(embeddings.document_id),
        "index_generation": str(embeddings.index_generation),
        "source_digest": embeddings.source_digest,
        "model_id": embeddings.model_id,
        "model_revision": embeddings.model_revision,
    }


def resolve_search_hits(
    points: list[models.ScoredPoint],
    chunks: DocumentChunks,
    embeddings: DocumentEmbeddings,
    expected_identity: dict[str, str],
    limit: int,
) -> list[SemanticSearchHit]:
    """Parse remote fields once, then verify their relationships to local data.

    The caller already validated the chunks/index pair. Pydantic owns field
    types/ranges; this join owns source identity, indices, and duplicates.
    """
    if len(points) > limit:
        raise ValueError("Qdrant returned more hits than requested")
    hits = []
    seen = set()
    for point in points:
        parsed = VectorSearchPoint.model_validate(
            {
                "id": point.id,
                "payload": point.payload,
                "score": point.score,
            }
        )
        identity = parsed.payload.model_dump(
            mode="json", exclude={"chunk_index", "page_number"}
        )
        if identity != expected_identity:
            raise ValueError("Qdrant hit does not match the requested index identity")
        chunk_index = parsed.payload.chunk_index
        if chunk_index >= len(chunks.chunks):
            raise ValueError("Qdrant hit references an unknown chunk")
        chunk = chunks.chunks[chunk_index]
        if parsed.payload.page_number != chunk.page_number:
            raise ValueError("Qdrant hit page does not match its source chunk")
        expected_id = vector_point_id(
            chunks.document_id,
            embeddings.source_digest,
            chunk_index,
            embeddings.index_generation,
        )
        if parsed.id != expected_id or chunk_index in seen:
            raise ValueError("Qdrant hit has an invalid or duplicate point identity")
        seen.add(chunk_index)
        hits.append(SemanticSearchHit(chunk=chunk, similarity=parsed.score))
    return sorted(hits, key=lambda hit: (-hit.similarity, hit.chunk.chunk_index))
