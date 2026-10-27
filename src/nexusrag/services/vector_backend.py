"""Generation-scoped Qdrant indexing and search; application wiring is separate."""

from uuid import UUID

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from nexusrag import config
from nexusrag.schemas.chunk import DocumentChunks
from nexusrag.schemas.embedding import DocumentEmbeddings
from nexusrag.schemas.retrieval import SemanticSearchHit
from nexusrag.schemas.vector import VectorQuery
from nexusrag.services.vector_points import (
    build_points,
    index_identity,
    resolve_search_hits,
)


class QdrantBackend:
    def __init__(
        self,
        client: AsyncQdrantClient | None = None,
        *,
        collection: str | None = None,
    ) -> None:
        self.client = (
            client
            if client is not None
            else AsyncQdrantClient(url=config.QDRANT_URL, timeout=10)
        )
        self.collection = (
            collection if collection is not None else config.QDRANT_COLLECTION
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close clients created by this adapter; injected clients belong to callers."""
        if self._owns_client:
            await self.client.close()

    async def delete_document(self, document_id: UUID) -> None:
        """Delete every generation for this document, preserving other documents."""
        if not await self.client.collection_exists(self.collection):
            return
        result = await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
            wait=True,
        )
        if result.status != models.UpdateStatus.COMPLETED:
            raise RuntimeError("Qdrant point deletion was not completed")

    async def index_document(
        self,
        chunks: DocumentChunks,
        embeddings: DocumentEmbeddings,
    ) -> None:
        """Write one generation without publishing it or deleting older points.

        The caller assigns a fresh index_generation for each new build and saves
        the embedding record only after this method succeeds. Retrying writes of
        that same unpublished generation is idempotent. Partial uploads may
        remain in Qdrant but must not be selected by the published generation.
        """
        index_identity(embeddings)  # A Qdrant write requires a generation.
        points = build_points(chunks, embeddings)  # Owns source/index validation.
        await self._ensure_collection(embeddings.dimensions)
        await self._upsert_points(points)

    async def _ensure_collection(self, dimensions: int) -> None:
        if not await self.client.collection_exists(self.collection):
            try:
                await self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=dimensions,
                        distance=models.Distance.COSINE,
                    ),
                )
            except UnexpectedResponse as error:
                # Another worker may have created this collection since checking.
                # Only that conflict is recoverable, and the winner is validated.
                if error.status_code != 409:
                    raise

        await self._validate_collection(dimensions)

    async def _upsert_points(self, points: list[models.PointStruct]) -> None:
        for offset in range(0, len(points), 128):
            result = await self.client.upsert(
                collection_name=self.collection,
                wait=True,
                points=points[offset : offset + 128],
            )
            if result.status != models.UpdateStatus.COMPLETED:
                raise RuntimeError("Qdrant point update was not completed")

    async def _validate_collection(self, dimensions: int) -> None:
        info = await self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if (
            not isinstance(vectors, models.VectorParams)
            or vectors.size != dimensions
            or vectors.distance != models.Distance.COSINE
        ):
            raise ValueError(
                "Qdrant collection must use matching dimensions and cosine distance"
            )

    async def search(
        self,
        chunks: DocumentChunks,
        embeddings: DocumentEmbeddings,
        query_vector: list[float],
        limit: int,
    ) -> list[SemanticSearchHit]:
        """Return verified local source chunks from this published generation.

        The caller (semantic_search) validates source/index alignment before
        encoding. This adapter validates the query, collection, and remote hits.
        No index is created/rebuilt and no remote text is used as source text.
        """
        expected = index_identity(embeddings)
        query = VectorQuery(
            query_vector=query_vector, dimensions=embeddings.dimensions, limit=limit
        )
        if not chunks.chunks:
            return []
        await self._validate_collection(embeddings.dimensions)
        result = await self.client.query_points(
            collection_name=self.collection,
            query=query.query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in expected.items()
                ]
            ),
            search_params=models.SearchParams(exact=True),
            limit=query.limit,
            with_payload=True,
            with_vectors=False,
        )
        return resolve_search_hits(
            result.points, chunks, embeddings, expected, query.limit
        )
