import asyncio
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from nexusrag.schemas.chunk import DocumentChunk, DocumentChunks
from nexusrag.schemas.embedding import ChunkEmbedding, DocumentEmbeddings
from nexusrag.services.retrieval import StaleSemanticIndexError, validate_semantic_index
from nexusrag.services.vector_backend import QdrantBackend
from nexusrag.services.vector_points import build_points


def records(count=2):
    document_id = uuid4()
    chunks = DocumentChunks(
        document_id=document_id,
        chunk_size=10,
        overlap=0,
        chunks=[
            DocumentChunk(
                document_id=document_id,
                page_number=1,
                chunk_index=i,
                start_char=0,
                end_char=4,
                text="text",
            )
            for i in range(count)
        ],
    )
    embeddings = DocumentEmbeddings(
        document_id=document_id,
        model_id="test",
        model_revision="v1",
        dimensions=3,
        source_digest=sha256(chunks.model_dump_json().encode()).hexdigest(),
        embeddings=[
            ChunkEmbedding(chunk_index=i, vector=[1.0, 0.0, 0.0]) for i in range(count)
        ],
        index_generation=uuid4(),
    )
    return chunks, embeddings


@pytest.fixture
def client():
    client = AsyncMock(spec=AsyncQdrantClient)
    client.collection_exists.return_value = False
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=models.VectorParams(size=3, distance=models.Distance.COSINE)
            )
        )
    )
    client.upsert.return_value = models.UpdateResult(
        operation_id=1, status=models.UpdateStatus.COMPLETED
    )
    return client


def test_creates_collection_with_record_dimensions_and_uploads_all_points(client):
    chunks, embeddings = records(130)
    backend = QdrantBackend(client, collection="test-vectors")
    asyncio.run(backend.index_document(chunks, embeddings))
    client.create_collection.assert_awaited_once_with(
        collection_name="test-vectors",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    assert client.upsert.await_count == 2
    assert [len(call.kwargs["points"]) for call in client.upsert.await_args_list] == [
        128,
        2,
    ]
    assert all(call.kwargs["wait"] is True for call in client.upsert.await_args_list)
    points = [
        point
        for call in client.upsert.await_args_list
        for point in call.kwargs["points"]
    ]
    assert points == build_points(chunks, embeddings)


def test_delete_missing_collection_is_idempotent(client):
    asyncio.run(QdrantBackend(client).delete_document(uuid4()))
    client.delete.assert_not_awaited()
    client.create_collection.assert_not_awaited()


def test_delete_waits_for_completion_and_filters_only_document_id(client):
    document_id = uuid4()
    client.collection_exists.return_value = True
    client.delete.return_value = models.UpdateResult(
        status=models.UpdateStatus.COMPLETED
    )
    asyncio.run(
        QdrantBackend(client, collection="test-vectors").delete_document(document_id)
    )
    call = client.delete.await_args.kwargs
    assert call["collection_name"] == "test-vectors"
    assert call["wait"] is True
    assert call["points_selector"].filter == models.Filter(
        must=[
            models.FieldCondition(
                key="document_id", match=models.MatchValue(value=str(document_id))
            )
        ]
    )


def test_unconfirmed_vector_deletion_is_not_success(client):
    client.collection_exists.return_value = True
    client.delete.return_value = models.UpdateResult(
        status=models.UpdateStatus.ACKNOWLEDGED
    )
    with pytest.raises(RuntimeError, match="deletion was not completed"):
        asyncio.run(QdrantBackend(client).delete_document(uuid4()))


def test_real_local_delete_removes_all_generations_and_keeps_other_documents():
    async def exercise():
        client = AsyncQdrantClient(":memory:")
        try:
            backend = QdrantBackend(client, collection="delete-test")
            chunks, embeddings = records()
            other_chunks, other_embeddings = records()
            await backend.index_document(chunks, embeddings)
            await backend.index_document(
                chunks, embeddings.model_copy(update={"index_generation": uuid4()})
            )
            await backend.index_document(other_chunks, other_embeddings)
            await backend.delete_document(chunks.document_id)
            await backend.delete_document(chunks.document_id)
            points, _ = await client.scroll("delete-test", limit=20, with_payload=True)
            assert len(points) == len(other_chunks.chunks)
            assert all(
                point.payload["document_id"] == str(other_chunks.document_id)
                for point in points
            )
            assert await backend.search(other_chunks, other_embeddings, [1, 0, 0], 5)
            assert await backend.search(chunks, embeddings, [1, 0, 0], 5) == []
        finally:
            await client.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "vectors",
    [
        models.VectorParams(size=4, distance=models.Distance.COSINE),
        models.VectorParams(size=3, distance=models.Distance.DOT),
        {"named": models.VectorParams(size=3, distance=models.Distance.COSINE)},
    ],
)
def test_incompatible_existing_collection_rejected_before_upsert(client, vectors):
    chunks, embeddings = records()
    client.collection_exists.return_value = True
    client.get_collection.return_value.config.params.vectors = vectors
    with pytest.raises(ValueError, match="matching dimensions"):
        asyncio.run(QdrantBackend(client).index_document(chunks, embeddings))
    client.create_collection.assert_not_awaited()
    client.upsert.assert_not_awaited()


def test_stale_embeddings_rejected_before_network(client):
    chunks, embeddings = records()
    embeddings.source_digest = "0" * 64
    with pytest.raises(StaleSemanticIndexError):
        asyncio.run(QdrantBackend(client).index_document(chunks, embeddings))
    client.collection_exists.assert_not_awaited()


def test_generation_required_without_mutating_legacy_record(client):
    chunks, embeddings = records()
    embeddings.index_generation = None
    before = embeddings.model_dump()
    with pytest.raises(ValueError, match="index_generation"):
        asyncio.run(QdrantBackend(client).index_document(chunks, embeddings))
    assert embeddings.model_dump() == before
    client.collection_exists.assert_not_awaited()


def test_empty_document_creates_and_validates_collection_without_upsert(client):
    chunks, embeddings = records(0)
    asyncio.run(QdrantBackend(client).index_document(chunks, embeddings))
    client.get_collection.assert_awaited_once()
    client.upsert.assert_not_awaited()


def test_collection_creation_race_validates_winner(client):
    chunks, embeddings = records()
    client.create_collection.side_effect = UnexpectedResponse(
        409, "Conflict", b"exists", httpx.Headers()
    )
    asyncio.run(QdrantBackend(client).index_document(chunks, embeddings))
    client.get_collection.assert_awaited_once()
    client.upsert.assert_awaited_once()


def test_collection_errors_are_not_swallowed(client):
    chunks, embeddings = records()
    client.create_collection.side_effect = UnexpectedResponse(
        500, "Error", b"error", httpx.Headers()
    )
    with pytest.raises(UnexpectedResponse):
        asyncio.run(QdrantBackend(client).index_document(chunks, embeddings))
    client.upsert.assert_not_awaited()


def test_failed_batch_stops_and_keeps_input_unchanged(client):
    chunks, embeddings = records(260)
    before = embeddings.model_dump()
    client.upsert.side_effect = [
        models.UpdateResult(status=models.UpdateStatus.COMPLETED),
        RuntimeError("write failed"),
    ]
    with pytest.raises(RuntimeError, match="write failed"):
        asyncio.run(QdrantBackend(client).index_document(chunks, embeddings))
    assert client.upsert.await_count == 2
    assert embeddings.model_dump() == before


def test_acknowledged_but_incomplete_result_is_not_success(client):
    chunks, embeddings = records()
    client.upsert.return_value = models.UpdateResult(
        status=models.UpdateStatus.ACKNOWLEDGED
    )
    with pytest.raises(RuntimeError, match="not completed"):
        asyncio.run(QdrantBackend(client).index_document(chunks, embeddings))


def test_real_local_qdrant_generations_do_not_overwrite_each_other():
    async def exercise():
        client = AsyncQdrantClient(":memory:")
        try:
            backend = QdrantBackend(client, collection="test-vectors")
            chunks, embeddings = records()
            await backend.index_document(chunks, embeddings)
            await backend.index_document(chunks, embeddings)
            assert (await client.count("test-vectors", exact=True)).count == 2
            old_ids = {point.id for point in build_points(chunks, embeddings)}
            updated = DocumentEmbeddings.model_validate(
                embeddings.model_dump() | {"index_generation": uuid4()}
            )
            await backend.index_document(chunks, updated)
            assert (await client.count("test-vectors", exact=True)).count == 4
            assert old_ids.isdisjoint(
                {point.id for point in build_points(chunks, updated)}
            )
            # The caller's client remains usable after adapter cleanup.
            await backend.aclose()
            assert await client.collection_exists("test-vectors")
        finally:
            await client.close()

    asyncio.run(exercise())


def search_point(chunks, embeddings, index=0, score=0.8):
    point = build_points(chunks, embeddings)[index]
    return SimpleNamespace(id=point.id, payload=point.payload, score=score)


def test_search_filters_identity_and_returns_local_source_text(client):
    chunks, embeddings = records()
    point = search_point(chunks, embeddings)
    point.payload["text"] = "untrusted remote text"
    client.query_points.return_value = SimpleNamespace(points=[point])
    hits = asyncio.run(
        QdrantBackend(client, collection="test-vectors").search(
            chunks, embeddings, [1.0, 0.0, 0.0], 5
        )
    )
    assert len(hits) == 1
    assert hits[0].chunk.text == "text"
    assert hits[0].similarity == 0.8
    call = client.query_points.await_args.kwargs
    assert call["collection_name"] == "test-vectors"
    assert call["limit"] == 5
    assert call["search_params"].exact is True
    assert call["with_vectors"] is False
    assert {
        condition.key: condition.match.value for condition in call["query_filter"].must
    } == {
        "document_id": str(chunks.document_id),
        "index_generation": str(embeddings.index_generation),
        "source_digest": embeddings.source_digest,
        "model_id": embeddings.model_id,
        "model_revision": embeddings.model_revision,
    }


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_search_rejects_invalid_limits_before_network(client, limit):
    chunks, embeddings = records()
    with pytest.raises(ValidationError):
        asyncio.run(
            QdrantBackend(client).search(chunks, embeddings, [1.0, 0.0, 0.0], limit)
        )
    client.get_collection.assert_not_awaited()
    client.query_points.assert_not_awaited()


@pytest.mark.parametrize(
    "vector",
    [
        [],
        [1.0, 0.0],
        [0.0, 0.0, 0.0],
        [float("nan"), 0.0, 0.0],
        [float("inf"), 0.0, 0.0],
        [True, 0.0, 0.0],
        ["invalid", 0.0, 0.0],
    ],
)
def test_search_rejects_invalid_query_vectors(client, vector):
    chunks, embeddings = records()
    with pytest.raises(ValidationError):
        asyncio.run(QdrantBackend(client).search(chunks, embeddings, vector, 5))
    client.get_collection.assert_not_awaited()
    client.query_points.assert_not_awaited()


@pytest.mark.parametrize(
    "field",
    [
        "document_id",
        "index_generation",
        "source_digest",
        "model_id",
        "model_revision",
        "page_number",
        "chunk_index",
    ],
)
def test_search_rejects_mismatched_hit_metadata(client, field):
    chunks, embeddings = records()
    point = search_point(chunks, embeddings)
    point.payload[field] = -1 if field in {"page_number", "chunk_index"} else "wrong"
    client.query_points.return_value = SimpleNamespace(points=[point])
    with pytest.raises(ValueError):
        asyncio.run(
            QdrantBackend(client).search(chunks, embeddings, [1.0, 0.0, 0.0], 5)
        )


@pytest.mark.parametrize(
    "invalid",
    [
        "wrong_id",
        "missing_payload",
        "bool_chunk_index",
        "bool_page_number",
        "duplicate",
        "too_many",
    ],
)
def test_search_rejects_invalid_hit_identity(client, invalid):
    chunks, embeddings = records()
    point = search_point(chunks, embeddings)
    points = [point]
    limit = 5
    if invalid == "wrong_id":
        point.id = str(uuid4())
    elif invalid == "missing_payload":
        point.payload = None
    elif invalid == "bool_chunk_index":
        point.payload["chunk_index"] = False
    elif invalid == "bool_page_number":
        point.payload["page_number"] = True
    else:
        points = [point, point]
        limit = 1 if invalid == "too_many" else 5
    client.query_points.return_value = SimpleNamespace(points=points)
    with pytest.raises(ValueError):
        asyncio.run(
            QdrantBackend(client).search(chunks, embeddings, [1.0, 0.0, 0.0], limit)
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), 1.1, -1.1, True])
def test_search_rejects_invalid_scores(client, score):
    chunks, embeddings = records()
    client.query_points.return_value = SimpleNamespace(
        points=[search_point(chunks, embeddings, score=score)]
    )
    with pytest.raises(ValidationError):
        asyncio.run(
            QdrantBackend(client).search(chunks, embeddings, [1.0, 0.0, 0.0], 5)
        )


def test_search_allows_float32_rounding_at_cosine_boundary(client):
    chunks, embeddings = records()
    client.query_points.return_value = SimpleNamespace(
        points=[search_point(chunks, embeddings, score=1.0000001)]
    )
    assert (
        asyncio.run(
            QdrantBackend(client).search(chunks, embeddings, [1.0, 0.0, 0.0], 5)
        )[0].similarity
        == 1.0
    )


@pytest.mark.parametrize("case", ["empty_document", "empty_results"])
def test_empty_search_returns_list(client, case):
    chunks, embeddings = records(0 if case == "empty_document" else 2)
    client.query_points.return_value = SimpleNamespace(points=[])
    assert (
        asyncio.run(
            QdrantBackend(client).search(chunks, embeddings, [1.0, 0.0, 0.0], 5)
        )
        == []
    )
    if case == "empty_document":
        client.get_collection.assert_not_awaited()
        client.query_points.assert_not_awaited()


def test_search_requires_published_generation(client):
    chunks, embeddings = records()
    embeddings.index_generation = None
    with pytest.raises(ValueError, match="published index_generation"):
        asyncio.run(
            QdrantBackend(client).search(chunks, embeddings, [1.0, 0.0, 0.0], 5)
        )
    client.get_collection.assert_not_awaited()


def test_real_local_search_matches_cosine_and_filters_documents_and_generations():
    from nexusrag.services.retrieval import search_semantic_chunks

    async def exercise():
        client = AsyncQdrantClient(":memory:")
        try:
            backend = QdrantBackend(client, collection="test-vectors")
            chunks, embeddings = records(3)
            for record, vector in zip(
                embeddings.embeddings,
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                strict=True,
            ):
                record.vector = vector
            await backend.index_document(chunks, embeddings)
            old = DocumentEmbeddings.model_validate(
                embeddings.model_dump() | {"index_generation": uuid4()}
            )
            await backend.index_document(chunks, old)
            other_chunks, other_embeddings = records()
            await backend.index_document(other_chunks, other_embeddings)
            query = [1.0, 0.2, 0.0]
            expected = search_semantic_chunks(chunks, embeddings, query, 3)
            hits = await backend.search(chunks, embeddings, query, 3)
            assert [hit.chunk for hit in hits] == [hit.chunk for hit in expected]
            assert [hit.similarity for hit in hits] == pytest.approx(
                [hit.similarity for hit in expected], abs=1e-6
            )
        finally:
            await client.close()

    asyncio.run(exercise())


def test_index_checks_source_alignment_once(client, monkeypatch):
    from unittest.mock import Mock

    from nexusrag.services import vector_points

    chunks, embeddings = records()
    validate = Mock(wraps=validate_semantic_index)
    monkeypatch.setattr(vector_points, "validate_semantic_index", validate)
    asyncio.run(QdrantBackend(client).index_document(chunks, embeddings))
    validate.assert_called_once_with(chunks, embeddings)


def test_search_service_checks_source_alignment_once(client, monkeypatch):
    from unittest.mock import Mock

    from nexusrag import config
    from nexusrag.services import search
    from nexusrag.services.embedding_runtime import EmbeddingRuntime

    chunks, embeddings = records()
    monkeypatch.setattr(config, "EMBEDDING_MODEL_ID", embeddings.model_id)
    monkeypatch.setattr(config, "EMBEDDING_MODEL_REVISION", embeddings.model_revision)
    embedder = Mock(
        model_id=embeddings.model_id, model_revision=embeddings.model_revision
    )
    embedder.get_dimensions.return_value = embeddings.dimensions
    embedder.embed_query.return_value = [1.0, 0.0, 0.0]
    client.query_points.return_value = SimpleNamespace(
        points=[search_point(chunks, embeddings)]
    )
    validate = Mock(wraps=validate_semantic_index)
    monkeypatch.setattr(search.retrieval, "validate_semantic_index", validate)

    async def exercise():
        return await search.semantic_search(
            chunks,
            embeddings,
            "question",
            EmbeddingRuntime(lambda: embedder),
            backend=QdrantBackend(client),
        )

    asyncio.run(exercise())
    validate.assert_called_once_with(chunks, embeddings)
