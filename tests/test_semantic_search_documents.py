import asyncio
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from threading import get_ident
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from nexusrag import config, storage
from nexusrag.dependencies import get_embedding_runtime
from nexusrag.schemas.chunk import DocumentChunk, DocumentChunks
from nexusrag.schemas.embedding import ChunkEmbedding, DocumentEmbeddings
from nexusrag.schemas.retrieval import SemanticSearchResponse
from nexusrag.services import retrieval
from nexusrag.services.embedding_model import EmbeddingInputError, MiniEmbedder
from nexusrag.services.embedding_runtime import EmbeddingRuntime

main = import_module("nexusrag.main")


@pytest.fixture(autouse=True)
def directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", tmp_path / "documents")
    monkeypatch.setattr(config, "EMBEDDING_DIRECTORY", tmp_path / "embeddings")
    monkeypatch.setattr(config, "EMBEDDING_MODEL_ID", "test-encoder")
    monkeypatch.setattr(config, "EMBEDDING_MODEL_REVISION", "test-v1")


@pytest.fixture
def embedder() -> Mock:
    encoder = Mock(spec=MiniEmbedder, model_id="test-encoder", model_revision="test-v1")
    encoder.get_dimensions.return_value = 3
    encoder.embed_query.return_value = [1.0, 0.0, 0.0]
    encoder.embed_documents.side_effect = AssertionError(
        "Search must not embed documents"
    )
    return encoder


@pytest.fixture
def loader(embedder: Mock) -> Mock:
    return Mock(return_value=embedder)


@pytest.fixture
def client(loader: Mock):
    runtime = None

    async def dependency():
        nonlocal runtime
        if runtime is None:
            runtime = EmbeddingRuntime(loader)
        return runtime

    previous = main.app.dependency_overrides.get(get_embedding_runtime)
    main.app.dependency_overrides[get_embedding_runtime] = dependency
    try:
        with TestClient(main.app) as test_client:
            yield test_client
    finally:
        if previous is None:
            main.app.dependency_overrides.pop(get_embedding_runtime, None)
        else:
            main.app.dependency_overrides[get_embedding_runtime] = previous


def save_records(
    vectors: list[list[float]] | None = None, dimensions: int = 3
) -> tuple[DocumentChunks, DocumentEmbeddings]:
    if vectors is None:
        vectors = [[-1, 0, 0], [0, 1, 0], [1, 0, 0]]
    document_id = uuid4()
    texts = [f"Passage {index}: café" for index in range(len(vectors))]
    chunks = DocumentChunks(
        document_id=document_id,
        chunk_size=100,
        overlap=0,
        chunks=[
            DocumentChunk(
                document_id=document_id,
                page_number=index + 1,
                chunk_index=index,
                start_char=0,
                end_char=len(text),
                text=text,
            )
            for index, text in enumerate(texts)
        ],
    )
    embeddings = DocumentEmbeddings(
        document_id=document_id,
        model_id="test-encoder",
        model_revision="test-v1",
        dimensions=dimensions,
        source_digest=sha256(chunks.model_dump_json().encode("utf-8")).hexdigest(),
        embeddings=[
            ChunkEmbedding(chunk_index=index, vector=vector)
            for index, vector in enumerate(vectors)
        ],
    )
    asyncio.run(storage.save_chunks(chunks, config.CHUNK_DIRECTORY))
    asyncio.run(storage.save_embeddings(embeddings, config.EMBEDDING_DIRECTORY))
    return chunks, embeddings


def test_search_embeds_only_query_and_returns_ranked_original_chunks(
    client: TestClient, loader: Mock, embedder: Mock
) -> None:
    chunks, _ = save_records()
    paths = [
        directory / f"{chunks.document_id}.json"
        for directory in [config.CHUNK_DIRECTORY, config.EMBEDDING_DIRECTORY]
    ]
    originals = [path.read_bytes() for path in paths]
    query = " Vacation days? "

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": query}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == str(chunks.document_id)
    assert body["query"] == query
    assert [hit["chunk"] for hit in body["hits"]] == [
        chunks.chunks[index].model_dump(mode="json") for index in [2, 1, 0]
    ]
    assert [hit["similarity"] for hit in body["hits"]] == pytest.approx([1, 0, -1])
    assert all("score" not in hit for hit in body["hits"])
    SemanticSearchResponse.model_validate(body)
    embedder.embed_query.assert_called_once_with(query)
    embedder.embed_documents.assert_not_called()
    loader.assert_called_once_with()
    assert [path.read_bytes() for path in paths] == originals


@pytest.mark.parametrize("limit", [1, 20])
def test_limit_selects_highest_similarity_hits(client: TestClient, limit: int) -> None:
    chunks, _ = save_records()

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search",
        params={"query": "vacation", "limit": limit},
    )

    assert response.status_code == 200
    assert [hit["chunk"]["chunk_index"] for hit in response.json()["hits"]] == [
        2,
        1,
        0,
    ][:limit]


def test_default_limit_is_five_and_ties_use_chunk_index(client: TestClient) -> None:
    chunks, _ = save_records([[1, 0, 0]] * 6)

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "vacation"}
    )

    assert response.status_code == 200
    assert [hit["chunk"]["chunk_index"] for hit in response.json()["hits"]] == list(
        range(5)
    )


def test_empty_index_returns_empty_hits(client: TestClient, embedder: Mock) -> None:
    chunks, _ = save_records([])

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "vacation"}
    )

    assert response.status_code == 200
    assert response.json()["hits"] == []
    embedder.embed_documents.assert_not_called()


@pytest.mark.parametrize("query", ["?!___🚀", "x" * 500])
def test_nonblank_query_does_not_use_keyword_tokenizer(
    client: TestClient, embedder: Mock, query: str
) -> None:
    chunks, _ = save_records()

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": query}
    )

    assert response.status_code == 200
    embedder.embed_query.assert_called_once_with(query)


def test_repeated_queries_reuse_loaded_model(client: TestClient, loader: Mock) -> None:
    chunks, _ = save_records()
    url = f"/documents/{chunks.document_id}/semantic-search"

    assert client.get(url, params={"query": "vacation"}).status_code == 200
    assert client.get(url, params={"query": "annual leave"}).status_code == 200

    loader.assert_called_once_with()
    assert loader.return_value.embed_query.call_count == 2
    loader.return_value.embed_documents.assert_not_called()


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"query": ""},
        {"query": " \t\n"},
        {"query": "x" * 501},
        {"query": "leave", "limit": 0},
        {"query": "leave", "limit": -1},
        {"query": "leave", "limit": 21},
        {"query": "leave", "limit": "abc"},
    ],
)
def test_invalid_inputs_are_rejected_before_storage_or_model_access(
    client: TestClient, loader: Mock, monkeypatch: pytest.MonkeyPatch, params: dict
) -> None:
    load = AsyncMock(side_effect=AssertionError("Invalid input must not read storage"))
    monkeypatch.setattr(main, "load_chunks", load)

    response = client.get(f"/documents/{uuid4()}/semantic-search", params=params)

    assert response.status_code == 422
    load.assert_not_awaited()
    loader.assert_not_called()


def test_invalid_uuid_is_rejected(client: TestClient, loader: Mock) -> None:
    response = client.get(
        "/documents/not-a-uuid/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 422
    loader.assert_not_called()


@pytest.mark.parametrize("pdf_exists, expected_status", [(False, 404), (True, 409)])
def test_document_without_chunks_does_not_load_model(
    client: TestClient, loader: Mock, pdf_exists: bool, expected_status: int
) -> None:
    document_id = uuid4()
    if pdf_exists:
        config.STORAGE_DIRECTORY.mkdir()
        (config.STORAGE_DIRECTORY / f"{document_id}.pdf").write_bytes(b"%PDF-example")

    response = client.get(
        f"/documents/{document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == expected_status
    loader.assert_not_called()


def test_missing_index_is_409_even_when_only_chunks_remain(
    client: TestClient, loader: Mock
) -> None:
    chunks, _ = save_records()
    (config.EMBEDDING_DIRECTORY / f"{chunks.document_id}.json").unlink()

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 409
    assert "Semantic index is not available" in response.json()["detail"]
    loader.assert_not_called()
    assert list(config.EMBEDDING_DIRECTORY.iterdir()) == []


@pytest.mark.parametrize("kind", ["chunks", "embeddings"])
@pytest.mark.parametrize("contents", [b"not JSON", b"{}", b"\xff\xfe"])
def test_corrupt_storage_is_generic_500_before_model_loading(
    client: TestClient, loader: Mock, kind: str, contents: bytes
) -> None:
    chunks, _ = save_records()
    directory = (
        config.CHUNK_DIRECTORY if kind == "chunks" else config.EMBEDDING_DIRECTORY
    )
    (directory / f"{chunks.document_id}.json").write_bytes(contents)

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 500
    expected = (
        "Could not load chunks" if kind == "chunks" else "Could not load semantic index"
    )
    assert response.json() == {"detail": expected}
    loader.assert_not_called()


@pytest.mark.parametrize("change", ["text", "settings", "count", "model", "revision"])
def test_stale_index_is_409_without_model_or_query_encoding(
    client: TestClient, loader: Mock, change: str
) -> None:
    chunks, embeddings = save_records()
    if change == "text":
        chunks.chunks[0].text = chunks.chunks[0].text.replace("café", "tea!")
    elif change == "settings":
        chunks.overlap = 1
    elif change == "count":
        embeddings.embeddings.pop()
    elif change == "model":
        embeddings.model_id = "old-encoder"
    else:
        embeddings.model_revision = "old-revision"
    asyncio.run(storage.save_chunks(chunks, config.CHUNK_DIRECTORY))
    path = asyncio.run(storage.save_embeddings(embeddings, config.EMBEDDING_DIRECTORY))
    original = path.read_bytes()

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 409
    assert "rebuild" in response.json()["detail"]
    loader.assert_not_called()
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "attribute, value",
    [("model_id", "wrong-model"), ("model_revision", "wrong-revision")],
)
def test_loaded_model_identity_must_match_saved_index(
    client: TestClient, embedder: Mock, attribute: str, value: str
) -> None:
    chunks, _ = save_records()
    setattr(embedder, attribute, value)

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 409
    embedder.embed_query.assert_not_called()


def test_loaded_model_dimensions_must_match_even_with_empty_index(
    client: TestClient, embedder: Mock
) -> None:
    chunks, _ = save_records([], dimensions=2)

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 409
    embedder.embed_query.assert_not_called()


def test_missing_model_is_safe_503_without_modifying_index(
    client: TestClient, loader: Mock
) -> None:
    chunks, _ = save_records()
    path = config.EMBEDDING_DIRECTORY / f"{chunks.document_id}.json"
    original = path.read_bytes()
    loader.side_effect = OSError("/secret/model-cache")

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 503
    assert "secret" not in response.text
    assert path.read_bytes() == original


def test_query_over_token_limit_is_422(client: TestClient, embedder: Mock) -> None:
    chunks, _ = save_records()
    embedder.embed_query.side_effect = EmbeddingInputError(
        "query has 300 tokens; limit is 256"
    )

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "query has 300 tokens; limit is 256"}


@pytest.mark.parametrize(
    "vector", [[], [0, 0, 0], [float("nan"), 1, 0], [float("inf"), 1, 0], [1]]
)
def test_invalid_query_vector_from_model_is_500_not_bad_client_input(
    client: TestClient, embedder: Mock, vector: list[float]
) -> None:
    chunks, _ = save_records()
    embedder.embed_query.return_value = vector

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not search document"}


@pytest.mark.parametrize("kind", ["chunks", "embeddings"])
def test_storage_permission_errors_are_generic_500(
    client: TestClient, loader: Mock, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    chunks, _ = save_records()
    name = "load_chunks" if kind == "chunks" else "load_embeddings"
    monkeypatch.setattr(
        main, name, AsyncMock(side_effect=PermissionError("/secret/path"))
    )

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 500
    assert "secret" not in response.text
    loader.assert_not_called()


def test_internal_ranker_failure_is_safe_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks, _ = save_records()
    monkeypatch.setattr(
        retrieval,
        "search_semantic_chunks",
        Mock(side_effect=ValueError("internal bug")),
    )

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not search document"}


def test_query_encoding_and_ranking_run_in_worker_threads(
    client: TestClient, embedder: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks, embeddings = save_records()
    request_threads = []
    worker_threads = []
    original_load = main.load_chunks
    original_rank = retrieval.search_semantic_chunks

    async def load(*args):
        request_threads.append(get_ident())
        return await original_load(*args)

    def encode(query):
        worker_threads.append(get_ident())
        return [1.0, 0.0, 0.0]

    def rank(chunk_record, embedding_record, query_vector, limit):
        worker_threads.append(get_ident())
        assert chunk_record == chunks
        assert embedding_record == embeddings
        return original_rank(chunk_record, embedding_record, query_vector, limit)

    monkeypatch.setattr(main, "load_chunks", load)
    monkeypatch.setattr(retrieval, "search_semantic_chunks", rank)
    embedder.embed_query.side_effect = encode

    response = client.get(
        f"/documents/{chunks.document_id}/semantic-search", params={"query": "leave"}
    )

    assert response.status_code == 200
    assert len(worker_threads) == 2
    assert set(worker_threads).isdisjoint(request_threads)


def test_docs_declare_semantic_response_schema(client: TestClient) -> None:
    operation = client.get("/openapi.json").json()["paths"][
        "/documents/{document_id}/semantic-search"
    ]["get"]

    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SemanticSearchResponse"
    }
