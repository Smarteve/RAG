import asyncio
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
from nexusrag.schemas.embedding import SemanticIndexResponse
from nexusrag.services.embedding import embed_chunks
from nexusrag.services.embedding_model import EmbeddingInputError, MiniEmbedder
from nexusrag.services.embedding_runtime import EmbeddingRuntime

main = import_module("nexusrag.main")


@pytest.fixture(autouse=True)
def directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", tmp_path / "documents")
    monkeypatch.setattr(config, "EMBEDDING_DIRECTORY", tmp_path / "embeddings")


@pytest.fixture
def embedder() -> Mock:
    encoder = Mock(spec=MiniEmbedder, model_id="test-encoder", model_revision="test-v1")
    encoder.get_dimensions.return_value = 3
    encoder.embed_documents.side_effect = lambda texts: [
        [float(len(text)), 1.0, 0.0] for text in texts
    ]
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


def save_source(count: int = 2) -> DocumentChunks:
    document_id = uuid4()
    texts = [f"Passage {index}: café" for index in range(count)]
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
    asyncio.run(storage.save_chunks(chunks, config.CHUNK_DIRECTORY))
    return chunks


@pytest.mark.parametrize("count", [0, 2, 17])
def test_post_builds_complete_index_and_returns_only_metadata(
    client: TestClient, loader: Mock, count: int
) -> None:
    chunks = save_source(count)
    source = config.CHUNK_DIRECTORY / f"{chunks.document_id}.json"
    original = source.read_bytes()

    response = client.post(f"/documents/{chunks.document_id}/semantic-index")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": str(chunks.document_id),
        "model_id": "test-encoder",
        "model_revision": "test-v1",
        "dimensions": 3,
        "embedding_count": count,
    }
    SemanticIndexResponse.model_validate(response.json())
    saved = asyncio.run(
        storage.load_embeddings(chunks.document_id, config.EMBEDDING_DIRECTORY)
    )
    assert saved == embed_chunks(chunks, loader.return_value)
    assert source.read_bytes() == original
    loader.assert_called_once_with()


def test_rebuild_reuses_model_and_replaces_index_for_current_chunks(
    client: TestClient, loader: Mock
) -> None:
    chunks = save_source()
    url = f"/documents/{chunks.document_id}/semantic-index"
    assert client.post(url).status_code == 200
    previous = asyncio.run(
        storage.load_embeddings(chunks.document_id, config.EMBEDDING_DIRECTORY)
    )
    chunks.chunks[0].text = chunks.chunks[0].text.replace("café", "tea!")
    asyncio.run(storage.save_chunks(chunks, config.CHUNK_DIRECTORY))

    response = client.post(url)

    assert response.status_code == 200
    current = asyncio.run(
        storage.load_embeddings(chunks.document_id, config.EMBEDDING_DIRECTORY)
    )
    assert current.source_digest != previous.source_digest
    assert current == embed_chunks(chunks, loader.return_value)
    loader.assert_called_once_with()


def test_invalid_uuid_does_not_load_model(client: TestClient, loader: Mock) -> None:
    response = client.post("/documents/not-a-uuid/semantic-index")

    assert response.status_code == 422
    loader.assert_not_called()


def test_get_cannot_trigger_index_build(client: TestClient, loader: Mock) -> None:
    response = client.get(f"/documents/{uuid4()}/semantic-index")

    assert response.status_code == 405
    loader.assert_not_called()


@pytest.mark.parametrize("pdf_exists, expected_status", [(False, 404), (True, 409)])
def test_missing_chunks_do_not_load_model(
    client: TestClient, loader: Mock, pdf_exists: bool, expected_status: int
) -> None:
    document_id = uuid4()
    if pdf_exists:
        config.STORAGE_DIRECTORY.mkdir()
        (config.STORAGE_DIRECTORY / f"{document_id}.pdf").write_bytes(b"%PDF-example")

    response = client.post(f"/documents/{document_id}/semantic-index")

    assert response.status_code == expected_status
    loader.assert_not_called()
    assert not config.EMBEDDING_DIRECTORY.exists()


@pytest.mark.parametrize("contents", [b"not JSON", b"{}", b"\xff\xfe"])
def test_corrupt_chunks_return_generic_500_before_loading_model(
    client: TestClient, loader: Mock, contents: bytes
) -> None:
    document_id = uuid4()
    config.CHUNK_DIRECTORY.mkdir()
    (config.CHUNK_DIRECTORY / f"{document_id}.json").write_bytes(contents)

    response = client.post(f"/documents/{document_id}/semantic-index")

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load chunks"}
    loader.assert_not_called()


def test_chunk_document_id_mismatch_is_internal_error(
    client: TestClient, loader: Mock
) -> None:
    chunks = save_source()
    requested_id = uuid4()
    (config.CHUNK_DIRECTORY / f"{requested_id}.json").write_text(
        chunks.model_dump_json(), encoding="utf-8"
    )

    response = client.post(f"/documents/{requested_id}/semantic-index")

    assert response.status_code == 500
    loader.assert_not_called()


@pytest.mark.parametrize(
    "failure", [ImportError("secret dependency"), OSError("/secret/cache")]
)
def test_unavailable_model_returns_safe_503_and_preserves_previous_index(
    client: TestClient, loader: Mock, embedder: Mock, failure: Exception
) -> None:
    chunks = save_source()
    previous = embed_chunks(chunks, embedder)
    path = asyncio.run(storage.save_embeddings(previous, config.EMBEDDING_DIRECTORY))
    original = path.read_bytes()
    loader.side_effect = failure

    response = client.post(f"/documents/{chunks.document_id}/semantic-index")

    assert response.status_code == 503
    assert "secret" not in response.text
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "failure, expected_status",
    [
        (EmbeddingInputError("document 0 has too many tokens"), 409),
        (RuntimeError("/secret/model failed"), 500),
        (ValueError("invalid provider output"), 500),
    ],
)
def test_encoding_failures_do_not_replace_existing_index(
    client: TestClient, embedder: Mock, failure: Exception, expected_status: int
) -> None:
    chunks = save_source()
    path = asyncio.run(
        storage.save_embeddings(
            embed_chunks(chunks, embedder), config.EMBEDDING_DIRECTORY
        )
    )
    original = path.read_bytes()
    embedder.embed_documents.side_effect = failure

    response = client.post(f"/documents/{chunks.document_id}/semantic-index")

    assert response.status_code == expected_status
    assert "secret" not in response.text
    assert path.read_bytes() == original


@pytest.mark.parametrize("vector", [[0.0, 0.0, 0.0], [1.0, 0.0]])
def test_invalid_provider_vectors_are_500_not_request_validation_errors(
    client: TestClient, embedder: Mock, vector: list[float]
) -> None:
    chunks = save_source(1)
    embedder.embed_documents.side_effect = lambda texts: [vector]

    response = client.post(f"/documents/{chunks.document_id}/semantic-index")

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not build semantic index"}
    assert not config.EMBEDDING_DIRECTORY.exists()


def test_failed_publish_returns_500_and_keeps_previous_index(
    client: TestClient, embedder: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = save_source()
    path = asyncio.run(
        storage.save_embeddings(
            embed_chunks(chunks, embedder), config.EMBEDDING_DIRECTORY
        )
    )
    original = path.read_bytes()

    def fail_replace(self, target):
        raise FileNotFoundError("/secret/destination disappeared")

    monkeypatch.setattr(Path, "replace", fail_replace)

    response = client.post(f"/documents/{chunks.document_id}/semantic-index")

    # A save failure must not be misclassified as an unknown document (404).
    assert response.status_code == 500
    assert response.json() == {"detail": "Could not build semantic index"}
    assert path.read_bytes() == original
    assert list(config.EMBEDDING_DIRECTORY.iterdir()) == [path]


def test_unreadable_chunks_return_generic_500(
    client: TestClient, loader: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main, "load_chunks", AsyncMock(side_effect=OSError("/secret/path"))
    )

    response = client.post(f"/documents/{uuid4()}/semantic-index")

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load chunks"}
    loader.assert_not_called()


def test_encoding_is_offloaded_from_request_event_loop(
    client: TestClient, embedder: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = save_source()
    request_threads = []
    worker_threads = []
    original = main.load_chunks

    async def load(*args):
        request_threads.append(get_ident())
        return await original(*args)

    def encode(texts):
        worker_threads.append(get_ident())
        return [[1.0, 0.0, 0.0] for text in texts]

    monkeypatch.setattr(main, "load_chunks", load)
    embedder.embed_documents.side_effect = encode

    response = client.post(f"/documents/{chunks.document_id}/semantic-index")

    assert response.status_code == 200
    assert request_threads
    assert worker_threads
    assert set(worker_threads).isdisjoint(request_threads)


def test_manual_indexing_respects_processing_lock(client, loader):
    from nexusrag.services.processing_lock import document_lock

    chunks = save_source()
    with document_lock(chunks.document_id, config.JOB_DATABASE_PATH):
        response = client.post(f"/documents/{chunks.document_id}/semantic-index")
    assert response.status_code == 409
    loader.assert_not_called()
    assert (
        client.post(f"/documents/{chunks.document_id}/semantic-index").status_code
        == 200
    )


def test_manual_indexing_holds_lock_through_publish(client, monkeypatch):
    from nexusrag.services.processing_lock import DocumentBusyError, document_lock

    chunks = save_source()
    original = main.indexing.build_semantic_index

    async def build(*args, **kwargs):
        with (
            pytest.raises(DocumentBusyError),
            document_lock(chunks.document_id, config.JOB_DATABASE_PATH),
        ):
            pytest.fail("Reindexing did not hold document lock")
        return await original(*args, **kwargs)

    monkeypatch.setattr(main.indexing, "build_semantic_index", build)
    assert (
        client.post(f"/documents/{chunks.document_id}/semantic-index").status_code
        == 200
    )


@pytest.mark.parametrize("status", ["queued", "running", "failed"])
def test_manual_indexing_rejects_unfinished_jobs(client, loader, status):
    from datetime import UTC, datetime

    from nexusrag.schemas.job import ProcessingJob
    from nexusrag.services.job_storage import create_job, transition_job

    chunks = save_source()
    now = datetime.now(UTC)
    job = ProcessingJob(
        job_id=uuid4(),
        document_id=chunks.document_id,
        status="queued",
        created_at=now,
        updated_at=now,
    )
    asyncio.run(create_job(job, config.JOB_DATABASE_PATH))
    if status != "queued":
        asyncio.run(transition_job(job.job_id, "running", config.JOB_DATABASE_PATH))
    if status == "failed":
        asyncio.run(
            transition_job(
                job.job_id, "failed", config.JOB_DATABASE_PATH, "Processing failed"
            )
        )
    response = client.post(f"/documents/{chunks.document_id}/semantic-index")
    assert response.status_code == 409
    loader.assert_not_called()
