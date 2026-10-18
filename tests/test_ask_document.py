import asyncio
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from nexusrag import config, storage
from nexusrag.dependencies import get_answer_generator, get_embedding_runtime
from nexusrag.schemas.answer import AskDocumentResponse, GeneratedAnswer
from nexusrag.schemas.chunk import DocumentChunk, DocumentChunks
from nexusrag.schemas.embedding import ChunkEmbedding, DocumentEmbeddings
from nexusrag.services import context, generation, generation_runtime, retrieval
from nexusrag.services.embedding_model import EmbeddingInputError, MiniEmbedder
from nexusrag.services.embedding_runtime import EmbeddingRuntime
from nexusrag.services.generation_runtime import (
    GenerationProviderError,
    GenerationTimeoutError,
    GenerationUnavailableError,
    GroqAnswerGenerator,
)

main = import_module("nexusrag.main")
QUESTION = "How many vacation days do employees get?"


@pytest.fixture(autouse=True)
def directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", tmp_path / "documents")
    monkeypatch.setattr(config, "EMBEDDING_DIRECTORY", tmp_path / "embeddings")
    monkeypatch.setattr(config, "EMBEDDING_MODEL_ID", "test-encoder")
    monkeypatch.setattr(config, "EMBEDDING_MODEL_REVISION", "test-v1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


@pytest.fixture
def embedder() -> Mock:
    model = Mock(spec=MiniEmbedder, model_id="test-encoder", model_revision="test-v1")
    model.get_dimensions.return_value = 3
    model.embed_query.return_value = [1.0, 0.0, 0.0]
    model.embed_documents.side_effect = AssertionError(
        "Asking must not embed document text"
    )
    return model


@pytest.fixture
def loader(embedder: Mock) -> Mock:
    return Mock(return_value=embedder)


@pytest.fixture
def generator():
    return SimpleNamespace(
        generate=AsyncMock(
            return_value=GeneratedAnswer(
                answer="Employees receive 25 days of annual leave.",
                status="answered",
                citation_ids=["S1"],
            )
        )
    )


@pytest.fixture
def runtime_holder() -> list[EmbeddingRuntime]:
    return []


@pytest.fixture
def client(loader, generator, runtime_holder):
    async def embedding_dependency():
        if not runtime_holder:
            runtime_holder.append(EmbeddingRuntime(loader))
        return runtime_holder[0]

    async def generation_dependency():
        return generator

    overrides = main.app.dependency_overrides.copy()
    main.app.dependency_overrides[get_embedding_runtime] = embedding_dependency
    main.app.dependency_overrides[get_answer_generator] = generation_dependency
    try:
        with TestClient(main.app) as test_client:
            yield test_client
    finally:
        main.app.dependency_overrides.clear()
        main.app.dependency_overrides.update(overrides)


def save_records(count: int = 3) -> tuple[DocumentChunks, DocumentEmbeddings]:
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
    vectors = [[-1, 0, 0], [0, 1, 0], [1, 0, 0]]
    embeddings = DocumentEmbeddings(
        document_id=document_id,
        model_id="test-encoder",
        model_revision="test-v1",
        dimensions=3,
        source_digest=sha256(chunks.model_dump_json().encode("utf-8")).hexdigest(),
        embeddings=[
            ChunkEmbedding(chunk_index=index, vector=vectors[index % 3])
            for index in range(count)
        ],
    )
    asyncio.run(storage.save_chunks(chunks, config.CHUNK_DIRECTORY))
    asyncio.run(storage.save_embeddings(embeddings, config.EMBEDDING_DIRECTORY))
    return chunks, embeddings


def ask(client: TestClient, document_id, question: str = QUESTION):
    return client.post(f"/documents/{document_id}/ask", json={"question": question})


def test_ask_returns_answer_with_server_resolved_highest_ranked_source(
    client, generator, embedder, loader
) -> None:
    chunks, _ = save_records()
    paths = [
        directory / f"{chunks.document_id}.json"
        for directory in [config.CHUNK_DIRECTORY, config.EMBEDDING_DIRECTORY]
    ]
    original = [path.read_bytes() for path in paths]

    response = ask(client, chunks.document_id, f" \n{QUESTION} ")

    assert response.status_code == 200
    body = response.json()
    AskDocumentResponse.model_validate(body)
    assert body == {
        "document_id": str(chunks.document_id),
        "question": QUESTION,
        "answer": "Employees receive 25 days of annual leave.",
        "status": "answered",
        "sources": [
            {
                "source_id": "S1",
                "document_id": str(chunks.document_id),
                "page_number": 3,
                "chunk_index": 2,
            }
        ],
    }
    embedder.embed_query.assert_called_once_with(QUESTION)
    embedder.embed_documents.assert_not_called()
    loader.assert_called_once_with()
    generator.generate.assert_awaited_once()
    question, prepared = generator.generate.await_args.args
    assert question == QUESTION
    assert list(prepared.sources) == ["S1", "S2", "S3"]
    assert [source.chunk_index for source in prepared.sources.values()] == [2, 1, 0]
    assert "Passage 2: café" in prepared.text
    assert (
        generation.count_source_token_upper_bound(prepared.text)
        <= config.GENERATION_CONTEXT_TOKEN_BUDGET
    )
    assert [path.read_bytes() for path in paths] == original
    assert "vector" not in body and "text" not in body and "path" not in body


def test_multiple_citations_keep_model_order_but_use_stored_metadata(
    client, generator
) -> None:
    chunks, _ = save_records()
    generator.generate.return_value = GeneratedAnswer(
        answer="A sourced answer.", status="answered", citation_ids=["S2", "S1"]
    )

    response = ask(client, chunks.document_id)

    assert response.status_code == 200
    assert [source["page_number"] for source in response.json()["sources"]] == [2, 3]
    assert [source["chunk_index"] for source in response.json()["sources"]] == [1, 2]


def test_insufficient_evidence_returns_200_without_sources(client, generator) -> None:
    chunks, _ = save_records()
    generator.generate.return_value = GeneratedAnswer(
        answer="No parking policy is supplied.",
        status="insufficient_evidence",
        citation_ids=[],
    )

    response = ask(client, chunks.document_id, "What are the parking fees?")

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert response.json()["sources"] == []


@pytest.mark.parametrize(
    "empty", [False, True], ids=["budget_excluded", "empty_document"]
)
def test_empty_context_is_delegated_without_credentials_or_provider_call(
    client, generator, monkeypatch, empty
) -> None:
    chunks, _ = save_records(0 if empty else 3)
    if not empty:
        monkeypatch.setattr(config, "GENERATION_CONTEXT_TOKEN_BUDGET", 1)
    provider_constructor = Mock(
        side_effect=AssertionError("Empty context must not create a provider client")
    )
    # Replace the lazy binding without importing the SDK or making HTTP requests.
    monkeypatch.setitem(
        generation_runtime.__dict__,
        "groq",
        SimpleNamespace(AsyncGroq=provider_constructor),
    )
    generator.generate.side_effect = GroqAnswerGenerator().generate

    response = ask(client, chunks.document_id)

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert response.json()["sources"] == []
    assert response.json()["answer"] == (
        "The supplied sources do not provide enough evidence to answer this question."
    )
    generator.generate.assert_awaited_once_with(
        QUESTION, context.PreparedContext("", {})
    )
    provider_constructor.assert_not_called()


def test_default_retrieval_limit_is_five(client, generator) -> None:
    chunks, _ = save_records(8)

    assert ask(client, chunks.document_id).status_code == 200
    _, prepared = generator.generate.await_args.args
    assert len(prepared.sources) == 5
    assert [source.chunk_index for source in prepared.sources.values()] == [
        2,
        5,
        1,
        4,
        7,
    ]


@pytest.mark.parametrize("question", ["???", "你好？", "x" * 500])
def test_nonblank_questions_are_not_validated_with_keyword_tokenizer(
    client, embedder, question
) -> None:
    chunks, _ = save_records()

    assert ask(client, chunks.document_id, question).status_code == 200
    embedder.embed_query.assert_called_once_with(question)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": " \n"},
        {"question": "x" * 501},
        {"question": None},
        {"question": 123},
        {"question": QUESTION, "limit": 5},
    ],
)
def test_bad_body_is_422_before_storage_or_model_use(
    client, generator, loader, monkeypatch, body
) -> None:
    load = AsyncMock(side_effect=AssertionError("Bad input must not load storage"))
    monkeypatch.setattr(storage, "load_chunks", load)

    response = client.post(f"/documents/{uuid4()}/ask", json=body)

    assert response.status_code == 422
    load.assert_not_awaited()
    loader.assert_not_called()
    generator.generate.assert_not_awaited()


def test_missing_body_and_invalid_uuid_are_422(client, loader, generator) -> None:
    assert client.post(f"/documents/{uuid4()}/ask").status_code == 422
    assert ask(client, "invalid").status_code == 422
    loader.assert_not_called()
    generator.generate.assert_not_awaited()


@pytest.mark.parametrize("pdf_exists, expected", [(False, 404), (True, 409)])
def test_missing_document_or_chunks_does_not_call_models(
    client, loader, generator, pdf_exists, expected
) -> None:
    document_id = uuid4()
    if pdf_exists:
        config.STORAGE_DIRECTORY.mkdir()
        (config.STORAGE_DIRECTORY / f"{document_id}.pdf").write_bytes(b"%PDF-example")

    assert ask(client, document_id).status_code == expected
    loader.assert_not_called()
    generator.generate.assert_not_awaited()


def test_missing_index_is_409_and_is_not_rebuilt(client, loader, generator) -> None:
    chunks, _ = save_records()
    (config.EMBEDDING_DIRECTORY / f"{chunks.document_id}.json").unlink()

    response = ask(client, chunks.document_id)

    assert response.status_code == 409
    assert "build it before asking" in response.json()["detail"]
    assert list(config.EMBEDDING_DIRECTORY.iterdir()) == []
    loader.assert_not_called()
    generator.generate.assert_not_awaited()


@pytest.mark.parametrize("kind", ["chunks", "embeddings"])
@pytest.mark.parametrize("content", [b"not JSON", b"{}", b"\xff\xfe"])
def test_corrupt_records_are_safe_500_without_model_calls(
    client, loader, generator, kind, content
) -> None:
    chunks, _ = save_records()
    directory = (
        config.CHUNK_DIRECTORY if kind == "chunks" else config.EMBEDDING_DIRECTORY
    )
    (directory / f"{chunks.document_id}.json").write_bytes(content)

    response = ask(client, chunks.document_id)

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not answer document"}
    loader.assert_not_called()
    generator.generate.assert_not_awaited()


@pytest.mark.parametrize("change", ["digest", "model", "revision", "count"])
def test_stale_index_is_409_before_any_model_calls(
    client, loader, generator, change
) -> None:
    chunks, embeddings = save_records()
    if change == "digest":
        embeddings.source_digest = "0" * 64
    elif change == "model":
        embeddings.model_id = "old-model"
    elif change == "revision":
        embeddings.model_revision = "old-revision"
    else:
        embeddings.embeddings.pop()
    asyncio.run(storage.save_embeddings(embeddings, config.EMBEDDING_DIRECTORY))

    assert ask(client, chunks.document_id).status_code == 409
    loader.assert_not_called()
    generator.generate.assert_not_awaited()


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("model_id", "wrong-model"),
        ("model_revision", "wrong-revision"),
        ("dimensions", 2),
    ],
)
def test_loaded_model_mismatch_is_409_before_encoding(
    client, embedder, generator, attribute, value
) -> None:
    chunks, _ = save_records()
    if attribute == "dimensions":
        embedder.get_dimensions.return_value = value
    else:
        setattr(embedder, attribute, value)

    assert ask(client, chunks.document_id).status_code == 409
    embedder.embed_query.assert_not_called()
    generator.generate.assert_not_awaited()


def test_embedding_model_unavailable_is_safe_503(client, loader, generator) -> None:
    chunks, _ = save_records()
    loader.side_effect = OSError("/secret/cache")

    response = ask(client, chunks.document_id)

    assert response.status_code == 503
    assert "secret" not in response.text
    generator.generate.assert_not_awaited()


def test_over_embedding_token_limit_is_422(client, embedder, generator) -> None:
    chunks, _ = save_records()
    embedder.embed_query.side_effect = EmbeddingInputError(
        "query exceeds the model token limit"
    )

    assert ask(client, chunks.document_id).status_code == 422
    generator.generate.assert_not_awaited()


@pytest.mark.parametrize("vector", [[], [0, 0, 0], [1], [float("nan"), 0, 1]])
def test_invalid_embedding_output_is_500_not_bad_client_input(
    client, embedder, generator, vector
) -> None:
    chunks, _ = save_records()
    embedder.embed_query.return_value = vector

    assert ask(client, chunks.document_id).status_code == 500
    generator.generate.assert_not_awaited()


@pytest.mark.parametrize("name", ["load_chunks", "load_embeddings"])
def test_storage_errors_are_safe_500(
    client, generator, loader, monkeypatch, name
) -> None:
    chunks, _ = save_records()
    monkeypatch.setattr(
        storage, name, AsyncMock(side_effect=PermissionError("/secret/path"))
    )

    response = ask(client, chunks.document_id)

    assert response.status_code == 500
    assert "secret" not in response.text
    loader.assert_not_called()
    generator.generate.assert_not_awaited()


@pytest.mark.parametrize(
    "error, expected",
    [
        (GenerationUnavailableError("secret-key"), 503),
        (GenerationTimeoutError("secret-key"), 504),
        (GenerationProviderError("secret-key"), 502),
        (generation.InvalidGeneratedAnswerError("secret-key"), 502),
        (RuntimeError("secret-key"), 500),
    ],
)
def test_generation_errors_are_safe_and_do_not_modify_saved_data(
    client, generator, caplog, error, expected
) -> None:
    chunks, _ = save_records()
    path = config.EMBEDDING_DIRECTORY / f"{chunks.document_id}.json"
    original = path.read_bytes()
    generator.generate.side_effect = error

    response = ask(client, chunks.document_id)

    assert response.status_code == expected
    assert "secret-key" not in response.text
    assert "secret-key" not in caplog.text
    assert path.read_bytes() == original


def test_unknown_citation_from_injected_generator_is_rejected(
    client, generator
) -> None:
    chunks, _ = save_records()
    generator.generate.return_value = GeneratedAnswer(
        answer="An answer.", status="answered", citation_ids=["S999"]
    )

    response = ask(client, chunks.document_id)

    assert response.status_code == 502
    assert response.json() == {"detail": "Could not generate a valid answer"}


def test_embedding_lock_is_released_before_generation(
    client, generator, runtime_holder
) -> None:
    chunks, _ = save_records()

    async def generate(question, prepared):
        runtime = runtime_holder[0]
        assert not runtime._lock.locked()

        async def acquire_again():
            async with runtime.use_embedder():
                pass

        await asyncio.wait_for(acquire_again(), timeout=1)
        return GeneratedAnswer(
            answer="An answer.", status="answered", citation_ids=["S1"]
        )

    generator.generate.side_effect = generate

    assert ask(client, chunks.document_id).status_code == 200


def test_repeated_questions_reuse_embedding_model_but_generate_separately(
    client, loader, generator, embedder
) -> None:
    chunks, _ = save_records()

    assert ask(client, chunks.document_id).status_code == 200
    assert ask(client, chunks.document_id, "What about remote work?").status_code == 200
    loader.assert_called_once_with()
    assert embedder.embed_query.call_count == 2
    assert generator.generate.await_count == 2


def test_encoding_ranking_and_context_building_are_offloaded(
    client, embedder, monkeypatch
) -> None:
    chunks, _ = save_records()
    event_loop_threads = []
    worker_threads = []
    original_load = storage.load_chunks
    original_rank = retrieval.search_semantic_chunks
    original_context = context.build_context

    async def load(*args):
        event_loop_threads.append(get_ident())
        return await original_load(*args)

    def encode(query):
        worker_threads.append(get_ident())
        return [1, 0, 0]

    def rank(*args):
        worker_threads.append(get_ident())
        return original_rank(*args)

    def prepare(*args, **kwargs):
        worker_threads.append(get_ident())
        return original_context(*args, **kwargs)

    monkeypatch.setattr(storage, "load_chunks", load)
    monkeypatch.setattr(retrieval, "search_semantic_chunks", rank)
    monkeypatch.setattr(context, "build_context", prepare)
    embedder.embed_query.side_effect = encode

    assert ask(client, chunks.document_id).status_code == 200
    assert len(worker_threads) == 3
    assert set(worker_threads).isdisjoint(event_loop_threads)


def test_ask_does_not_call_other_http_endpoint_functions(client, monkeypatch) -> None:
    chunks, _ = save_records()
    monkeypatch.setattr(
        main,
        "semantic_search_document",
        AsyncMock(side_effect=AssertionError("Don't call an HTTP function")),
    )

    assert ask(client, chunks.document_id).status_code == 200


def test_openapi_declares_body_and_response_schemas(client) -> None:
    operation = client.get("/openapi.json").json()["paths"][
        "/documents/{document_id}/ask"
    ]["post"]

    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AskDocumentRequest"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AskDocumentResponse"
    }


def test_missing_generation_key_is_503_for_nonempty_context(client, loader) -> None:
    chunks, _ = save_records()
    main.app.dependency_overrides.pop(get_answer_generator)

    response = ask(client, chunks.document_id)

    assert response.status_code == 503
    assert "API key" in response.json()["detail"]
    loader.assert_called_once()


def test_blank_document_needs_no_generation_configuration(client) -> None:
    chunks, _ = save_records(0)
    main.app.dependency_overrides.pop(get_answer_generator)

    response = ask(client, chunks.document_id)

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"


@pytest.mark.parametrize("status", ["queued", "running", "failed", "succeeded"])
def test_processing_status_gates_answers(client, generator, loader, status):
    from datetime import UTC, datetime

    from nexusrag.schemas.job import ProcessingJob
    from nexusrag.services.job_storage import create_job, transition_job

    # Complete artifacts must not bypass queued/running/failed job status.
    chunks, _ = save_records()
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
    if status in {"succeeded", "failed"}:
        asyncio.run(
            transition_job(
                job.job_id,
                status,
                config.JOB_DATABASE_PATH,
                error="Document processing failed" if status == "failed" else None,
            )
        )
    response = client.post(
        f"/documents/{chunks.document_id}/ask", json={"question": QUESTION}
    )
    if status == "succeeded":
        assert response.status_code == 200
        generator.generate.assert_awaited_once()
    else:
        assert response.status_code == 409
        assert response.json()["detail"] == {
            "message": "Document processing failed"
            if status == "failed"
            else "Document is still processing",
            "job_id": str(job.job_id),
            "status": status,
        }
        generator.generate.assert_not_awaited()
        loader.assert_not_called()


def test_readiness_database_failure_is_safe(client, monkeypatch, generator):
    import sqlite3

    monkeypatch.setattr(
        main,
        "load_document_job",
        AsyncMock(side_effect=sqlite3.OperationalError("private path")),
    )
    response = client.post(f"/documents/{uuid4()}/ask", json={"question": QUESTION})
    assert response.status_code == 500
    assert response.json() == {"detail": "Could not check document readiness"}
    generator.generate.assert_not_awaited()
