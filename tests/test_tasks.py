import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from nexusrag import tasks
from nexusrag.schemas.job import ProcessingJob
from nexusrag.services.embedding_model import MiniEmbedder


@pytest.fixture
def workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ProcessingJob, AsyncMock, AsyncMock]:
    now = datetime.now(UTC)
    job = ProcessingJob(
        job_id=uuid4(),
        document_id=uuid4(),
        status="queued",
        created_at=now,
        updated_at=now,
    )
    monkeypatch.setattr(tasks, "load_job", AsyncMock(return_value=job))
    transition = AsyncMock()
    monkeypatch.setattr(tasks, "transition_job", transition)
    process = AsyncMock(return_value=[])
    monkeypatch.setattr(tasks, "run_in_threadpool", process)
    monkeypatch.setattr(tasks, "save_extraction", AsyncMock())
    monkeypatch.setattr(tasks, "save_chunks", AsyncMock())
    runtime = Mock()
    embedder = Mock(spec=MiniEmbedder)
    embedder.get_token_limit.return_value = 256
    runtime.use_embedder.return_value.__aenter__ = AsyncMock(return_value=embedder)
    runtime.use_embedder.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(tasks, "EmbeddingRuntime", Mock(return_value=runtime))
    monkeypatch.setattr(tasks, "build_semantic_index", AsyncMock())
    backend = Mock(index_document=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr(tasks, "QdrantBackend", Mock(return_value=backend))
    return job, transition, process


def test_success_marks_ready_after_indexing(workflow) -> None:
    job, transition, _ = workflow
    asyncio.run(tasks._process_document(job.job_id))
    assert [call.args[1] for call in transition.await_args_list] == [
        "running",
        "succeeded",
    ]
    tasks.save_extraction.assert_awaited_once()
    tasks.save_chunks.assert_awaited_once()
    tasks.build_semantic_index.assert_awaited_once()
    backend = tasks.QdrantBackend.return_value
    assert tasks.build_semantic_index.await_args.kwargs["backend"] is backend
    backend.aclose.assert_awaited_once()
    assert tasks.build_semantic_index.await_args.args[0] == job.document_id
    embedder = tasks.build_semantic_index.await_args.args[1]
    chunk_call = tasks.run_in_threadpool.await_args_list[1]
    assert chunk_call.args[0] is tasks.chunk_document
    assert chunk_call.kwargs == {
        "count_tokens": embedder.count_tokens,
        "max_tokens": embedder.get_token_limit.return_value,
    }
    tasks.EmbeddingRuntime.assert_called_once_with()
    tasks.EmbeddingRuntime.return_value.use_embedder.assert_called_once_with()


def test_claim_failure_does_not_process_or_fail_job(workflow) -> None:
    job, transition, process = workflow
    transition.side_effect = ValueError("Invalid job transition")
    with pytest.raises(
        tasks.JobNotClaimedError, match="Could not claim processing job"
    ):
        asyncio.run(tasks._process_document(job.job_id))
    assert transition.await_count == 1
    process.assert_not_awaited()


@pytest.mark.parametrize(
    "stage",
    ["run_in_threadpool", "save_extraction", "save_chunks", "build_semantic_index"],
)
def test_processing_failure_is_recorded_and_reraised(workflow, stage: str) -> None:
    job, transition, _ = workflow
    error = RuntimeError("private provider/path detail")
    getattr(tasks, stage).side_effect = error
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(tasks._process_document(job.job_id))
    assert caught.value is error
    assert [call.args[1] for call in transition.await_args_list] == [
        "running",
        "failed",
    ]
    assert transition.await_args.kwargs["error"] == "Document processing failed"


def test_success_write_failure_does_not_mark_processing_failed(workflow) -> None:
    job, transition, _ = workflow
    transition.side_effect = [None, OSError("Database unavailable")]
    with pytest.raises(OSError, match="Database unavailable"):
        asyncio.run(tasks._process_document(job.job_id))
    assert [call.args[1] for call in transition.await_args_list] == [
        "running",
        "succeeded",
    ]


def test_task_entry_point_holds_lock_through_work(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from uuid import uuid4

    from nexusrag.services.processing_lock import DocumentBusyError, document_lock

    job_id, document_id = uuid4(), uuid4()
    path = tmp_path / "jobs.sqlite3"
    monkeypatch.setattr(tasks.config, "JOB_DATABASE_PATH", path)
    monkeypatch.setattr(
        tasks, "_load_job", Mock(return_value=SimpleNamespace(document_id=document_id))
    )

    async def process(identifier):
        assert identifier == job_id
        with pytest.raises(DocumentBusyError), document_lock(document_id, path):
            pytest.fail("Task did not hold lock")

    monkeypatch.setattr(tasks, "_process_document", process)
    tasks.process_document(str(job_id))


def test_index_failure_closes_worker_backend(workflow):
    job, transition, _ = workflow
    tasks.build_semantic_index.side_effect = RuntimeError("Index failed")
    with pytest.raises(RuntimeError, match="Index failed"):
        asyncio.run(tasks._process_document(job.job_id))
    tasks.QdrantBackend.return_value.aclose.assert_awaited_once()
    assert transition.await_args.args[1] == "failed"


def test_backend_request_dependency_closes_client_on_error(monkeypatch):
    from contextlib import aclosing

    from nexusrag import dependencies

    backend = Mock(aclose=AsyncMock())
    monkeypatch.setattr(dependencies, "QdrantBackend", Mock(return_value=backend))

    async def request():
        async with aclosing(dependencies.get_vector_backend()) as dependency:
            assert await anext(dependency) is backend
            raise RuntimeError("Request failed")

    with pytest.raises(RuntimeError, match="Request failed"):
        asyncio.run(request())
    backend.aclose.assert_awaited_once()


def test_worker_chunks_to_model_budget_and_indexes_with_one_model_load(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from qdrant_client import AsyncQdrantClient

    from nexusrag import config, storage
    from nexusrag.database import initialize_database
    from nexusrag.schemas.extraction import ExtractedPage
    from nexusrag.services.embedding_runtime import EmbeddingRuntime
    from nexusrag.services.job_storage import create_job, load_job
    from nexusrag.services.retrieval import validate_semantic_index
    from nexusrag.services.vector_backend import QdrantBackend

    initialize_database(config.JOB_DATABASE_PATH)
    for name, directory in [
        ("STORAGE_DIRECTORY", "documents"),
        ("EXTRACTION_DIRECTORY", "extracted"),
        ("EMBEDDING_DIRECTORY", "embeddings"),
    ]:
        monkeypatch.setattr(config, name, tmp_path / directory)
    config.STORAGE_DIRECTORY.mkdir()
    pages = [
        ExtractedPage(page_number=1, text="Annual leave is twenty-five days."),
        ExtractedPage(page_number=2, text=" \n\t"),
        ExtractedPage(page_number=4, text="中文 café 🚀, punctuation!"),
    ]
    now = datetime.now(UTC)
    job = ProcessingJob(
        job_id=uuid4(),
        document_id=uuid4(),
        status="queued",
        created_at=now,
        updated_at=now,
    )
    (config.STORAGE_DIRECTORY / f"{job.document_id}.pdf").write_bytes(b"%PDF-saved")
    asyncio.run(create_job(job, config.JOB_DATABASE_PATH))
    monkeypatch.setattr(tasks, "extract_pages", Mock(return_value=pages))

    model = Mock(max_seq_length=8)
    model.get_sentence_embedding_dimension.return_value = 3

    def tokenize(texts, **kwargs):
        assert kwargs == {"add_special_tokens": True, "truncation": False}
        if isinstance(texts, str):
            return {"input_ids": [1] * (len(texts) + 2)}
        return {"input_ids": [[1] * (len(text) + 2) for text in texts]}

    def encode(texts, **kwargs):
        assert all(len(text) + 2 <= model.max_seq_length for text in texts)
        return SimpleNamespace(tolist=lambda: [[1.0, 0.0, 0.0] for text in texts])

    model.tokenizer.side_effect = tokenize
    model.encode_document.side_effect = encode
    embedder = MiniEmbedder(model, model_id="test", model_revision="v1")
    loader = Mock(return_value=embedder)
    monkeypatch.setattr(tasks, "EmbeddingRuntime", lambda: EmbeddingRuntime(loader))
    client = AsyncQdrantClient(":memory:")
    backend = QdrantBackend(client, collection="tokenized-worker")
    monkeypatch.setattr(tasks, "QdrantBackend", lambda: backend)
    try:
        tasks.process_document(str(job.job_id))
        assert (
            asyncio.run(load_job(job.job_id, config.JOB_DATABASE_PATH)).status
            == "succeeded"
        )
        chunks = asyncio.run(
            storage.load_chunks(job.document_id, config.CHUNK_DIRECTORY)
        )
        embeddings = asyncio.run(
            storage.load_embeddings(job.document_id, config.EMBEDDING_DIRECTORY)
        )
        validate_semantic_index(chunks, embeddings)
        assert len(chunks.chunks) > len(pages)
        source = {page.page_number: page.text for page in pages}
        assert {chunk.page_number for chunk in chunks.chunks} == {1, 4}
        for chunk in chunks.chunks:
            assert len(chunk.text) + 2 <= model.max_seq_length
            assert (
                chunk.text
                == source[chunk.page_number][chunk.start_char : chunk.end_char]
            )
        assert asyncio.run(client.count("tokenized-worker", exact=True)).count == len(
            chunks.chunks
        )
        loader.assert_called_once_with()
    finally:
        asyncio.run(client.close())
