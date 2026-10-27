import asyncio
import sqlite3
from datetime import UTC, datetime
from importlib import import_module
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from nexusrag import config
from nexusrag.database import initialize_database
from nexusrag.dependencies import get_vector_backend
from nexusrag.schemas.job import ProcessingJob
from nexusrag.services import document_deletion
from nexusrag.services.document_state import is_document_deleted
from nexusrag.services.job_retry import (
    RetryNotAllowedError,
    load_retry_candidates,
    retry_failed_job,
)
from nexusrag.services.job_storage import create_job, load_job, transition_job
from nexusrag.services.processing_lock import DocumentBusyError, document_lock

main = import_module("nexusrag.main")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for name, directory in [
        ("STORAGE_DIRECTORY", "documents"),
        ("EXTRACTION_DIRECTORY", "extracted"),
        ("CHUNK_DIRECTORY", "chunks"),
        ("EMBEDDING_DIRECTORY", "embeddings"),
        ("EMBEDDING_MODEL_CACHE_DIRECTORY", "models"),
    ]:
        path = tmp_path / directory
        path.mkdir(exist_ok=True)
        monkeypatch.setattr(config, name, path)
    backend = AsyncMock()
    previous = main.app.dependency_overrides.get(get_vector_backend)
    main.app.dependency_overrides[get_vector_backend] = lambda: backend
    try:
        with TestClient(main.app) as client:
            yield client, backend
    finally:
        if previous is None:
            main.app.dependency_overrides.pop(get_vector_backend, None)
        else:
            main.app.dependency_overrides[get_vector_backend] = previous


def artifacts(document_id):
    return [
        config.STORAGE_DIRECTORY / f"{document_id}.pdf",
        config.EXTRACTION_DIRECTORY / f"{document_id}.json",
        config.CHUNK_DIRECTORY / f"{document_id}.json",
        config.EMBEDDING_DIRECTORY / f"{document_id}.json",
    ]


def save_artifacts(document_id):
    paths = artifacts(document_id)
    for path in paths:
        path.write_bytes(b"document content")
    return paths


def save_job(document_id, status="succeeded"):
    now = datetime.now(UTC)
    job = ProcessingJob(
        job_id=uuid4(),
        document_id=document_id,
        status="queued",
        created_at=now,
        updated_at=now,
    )
    asyncio.run(create_job(job, config.JOB_DATABASE_PATH))
    if status != "queued":
        job = asyncio.run(
            transition_job(job.job_id, "running", config.JOB_DATABASE_PATH)
        )
    if status in {"succeeded", "failed"}:
        job = asyncio.run(
            transition_job(
                job.job_id,
                status,
                config.JOB_DATABASE_PATH,
                "Document processing was interrupted" if status == "failed" else None,
            )
        )
    return job


def test_delete_removes_only_selected_document_and_retains_history_and_shared_files(
    setup,
):
    client, backend = setup
    document_id, other_id = uuid4(), uuid4()
    paths = save_artifacts(document_id)
    others = save_artifacts(other_id)
    temporary_files = []
    for path in paths:
        temporary = path.parent / f".{document_id}.interrupted.tmp"
        temporary.write_bytes(b"partial document")
        temporary_files.append(temporary)
    other_temporary = others[0].parent / f".{other_id}.interrupted.tmp"
    other_temporary.write_bytes(b"another document")
    job = save_job(document_id)
    model = config.EMBEDDING_MODEL_CACHE_DIRECTORY / "weights"
    model.write_bytes(b"shared model")

    async def delete_vectors(identifier):
        assert identifier == document_id
        assert await is_document_deleted(identifier, config.JOB_DATABASE_PATH)
        assert all(path.exists() for path in paths)
        with (
            pytest.raises(DocumentBusyError),
            document_lock(identifier, config.JOB_DATABASE_PATH),
        ):
            pass

    backend.delete_document.side_effect = delete_vectors
    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 204
    assert response.content == b""
    assert not any(path.exists() for path in paths)
    assert not any(path.exists() for path in temporary_files)
    assert all(path.read_bytes() == b"document content" for path in others)
    assert other_temporary.read_bytes() == b"another document"
    assert model.read_bytes() == b"shared model"
    assert (
        config.JOB_DATABASE_PATH.parent / ".job-locks" / f"{document_id}.lock"
    ).exists()
    assert asyncio.run(load_job(job.job_id, config.JOB_DATABASE_PATH)) == job
    assert client.get(f"/jobs/{job.job_id}").status_code == 200
    backend.delete_document.assert_awaited_once_with(document_id)
    with document_lock(document_id, config.JOB_DATABASE_PATH):
        pass


def test_repeated_delete_and_unknown_ids_are_idempotent(setup):
    client, backend = setup
    document_id = uuid4()
    for _ in range(2):
        assert client.delete(f"/documents/{document_id}").status_code == 204
    backend.delete_document.assert_awaited_once_with(document_id)


@pytest.mark.parametrize("status", ["queued", "running"])
def test_active_jobs_reject_delete_without_hiding_or_removing_anything(setup, status):
    client, backend = setup
    document_id = uuid4()
    paths = save_artifacts(document_id)
    save_job(document_id, status)

    assert client.delete(f"/documents/{document_id}").status_code == 409
    assert not asyncio.run(is_document_deleted(document_id, config.JOB_DATABASE_PATH))
    assert all(path.exists() for path in paths)
    backend.delete_document.assert_not_awaited()


def test_live_processing_or_reindex_lock_rejects_delete(setup):
    client, backend = setup
    document_id = uuid4()
    paths = save_artifacts(document_id)
    with document_lock(document_id, config.JOB_DATABASE_PATH):
        assert client.delete(f"/documents/{document_id}").status_code == 409
    assert not asyncio.run(is_document_deleted(document_id, config.JOB_DATABASE_PATH))
    assert all(path.exists() for path in paths)
    backend.delete_document.assert_not_awaited()


@pytest.mark.parametrize("partial", [False, True])
def test_incomplete_deletion_hides_every_document_endpoint_and_can_resume(
    setup, monkeypatch, partial
):
    client, backend = setup
    document_id = uuid4()
    paths = save_artifacts(document_id)
    job = save_job(document_id, "failed")
    original_remove = document_deletion._remove_files
    if partial:

        def fail_remove(paths):
            paths[0].unlink()
            raise PermissionError("private filesystem details")

        monkeypatch.setattr(document_deletion, "_remove_files", fail_remove)
    else:
        backend.delete_document.side_effect = ConnectionError("private Qdrant URL")

    response = client.delete(f"/documents/{document_id}")
    assert response.status_code == (500 if partial else 503)
    assert "private" not in response.text
    assert all(path.exists() for path in paths[1:])
    for route in [
        "pages/1",
        "chunks",
        "search?query=leave",
        "semantic-search?query=leave",
    ]:
        result = client.get(f"/documents/{document_id}/{route}")
        assert result.status_code == 404
        assert result.json() == {"detail": "Document not found"}
    assert (
        client.post(
            f"/documents/{document_id}/ask", json={"question": "Leave?"}
        ).status_code
        == 404
    )
    assert client.post(f"/documents/{document_id}/semantic-index").status_code == 404
    assert asyncio.run(load_retry_candidates(config.JOB_DATABASE_PATH, 3)) == []
    enqueue = AsyncMock()
    with pytest.raises(RetryNotAllowedError, match="deleted"):
        asyncio.run(
            retry_failed_job(job.job_id, config.JOB_DATABASE_PATH, 3, enqueue=enqueue)
        )
    enqueue.assert_not_awaited()

    backend.delete_document.side_effect = None
    monkeypatch.setattr(document_deletion, "_remove_files", original_remove)
    assert client.delete(f"/documents/{document_id}").status_code == 204
    assert not any(path.exists() for path in paths)
    assert backend.delete_document.await_count == 2
    assert client.get(f"/jobs/{job.job_id}").json()["status"] == "failed"


def test_completion_write_failure_can_resume_without_restoring_document(
    setup, monkeypatch
):
    client, backend = setup
    document_id = uuid4()
    paths = save_artifacts(document_id)
    complete = document_deletion._complete_deletion

    def fail_complete(*args):
        raise sqlite3.OperationalError("private DB error")

    monkeypatch.setattr(document_deletion, "_complete_deletion", fail_complete)
    assert client.delete(f"/documents/{document_id}").status_code == 500
    assert not any(path.exists() for path in paths)
    assert asyncio.run(is_document_deleted(document_id, config.JOB_DATABASE_PATH))
    monkeypatch.setattr(document_deletion, "_complete_deletion", complete)
    assert client.delete(f"/documents/{document_id}").status_code == 204
    assert backend.delete_document.await_count == 2


def test_deleted_id_cannot_acquire_a_new_processing_job(setup):
    client, _ = setup
    document_id = uuid4()
    assert client.delete(f"/documents/{document_id}").status_code == 204
    with pytest.raises(ValueError, match="deleted"):
        save_job(document_id, "queued")


def test_job_claim_checks_deletion_even_for_a_stale_queued_record(setup):
    client, _ = setup
    document_id = uuid4()
    job = save_job(document_id)
    assert client.delete(f"/documents/{document_id}").status_code == 204
    # Model a stale external write/restore without allowing the worker to publish.
    stale = job.model_copy(update={"status": "queued"})
    with sqlite3.connect(config.JOB_DATABASE_PATH) as connection:
        connection.execute(
            "UPDATE processing_jobs SET record = ? WHERE job_id = ?",
            (stale.model_dump_json(), str(job.job_id)),
        )
    with pytest.raises(ValueError, match="deleted"):
        asyncio.run(transition_job(job.job_id, "running", config.JOB_DATABASE_PATH))


def test_unavailable_database_fails_before_vector_or_file_cleanup(setup):
    client, backend = setup
    document_id = uuid4()
    paths = save_artifacts(document_id)
    with sqlite3.connect(config.JOB_DATABASE_PATH) as connection:
        connection.execute("DROP TABLE document_deletions")
    assert client.delete(f"/documents/{document_id}").status_code == 500
    assert client.get(f"/documents/{document_id}/chunks").status_code == 500
    assert all(path.exists() for path in paths)
    backend.delete_document.assert_not_awaited()


def test_invalid_id_does_not_call_backend(setup):
    client, backend = setup
    assert client.delete("/documents/not-a-uuid").status_code == 422
    backend.delete_document.assert_not_awaited()


def test_startup_adds_deletion_table_to_existing_job_database(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE processing_jobs (job_id TEXT PRIMARY KEY NOT NULL, record TEXT NOT NULL)"
        )
    initialize_database(path)
    initialize_database(path)
    assert not asyncio.run(is_document_deleted(uuid4(), path))
