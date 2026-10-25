import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from nexusrag import config
from nexusrag.database import initialize_database
from nexusrag.schemas.job import ProcessingJob
from nexusrag.services.job_retry import (
    RetryAlreadyExistsError,
    RetryLimitReachedError,
    RetryNotAllowedError,
    RetrySubmissionError,
    retry_failed_job,
)
from nexusrag.services.job_storage import (
    create_job,
    load_document_job,
    load_job,
    transition_job,
)
from nexusrag.services.processing_lock import DocumentBusyError, document_lock


@pytest.fixture
def failed(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    initialize_database(path)
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", tmp_path / "documents")
    config.STORAGE_DIRECTORY.mkdir()
    now = datetime.now(UTC)
    job = ProcessingJob(
        job_id=uuid4(),
        document_id=uuid4(),
        status="queued",
        created_at=now,
        updated_at=now,
    )
    (config.STORAGE_DIRECTORY / f"{job.document_id}.pdf").write_bytes(b"%PDF-saved")
    asyncio.run(create_job(job, path))
    asyncio.run(transition_job(job.job_id, "running", path))
    job = asyncio.run(
        transition_job(
            job.job_id, "failed", path, "Document processing was interrupted"
        )
    )
    return path, job


def test_retry_persisted_before_enqueue_and_selected_as_current(failed):
    path, original = failed

    async def enqueue(job_id):
        saved = await load_job(job_id, path)
        assert saved.status == "queued"

    submit = AsyncMock(side_effect=enqueue)
    result = asyncio.run(retry_failed_job(original.job_id, path, 3, enqueue=submit))
    assert result.job_id != original.job_id
    assert result.document_id == original.document_id
    assert result.root_job_id == original.root_job_id
    assert result.previous_job_id == original.job_id
    assert result.attempt_number == 2
    assert asyncio.run(load_job(original.job_id, path)) == original
    assert asyncio.run(load_document_job(original.document_id, path)) == result
    submit.assert_awaited_once_with(result.job_id)


def test_ambiguous_enqueue_retains_retry_and_prevents_duplicate(failed):
    path, job = failed
    submit = AsyncMock(side_effect=ConnectionError("secret URL"))
    with pytest.raises(RetrySubmissionError) as caught:
        asyncio.run(retry_failed_job(job.job_id, path, 3, enqueue=submit))
    saved = asyncio.run(load_job(caught.value.job_id, path))
    assert saved.status == "queued"
    assert "secret" not in str(caught.value)
    with pytest.raises(RetryAlreadyExistsError) as duplicate:
        asyncio.run(retry_failed_job(job.job_id, path, 3, enqueue=submit))
    assert duplicate.value.job_id == saved.job_id
    assert submit.await_count == 1


def test_limit_applies_to_entire_chain(failed):
    path, job = failed
    submit = AsyncMock()
    retry = asyncio.run(retry_failed_job(job.job_id, path, 2, enqueue=submit))
    asyncio.run(transition_job(retry.job_id, "running", path))
    asyncio.run(
        transition_job(
            retry.job_id, "failed", path, "Document processing was interrupted"
        )
    )
    with pytest.raises(RetryLimitReachedError):
        asyncio.run(retry_failed_job(retry.job_id, path, 2, enqueue=submit))
    assert submit.await_count == 1


@pytest.mark.parametrize("status", ["queued", "running", "succeeded"])
def test_nonfailed_current_attempt_rejected(failed, status):
    path, original = failed
    retry = asyncio.run(retry_failed_job(original.job_id, path, 3, enqueue=AsyncMock()))
    if status != "queued":
        asyncio.run(transition_job(retry.job_id, "running", path))
    if status == "succeeded":
        asyncio.run(transition_job(retry.job_id, "succeeded", path))
    submit = AsyncMock()
    with pytest.raises(RetryNotAllowedError):
        asyncio.run(retry_failed_job(retry.job_id, path, 3, enqueue=submit))
    submit.assert_not_awaited()


def test_generic_failure_is_not_automatically_retryable(failed):
    path, original = failed
    retry = asyncio.run(retry_failed_job(original.job_id, path, 3, enqueue=AsyncMock()))
    asyncio.run(transition_job(retry.job_id, "running", path))
    asyncio.run(
        transition_job(retry.job_id, "failed", path, "Document processing failed")
    )
    with pytest.raises(RetryNotAllowedError, match="classified"):
        asyncio.run(retry_failed_job(retry.job_id, path, 3, enqueue=AsyncMock()))


def test_live_document_lock_prevents_retry(failed):
    path, job = failed
    submit = AsyncMock()
    with document_lock(job.document_id, path), pytest.raises(DocumentBusyError):
        asyncio.run(retry_failed_job(job.job_id, path, 3, enqueue=submit))
    submit.assert_not_awaited()


def test_concurrent_retry_requests_create_only_one_attempt(failed):
    path, job = failed
    submit = AsyncMock()

    def retry(_):
        try:
            return asyncio.run(retry_failed_job(job.job_id, path, 3, enqueue=submit))
        except DocumentBusyError, RetryAlreadyExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(retry, range(2)))
    assert sum(result is not None for result in results) == 1
    assert submit.await_count == 1


def test_sqlite_enforces_unique_retry_parent(failed):
    path, job = failed
    retry = asyncio.run(retry_failed_job(job.job_id, path, 3, enqueue=AsyncMock()))
    duplicate = ProcessingJob.model_validate(retry.model_dump() | {"job_id": uuid4()})
    with pytest.raises(sqlite3.IntegrityError):
        asyncio.run(create_job(duplicate, path))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_attempt_limit(failed, limit):
    path, job = failed
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(retry_failed_job(job.job_id, path, limit, enqueue=AsyncMock()))


def test_missing_source_prevents_retry(failed):
    path, job = failed
    (config.STORAGE_DIRECTORY / f"{job.document_id}.pdf").unlink()
    with pytest.raises(FileNotFoundError, match="Source document"):
        asyncio.run(retry_failed_job(job.job_id, path, 3, enqueue=AsyncMock()))


def test_database_prevents_another_active_attempt_for_document(failed):
    path, job = failed
    asyncio.run(retry_failed_job(job.job_id, path, 3, enqueue=AsyncMock()))
    now = datetime.now(UTC)
    other = ProcessingJob(
        job_id=uuid4(),
        document_id=job.document_id,
        status="queued",
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(sqlite3.IntegrityError):
        asyncio.run(create_job(other, path))


def test_unknown_retry_job(failed):
    path, _ = failed
    with pytest.raises(FileNotFoundError):
        asyncio.run(retry_failed_job(uuid4(), path, 3, enqueue=AsyncMock()))


def test_candidate_lookup_includes_hook_failure_and_excludes_existing_child(failed):
    from nexusrag.services.job_retry import load_retry_candidates

    path, job = failed
    assert asyncio.run(load_retry_candidates(path, 3)) == [job]
    assert asyncio.run(load_retry_candidates(path, 1)) == []
    retry = asyncio.run(retry_failed_job(job.job_id, path, 3, enqueue=AsyncMock()))
    assert asyncio.run(load_retry_candidates(path, 3)) == []
    asyncio.run(transition_job(retry.job_id, "running", path))
    asyncio.run(
        transition_job(retry.job_id, "failed", path, "Document processing failed")
    )
    assert asyncio.run(load_retry_candidates(path, 3)) == []
