import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag.database import initialize_database
from nexusrag.schemas.job import JobStatus, ProcessingJob
from nexusrag.services.job_storage import create_job, load_job, transition_job


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    path = tmp_path / "database" / "jobs.sqlite3"
    initialize_database(path)
    return path


@pytest.fixture
def job() -> ProcessingJob:
    now = datetime.now(UTC)
    return ProcessingJob(
        job_id=uuid4(),
        document_id=uuid4(),
        status="queued",
        created_at=now,
        updated_at=now,
    )


def test_create_and_load_job(database_path: Path, job: ProcessingJob) -> None:
    assert asyncio.run(create_job(job, database_path)) is None

    restored = asyncio.run(load_job(job.job_id, database_path))

    assert isinstance(restored, ProcessingJob)
    assert restored == job
    assert restored is not job


def test_duplicate_creation_preserves_original(
    database_path: Path, job: ProcessingJob
) -> None:
    asyncio.run(create_job(job, database_path))
    duplicate = ProcessingJob.model_validate(
        job.model_dump() | {"document_id": uuid4()}
    )

    with pytest.raises(sqlite3.IntegrityError):
        asyncio.run(create_job(duplicate, database_path))

    assert asyncio.run(load_job(job.job_id, database_path)) == job


def test_unknown_job(database_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Job not found"):
        asyncio.run(load_job(uuid4(), database_path))


@pytest.mark.parametrize("status", ["running", "succeeded", "failed"])
def test_creation_rejects_nonqueued_job(
    database_path: Path, job: ProcessingJob, status: JobStatus
) -> None:
    record = ProcessingJob.model_validate(
        job.model_dump()
        | {
            "status": status,
            "error": "Processing failed" if status == "failed" else None,
        }
    )

    with pytest.raises(ValueError, match="New jobs must be queued"):
        asyncio.run(create_job(record, database_path))

    with pytest.raises(FileNotFoundError):
        asyncio.run(load_job(job.job_id, database_path))


@pytest.mark.parametrize("record", ["not JSON", "{}"])
def test_load_rejects_corrupt_record(
    database_path: Path, job: ProcessingJob, record: str
) -> None:
    connection = sqlite3.connect(database_path)
    try:
        with connection:
            connection.execute(
                "INSERT INTO processing_jobs (job_id, record) VALUES (?, ?)",
                (str(job.job_id), record),
            )
    finally:
        connection.close()

    with pytest.raises(ValidationError):
        asyncio.run(load_job(job.job_id, database_path))


def test_load_rejects_mismatched_job_id(
    database_path: Path, job: ProcessingJob
) -> None:
    requested_id = uuid4()
    connection = sqlite3.connect(database_path)
    try:
        with connection:
            connection.execute(
                "INSERT INTO processing_jobs (job_id, record) VALUES (?, ?)",
                (str(requested_id), job.model_dump_json()),
            )
    finally:
        connection.close()

    with pytest.raises(ValueError, match="Stored job ID does not match requested job"):
        asyncio.run(load_job(requested_id, database_path))


@pytest.mark.parametrize("terminal_status", ["succeeded", "failed"])
def test_transitions_are_persisted(
    database_path: Path, job: ProcessingJob, terminal_status: JobStatus
) -> None:
    asyncio.run(create_job(job, database_path))
    running = asyncio.run(transition_job(job.job_id, "running", database_path))
    assert running.status == "running"
    assert asyncio.run(load_job(job.job_id, database_path)) == running

    error = "  Processing failed  " if terminal_status == "failed" else None
    terminal = asyncio.run(
        transition_job(job.job_id, terminal_status, database_path, error)
    )
    assert terminal.status == terminal_status
    assert terminal.error == ("Processing failed" if error else None)
    assert terminal.job_id == job.job_id
    assert terminal.document_id == job.document_id
    assert terminal.created_at == job.created_at
    assert terminal.updated_at >= running.updated_at >= job.updated_at
    assert asyncio.run(load_job(job.job_id, database_path)) == terminal


@pytest.mark.parametrize(
    ("status", "new_status"),
    [
        (status, new_status)
        for status in ("queued", "running", "succeeded", "failed")
        for new_status in ("queued", "running", "succeeded", "failed")
        if (status, new_status)
        not in {("queued", "running"), ("running", "succeeded"), ("running", "failed")}
    ],
)
def test_invalid_transition_preserves_stored_record(
    database_path: Path, job: ProcessingJob, status: JobStatus, new_status: JobStatus
) -> None:
    asyncio.run(create_job(job, database_path))
    if status != "queued":
        asyncio.run(transition_job(job.job_id, "running", database_path))
    if status in {"succeeded", "failed"}:
        asyncio.run(
            transition_job(
                job.job_id,
                status,
                database_path,
                "Processing failed" if status == "failed" else None,
            )
        )
    before = asyncio.run(load_job(job.job_id, database_path))

    with pytest.raises(ValueError, match="Invalid job transition"):
        asyncio.run(transition_job(job.job_id, new_status, database_path))

    assert asyncio.run(load_job(job.job_id, database_path)) == before


@pytest.mark.parametrize("error", [None, "", " \t\n"])
def test_invalid_failure_rolls_back(
    database_path: Path, job: ProcessingJob, error: str | None
) -> None:
    asyncio.run(create_job(job, database_path))
    running = asyncio.run(transition_job(job.job_id, "running", database_path))

    with pytest.raises(ValidationError, match="nonblank error"):
        asyncio.run(transition_job(job.job_id, "failed", database_path, error))

    assert asyncio.run(load_job(job.job_id, database_path)) == running
    # A subsequent successful write also verifies the failed operation released its lock.
    result = asyncio.run(transition_job(job.job_id, "succeeded", database_path))
    assert asyncio.run(load_job(job.job_id, database_path)) == result


def test_transition_unknown_job(database_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Job not found"):
        asyncio.run(transition_job(uuid4(), "running", database_path))


def test_competing_claims_only_one_succeeds(
    database_path: Path, job: ProcessingJob
) -> None:
    asyncio.run(create_job(job, database_path))

    async def claim() -> ProcessingJob:
        return await transition_job(job.job_id, "running", database_path)

    async def compete() -> list:
        return await asyncio.gather(claim(), claim(), return_exceptions=True)

    results = asyncio.run(compete())
    winners = [result for result in results if isinstance(result, ProcessingJob)]
    failures = [result for result in results if isinstance(result, ValueError)]
    assert len(winners) == 1
    assert len(failures) == 1
    assert "running -> running" in str(failures[0])
    assert asyncio.run(load_job(job.job_id, database_path)) == winners[0]
