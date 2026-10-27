import sqlite3
from itertools import pairwise
from pathlib import Path
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from nexusrag.schemas.job import JobStatus, ProcessingJob
from nexusrag.services.document_state import deletion_status
from nexusrag.services.jobs import job_transition


async def create_job(
    job: ProcessingJob,
    database_path: Path,
) -> None:
    await run_in_threadpool(_create_job, job=job, database_path=database_path)


async def load_job(
    job_id: UUID,
    database_path: Path,
) -> ProcessingJob:
    return await run_in_threadpool(
        _load_job, job_id=job_id, database_path=database_path
    )


async def transition_job(
    job_id: UUID,
    new_status: JobStatus,
    database_path: Path,
    error: str | None = None,
) -> ProcessingJob:
    return await run_in_threadpool(
        _transition_job, job_id, new_status, database_path, error
    )


def _create_job(
    job: ProcessingJob,
    database_path: Path,
) -> None:

    if job.status != "queued":
        raise ValueError("New jobs must be queued")

    connection = sqlite3.connect(database_path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if deletion_status(connection, job.document_id) is not None:
                raise ValueError("Cannot create a job for a deleted document")
            connection.execute(
                "INSERT INTO processing_jobs (job_id, record) VALUES (?, ?)",
                (str(job.job_id), job.model_dump_json()),
            )
    finally:
        connection.close()


def _load_job(
    job_id: UUID,
    database_path: Path,
) -> ProcessingJob:

    connection = sqlite3.connect(database_path)

    try:
        row = connection.execute(
            "SELECT record FROM processing_jobs WHERE job_id = ?",
            (str(job_id),),
        ).fetchone()

        if row is None:
            raise FileNotFoundError("Job not found")

        job = ProcessingJob.model_validate_json(row[0])
        if job.job_id != job_id:
            raise ValueError("Stored job ID does not match requested job")

        return job

    finally:
        connection.close()


def _transition_job(
    job_id: UUID,
    new_status: JobStatus,
    database_path: Path,
    error: str | None = None,
) -> ProcessingJob:
    connection = sqlite3.connect(database_path)
    try:
        with connection:
            # Acquire the write lock before reading the current status.
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record FROM processing_jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()

            if row is None:
                raise FileNotFoundError("Job not found")

            current_job = ProcessingJob.model_validate_json(row[0])
            if current_job.job_id != job_id:
                raise ValueError("Stored job ID does not match requested job")
            if (
                new_status == "running"
                and deletion_status(connection, current_job.document_id) is not None
            ):
                raise ValueError("Cannot process a deleted document")

            updated_job = job_transition(current_job, new_status, error)
            connection.execute(
                "UPDATE processing_jobs SET record = ? WHERE job_id = ?",
                (updated_job.model_dump_json(), str(job_id)),
            )

        return updated_job
    finally:
        connection.close()


async def load_document_job(
    document_id: UUID, database_path: Path
) -> ProcessingJob | None:
    """Find a document's processing job; legacy documents may have none."""
    return await run_in_threadpool(_load_document_job, document_id, database_path)


def _document_jobs(
    connection: sqlite3.Connection, document_id: UUID
) -> list[ProcessingJob]:
    rows = connection.execute(
        "SELECT job_id, record FROM processing_jobs "
        "WHERE json_extract(record, '$.document_id') = ?",
        (str(document_id),),
    ).fetchall()
    jobs = []
    for stored_id, record in rows:
        job = ProcessingJob.model_validate_json(record)
        if str(job.job_id) != stored_id or job.document_id != document_id:
            raise ValueError("Stored job identity mismatch")
        jobs.append(job)
    return jobs


def _latest_attempt(jobs: list[ProcessingJob]) -> ProcessingJob | None:
    if not jobs:
        return None
    ordered = sorted(jobs, key=lambda job: job.attempt_number)
    root = ordered[0]
    if root.attempt_number != 1 or any(
        job.root_job_id != root.job_id for job in ordered
    ):
        raise ValueError("Invalid or ambiguous retry chain")
    for previous, current in pairwise(ordered):
        if (
            current.attempt_number != previous.attempt_number + 1
            or current.previous_job_id != previous.job_id
            or previous.status != "failed"
        ):
            raise ValueError("Invalid or ambiguous retry chain")
    return ordered[-1]


def _load_document_job(document_id: UUID, database_path: Path) -> ProcessingJob | None:
    connection = sqlite3.connect(database_path)
    try:
        return _latest_attempt(_document_jobs(connection, document_id))
    finally:
        connection.close()


async def fail_running_job(job_id: UUID, database_path: Path, error: str) -> bool:
    """Fail a running job atomically; preserve queued and terminal records."""
    return await run_in_threadpool(_fail_running_job, job_id, database_path, error)


def _fail_running_job(
    job_id: UUID,
    database_path: Path,
    error: str,
    *,
    expected_job: ProcessingJob | None = None,
) -> bool:
    connection = sqlite3.connect(database_path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record FROM processing_jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise FileNotFoundError("Job not found")
            current = ProcessingJob.model_validate_json(row[0])
            if current.job_id != job_id:
                raise ValueError("Stored job ID does not match requested job")
            if expected_job is not None and current != expected_job:
                return False
            if current.status != "running":
                return False
            updated = job_transition(current, "failed", error)
            connection.execute(
                "UPDATE processing_jobs SET record = ? WHERE job_id = ?",
                (updated.model_dump_json(), str(job_id)),
            )
        return True
    finally:
        connection.close()
