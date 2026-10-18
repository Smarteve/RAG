import sqlite3
from pathlib import Path
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from nexusrag.schemas.job import JobStatus, ProcessingJob
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

            updated_job = job_transition(current_job, new_status, error)
            connection.execute(
                "UPDATE processing_jobs SET record = ? WHERE job_id = ?",
                (updated_job.model_dump_json(), str(job_id)),
            )

        return updated_job
    finally:
        connection.close()
