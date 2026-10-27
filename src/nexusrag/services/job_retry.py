"""Explicit retry submission; RQ automatic retries remain disabled."""

import sqlite3
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.concurrency import run_in_threadpool

from nexusrag import config
from nexusrag.schemas.job import ProcessingJob
from nexusrag.services.document_state import deletion_status
from nexusrag.services.job_storage import _document_jobs, _latest_attempt, _load_job
from nexusrag.services.processing_lock import document_lock
from nexusrag.services.queue import enqueue_processing


class RetryLimitReachedError(ValueError):
    """The processing chain has exhausted its total attempts."""


class RetryNotAllowedError(ValueError):
    """The job is not a retryable failed current attempt."""


class RetryAlreadyExistsError(ValueError):
    def __init__(self, job_id: UUID):
        super().__init__("A later attempt already exists; inspect it instead")
        self.job_id = job_id


class RetrySubmissionError(RuntimeError):
    def __init__(self, job_id: UUID):
        super().__init__("Retry saved but queue acceptance could not be confirmed")
        self.job_id = job_id


async def retry_failed_job(
    job_id: UUID,
    database_path: Path,
    max_attempts: int,
    *,
    enqueue: Callable[[UUID], Awaitable[None]] | None = None,
) -> ProcessingJob:
    """Persist one new attempt, then enqueue it; retain it on ambiguous failure.

    Only confirmed interruptions are retryable for now. Generic processing
    failures may be malformed input and require diagnosis instead. The PDF and
    database must be shared with workers. Duplicate requests never resubmit an
    existing retry; queued-delivery reconciliation is a separate operation.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    retry = await run_in_threadpool(_create_retry, job_id, database_path, max_attempts)
    try:
        await (enqueue if enqueue is not None else enqueue_processing)(retry.job_id)
    except Exception as error:
        raise RetrySubmissionError(retry.job_id) from error
    return retry


def _create_retry(
    job_id: UUID, database_path: Path, max_attempts: int
) -> ProcessingJob:
    snapshot = _load_job(job_id, database_path)
    with document_lock(snapshot.document_id, database_path):
        connection = sqlite3.connect(database_path)
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                if deletion_status(connection, snapshot.document_id) is not None:
                    raise RetryNotAllowedError("Document has been deleted")
                jobs = _document_jobs(connection, snapshot.document_id)
                latest = _latest_attempt(jobs)
                current = next((job for job in jobs if job.job_id == job_id), None)
                if current is None:
                    raise FileNotFoundError("Job not found")
                if latest.job_id != job_id:
                    raise RetryAlreadyExistsError(latest.job_id)
                if current.status != "failed":
                    raise RetryNotAllowedError("Only failed jobs can be retried")
                if current.attempt_number >= max_attempts:
                    raise RetryLimitReachedError("Maximum processing attempts reached")
                if current.error != "Document processing was interrupted":
                    raise RetryNotAllowedError("Failure is not classified as retryable")
                if not (
                    config.STORAGE_DIRECTORY / f"{current.document_id}.pdf"
                ).is_file():
                    raise FileNotFoundError("Source document is unavailable")
                now = datetime.now(UTC)
                retry_job = ProcessingJob(
                    job_id=uuid4(),
                    document_id=current.document_id,
                    status="queued",
                    created_at=now,
                    updated_at=now,
                    root_job_id=current.root_job_id,
                    previous_job_id=current.job_id,
                    attempt_number=current.attempt_number + 1,
                )
                connection.execute(
                    "INSERT INTO processing_jobs (job_id, record) VALUES (?, ?)",
                    (str(retry_job.job_id), retry_job.model_dump_json()),
                )
            return retry_job
        finally:
            connection.close()


async def load_retry_candidates(
    database_path: Path, max_attempts: int
) -> list[ProcessingJob]:
    """Return eligible latest failed attempts, including failures from hooks.

    This is only a snapshot; retry_failed_job repeats eligibility checks under
    the document lock and write transaction before creating an attempt.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    return await run_in_threadpool(_load_retry_candidates, database_path, max_attempts)


def _load_retry_candidates(
    database_path: Path, max_attempts: int
) -> list[ProcessingJob]:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT job_id, record FROM processing_jobs WHERE NOT EXISTS ("
            "SELECT 1 FROM document_deletions WHERE document_id = "
            "json_extract(processing_jobs.record, '$.document_id'))"
        ).fetchall()
        documents: dict[UUID, list[ProcessingJob]] = {}
        for stored_id, record in rows:
            job = ProcessingJob.model_validate_json(record)
            if str(job.job_id) != stored_id:
                raise ValueError("Stored job identity mismatch")
            documents.setdefault(job.document_id, []).append(job)
        candidates = []
        for jobs in documents.values():
            latest = _latest_attempt(jobs)
            if (
                latest.status == "failed"
                and latest.error == "Document processing was interrupted"
                and latest.attempt_number < max_attempts
            ):
                candidates.append(latest)
        return candidates
    finally:
        connection.close()
