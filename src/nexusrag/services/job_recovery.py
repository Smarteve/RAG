"""Guarded status recovery for failed RQ executions; does not retry work."""

import sqlite3
from pathlib import Path
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from nexusrag.schemas.job import ProcessingJob
from nexusrag.services.jobs import job_transition


async def fail_running_job(job_id: UUID, database_path: Path, error: str) -> bool:
    """Fail a running job atomically; preserve queued and terminal records."""
    return await run_in_threadpool(_fail_running_job, job_id, database_path, error)


def _fail_running_job(job_id: UUID, database_path: Path, error: str) -> bool:
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
