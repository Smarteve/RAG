from datetime import UTC, datetime

from nexusrag.schemas.job import JobStatus, ProcessingJob

ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    "queued": {"running"},
    "running": {"succeeded", "failed"},
    "succeeded": set(),
    "failed": set(),
}


def job_transition(
    current_job: ProcessingJob, new_status: JobStatus, error: str | None = None
) -> ProcessingJob:
    if new_status not in ALLOWED_TRANSITIONS[current_job.status]:
        raise ValueError(
            f"Invalid job transition: {current_job.status} -> {new_status}"
        )

    new_job = ProcessingJob(
        job_id=current_job.job_id,
        document_id=current_job.document_id,
        status=new_status,
        created_at=current_job.created_at,
        updated_at=datetime.now(UTC),
        error=error,
    )

    return new_job
