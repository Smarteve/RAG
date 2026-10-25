"""Reconcile queued SQLite attempts with RQ, preserving IDs and attempt counts."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue
from rq.exceptions import DuplicateJobError, InvalidJobOperation, NoSuchJobError
from rq.job import Job, JobStatus
from rq.serializers import JSONSerializer

from nexusrag import config
from nexusrag.schemas.job import ProcessingJob
from nexusrag.schemas.recovery import DeliveryDecision
from nexusrag.services.job_storage import _load_document_job
from nexusrag.services.processing_lock import DocumentBusyError, document_lock


async def reconcile_queued_jobs(
    database_path: Path,
    redis_url: str,
    *,
    apply: bool = False,
    grace_period: timedelta = timedelta(seconds=60),
    queue_name: str | None = None,
) -> list[DeliveryDecision]:
    """Redeliver only old, unchanged queued attempts whose RQ record is absent.

    Dry runs do not acquire locks or write. Apply mode holds the same document
    lock used by workers through rechecking and atomic unique enqueue. Missing
    records with residual queue entries and existing terminal RQ records remain
    uncertain. Database corruption propagates; per-job failures are reported.
    """
    if grace_period < timedelta(0):
        raise ValueError("grace_period must not be negative")
    return await run_in_threadpool(
        _reconcile_queued_jobs,
        database_path,
        redis_url,
        apply,
        grace_period,
        queue_name if queue_name is not None else config.INGESTION_QUEUE_NAME,
    )


def _queued_snapshots(database_path: Path) -> list[ProcessingJob]:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT job_id, record FROM processing_jobs"
        ).fetchall()
        jobs = []
        for stored_id, record in rows:
            job = ProcessingJob.model_validate_json(record)
            if str(job.job_id) != stored_id:
                raise ValueError("Stored job identity mismatch")
            if job.status == "queued":
                jobs.append(job)
        return jobs
    finally:
        connection.close()


def _decision(job: ProcessingJob, action: str, reason: str) -> DeliveryDecision:
    return DeliveryDecision(job_id=job.job_id, action=action, reason=reason)


def _rq_observation(job: ProcessingJob, queue: Queue) -> DeliveryDecision:
    try:
        rq_job = Job.fetch(
            str(job.job_id), connection=queue.connection, serializer=JSONSerializer
        )
    except NoSuchJobError:
        # Older producers or external deletion may leave an ID without its hash.
        if queue.get_job_position(str(job.job_id)) is not None:
            return _decision(
                job, "uncertain", "RQ queue contains an ID with no job record"
            )
        return _decision(job, "would_enqueue", "RQ record is absent")
    if (
        rq_job.origin != queue.name
        or rq_job.func_name
        not in {"nexusrag.tasks.process_document", "nexusrag.worker.process_document"}
        or tuple(rq_job.args) != (str(job.job_id),)
        or rq_job.kwargs
    ):
        return _decision(
            job, "uncertain", "Existing RQ record does not match the processing task"
        )
    status = rq_job.get_status(refresh=True)
    executions = rq_job.get_executions()
    if rq_job.get_status(refresh=True) != status:
        return _decision(
            job, "uncertain", "RQ status changed during delivery inspection"
        )
    if status == JobStatus.QUEUED:
        if executions or queue.get_job_position(str(job.job_id)) is None:
            return _decision(
                job, "uncertain", "RQ queued record has inconsistent delivery state"
            )
        return _decision(job, "present", "Job is already queued in RQ")
    if status == JobStatus.SCHEDULED and not executions:
        return _decision(job, "present", "Job is already scheduled in RQ")
    if status == JobStatus.STARTED and executions:
        return _decision(
            job, "present", "RQ reports an existing execution; not resubmitting"
        )
    return _decision(job, "uncertain", "RQ state disagrees with queued SQLite status")


def _check_and_deliver(
    job: ProcessingJob, database_path: Path, queue: Queue, apply: bool
) -> DeliveryDecision:
    if _load_document_job(job.document_id, database_path) != job:
        return _decision(
            job, "skipped", "SQLite job snapshot or current attempt changed"
        )
    if not (config.STORAGE_DIRECTORY / f"{job.document_id}.pdf").is_file():
        return _decision(job, "uncertain", "Source document is unavailable")
    observation = _rq_observation(job, queue)
    if observation.action != "would_enqueue" or not apply:
        return observation
    # Recheck both systems under document ownership. RQ's unique script closes
    # the remaining race with the initial HTTP producer or another submitter.
    if _load_document_job(job.document_id, database_path) != job:
        return _decision(job, "skipped", "SQLite job snapshot changed before delivery")
    observation = _rq_observation(job, queue)
    if observation.action != "would_enqueue":
        return observation
    try:
        queue.enqueue_call(
            "nexusrag.tasks.process_document",
            args=(str(job.job_id),),
            job_id=str(job.job_id),
            unique=True,
        )
    except DuplicateJobError:
        # Don't overwrite the winner; classify its record afresh.
        return _rq_observation(job, queue)
    except RedisError:
        return _decision(
            job,
            "delivery_unconfirmed",
            "Queue acceptance could not be confirmed; queued attempt retained",
        )
    return _decision(
        job, "enqueued", "Existing queued attempt submitted with its original job ID"
    )


def _reconcile_queued_jobs(
    database_path: Path,
    redis_url: str,
    apply: bool,
    grace_period: timedelta,
    queue_name: str,
) -> list[DeliveryDecision]:
    jobs = _queued_snapshots(database_path)
    if not jobs:
        return []
    connection = Redis.from_url(redis_url, socket_connect_timeout=5, socket_timeout=5)
    queue = Queue(queue_name, connection=connection, serializer=JSONSerializer)
    decisions = []
    try:
        for job in jobs:
            age = datetime.now(UTC) - job.updated_at
            if age < grace_period:
                decisions.append(
                    _decision(
                        job,
                        "skipped",
                        "Queued attempt is within the delivery grace period",
                    )
                )
                continue
            try:
                if apply:
                    with document_lock(job.document_id, database_path):
                        decision = _check_and_deliver(job, database_path, queue, True)
                else:
                    decision = _check_and_deliver(job, database_path, queue, False)
            except DocumentBusyError:
                decision = _decision(
                    job, "skipped", "Document is owned by processing or recovery"
                )
            except RedisError:
                decision = _decision(job, "uncertain", "RQ backend is unavailable")
            except InvalidJobOperation, sqlite3.Error, OSError, ValueError, TypeError:
                decision = _decision(
                    job, "uncertain", "Job records changed or could not be verified"
                )
            decisions.append(decision)
        return decisions
    finally:
        connection.close()
