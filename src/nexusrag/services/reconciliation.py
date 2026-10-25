"""Inspect RQ executions and optionally recover confirmed interrupted jobs."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from redis import Redis
from redis.exceptions import RedisError
from rq.exceptions import InvalidJobOperation, NoSuchJobError
from rq.job import Job, JobStatus
from rq.serializers import JSONSerializer

from nexusrag.schemas.job import ProcessingJob
from nexusrag.schemas.recovery import RecoveryDecision
from nexusrag.services.job_storage import _fail_running_job
from nexusrag.services.processing_lock import DocumentBusyError, document_lock


async def inspect_running_jobs(
    database_path: Path,
    redis_url: str,
    *,
    heartbeat_max_age: timedelta = timedelta(seconds=120),
    apply_recovery: bool = False,
) -> list[RecoveryDecision]:
    """Inspect once; optionally fail interrupted jobs under a document lock.

    The heartbeat age is an inspection threshold, not a crash timeout. Missing
    records, stale heartbeats, and Redis failures all remain uncertain.
    Apply mode requires all workers to use the document lock; restart older workers.
    """
    if heartbeat_max_age <= timedelta(0):
        raise ValueError("heartbeat_max_age must be positive")
    return await run_in_threadpool(
        _inspect_running_jobs,
        database_path,
        redis_url,
        heartbeat_max_age,
        apply_recovery,
    )


def _load_running_jobs(database_path: Path) -> list[ProcessingJob]:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT job_id, record FROM processing_jobs"
        ).fetchall()
        jobs = []
        # Validate before filtering so malformed records are not silently skipped.
        for stored_id, record in rows:
            job = ProcessingJob.model_validate_json(record)
            if str(job.job_id) != stored_id:
                raise ValueError("Stored job ID does not match record")
            if job.status == "running":
                jobs.append(job)
        return jobs
    finally:
        connection.close()


def _inspect_running_jobs(
    database_path: Path,
    redis_url: str,
    heartbeat_max_age: timedelta,
    apply_recovery: bool,
) -> list[RecoveryDecision]:
    jobs = _load_running_jobs(database_path)
    if not jobs:
        return []
    connection = Redis.from_url(redis_url, socket_connect_timeout=5, socket_timeout=5)
    try:
        decisions = []
        for job in jobs:
            decision = _inspect_job(job, connection, heartbeat_max_age)
            if decision.decision == "interrupted":
                if apply_recovery:
                    decision = _recover_interrupted(
                        job, database_path, connection, heartbeat_max_age
                    )
                else:
                    decision = RecoveryDecision(
                        job_id=job.job_id,
                        decision="interrupted",
                        recovery_action="skipped",
                        reason="Dry run: interrupted candidate; no changes applied",
                    )
            decisions.append(decision)
        return decisions
    finally:
        connection.close()


def _inspect_job(
    job: ProcessingJob, connection: Redis, heartbeat_max_age: timedelta
) -> RecoveryDecision:
    decision = "uncertain"
    reason = "RQ state does not confirm execution liveness"
    try:
        rq_job = Job.fetch(
            str(job.job_id), connection=connection, serializer=JSONSerializer
        )
        status = rq_job.get_status(refresh=True)
        executions = rq_job.get_executions()
        # These are separate reads; flag a visible status change as uncertainty.
        if rq_job.get_status(refresh=True) != status:
            reason = "RQ status changed during inspection"
        elif status == JobStatus.STARTED:
            now = datetime.now(UTC)
            if any(
                isinstance(execution.last_heartbeat, datetime)
                and execution.last_heartbeat.utcoffset() is not None
                and timedelta(0) <= now - execution.last_heartbeat <= heartbeat_max_age
                for execution in executions
            ):
                decision = "active"
                reason = "RQ execution has a recent heartbeat"
            else:
                reason = "RQ execution heartbeat is missing or stale"
        elif status in {JobStatus.FAILED, JobStatus.STOPPED} and not executions:
            decision = "interrupted"
            reason = "RQ reports failed or stopped with no execution records"
        elif status == JobStatus.FINISHED:
            reason = "RQ finished but SQLite still reports running"
    except NoSuchJobError:
        reason = "RQ record is missing or expired"
    except RedisError:
        reason = "RQ backend is unavailable"
    except InvalidJobOperation, ValueError:
        # Execution records can expire between listing and fetching them.
        reason = "RQ records are inconsistent or changed during inspection"
    return RecoveryDecision(job_id=job.job_id, decision=decision, reason=reason)


def _recover_interrupted(
    job: ProcessingJob,
    database_path: Path,
    connection: Redis,
    heartbeat_max_age: timedelta,
) -> RecoveryDecision:
    try:
        with document_lock(job.document_id, database_path):
            # Recheck after acquiring ownership, before opening a write transaction.
            fresh = _inspect_job(job, connection, heartbeat_max_age)
            if fresh.decision != "interrupted":
                return RecoveryDecision(
                    job_id=job.job_id,
                    decision=fresh.decision,
                    recovery_action="skipped",
                    reason="Recovery skipped: RQ evidence changed",
                )
            applied = _fail_running_job(
                job.job_id,
                database_path,
                "Document processing was interrupted",
                expected_job=job,
            )
            return RecoveryDecision(
                job_id=job.job_id,
                decision="interrupted",
                recovery_action="failed" if applied else "skipped",
                reason="Interrupted job marked failed"
                if applied
                else "Recovery skipped: SQLite record changed",
            )
    except DocumentBusyError:
        return RecoveryDecision(
            job_id=job.job_id,
            decision="uncertain",
            recovery_action="skipped",
            reason="Recovery skipped: document execution lock is held",
        )
    except sqlite3.Error, OSError, ValueError:
        return RecoveryDecision(
            job_id=job.job_id,
            decision="uncertain",
            recovery_action="skipped",
            reason="Recovery skipped: storage or locking failure",
        )
