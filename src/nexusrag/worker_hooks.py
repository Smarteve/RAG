"""RQ failure hooks. Whole-worker loss still requires reconciliation."""

import asyncio
import logging
import sqlite3
from resource import struct_rusage
from types import TracebackType
from uuid import UUID

from rq.job import Job

from nexusrag import config
from nexusrag.services.job_recovery import fail_running_job
from nexusrag.tasks import JobNotClaimedError

logger = logging.getLogger(__name__)


def _record_failure(job: Job, message: str) -> None:
    try:
        asyncio.run(
            fail_running_job(UUID(job.id), config.JOB_DATABASE_PATH, error=message)
        )
    except (sqlite3.Error, OSError, ValueError, RuntimeError) as error:
        # Keep provider exception text, paths, and document content out of logs.
        # Do not prevent RQ's own failure handling if SQLite recovery fails.
        logger.error("Could not recover processing job (%s)", type(error).__name__)


def handle_job_exception(
    job: Job,
    exc_type: type[BaseException],
    exc_value: BaseException,
    traceback: TracebackType | None,
) -> bool:
    """Record failure without interrupting RQ's normal exception handlers."""
    # A duplicate execution rejected at claim time does not own the running job.
    if not issubclass(exc_type, JobNotClaimedError):
        _record_failure(job, "Document processing failed")
    return True


def handle_workhorse_killed(
    job: Job, retpid: int, ret_val: int, rusage: struct_rusage
) -> None:
    """Record unexpected child termination detected by the surviving parent."""
    _record_failure(job, "Document processing was interrupted")
