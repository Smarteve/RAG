import asyncio
import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag import config, run_worker, worker_hooks
from nexusrag.database import initialize_database
from nexusrag.schemas.job import ProcessingJob
from nexusrag.services.job_recovery import fail_running_job
from nexusrag.services.job_storage import create_job, load_job, transition_job
from nexusrag.tasks import JobNotClaimedError


@pytest.fixture
def job():
    initialize_database(config.JOB_DATABASE_PATH)
    now = datetime.now(UTC)
    record = ProcessingJob(
        job_id=uuid4(),
        document_id=uuid4(),
        status="queued",
        created_at=now,
        updated_at=now,
    )
    asyncio.run(create_job(record, config.JOB_DATABASE_PATH))
    return record


@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed"])
@pytest.mark.parametrize("hook", ["exception", "killed"])
def test_hooks_only_fail_running_jobs(job, status, hook):
    if status != "queued":
        asyncio.run(transition_job(job.job_id, "running", config.JOB_DATABASE_PATH))
    if status in {"succeeded", "failed"}:
        asyncio.run(
            transition_job(
                job.job_id,
                status,
                config.JOB_DATABASE_PATH,
                "Original failure" if status == "failed" else None,
            )
        )
    before = asyncio.run(load_job(job.job_id, config.JOB_DATABASE_PATH))
    rq_job = SimpleNamespace(id=str(job.job_id))
    if hook == "exception":
        assert (
            worker_hooks.handle_job_exception(
                rq_job, RuntimeError, RuntimeError("secret provider response"), None
            )
            is True
        )
        expected_error = "Document processing failed"
    else:
        worker_hooks.handle_workhorse_killed(rq_job, 123, 9, None)
        expected_error = "Document processing was interrupted"
    after = asyncio.run(load_job(job.job_id, config.JOB_DATABASE_PATH))
    if status == "running":
        assert after.status == "failed"
        assert after.error == expected_error
        assert after.job_id == before.job_id
        assert after.document_id == before.document_id
        assert after.created_at == before.created_at
    else:
        assert after == before
    # Notifications can repeat; don't replace the first error or timestamp.
    worker_hooks.handle_workhorse_killed(rq_job, 123, 9, None)
    assert asyncio.run(load_job(job.job_id, config.JOB_DATABASE_PATH)) == after


def test_rejected_claim_hook_preserves_other_execution(job):
    asyncio.run(transition_job(job.job_id, "running", config.JOB_DATABASE_PATH))
    before = asyncio.run(load_job(job.job_id, config.JOB_DATABASE_PATH))
    result = worker_hooks.handle_job_exception(
        SimpleNamespace(id=str(job.job_id)),
        JobNotClaimedError,
        JobNotClaimedError(),
        None,
    )
    assert result is True
    assert asyncio.run(load_job(job.job_id, config.JOB_DATABASE_PATH)) == before


@pytest.mark.parametrize("hook", ["exception", "killed"])
def test_storage_failure_does_not_break_rq_or_log_secrets(monkeypatch, caplog, hook):
    monkeypatch.setattr(
        worker_hooks,
        "fail_running_job",
        AsyncMock(side_effect=sqlite3.OperationalError("private path")),
    )
    job = SimpleNamespace(id=str(uuid4()))
    if hook == "exception":
        assert (
            worker_hooks.handle_job_exception(
                job, RuntimeError, RuntimeError("secret"), None
            )
            is True
        )
    else:
        worker_hooks.handle_workhorse_killed(job, 123, 9, None)
    assert "Could not recover processing job" in caplog.text
    assert "private path" not in caplog.text
    assert "secret" not in caplog.text


def test_recovery_unknown_job_raises(job):
    with pytest.raises(FileNotFoundError):
        asyncio.run(fail_running_job(uuid4(), config.JOB_DATABASE_PATH, "Interrupted"))


def test_invalid_error_rolls_back_and_releases_lock(job):
    running = asyncio.run(
        transition_job(job.job_id, "running", config.JOB_DATABASE_PATH)
    )
    with pytest.raises(ValidationError):
        asyncio.run(fail_running_job(job.job_id, config.JOB_DATABASE_PATH, " "))
    assert asyncio.run(load_job(job.job_id, config.JOB_DATABASE_PATH)) == running
    assert (
        asyncio.run(
            fail_running_job(job.job_id, config.JOB_DATABASE_PATH, "Interrupted")
        )
        is True
    )


def test_worker_startup_registers_hooks_and_json(monkeypatch):
    connection = Mock()
    worker = Mock()
    factory = Mock(return_value=worker)
    monkeypatch.setattr(run_worker.Redis, "from_url", Mock(return_value=connection))
    queue = Mock()
    monkeypatch.setattr(run_worker, "Queue", Mock(return_value=queue))
    monkeypatch.setattr(run_worker, "Worker", factory)
    run_worker.main()
    kwargs = factory.call_args.kwargs
    assert factory.call_args.args == ([queue],)
    assert kwargs["serializer"] is run_worker.JSONSerializer
    assert kwargs["exception_handlers"] == [worker_hooks.handle_job_exception]
    assert kwargs["work_horse_killed_handler"] is worker_hooks.handle_workhorse_killed
    worker.work.assert_called_once_with()
    connection.close.assert_called_once_with()
