import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from redis.exceptions import ConnectionError, TimeoutError
from rq.exceptions import DuplicateJobError, NoSuchJobError
from rq.job import JobStatus

from nexusrag import config
from nexusrag.database import initialize_database
from nexusrag.schemas.job import ProcessingJob
from nexusrag.services import delivery_reconciliation as delivery
from nexusrag.services.job_storage import _load_job, _transition_job
from nexusrag.services.processing_lock import document_lock


@pytest.fixture
def queued(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    initialize_database(path)
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", tmp_path / "documents")
    config.STORAGE_DIRECTORY.mkdir()
    now = datetime.now(UTC) - timedelta(minutes=5)
    job = ProcessingJob(
        job_id=uuid4(),
        document_id=uuid4(),
        status="queued",
        created_at=now,
        updated_at=now,
    )
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute(
                "INSERT INTO processing_jobs VALUES (?, ?)",
                (str(job.job_id), job.model_dump_json()),
            )
    finally:
        connection.close()
    (config.STORAGE_DIRECTORY / f"{job.document_id}.pdf").write_bytes(b"%PDF-saved")
    client = Mock()
    monkeypatch.setattr(delivery.Redis, "from_url", Mock(return_value=client))
    queue = Mock(name="ingestion_queue")
    queue.name = "ingestion"
    queue.connection = client
    queue.get_job_position.return_value = None
    monkeypatch.setattr(delivery, "Queue", Mock(return_value=queue))
    fetch = Mock(side_effect=NoSuchJobError())
    monkeypatch.setattr(delivery.Job, "fetch", fetch)
    return path, job, client, queue, fetch


def rq_record(job, status):
    record = Mock()
    record.origin = "ingestion"
    record.func_name = "nexusrag.tasks.process_document"
    record.args = (str(job.job_id),)
    record.kwargs = {}
    record.get_status.return_value = status
    record.get_executions.return_value = []
    return record


def test_missing_record_dry_run_is_read_only(queued):
    path, _, client, queue, _ = queued
    before = path.read_bytes()
    decisions = asyncio.run(delivery.reconcile_queued_jobs(path, "redis://test"))
    assert decisions[0].action == "would_enqueue"
    queue.enqueue_call.assert_not_called()
    assert path.read_bytes() == before
    assert not (path.parent / ".job-locks").exists()
    client.close.assert_called_once_with()


def test_missing_record_redelivers_same_attempt(queued):
    path, job, client, queue, _ = queued
    decisions = asyncio.run(
        delivery.reconcile_queued_jobs(path, "redis://test", apply=True)
    )
    assert decisions[0].action == "enqueued"
    queue.enqueue_call.assert_called_once_with(
        "nexusrag.tasks.process_document",
        args=(str(job.job_id),),
        job_id=str(job.job_id),
        unique=True,
    )
    assert _load_job(job.job_id, path) == job
    client.close.assert_called_once_with()


@pytest.mark.parametrize(
    "status", [JobStatus.QUEUED, JobStatus.SCHEDULED, JobStatus.STARTED]
)
def test_existing_delivery_is_not_resubmitted(queued, status):
    path, job, _, queue, fetch = queued
    record = rq_record(job, status)
    if status == JobStatus.QUEUED:
        queue.get_job_position.return_value = 0
    if status == JobStatus.STARTED:
        record.get_executions.return_value = [
            SimpleNamespace(last_heartbeat=datetime.now(UTC))
        ]
    fetch.side_effect = None
    fetch.return_value = record
    result = asyncio.run(
        delivery.reconcile_queued_jobs(path, "redis://test", apply=True)
    )
    assert result[0].action == "present"
    queue.enqueue_call.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.STOPPED, JobStatus.STARTED],
)
def test_inconsistent_terminal_or_inactive_state_is_uncertain(queued, status):
    path, job, _, queue, fetch = queued
    fetch.side_effect = None
    fetch.return_value = rq_record(job, status)
    assert (
        asyncio.run(delivery.reconcile_queued_jobs(path, "redis://test", apply=True))[
            0
        ].action
        == "uncertain"
    )
    queue.enqueue_call.assert_not_called()


def test_missing_record_with_stale_queue_id_is_uncertain(queued):
    path, _, _, queue, _ = queued
    queue.get_job_position.return_value = 0
    assert (
        asyncio.run(delivery.reconcile_queued_jobs(path, "redis://test", apply=True))[
            0
        ].action
        == "uncertain"
    )
    queue.enqueue_call.assert_not_called()


def test_live_document_is_skipped(queued):
    path, job, _, queue, fetch = queued
    with document_lock(job.document_id, path):
        result = asyncio.run(
            delivery.reconcile_queued_jobs(path, "redis://test", apply=True)
        )
    assert result[0].action == "skipped"
    fetch.assert_not_called()
    queue.enqueue_call.assert_not_called()


def test_changed_snapshot_is_skipped(queued, monkeypatch):
    path, job, _, queue, _ = queued
    original = delivery._queued_snapshots

    def snapshots(database_path):
        records = original(database_path)
        _transition_job(job.job_id, "running", path)
        return records

    monkeypatch.setattr(delivery, "_queued_snapshots", snapshots)
    result = asyncio.run(
        delivery.reconcile_queued_jobs(path, "redis://test", apply=True)
    )
    assert result[0].action == "skipped"
    queue.enqueue_call.assert_not_called()


def test_grace_period_skips_fresh_jobs(queued):
    path, _, _, queue, fetch = queued
    result = asyncio.run(
        delivery.reconcile_queued_jobs(
            path, "redis://test", apply=True, grace_period=timedelta(hours=1)
        )
    )
    assert result[0].action == "skipped"
    fetch.assert_not_called()
    queue.enqueue_call.assert_not_called()


def test_duplicate_enqueue_race_rechecks_winning_record(queued):
    path, job, _, queue, fetch = queued
    record = rq_record(job, JobStatus.QUEUED)
    fetch.side_effect = [NoSuchJobError(), NoSuchJobError(), record]
    queue.enqueue_call.side_effect = DuplicateJobError()
    queue.get_job_position.side_effect = [None, None, 0]
    assert (
        asyncio.run(delivery.reconcile_queued_jobs(path, "redis://test", apply=True))[
            0
        ].action
        == "present"
    )
    assert _load_job(job.job_id, path) == job


def test_ambiguous_enqueue_retains_sqlite_attempt(queued):
    path, job, _, queue, _ = queued
    queue.enqueue_call.side_effect = TimeoutError("secret URL")
    result = asyncio.run(
        delivery.reconcile_queued_jobs(path, "redis://test", apply=True)
    )
    assert result[0].action == "delivery_unconfirmed"
    assert "secret" not in result[0].reason
    assert _load_job(job.job_id, path) == job


def test_redis_outage_is_uncertain(queued):
    path, _, _, queue, fetch = queued
    fetch.side_effect = ConnectionError("private server")
    result = asyncio.run(
        delivery.reconcile_queued_jobs(path, "redis://test", apply=True)
    )
    assert result[0].action == "uncertain"
    assert "private" not in result[0].reason
    queue.enqueue_call.assert_not_called()


def test_missing_source_is_not_submitted(queued):
    path, job, _, queue, _ = queued
    (config.STORAGE_DIRECTORY / f"{job.document_id}.pdf").unlink()
    result = asyncio.run(
        delivery.reconcile_queued_jobs(path, "redis://test", apply=True)
    )
    assert result[0].action == "uncertain"
    queue.enqueue_call.assert_not_called()


def test_one_lookup_failure_does_not_stop_other_queued_jobs(queued):
    path, _, _, queue, fetch = queued
    now = datetime.now(UTC) - timedelta(minutes=5)
    other = ProcessingJob(
        job_id=uuid4(),
        document_id=uuid4(),
        status="queued",
        created_at=now,
        updated_at=now,
    )
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute(
                "INSERT INTO processing_jobs VALUES (?, ?)",
                (str(other.job_id), other.model_dump_json()),
            )
    finally:
        connection.close()
    (config.STORAGE_DIRECTORY / f"{other.document_id}.pdf").write_bytes(b"%PDF-saved")
    fetch.side_effect = [
        ConnectionError("private URL"),
        NoSuchJobError(),
        NoSuchJobError(),
    ]
    results = asyncio.run(
        delivery.reconcile_queued_jobs(path, "redis://test", apply=True)
    )
    assert [d.action for d in results] == ["uncertain", "enqueued"]
    assert queue.enqueue_call.call_count == 1


def test_existing_task_with_wrong_identity_is_uncertain(queued):
    path, job, _, queue, fetch = queued
    record = rq_record(job, JobStatus.QUEUED)
    record.args = (str(uuid4()),)
    fetch.side_effect = None
    fetch.return_value = record
    result = asyncio.run(
        delivery.reconcile_queued_jobs(path, "redis://test", apply=True)
    )
    assert result[0].action == "uncertain"
    queue.enqueue_call.assert_not_called()
