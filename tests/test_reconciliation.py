import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from redis.exceptions import ConnectionError
from rq.exceptions import NoSuchJobError
from rq.job import JobStatus

from nexusrag.database import initialize_database
from nexusrag.schemas.job import ProcessingJob
from nexusrag.services import reconciliation


@pytest.fixture
def inspection(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    initialize_database(path)
    now = datetime.now(UTC)
    jobs = [
        ProcessingJob(
            job_id=uuid4(),
            document_id=uuid4(),
            status="running",
            created_at=now,
            updated_at=now,
        )
        for _ in range(2)
    ]
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.executemany(
                "INSERT INTO processing_jobs VALUES (?, ?)",
                [(str(job.job_id), job.model_dump_json()) for job in jobs],
            )
    finally:
        connection.close()
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(reconciliation.Redis, "from_url", factory)
    fetch = Mock()
    monkeypatch.setattr(reconciliation.Job, "fetch", fetch)
    return path, jobs, client, factory, fetch


@pytest.mark.parametrize(
    ("status", "age", "expected"),
    [
        (JobStatus.STARTED, 10, "active"),
        (JobStatus.STARTED, 300, "uncertain"),
        (JobStatus.STARTED, -10, "uncertain"),
        (JobStatus.STARTED, None, "uncertain"),
        (JobStatus.FAILED, None, "interrupted"),
        (JobStatus.STOPPED, None, "interrupted"),
        (JobStatus.FAILED, 10, "uncertain"),
        (JobStatus.FINISHED, None, "uncertain"),
        (JobStatus.QUEUED, None, "uncertain"),
    ],
)
def test_status_decisions_are_read_only(inspection, status, age, expected):
    path, jobs, client, factory, fetch = inspection
    before = path.read_bytes()
    rq_job = Mock()
    rq_job.get_status.return_value = status
    rq_job.get_executions.return_value = (
        []
        if age is None
        else [
            SimpleNamespace(last_heartbeat=datetime.now(UTC) - timedelta(seconds=age))
        ]
    )
    fetch.return_value = rq_job
    decisions = asyncio.run(reconciliation.inspect_running_jobs(path, "redis://test"))
    assert [d.job_id for d in decisions] == [j.job_id for j in jobs]
    assert all(d.decision == expected for d in decisions)
    factory.assert_called_once_with(
        "redis://test", socket_connect_timeout=5, socket_timeout=5
    )
    client.close.assert_called_once_with()
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "error",
    [
        NoSuchJobError("private id"),
        ConnectionError("secret URL"),
        ValueError("expired execution"),
    ],
)
def test_failed_lookup_does_not_stop_remaining_jobs(inspection, error):
    path, _, client, _, fetch = inspection
    rq_job = Mock()
    rq_job.get_status.return_value = JobStatus.FAILED
    rq_job.get_executions.return_value = []
    fetch.side_effect = [error, rq_job]
    decisions = asyncio.run(reconciliation.inspect_running_jobs(path, "redis://test"))
    assert [d.decision for d in decisions] == ["uncertain", "interrupted"]
    assert str(error) not in decisions[0].reason
    client.close.assert_called_once_with()


def test_status_changing_during_inspection_is_uncertain(inspection):
    path, _, _, _, fetch = inspection
    rq_job = Mock()
    rq_job.get_status.side_effect = [JobStatus.FAILED, JobStatus.STARTED] * 2
    rq_job.get_executions.return_value = []
    fetch.return_value = rq_job
    assert all(
        d.decision == "uncertain"
        for d in asyncio.run(reconciliation.inspect_running_jobs(path, "redis://test"))
    )


def test_no_running_jobs_does_not_connect_to_redis(inspection):
    path, _, _, factory, _ = inspection
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute("DELETE FROM processing_jobs")
    finally:
        connection.close()
    assert asyncio.run(reconciliation.inspect_running_jobs(path, "redis://test")) == []
    factory.assert_not_called()


def test_corrupt_sqlite_record_is_not_silently_skipped(inspection):
    path, _, _, factory, _ = inspection
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute("UPDATE processing_jobs SET record = '{}' ")
    finally:
        connection.close()
    with pytest.raises(ValidationError):
        asyncio.run(reconciliation.inspect_running_jobs(path, "redis://test"))
    factory.assert_not_called()


def test_invalid_heartbeat_threshold(inspection):
    path, _, _, factory, _ = inspection
    with pytest.raises(ValueError, match="must be positive"):
        asyncio.run(
            reconciliation.inspect_running_jobs(
                path, "redis://test", heartbeat_max_age=timedelta(0)
            )
        )
    factory.assert_not_called()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (JobStatus.STARTED, "active"),
        (JobStatus.FAILED, "interrupted"),
        (JobStatus.FINISHED, "uncertain"),
    ],
)
def test_inspection_reports_without_writing(inspection, status, expected):
    path, jobs, client, _, fetch = inspection
    before = path.read_bytes()
    rq_job = Mock()
    rq_job.get_status.return_value = status
    rq_job.get_executions.return_value = (
        [SimpleNamespace(last_heartbeat=datetime.now(UTC))]
        if status == JobStatus.STARTED
        else []
    )
    fetch.return_value = rq_job

    decisions = asyncio.run(reconciliation.inspect_running_jobs(path, "redis://test"))

    assert [d.job_id for d in decisions] == [job.job_id for job in jobs]
    assert all(d.decision == expected for d in decisions)
    if expected == "interrupted":
        assert all("Dry run" in d.reason for d in decisions)
    assert path.read_bytes() == before
    client.close.assert_called_once_with()


def test_inspection_forwards_heartbeat_threshold(inspection):
    path, _, _, _, fetch = inspection
    rq_job = Mock()
    rq_job.get_status.return_value = JobStatus.STARTED
    rq_job.get_executions.return_value = [
        SimpleNamespace(last_heartbeat=datetime.now(UTC) - timedelta(seconds=30))
    ]
    fetch.return_value = rq_job
    decisions = asyncio.run(
        reconciliation.inspect_running_jobs(
            path, "redis://test", heartbeat_max_age=timedelta(seconds=10)
        )
    )
    assert all(d.decision == "uncertain" for d in decisions)


def _failed_rq_job():
    rq_job = Mock()
    rq_job.get_status.return_value = JobStatus.FAILED
    rq_job.get_executions.return_value = []
    return rq_job


def test_apply_recovers_interrupted_jobs_without_retrying(inspection):
    path, jobs, _, _, fetch = inspection
    fetch.return_value = _failed_rq_job()
    decisions = asyncio.run(
        reconciliation.inspect_running_jobs(path, "redis://test", apply_recovery=True)
    )
    assert all(d.recovery_action == "failed" for d in decisions)
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("SELECT record FROM processing_jobs").fetchall()
    finally:
        connection.close()
    records = [ProcessingJob.model_validate_json(row[0]) for row in rows]
    assert len(records) == len(jobs)
    assert all(
        job.status == "failed" and job.error == "Document processing was interrupted"
        for job in records
    )
    # Terminal jobs are not candidates on another pass.
    assert (
        asyncio.run(
            reconciliation.inspect_running_jobs(
                path, "redis://test", apply_recovery=True
            )
        )
        == []
    )


def test_apply_skips_document_owned_by_live_or_paused_worker(inspection):
    from nexusrag.services.processing_lock import document_lock

    path, jobs, _, _, fetch = inspection
    fetch.return_value = _failed_rq_job()
    with document_lock(jobs[0].document_id, path):
        decisions = asyncio.run(
            reconciliation.inspect_running_jobs(
                path, "redis://test", apply_recovery=True
            )
        )
    assert decisions[0].recovery_action == "skipped"
    assert decisions[0].decision == "uncertain"
    from nexusrag.services.job_storage import _load_job

    assert _load_job(jobs[0].job_id, path) == jobs[0]


def test_apply_rechecks_rq_under_document_lock(inspection):
    path, _, _, _, fetch = inspection
    failed = _failed_rq_job()
    started = Mock()
    started.get_status.return_value = JobStatus.STARTED
    started.get_executions.return_value = [
        SimpleNamespace(last_heartbeat=datetime.now(UTC))
    ]
    fetch.side_effect = [failed, started, failed, started]
    before = path.read_bytes()
    decisions = asyncio.run(
        reconciliation.inspect_running_jobs(path, "redis://test", apply_recovery=True)
    )
    assert all(d.recovery_action == "skipped" for d in decisions)
    assert path.read_bytes() == before


def test_apply_rejects_changed_sqlite_snapshot(inspection):
    from nexusrag.services.job_storage import _load_job, _transition_job

    path, jobs, _, _, fetch = inspection
    rq_job = _failed_rq_job()
    calls = 0

    def fetch_job(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            _transition_job(jobs[0].job_id, "succeeded", path)
        return rq_job

    fetch.side_effect = fetch_job
    decisions = asyncio.run(
        reconciliation.inspect_running_jobs(path, "redis://test", apply_recovery=True)
    )
    assert decisions[0].recovery_action == "skipped"
    assert _load_job(jobs[0].job_id, path).status == "succeeded"


@pytest.mark.parametrize("status", [JobStatus.STARTED, JobStatus.FINISHED])
def test_apply_does_not_write_uncertain_jobs(inspection, status):
    path, _, _, _, fetch = inspection
    rq_job = Mock()
    rq_job.get_status.return_value = status
    rq_job.get_executions.return_value = []
    fetch.return_value = rq_job
    before = path.read_bytes()
    decisions = asyncio.run(
        reconciliation.inspect_running_jobs(path, "redis://test", apply_recovery=True)
    )
    assert all(
        d.decision == "uncertain" and d.recovery_action == "none" for d in decisions
    )
    assert path.read_bytes() == before
