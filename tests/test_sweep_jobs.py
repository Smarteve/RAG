import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from nexusrag import sweep_jobs
from nexusrag.schemas.job import ProcessingJob
from nexusrag.schemas.recovery import RecoveryDecision
from nexusrag.services.job_retry import RetryAlreadyExistsError, RetrySubmissionError
from nexusrag.services.processing_lock import DocumentBusyError


@pytest.fixture
def sweep(monkeypatch):
    now = datetime.now(UTC)
    jobs = [
        ProcessingJob(
            job_id=uuid4(),
            document_id=uuid4(),
            status="failed",
            created_at=now,
            updated_at=now,
            error="Document processing was interrupted",
        )
        for _ in range(2)
    ]
    inspect = AsyncMock(return_value=[])
    candidates = AsyncMock(return_value=jobs)
    retry = AsyncMock(side_effect=jobs)
    monkeypatch.setattr(sweep_jobs, "inspect_running_jobs", inspect)
    monkeypatch.setattr(sweep_jobs, "load_retry_candidates", candidates)
    monkeypatch.setattr(sweep_jobs, "retry_failed_job", retry)
    return jobs, inspect, candidates, retry


def events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_dry_run_never_retries(sweep, capsys):
    _, inspect, _, retry = sweep
    assert asyncio.run(sweep_jobs.sweep_once()) == 0
    assert inspect.await_args.kwargs["apply_recovery"] is False
    retry.assert_not_awaited()
    output = events(capsys)
    assert sum(row["event"] == "would_retry" for row in output) == 2
    assert output[-1]["proposed"] == 2


def test_apply_loads_candidates_after_recovery_and_includes_hook_failures(
    sweep, capsys
):
    jobs, inspect, candidates, retry = sweep
    order = []

    async def recover(*args, **kwargs):
        order.append("inspect")
        assert kwargs["apply_recovery"] is True
        return [
            RecoveryDecision(
                job_id=jobs[0].job_id,
                decision="interrupted",
                recovery_action="failed",
                reason="Recovered",
            )
        ]

    async def load(*args):
        order.append("candidates")
        return jobs

    inspect.side_effect = recover
    candidates.side_effect = load
    assert asyncio.run(sweep_jobs.sweep_once(apply=True)) == 0
    assert order == ["inspect", "candidates"]
    assert retry.await_count == 2
    summary = events(capsys)[-1]
    assert summary["recovered"] == 1
    assert summary["retried"] == 2


@pytest.mark.parametrize(
    "error", [DocumentBusyError(), RetryAlreadyExistsError(uuid4())]
)
def test_expected_skip_does_not_stop_other_jobs(sweep, capsys, error):
    jobs, _, _, retry = sweep
    retry.side_effect = [error, jobs[1]]
    assert asyncio.run(sweep_jobs.sweep_once(apply=True)) == 0
    summary = events(capsys)[-1]
    assert summary["skipped"] == 1
    assert summary["retried"] == 1


@pytest.mark.parametrize(
    "error", [RetrySubmissionError(uuid4()), sqlite3.OperationalError("secret URL")]
)
def test_retry_failure_is_reported_and_other_jobs_continue(sweep, capsys, error):
    jobs, _, _, retry = sweep
    retry.side_effect = [error, jobs[1]]
    assert asyncio.run(sweep_jobs.sweep_once(apply=True)) == 1
    output = events(capsys)
    assert output[-1]["failures"] == 1
    assert output[-1]["retried"] == 1
    assert "secret" not in str(output)
    if isinstance(error, RetrySubmissionError):
        assert output[0]["job_id"] == str(error.job_id)
        assert output[0]["event"] == "retry_delivery_unconfirmed"


def test_inspection_failure_stops_before_retries(sweep, capsys):
    _, inspect, candidates, retry = sweep
    inspect.side_effect = sqlite3.OperationalError("private path")
    assert asyncio.run(sweep_jobs.sweep_once(apply=True)) == 1
    candidates.assert_not_awaited()
    retry.assert_not_awaited()
    assert events(capsys) == [
        {"event": "inspection_failed", "reason": "Could not inspect job records"}
    ]


def test_cli_passes_options(monkeypatch):
    run = AsyncMock(return_value=1)
    monkeypatch.setattr(sweep_jobs, "sweep_once", run)
    monkeypatch.setattr("sys.argv", ["sweep_jobs", "--apply", "--max-attempts", "2"])
    assert sweep_jobs.main() == 1
    run.assert_awaited_once_with(apply=True, max_attempts=2)
