import asyncio
import sqlite3
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from nexusrag import config
from nexusrag.main import app
from nexusrag.schemas.job import ProcessingJob
from nexusrag.services.job_storage import create_job, transition_job


@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed"])
def test_fetch_persisted_job(status):
    now = datetime.now(UTC)
    job = ProcessingJob(
        job_id=uuid4(),
        document_id=uuid4(),
        status="queued",
        created_at=now,
        updated_at=now,
    )
    with TestClient(app) as client:
        asyncio.run(create_job(job, config.JOB_DATABASE_PATH))
        if status != "queued":
            job = asyncio.run(
                transition_job(job.job_id, "running", config.JOB_DATABASE_PATH)
            )
        if status in {"succeeded", "failed"}:
            job = asyncio.run(
                transition_job(
                    job.job_id,
                    status,
                    config.JOB_DATABASE_PATH,
                    error="Document processing failed" if status == "failed" else None,
                )
            )
        response = client.get(f"/jobs/{job.job_id}")
    assert response.status_code == 200
    assert response.json() == job.model_dump(mode="json")


def test_unknown_job():
    with TestClient(app) as client:
        response = client.get(f"/jobs/{uuid4()}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


def test_invalid_id_does_not_query_storage(monkeypatch):
    load = AsyncMock()
    monkeypatch.setattr("nexusrag.main.load_job", load)
    with TestClient(app) as client:
        response = client.get("/jobs/not-a-uuid")
    assert response.status_code == 422
    load.assert_not_awaited()


@pytest.mark.parametrize("record", ["not JSON", "{}"])
def test_corrupt_record_returns_safe_error(record):
    job_id = uuid4()
    with TestClient(app) as client:
        connection = sqlite3.connect(config.JOB_DATABASE_PATH)
        try:
            with connection:
                connection.execute(
                    "INSERT INTO processing_jobs (job_id, record) VALUES (?, ?)",
                    (str(job_id), record),
                )
        finally:
            connection.close()
        response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load job"}


@pytest.mark.parametrize(
    "error",
    [
        sqlite3.OperationalError("private database path"),
        PermissionError("private path"),
    ],
)
def test_storage_failure_returns_safe_error(monkeypatch, error):
    monkeypatch.setattr("nexusrag.main.load_job", AsyncMock(side_effect=error))
    with TestClient(app) as client:
        response = client.get(f"/jobs/{uuid4()}")
    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load job"}
