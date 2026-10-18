import asyncio
import sqlite3
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from nexusrag import config
from nexusrag.database import initialize_database
from nexusrag.main import app, upload_document
from nexusrag.schemas.document import MAX_DOCUMENT_SIZE_BYTES
from nexusrag.services import ingestion
from nexusrag.services.job_storage import load_job


@pytest.fixture(autouse=True)
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, job_database_path):
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", tmp_path / "documents")
    monkeypatch.setattr(config, "JOB_DATABASE_PATH", tmp_path / "jobs.sqlite3")
    initialize_database(config.JOB_DATABASE_PATH)
    enqueue = AsyncMock()
    monkeypatch.setattr(ingestion, "enqueue_processing", enqueue)
    return enqueue


def test_upload_persists_before_enqueueing(setup):
    contents = b"%PDF-1.4\nbroken PDF; parsing belongs to worker"

    async def check_saved(job_id):
        job = await load_job(job_id, config.JOB_DATABASE_PATH)
        assert job.status == "queued"
        assert (
            config.STORAGE_DIRECTORY / f"{job.document_id}.pdf"
        ).read_bytes() == contents

    setup.side_effect = check_saved
    with TestClient(app) as client:
        response = client.post(
            "/documents", files={"file": ("sample.pdf", contents, "application/pdf")}
        )
    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"document_id", "job_id", "status"}
    assert body["status"] == "queued"
    setup.assert_awaited_once_with(UUID(body["job_id"]))
    assert not config.CHUNK_DIRECTORY.exists()


@pytest.mark.parametrize(
    ("filename", "contents", "media_type", "status"),
    [
        ("notes.txt", b"notes", "text/plain", 415),
        ("fake.pdf", b"fake", "application/pdf", 415),
        ("../fake.pdf", b"%PDF-1.4", "application/pdf", 422),
        ("big.pdf", b"%PDF-" + b"x" * MAX_DOCUMENT_SIZE_BYTES, "application/pdf", 413),
    ],
    ids=["media-type", "signature", "filename", "oversize"],
)
def test_invalid_upload_is_not_enqueued(setup, filename, contents, media_type, status):
    response = TestClient(app).post(
        "/documents", files={"file": (filename, contents, media_type)}
    )
    assert response.status_code == status
    setup.assert_not_awaited()
    assert not config.STORAGE_DIRECTORY.exists()


@pytest.mark.parametrize("stage", ["pdf", "job", "queue"])
def test_submission_failure_is_not_accepted(setup, monkeypatch, stage):
    if stage == "pdf":
        config.STORAGE_DIRECTORY.write_text("not a directory")
    elif stage == "job":
        monkeypatch.setattr(
            ingestion,
            "create_job",
            AsyncMock(side_effect=sqlite3.OperationalError("private path")),
        )
    else:
        setup.side_effect = ConnectionError("private connection details")
    response = TestClient(app).post(
        "/documents", files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert response.status_code == (503 if stage == "queue" else 500)
    assert "private" not in response.text
    if stage != "queue":
        setup.assert_not_awaited()
    if stage == "queue":
        job_id = setup.await_args.args[0]
        job = asyncio.run(load_job(job_id, config.JOB_DATABASE_PATH))
        assert job.status == "queued"
        assert (config.STORAGE_DIRECTORY / f"{job.document_id}.pdf").exists()


@pytest.mark.parametrize("failure", [False, True])
def test_upload_is_closed(setup, failure):
    if failure:
        setup.side_effect = ConnectionError("unavailable")
    source = BytesIO(b"%PDF-1.4")
    upload = UploadFile(
        file=source,
        filename="sample.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )
    if failure:
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            asyncio.run(upload_document(upload))
    else:
        asyncio.run(upload_document(upload))
    assert source.closed
