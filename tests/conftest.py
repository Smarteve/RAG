from pathlib import Path

import pytest

from nexusrag import config
from nexusrag.database import initialize_database


@pytest.fixture(autouse=True)
def chunk_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep upload tests from writing chunks into real application storage."""
    directory = tmp_path / "chunks"
    monkeypatch.setattr(config, "CHUNK_DIRECTORY", directory)
    return directory


@pytest.fixture
def prepare_document():
    """Prepare retrieval fixtures without depending on queued HTTP uploads."""
    import asyncio
    from io import BytesIO

    from fastapi import UploadFile
    from starlette.datastructures import Headers

    from nexusrag.services.ingestion import ingest_document

    def prepare(contents: bytes):
        upload = UploadFile(
            file=BytesIO(contents),
            filename="sample.pdf",
            headers=Headers({"content-type": "application/pdf"}),
        )
        try:
            result = asyncio.run(
                ingest_document(
                    upload,
                    config.STORAGE_DIRECTORY,
                    config.EXTRACTION_DIRECTORY,
                    config.CHUNK_DIRECTORY,
                )
            )
            return result.extraction.document_id
        finally:
            upload.file.close()

    return prepare


@pytest.fixture(autouse=True)
def job_database_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep application startup from creating real job records in tests."""
    path = tmp_path / "jobs.sqlite3"
    monkeypatch.setattr(config, "JOB_DATABASE_PATH", path)
    return path


@pytest.fixture
def initialized_job_database(job_database_path: Path) -> Path:
    """Supply startup schema for HTTP tests whose clients do not enter lifespan."""
    initialize_database(job_database_path)
    return job_database_path


@pytest.fixture
def vector_search_backend():
    """Offline reference backend for HTTP orchestration tests, never Qdrant IO."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from fastapi.concurrency import run_in_threadpool

    from nexusrag.services import retrieval

    async def search(chunks, embeddings, query_vector, limit):
        return await run_in_threadpool(
            retrieval.search_semantic_chunks, chunks, embeddings, query_vector, limit
        )

    return SimpleNamespace(search=AsyncMock(side_effect=search))
