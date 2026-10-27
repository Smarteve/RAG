import asyncio
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from nexusrag import config, storage
from nexusrag.main import app
from nexusrag.schemas.chunk import DocumentChunks
from nexusrag.schemas.extraction import DocumentExtraction, ExtractedPage
from nexusrag.services.chunking import chunk_document

client = TestClient(app)
pytestmark = pytest.mark.usefixtures("initialized_job_database")
SAMPLE_PDF = Path(__file__).resolve().parents[1] / "examples/employee-handbook.pdf"


@pytest.fixture(autouse=True)
def directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", tmp_path / "documents")
    monkeypatch.setattr(config, "EXTRACTION_DIRECTORY", tmp_path / "extracted")


@pytest.fixture
def record() -> DocumentChunks:
    extraction = DocumentExtraction(
        document_id=uuid4(),
        pages=[
            ExtractedPage(page_number=1, text="ABCDEFGHIJK"),
            ExtractedPage(page_number=2, text="中文 café"),
        ],
    )
    result = DocumentChunks(
        document_id=extraction.document_id,
        chunk_size=5,
        overlap=2,
        chunks=chunk_document(extraction, chunk_size=5, overlap=2),
    )
    asyncio.run(storage.save_chunks(result, config.CHUNK_DIRECTORY))
    return result


@pytest.mark.parametrize("offset, limit", [(0, 2), (2, 2), (5, 2), (99, 100), (0, 100)])
def test_fetch_chunks_paginates_without_renumbering(
    record: DocumentChunks, offset: int, limit: int
) -> None:
    response = client.get(
        f"/documents/{record.document_id}/chunks",
        params={"offset": offset, "limit": limit},
    )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": str(record.document_id),
        "total_chunks": len(record.chunks),
        "offset": offset,
        "limit": limit,
        "chunks": [
            chunk.model_dump(mode="json")
            for chunk in record.chunks[offset : offset + limit]
        ],
    }


def test_fetch_default_pagination(record: DocumentChunks) -> None:
    response = client.get(f"/documents/{record.document_id}/chunks")

    assert response.status_code == 200
    assert response.json()["offset"] == 0
    assert response.json()["limit"] == 10
    assert len(response.json()["chunks"]) == len(record.chunks)


@pytest.mark.parametrize(
    "params",
    [{"offset": -1}, {"offset": "abc"}, {"limit": 0}, {"limit": 101}, {"limit": "abc"}],
)
def test_invalid_pagination_is_rejected(params: dict) -> None:
    response = client.get(f"/documents/{uuid4()}/chunks", params=params)

    assert response.status_code == 422
    assert not config.CHUNK_DIRECTORY.exists()


def test_invalid_document_id_is_rejected() -> None:
    assert client.get("/documents/not-a-uuid/chunks").status_code == 422


def test_unknown_document_returns_not_found() -> None:
    response = client.get(f"/documents/{uuid4()}/chunks")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}
    assert not config.CHUNK_DIRECTORY.exists()
    assert not config.STORAGE_DIRECTORY.exists()


def test_saved_pdf_without_chunks_returns_conflict() -> None:
    document_id = uuid4()
    config.STORAGE_DIRECTORY.mkdir()
    pdf = config.STORAGE_DIRECTORY / f"{document_id}.pdf"
    pdf.write_bytes(b"saved original")

    response = client.get(f"/documents/{document_id}/chunks")

    assert response.status_code == 409
    assert response.json() == {"detail": "Document chunks are not available"}
    assert pdf.read_bytes() == b"saved original"


@pytest.mark.parametrize("case", ["invalid_json", "invalid_utf8", "wrong_id"])
def test_corrupt_chunks_return_safe_server_error(case: str) -> None:
    document_id = uuid4()
    if case == "invalid_json":
        contents = b"private invalid JSON"
    elif case == "invalid_utf8":
        contents = b"\xff\xfe"
    else:
        contents = (
            DocumentChunks(document_id=uuid4(), chunk_size=5, overlap=2, chunks=[])
            .model_dump_json()
            .encode()
        )
    config.CHUNK_DIRECTORY.mkdir()
    source = config.CHUNK_DIRECTORY / f"{document_id}.json"
    source.write_bytes(contents)

    response = client.get(f"/documents/{document_id}/chunks")

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load chunks"}
    assert source.read_bytes() == contents


def test_chunk_read_permission_error_returns_safe_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_read(self, *args, **kwargs):
        raise PermissionError("Private server path")

    monkeypatch.setattr(Path, "read_text", fail_read)

    response = client.get(f"/documents/{uuid4()}/chunks")

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load chunks"}


def test_missing_chunks_existence_check_is_offloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_stat(self):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        raise PermissionError("Private server path")

    monkeypatch.setattr(Path, "is_file", fail_stat)

    response = client.get(f"/documents/{uuid4()}/chunks")

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load chunks"}


def test_upload_then_fetch_chunks_preserves_source_text(prepare_document) -> None:
    contents = SAMPLE_PDF.read_bytes()
    document_id = prepare_document(contents)

    response = client.get(f"/documents/{document_id}/chunks", params={"limit": 100})

    assert response.status_code == 200
    record = asyncio.run(storage.load_chunks(document_id, config.CHUNK_DIRECTORY))
    extraction = asyncio.run(
        storage.load_extraction(document_id, config.EXTRACTION_DIRECTORY)
    )
    assert response.json()["total_chunks"] == len(record.chunks)
    assert response.json()["chunks"] == [
        chunk.model_dump(mode="json") for chunk in record.chunks
    ]
    assert record.chunks
    for chunk in record.chunks:
        page = extraction.pages[chunk.page_number - 1]
        assert chunk.text == page.text[chunk.start_char : chunk.end_char]
    assert (config.STORAGE_DIRECTORY / f"{document_id}.pdf").read_bytes() == contents


def test_blank_upload_saves_empty_chunks_and_fetches_successfully(
    prepare_document,
) -> None:
    with BytesIO() as output, PdfWriter() as writer:
        writer.add_blank_page(width=595, height=842)
        writer.write(output)
        contents = output.getvalue()
    document_id = prepare_document(contents)
    response = client.get(f"/documents/{document_id}/chunks")
    assert response.status_code == 200
    assert response.json()["total_chunks"] == 0
    assert response.json()["chunks"] == []


def test_failed_extraction_does_not_create_chunks(prepare_document) -> None:
    document_id = prepare_document(b"%PDF-1.4\nbroken")
    response = client.get(f"/documents/{document_id}/chunks")
    assert response.status_code == 409
    assert not config.CHUNK_DIRECTORY.exists()


def test_chunk_storage_failure_retains_pdf_and_extraction(
    chunk_directory: Path,
    prepare_document,
) -> None:
    from nexusrag.services.ingestion import ChunkStorageError

    chunk_directory.write_bytes(b"existing data")
    contents = SAMPLE_PDF.read_bytes()
    with pytest.raises(ChunkStorageError) as caught:
        prepare_document(contents)
    document_id = caught.value.document_id
    assert (config.STORAGE_DIRECTORY / f"{document_id}.pdf").read_bytes() == contents
    assert (
        asyncio.run(
            storage.load_extraction(document_id, config.EXTRACTION_DIRECTORY)
        ).status
        == "completed"
    )
    assert chunk_directory.read_bytes() == b"existing data"
