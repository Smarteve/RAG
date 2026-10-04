import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from pydantic import ValidationError
from starlette.datastructures import Headers

from nexusrag.schemas.chunk import DocumentChunks
from nexusrag.schemas.extraction import DocumentExtraction
from nexusrag.services import ingestion

SAMPLE_PDF = Path(__file__).resolve().parents[1] / "examples/employee-handbook.pdf"


@pytest.mark.parametrize("valid_pdf", [True, False])
def test_ingestion_persists_result_and_leaves_upload_open(
    tmp_path: Path, valid_pdf: bool
) -> None:
    contents = SAMPLE_PDF.read_bytes() if valid_pdf else b"%PDF-1.4\nbroken\n"
    with BytesIO(contents) as source:
        upload = UploadFile(
            file=source,
            filename="sample.pdf",
            headers=Headers({"content-type": "application/pdf"}),
        )
        result = asyncio.run(
            ingestion.ingest_document(
                upload, tmp_path / "pdfs", tmp_path / "results", tmp_path / "chunks"
            )
        )
        assert not source.closed

    assert isinstance(result, ingestion.IngestionResult)
    assert result.metadata.filename == "sample.pdf"
    assert result.metadata.size_bytes == len(contents)
    document_id = result.extraction.document_id
    assert (tmp_path / "pdfs" / f"{document_id}.pdf").read_bytes() == contents
    saved = DocumentExtraction.model_validate_json(
        (tmp_path / "results" / f"{document_id}.json").read_bytes()
    )
    assert saved == result.extraction
    assert saved.status == ("completed" if valid_pdf else "failed")
    assert len(saved.pages) == (2 if valid_pdf else 0)
    assert saved.error == (None if valid_pdf else "pdf_read_failed")
    if valid_pdf:
        chunks = DocumentChunks.model_validate_json(
            (tmp_path / "chunks" / f"{document_id}.json").read_bytes()
        )
        assert chunks == result.chunks
        assert chunks.chunks
        for chunk in chunks.chunks:
            page = saved.pages[chunk.page_number - 1]
            assert chunk.text == page.text[chunk.start_char : chunk.end_char]
    else:
        assert result.chunks is None
        assert not (tmp_path / "chunks").exists()


@pytest.mark.parametrize(
    ("filename", "media_type", "contents", "error_type", "message"),
    [
        (
            "sample.txt",
            "text/plain",
            b"notes",
            ingestion.UnsupportedDocumentError,
            "Only PDF uploads are supported",
        ),
        (
            "sample.pdf",
            "application/pdf",
            b"not PDF",
            ingestion.UnsupportedDocumentError,
            "Uploaded content is not a valid PDF",
        ),
        (
            None,
            "application/pdf",
            b"%PDF-1.4",
            ingestion.InvalidDocumentMetadataError,
            "A filename is required",
        ),
        (
            "../sample.pdf",
            "application/pdf",
            b"%PDF-1.4",
            ingestion.InvalidDocumentMetadataError,
            "Invalid document metadata",
        ),
    ],
)
def test_rejected_upload_raises_application_error_without_saving(
    tmp_path: Path,
    filename: str | None,
    media_type: str,
    contents: bytes,
    error_type: type[Exception],
    message: str,
) -> None:
    with BytesIO(contents) as source:
        upload = UploadFile(
            file=source,
            filename=filename,
            headers=Headers({"content-type": media_type}),
        )
        with pytest.raises(error_type, match=message) as exc:
            asyncio.run(
                ingestion.ingest_document(
                    upload, tmp_path / "pdfs", tmp_path / "results", tmp_path / "chunks"
                )
            )
        assert not source.closed

    if message == "Invalid document metadata":
        assert isinstance(exc.value.__cause__, ValidationError)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failed_stage", ["pdf", "json", "chunks"])
def test_storage_errors_keep_cause_and_leave_upload_open(
    tmp_path: Path, failed_stage: str
) -> None:
    pdf_directory = tmp_path / "pdfs"
    extraction_directory = tmp_path / "results"
    chunk_directory = tmp_path / "chunks"
    blocked = {
        "pdf": pdf_directory,
        "json": extraction_directory,
        "chunks": chunk_directory,
    }[failed_stage]
    blocked.write_bytes(b"existing data")
    error_type = {
        "pdf": ingestion.DocumentStorageError,
        "json": ingestion.ExtractionStorageError,
        "chunks": ingestion.ChunkStorageError,
    }[failed_stage]
    contents = SAMPLE_PDF.read_bytes()
    with BytesIO(contents) as source:
        upload = UploadFile(
            file=source,
            filename="sample.pdf",
            headers=Headers({"content-type": "application/pdf"}),
        )
        with pytest.raises(error_type) as exc:
            asyncio.run(
                ingestion.ingest_document(
                    upload, pdf_directory, extraction_directory, chunk_directory
                )
            )
        assert not source.closed

    assert isinstance(exc.value.__cause__, OSError)
    assert blocked.read_bytes() == b"existing data"
    if failed_stage != "pdf":
        assert (pdf_directory / f"{exc.value.document_id}.pdf").read_bytes() == contents
    if failed_stage == "chunks":
        record = DocumentExtraction.model_validate_json(
            (extraction_directory / f"{exc.value.document_id}.json").read_bytes()
        )
        assert record.status == "completed"
