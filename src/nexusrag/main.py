import logging
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ValidationError

from nexusrag import config
from nexusrag.constants import CHUNK_SIZE_BYTES
from nexusrag.schemas.document import (
    MAX_DOCUMENT_SIZE_BYTES,
    DocumentMetadata,
    UploadDocumentResponse,
)
from nexusrag.schemas.extraction import DocumentExtraction
from nexusrag.services.extraction import DocumentExtractionError, extract_pages
from nexusrag.storage import save_document, save_extraction

logger = logging.getLogger(__name__)
PDF_SIGNATURE = b"%PDF-"

app = FastAPI(title="NexusRAG")


class HealthCheckResponse(BaseModel):
    status: str = "ok"


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok")


@app.post("/documents", response_model=UploadDocumentResponse, status_code=201)
async def upload_document(file: UploadFile) -> UploadDocumentResponse:
    """Coordinate ingestion and always close the request-owned upload."""
    try:
        metadata = await _validate_pdf_upload(file)
        document_id, pdf_path = await _save_uploaded_pdf(file)
        extraction = await _extract_document(document_id, pdf_path)
        await _save_extraction_result(extraction)

        return UploadDocumentResponse(
            document_id=document_id,
            metadata=metadata,
            extraction_status=extraction.status,
            page_count=len(extraction.pages)
            if extraction.status == "completed"
            else None,
            extraction_error=extraction.error,
        )
    finally:
        await file.close()


async def _validate_pdf_upload(file: UploadFile) -> DocumentMetadata:
    """Validate the upload without saving or closing it; leave the cursor at EOF."""
    if file.content_type != "application/pdf":
        raise HTTPException(415, detail="Only PDF uploads are supported")

    if not file.filename:
        raise HTTPException(422, detail="A filename is required")

    # A signature check filters obvious non-PDFs before storage and parsing.
    header = await file.read(len(PDF_SIGNATURE))
    if header != PDF_SIGNATURE:
        raise HTTPException(415, detail="Uploaded content is not a valid PDF")

    # Count the entire file, including the header just read.
    await file.seek(0)
    total_size_bytes = 0
    while True:
        chunk = await file.read(CHUNK_SIZE_BYTES)
        if not chunk:
            break
        total_size_bytes += len(chunk)
        if total_size_bytes > MAX_DOCUMENT_SIZE_BYTES:
            raise HTTPException(413, detail="File must not exceed 25 MiB")

    # Only invalid upload metadata should become a client validation error.
    try:
        return DocumentMetadata(
            filename=file.filename,
            media_type=file.content_type,
            size_bytes=total_size_bytes,
        )
    except ValidationError as error:
        raise HTTPException(422, detail="Invalid document metadata") from error


async def _save_uploaded_pdf(file: UploadFile) -> tuple[UUID, Path]:
    """Store the PDF and translate storage errors into an HTTP response."""
    try:
        # Storage rewinds the upload and completes copying before returning.
        return await save_document(file, config.STORAGE_DIRECTORY)
    except OSError as error:
        logger.exception("Failed to save uploaded document")
        raise HTTPException(500, detail="Could not save document") from error


async def _extract_document(document_id: UUID, pdf_path: Path) -> DocumentExtraction:
    """Build either a successful result or a record of a known extraction failure."""
    try:
        pages = await run_in_threadpool(extract_pages, pdf_path)
        return DocumentExtraction(document_id=document_id, pages=pages)
    except DocumentExtractionError:
        logger.exception("Could not parse document %s", document_id)
        return DocumentExtraction(
            document_id=document_id,
            pages=[],
            status="failed",
            error="pdf_read_failed",
        )
    except OSError:
        logger.exception("Could not read stored document %s", document_id)
        return DocumentExtraction(
            document_id=document_id,
            pages=[],
            status="failed",
            error="source_read_failed",
        )


async def _save_extraction_result(extraction: DocumentExtraction) -> None:
    """Persist success or failure without deleting the original PDF."""
    try:
        await save_extraction(extraction, config.EXTRACTION_DIRECTORY)
    except OSError as error:
        logger.exception("Failed to save extraction for %s", extraction.document_id)
        raise HTTPException(
            500,
            detail={
                "message": "Could not save extraction result",
                "document_id": str(extraction.document_id),
            },
        ) from error
