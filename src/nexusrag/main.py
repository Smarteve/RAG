import logging

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
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(415, detail="Only PDF uploads are supported")

        if not file.filename:
            raise HTTPException(422, detail="A filename is required")

        # A signature check filters obvious non-PDFs before storage and parsing.
        header = await file.read(len(PDF_SIGNATURE))
        if header != PDF_SIGNATURE:
            raise HTTPException(415, detail="Uploaded content is not a valid PDF")

        # Reset the cursor so the size includes the header we just read.
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
            metadata = DocumentMetadata(
                filename=file.filename,
                media_type=file.content_type,
                size_bytes=total_size_bytes,
            )
        except ValidationError as error:
            raise HTTPException(422, detail="Invalid document metadata") from error
        # Storage rewinds the upload and completes the copy before returning.
        try:
            document_id, pdf_path = await save_document(file, config.STORAGE_DIRECTORY)
        except OSError as error:
            logger.exception("Failed to save uploaded document")
            raise HTTPException(500, detail="Could not save document") from error

        try:
            pages = await run_in_threadpool(extract_pages, pdf_path)
            extraction = DocumentExtraction(document_id=document_id, pages=pages)
        except DocumentExtractionError:
            logger.exception("Could not parse document %s", document_id)
            extraction = DocumentExtraction(
                document_id=document_id,
                pages=[],
                status="failed",
                error="pdf_read_failed",
            )
        except OSError:
            logger.exception("Could not read stored document %s", document_id)
            extraction = DocumentExtraction(
                document_id=document_id,
                pages=[],
                status="failed",
                error="source_read_failed",
            )

        # Persist both successful results and known failures. Keep the original
        # PDF in either case so it can be inspected or reprocessed later.
        try:
            await save_extraction(extraction, config.EXTRACTION_DIRECTORY)
        except OSError as error:
            logger.exception("Failed to save extraction for %s", document_id)
            raise HTTPException(
                500,
                detail={
                    "message": "Could not save extraction result",
                    "document_id": str(document_id),
                },
            ) from error

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
