import logging

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, ValidationError

from nexusrag import config
from nexusrag.constants import CHUNK_SIZE_BYTES
from nexusrag.schemas.document import (
    MAX_DOCUMENT_SIZE_BYTES,
    DocumentMetadata,
    UploadDocumentResponse,
)
from nexusrag.storage import save_document

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

        # A signature check filters obvious non-PDFs; full parsing comes later.
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

        # Construct after reading every chunk, but inside the validation handler.
        metadata = DocumentMetadata(
            filename=file.filename,
            media_type=file.content_type,
            size_bytes=total_size_bytes,
        )
        # Storage rewinds the upload and completes the copy before returning.
        try:
            document_id, _ = await save_document(file, config.STORAGE_DIRECTORY)
        except OSError as error:
            logger.exception("Failed to save uploaded document")
            raise HTTPException(500, detail="Could not save document") from error

        return UploadDocumentResponse(document_id=document_id, metadata=metadata)
    except ValidationError as error:
        raise HTTPException(422, detail="Invalid document metadata") from error
    finally:
        await file.close()
