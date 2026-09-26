from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, ValidationError

from nexusrag.schemas.document import (
    MAX_DOCUMENT_SIZE_BYTES,
    DocumentMetadata,
)

CHUNK_SIZE_BYTES = 1024 * 1024
PDF_SIGNATURE = b"%PDF-"

app = FastAPI(title="NexusRAG")


class HealthCheckResponse(BaseModel):
    status: str = "ok"


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok")


@app.post("/documents", response_model=DocumentMetadata)
async def upload_document(file: UploadFile) -> DocumentMetadata:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail="Only PDF uploads are supported",
        )

    if not file.filename:
        raise HTTPException(
            status_code=422,
            detail="A filename is required",
        )

    try:
        # This filters obvious non-PDFs; it does not fully parse a PDF.
        header = await file.read(len(PDF_SIGNATURE))
        if header != PDF_SIGNATURE:
            raise HTTPException(
                status_code=415,
                detail="Uploaded content is not a valid PDF",
            )

        # Count the header bytes as part of the complete upload below.
        await file.seek(0)
        size_bytes = 0

        while True:
            chunk = await file.read(CHUNK_SIZE_BYTES)

            if not chunk:
                break

            size_bytes += len(chunk)

            if size_bytes > MAX_DOCUMENT_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="File must not exceed 25 MiB",
                )

        return DocumentMetadata(
            filename=file.filename,
            media_type=file.content_type,
            size_bytes=size_bytes,
        )
    except ValidationError as error:
        raise HTTPException(
            status_code=422,
            detail="Invalid document metadata",
        ) from error
    finally:
        await file.close()
