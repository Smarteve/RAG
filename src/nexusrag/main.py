import logging
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi import Path as PathParameter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from nexusrag import config
from nexusrag.schemas.document import UploadDocumentResponse
from nexusrag.schemas.extraction import FetchPageResponse
from nexusrag.services import ingestion
from nexusrag.storage import load_extraction

logger = logging.getLogger(__name__)
app = FastAPI(title="NexusRAG")


class HealthCheckResponse(BaseModel):
    status: str = "ok"


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok")


@app.post("/documents", response_model=UploadDocumentResponse, status_code=201)
async def upload_document(file: UploadFile) -> UploadDocumentResponse:
    """Translate ingestion results/errors into HTTP and close the owned upload."""
    try:
        result = await ingestion.ingest_document(
            file, config.STORAGE_DIRECTORY, config.EXTRACTION_DIRECTORY
        )
        extraction = result.extraction

        return UploadDocumentResponse(
            document_id=extraction.document_id,
            metadata=result.metadata,
            extraction_status=extraction.status,
            page_count=len(extraction.pages)
            if extraction.status == "completed"
            else None,
            extraction_error=extraction.error,
        )
    except ingestion.UnsupportedDocumentError as error:
        raise HTTPException(415, detail=str(error)) from error
    except ingestion.InvalidDocumentMetadataError as error:
        raise HTTPException(422, detail=str(error)) from error
    except ingestion.UploadTooLargeError as error:
        raise HTTPException(413, detail=str(error)) from error
    except ingestion.DocumentStorageError as error:
        raise HTTPException(500, detail=str(error)) from error
    except ingestion.ExtractionStorageError as error:
        raise HTTPException(
            500,
            detail={
                "message": str(error),
                "document_id": str(error.document_id),
            },
        ) from error
    finally:
        await file.close()


@app.get(
    "/documents/{document_id}/pages/{page_number}",
    response_model=FetchPageResponse,
)
async def fetch_page(
    document_id: UUID,
    page_number: Annotated[int, PathParameter(ge=1)],
) -> FetchPageResponse:
    """Read one saved page using its original, one-based PDF page number."""
    try:
        document = await load_extraction(document_id, config.EXTRACTION_DIRECTORY)
    except FileNotFoundError as error:
        # A PDF may have been saved before JSON storage failed or was interrupted.
        # Check in a worker thread because even a filesystem stat is synchronous.
        pdf_path = config.STORAGE_DIRECTORY / f"{document_id}.pdf"
        try:
            pdf_exists = await run_in_threadpool(pdf_path.is_file)
        except OSError as storage_error:
            logger.exception("Could not check stored document %s", document_id)
            raise HTTPException(
                500, detail="Could not load document"
            ) from storage_error
        if pdf_exists:
            raise HTTPException(
                409, detail="Document extraction result is not available"
            ) from error
        raise HTTPException(404, detail="Document not found") from error
    except (ValueError, OSError) as error:
        # ValidationError and UnicodeDecodeError are ValueError subclasses.
        # Corrupt stored data is a server error; don't expose internal details.
        logger.exception("Could not load extraction for %s", document_id)
        raise HTTPException(500, detail="Could not load document") from error

    if document.status == "failed":
        raise HTTPException(409, detail="Document extraction failed")

    # Use the stored page number rather than assuming list order matches it.
    for page in document.pages:
        if page.page_number == page_number:
            return FetchPageResponse(document_id=document_id, page=page)

    raise HTTPException(404, detail="Requested page does not exist")
