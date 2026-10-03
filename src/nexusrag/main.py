from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from nexusrag import config
from nexusrag.schemas.document import UploadDocumentResponse
from nexusrag.services import ingestion

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
