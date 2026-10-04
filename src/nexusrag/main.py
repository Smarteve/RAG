import logging
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi import Path as PathParameter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from nexusrag import config
from nexusrag.schemas.chunk import FetchChunksResponse
from nexusrag.schemas.document import UploadDocumentResponse
from nexusrag.schemas.extraction import FetchPageResponse
from nexusrag.schemas.retrieval import SearchResponse
from nexusrag.services import ingestion, retrieval
from nexusrag.storage import load_chunks, load_extraction

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
            file,
            config.STORAGE_DIRECTORY,
            config.EXTRACTION_DIRECTORY,
            config.CHUNK_DIRECTORY,
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
            chunk_count=len(result.chunks.chunks)
            if result.chunks is not None
            else None,
        )
    except ingestion.UnsupportedDocumentError as error:
        raise HTTPException(415, detail=str(error)) from error
    except ingestion.InvalidDocumentMetadataError as error:
        raise HTTPException(422, detail=str(error)) from error
    except ingestion.UploadTooLargeError as error:
        raise HTTPException(413, detail=str(error)) from error
    except ingestion.DocumentStorageError as error:
        raise HTTPException(500, detail=str(error)) from error
    except (ingestion.ExtractionStorageError, ingestion.ChunkStorageError) as error:
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


@app.get("/documents/{document_id}/chunks", response_model=FetchChunksResponse)
async def fetch_chunks(
    document_id: UUID,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> FetchChunksResponse:
    """Return a bounded number of saved chunks; offsets are zero-based."""
    try:
        record = await load_chunks(document_id, config.CHUNK_DIRECTORY)
    except FileNotFoundError as error:
        pdf_path = config.STORAGE_DIRECTORY / f"{document_id}.pdf"
        try:
            pdf_exists = await run_in_threadpool(pdf_path.is_file)
        except OSError as storage_error:
            logger.exception("Could not check stored document %s", document_id)
            raise HTTPException(500, detail="Could not load chunks") from storage_error
        if pdf_exists:
            raise HTTPException(
                409, detail="Document chunks are not available"
            ) from error
        raise HTTPException(404, detail="Document not found") from error
    except (ValueError, OSError) as error:
        logger.exception("Could not load chunks for %s", document_id)
        raise HTTPException(500, detail="Could not load chunks") from error

    return FetchChunksResponse(
        document_id=document_id,
        total_chunks=len(record.chunks),
        offset=offset,
        limit=limit,
        chunks=record.chunks[offset : offset + limit],
    )


@app.get("/documents/{document_id}/search", response_model=SearchResponse)
async def search_document(
    document_id: UUID,
    query: Annotated[str, Query(min_length=1, max_length=500)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> SearchResponse:
    """Rank saved chunks by matching query words, returning text and page references."""
    # Validate before checking storage so a bad query consistently returns 422.
    # Catch only the explicit input error: ValidationError is also a ValueError,
    # but an invalid internally generated hit must not be treated as client input.
    try:
        retrieval.validate_query(query)
    except retrieval.InvalidSearchQueryError as error:
        raise HTTPException(422, detail=str(error)) from error

    try:
        record = await load_chunks(document_id, config.CHUNK_DIRECTORY)
    except FileNotFoundError as error:
        pdf_path = config.STORAGE_DIRECTORY / f"{document_id}.pdf"
        try:
            pdf_exists = await run_in_threadpool(pdf_path.is_file)
        except OSError as storage_error:
            logger.exception("Could not check stored document %s", document_id)
            raise HTTPException(500, detail="Could not load chunks") from storage_error
        if pdf_exists:
            raise HTTPException(
                409, detail="Document chunks are not available"
            ) from error
        raise HTTPException(404, detail="Document not found") from error
    except (ValueError, OSError) as error:
        logger.exception("Could not load chunks for search of %s", document_id)
        raise HTTPException(500, detail="Could not load chunks") from error

    try:
        hits = await run_in_threadpool(
            retrieval.search_chunks, record.chunks, query, limit
        )
    except Exception as error:
        # This HTTP boundary logs internal failures; the pure ranking function
        # continues to propagate them instead of hiding or skipping invalid hits.
        logger.exception("Could not search document %s", document_id)
        raise HTTPException(500, detail="Could not search document") from error

    return SearchResponse(document_id=document_id, query=query, hits=hits)
