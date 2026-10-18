import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi import Path as PathParameter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from nexusrag import config
from nexusrag.database import initialize_database
from nexusrag.dependencies import get_answer_generator, get_embedding_runtime
from nexusrag.schemas.answer import AskDocumentRequest, AskDocumentResponse
from nexusrag.schemas.chunk import FetchChunksResponse
from nexusrag.schemas.document import QueuedUploadResponse
from nexusrag.schemas.embedding import SemanticIndexResponse
from nexusrag.schemas.extraction import FetchPageResponse
from nexusrag.schemas.job import ProcessingJob
from nexusrag.schemas.retrieval import SearchResponse, SemanticSearchResponse
from nexusrag.services import answering, indexing, ingestion, retrieval, search
from nexusrag.services.embedding_model import EmbeddingInputError
from nexusrag.services.embedding_runtime import (
    EmbeddingModelUnavailableError,
    EmbeddingRuntime,
)
from nexusrag.services.generation import InvalidGeneratedAnswerError
from nexusrag.services.generation_runtime import (
    AnswerGenerator,
    GenerationProviderError,
    GenerationTimeoutError,
    GenerationUnavailableError,
)
from nexusrag.services.job_storage import load_job
from nexusrag.storage import load_chunks, load_embeddings, load_extraction

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(initialize_database, config.JOB_DATABASE_PATH)
    yield


app = FastAPI(title="NexusRAG", lifespan=lifespan)


class HealthCheckResponse(BaseModel):
    status: str = "ok"


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok")


@app.get("/jobs/{job_id}", response_model=ProcessingJob)
async def fetch_job(job_id: UUID) -> ProcessingJob:
    """Return the persisted status of a document-processing job."""
    try:
        return await load_job(job_id, config.JOB_DATABASE_PATH)
    except FileNotFoundError as error:
        raise HTTPException(404, detail="Job not found") from error
    except (sqlite3.Error, OSError, ValueError) as error:
        logger.exception("Could not load processing job %s", job_id)
        raise HTTPException(500, detail="Could not load job") from error


@app.post("/documents", response_model=QueuedUploadResponse, status_code=202)
async def upload_document(file: UploadFile) -> QueuedUploadResponse:
    """Accept a stored document for background processing; close the upload."""
    try:
        return await ingestion.submit_document(
            file, config.STORAGE_DIRECTORY, config.JOB_DATABASE_PATH
        )
    except ingestion.UnsupportedDocumentError as error:
        raise HTTPException(415, detail=str(error)) from error
    except ingestion.InvalidDocumentMetadataError as error:
        raise HTTPException(422, detail=str(error)) from error
    except ingestion.UploadTooLargeError as error:
        raise HTTPException(413, detail=str(error)) from error
    except (ingestion.DocumentStorageError, ingestion.JobStorageError) as error:
        logger.exception("Could not persist document submission")
        raise HTTPException(500, detail=str(error)) from error
    except ingestion.QueueSubmissionError as error:
        logger.exception("Could not confirm document queue acceptance")
        raise HTTPException(503, detail=str(error)) from error
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


@app.post(
    "/documents/{document_id}/semantic-index",
    response_model=SemanticIndexResponse,
    status_code=200,
)
async def build_document_semantic_index(
    document_id: UUID,
    runtime: Annotated[EmbeddingRuntime, Depends(get_embedding_runtime)],
) -> SemanticIndexResponse:
    """Create or rebuild saved embeddings without returning vectors or paths."""
    # Validate source availability before loading an expensive optional model.
    # The service rereads chunks after acquiring the model lock, so a queued
    # rebuild uses the current snapshot rather than this preflight snapshot.
    try:
        await load_chunks(document_id, config.CHUNK_DIRECTORY)
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
        logger.exception("Could not load chunks for indexing of %s", document_id)
        raise HTTPException(500, detail="Could not load chunks") from error

    try:
        async with runtime.use_embedder() as embedder:
            record = await indexing.build_semantic_index(
                document_id,
                embedder,
                config.CHUNK_DIRECTORY,
                config.EMBEDDING_DIRECTORY,
            )
            return SemanticIndexResponse(
                document_id=record.document_id,
                model_id=record.model_id,
                model_revision=record.model_revision,
                dimensions=record.dimensions,
                embedding_count=len(record.embeddings),
            )
    except EmbeddingModelUnavailableError as error:
        logger.exception("Could not load embedding model for %s", document_id)
        raise HTTPException(
            503,
            detail="Local embedding model unavailable; install the embeddings extra "
            "and cache the configured model before indexing",
        ) from error
    except EmbeddingInputError as error:
        raise HTTPException(
            409,
            detail="Document chunks contain blank or oversized text; "
            "rechunk before indexing",
        ) from error
    except Exception as error:
        # Internal schema/model bugs and filesystem failures are not user input.
        logger.exception("Could not build semantic index for %s", document_id)
        raise HTTPException(500, detail="Could not build semantic index") from error


@app.get(
    "/documents/{document_id}/semantic-search",
    response_model=SemanticSearchResponse,
)
async def semantic_search_document(
    document_id: UUID,
    query: Annotated[str, Query(min_length=1, max_length=500)],
    runtime: Annotated[EmbeddingRuntime, Depends(get_embedding_runtime)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> SemanticSearchResponse:
    """Embed only the query and rank an existing, compatible document index."""
    # Semantic inputs need not follow the keyword tokenizer's word rules.
    if not query.strip():
        raise HTTPException(422, detail="query must not be blank")

    try:
        chunk_record = await load_chunks(document_id, config.CHUNK_DIRECTORY)
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
        logger.exception("Could not load chunks for semantic search of %s", document_id)
        raise HTTPException(500, detail="Could not load chunks") from error

    # Chunks establish that the document exists. A missing index is a separate
    # state conflict, not an unknown document or a reason to embed it in a GET.
    try:
        embedding_record = await load_embeddings(
            document_id, config.EMBEDDING_DIRECTORY
        )
    except FileNotFoundError as error:
        raise HTTPException(
            409, detail="Semantic index is not available; build it before searching"
        ) from error
    except (ValueError, OSError) as error:
        logger.exception("Could not load semantic index for %s", document_id)
        raise HTTPException(500, detail="Could not load semantic index") from error

    try:
        hits = await search.semantic_search(
            chunk_record, embedding_record, query, runtime, limit=limit
        )
        return SemanticSearchResponse(document_id=document_id, query=query, hits=hits)
    except retrieval.StaleSemanticIndexError as error:
        raise HTTPException(
            409, detail="Semantic index is stale or incompatible; rebuild it"
        ) from error
    except EmbeddingInputError as error:
        raise HTTPException(422, detail=str(error)) from error
    except EmbeddingModelUnavailableError as error:
        logger.exception("Could not load embedding model for search of %s", document_id)
        raise HTTPException(
            503,
            detail="Local embedding model unavailable; install the embeddings extra "
            "and cache the configured model before searching",
        ) from error
    except Exception as error:
        # Provider/schema failures are internal errors, not bad user input.
        logger.exception("Could not semantically search document %s", document_id)
        raise HTTPException(500, detail="Could not search document") from error


@app.post("/documents/{document_id}/ask", response_model=AskDocumentResponse)
async def ask_document(
    document_id: UUID,
    request: AskDocumentRequest,
    runtime: Annotated[EmbeddingRuntime, Depends(get_embedding_runtime)],
    generator: Annotated[AnswerGenerator, Depends(get_answer_generator)],
) -> AskDocumentResponse:
    """Answer one question with server-resolved citations to existing PDF chunks."""
    try:
        return await answering.answer_document(
            document_id,
            request.question,
            runtime,
            generator,
            storage_directory=config.STORAGE_DIRECTORY,
            chunk_directory=config.CHUNK_DIRECTORY,
            embedding_directory=config.EMBEDDING_DIRECTORY,
        )
    except answering.DocumentNotFoundError as error:
        raise HTTPException(404, detail="Document not found") from error
    except answering.DocumentChunksUnavailableError as error:
        raise HTTPException(409, detail="Document chunks are not available") from error
    except answering.SemanticIndexUnavailableError as error:
        raise HTTPException(
            409, detail="Semantic index is not available; build it before asking"
        ) from error
    except retrieval.StaleSemanticIndexError as error:
        raise HTTPException(
            409, detail="Semantic index is stale or incompatible; rebuild it"
        ) from error
    except EmbeddingInputError as error:
        raise HTTPException(422, detail=str(error)) from error
    except EmbeddingModelUnavailableError as error:
        raise HTTPException(
            503,
            detail="Local embedding model unavailable; install the embeddings extra and cache the configured model",
        ) from error
    except GenerationUnavailableError as error:
        logger.warning("Generation unavailable for document %s", document_id)
        raise HTTPException(
            503,
            detail="Generation unavailable; check the generation extra, API key, model access, and quota",
        ) from error
    except GenerationTimeoutError as error:
        raise HTTPException(504, detail="Answer generation timed out") from error
    except (GenerationProviderError, InvalidGeneratedAnswerError) as error:
        # Provider exceptions can contain prompts/response bodies; don't log them.
        logger.warning("Generation failed for document %s", document_id)
        raise HTTPException(502, detail="Could not generate a valid answer") from error
    except Exception as error:
        logger.error(
            "Could not answer document %s (%s)", document_id, type(error).__name__
        )
        raise HTTPException(500, detail="Could not answer document") from error
