import sqlite3
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from nexusrag import config
from nexusrag.services.embedding_runtime import EmbeddingRuntime
from nexusrag.services.generation_runtime import AnswerGenerator, GroqAnswerGenerator
from nexusrag.services.job_storage import load_document_job
from nexusrag.services.processing_lock import DocumentBusyError, document_lock
from nexusrag.services.vector_backend import QdrantBackend


async def get_embedding_runtime(request: Request) -> EmbeddingRuntime:
    """Keep one lazy model runtime on this app; tests can override the dependency."""
    runtime = getattr(request.app.state, "embedding_runtime", None)
    if runtime is None:
        # No await between lookup and assignment: requests cannot interleave here.
        runtime = EmbeddingRuntime()
        request.app.state.embedding_runtime = runtime
    return runtime


async def get_answer_generator(request: Request) -> AnswerGenerator:
    """Reuse a lightweight adapter, not credentials or a live network client."""
    generator = getattr(request.app.state, "answer_generator", None)
    if generator is None:
        generator = GroqAnswerGenerator()
        request.app.state.answer_generator = generator
    return generator


async def require_reindex_lock(document_id: UUID) -> AsyncIterator[None]:
    """Serialize manual reindexing with queued processing and recovery."""
    guard = document_lock(document_id, config.JOB_DATABASE_PATH)
    try:
        await run_in_threadpool(guard.__enter__)
    except DocumentBusyError as error:
        raise HTTPException(
            409, detail="Document processing or reindexing is active"
        ) from error
    except OSError as error:
        raise HTTPException(500, detail="Could not lock document") from error
    try:
        try:
            job = await load_document_job(document_id, config.JOB_DATABASE_PATH)
        except (sqlite3.Error, OSError, ValueError) as error:
            raise HTTPException(
                500, detail="Could not check document readiness"
            ) from error
        if job is not None and job.status != "succeeded":
            raise HTTPException(
                409, detail="Document must finish processing before reindexing"
            )
        yield
    finally:
        await run_in_threadpool(guard.__exit__, None, None, None)


async def get_vector_backend() -> AsyncIterator[QdrantBackend]:
    """Own one asynchronous Qdrant client for this request's event loop."""
    backend = QdrantBackend()
    try:
        yield backend
    finally:
        await backend.aclose()
