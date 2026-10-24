"""RQ entry point for processing an already stored document.

The worker must use JSON serialization and the same storage configuration as
our API. Automatic retries remain disabled until retry/recovery states exist.
"""

import asyncio
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from nexusrag import config
from nexusrag.constants import DEFAULT_CHUNK_OVERLAP_CHARS, DEFAULT_CHUNK_SIZE_CHARS
from nexusrag.schemas.chunk import DocumentChunks
from nexusrag.schemas.extraction import DocumentExtraction
from nexusrag.services.chunking import chunk_document
from nexusrag.services.embedding_runtime import EmbeddingRuntime
from nexusrag.services.extraction import extract_pages
from nexusrag.services.indexing import build_semantic_index
from nexusrag.services.job_storage import load_job, transition_job
from nexusrag.storage import save_chunks, save_extraction


class JobNotClaimedError(RuntimeError):
    """Execution did not claim the job and must not change its status."""


def process_document(job_id: str) -> None:
    asyncio.run(_process_document(UUID(job_id)))


async def _process_document(job_id: UUID) -> None:
    try:
        job = await load_job(job_id, config.JOB_DATABASE_PATH)
        await transition_job(job_id, "running", config.JOB_DATABASE_PATH)
    except Exception as error:
        # RQ's exception hook must not fail another execution after claim rejection.
        raise JobNotClaimedError("Could not claim processing job") from error

    try:
        pdf_path = config.STORAGE_DIRECTORY / f"{job.document_id}.pdf"
        pages = await run_in_threadpool(extract_pages, pdf_path)
        extraction = DocumentExtraction(document_id=job.document_id, pages=pages)
        await save_extraction(extraction, config.EXTRACTION_DIRECTORY)
        chunks = DocumentChunks(
            document_id=job.document_id,
            chunk_size=DEFAULT_CHUNK_SIZE_CHARS,
            overlap=DEFAULT_CHUNK_OVERLAP_CHARS,
            chunks=await run_in_threadpool(chunk_document, extraction),
        )
        await save_chunks(chunks, config.CHUNK_DIRECTORY)

        # Create locks within this invocation's event loop. RQ's default worker
        # isolates jobs in child processes, so this does not cache across jobs.
        runtime = EmbeddingRuntime()
        async with runtime.use_embedder() as embedder:
            await build_semantic_index(
                job.document_id,
                embedder,
                config.CHUNK_DIRECTORY,
                config.EMBEDDING_DIRECTORY,
            )
    except Exception:
        await transition_job(
            job_id,
            "failed",
            config.JOB_DATABASE_PATH,
            error="Document processing failed",
        )
        raise
    else:
        # Persistence failure here is not a document-processing failure.
        await transition_job(job_id, "succeeded", config.JOB_DATABASE_PATH)
