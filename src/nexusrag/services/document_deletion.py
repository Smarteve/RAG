"""Resumable deletion across SQLite, Qdrant, and local document artifacts."""

import sqlite3
from pathlib import Path
from typing import Protocol
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from nexusrag.services.document_state import deletion_status
from nexusrag.services.job_storage import _document_jobs
from nexusrag.services.processing_lock import document_lock


class DocumentProcessingError(RuntimeError):
    """Queued or running work must finish before deleting its document."""


class VectorDeletionError(RuntimeError):
    """Vector cleanup was not confirmed; deletion remains pending."""


class VectorDeletionBackend(Protocol):
    async def delete_document(self, document_id: UUID) -> None: ...


async def delete_document(
    document_id: UUID,
    *,
    database_path: Path,
    storage_directory: Path,
    extraction_directory: Path,
    chunk_directory: Path,
    embedding_directory: Path,
    backend: VectorDeletionBackend,
) -> None:
    """Retire an ID, remove all its vectors/files, and retain job history.

    The document lock excludes processing, reindexing, and retries. Durable
    intent is committed before external cleanup: a crash or error leaves the
    document hidden and prevents resurrection. Repeating DELETE resumes cleanup;
    already-deleted and unknown IDs succeed idempotently. Reads already in flight
    may finish. Shared model files, lock files, and other documents are preserved.
    """
    guard = document_lock(document_id, database_path)
    await run_in_threadpool(guard.__enter__)
    try:
        complete = await run_in_threadpool(_begin_deletion, document_id, database_path)
        if complete:
            return
        try:
            await backend.delete_document(document_id)
        except Exception as error:
            raise VectorDeletionError("Could not confirm vector deletion") from error
        paths = (
            storage_directory / f"{document_id}.pdf",
            extraction_directory / f"{document_id}.json",
            chunk_directory / f"{document_id}.json",
            embedding_directory / f"{document_id}.json",
        )
        await run_in_threadpool(_remove_files, paths)
        await run_in_threadpool(_complete_deletion, document_id, database_path)
    finally:
        await run_in_threadpool(guard.__exit__, None, None, None)


def _begin_deletion(document_id: UUID, database_path: Path) -> bool:
    connection = sqlite3.connect(database_path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if any(
                job.status in {"queued", "running"}
                for job in _document_jobs(connection, document_id)
            ):
                raise DocumentProcessingError("Document is queued or running")
            status = deletion_status(connection, document_id)
            if status is None:
                connection.execute(
                    "INSERT INTO document_deletions (document_id, status) "
                    "VALUES (?, 'deleting')",
                    (str(document_id),),
                )
        return status == "deleted"
    finally:
        connection.close()


def _remove_files(paths: tuple[Path, ...]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        # Interrupted atomic writes may leave document-scoped temporary files.
        # Ownership is still held, so no application writer can be using these.
        try:
            for temporary in path.parent.iterdir():
                if temporary.name.startswith(
                    f".{path.stem}."
                ) and temporary.name.endswith(".tmp"):
                    temporary.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


def _complete_deletion(document_id: UUID, database_path: Path) -> None:
    connection = sqlite3.connect(database_path)
    try:
        with connection:
            connection.execute(
                "UPDATE document_deletions SET status = 'deleted' WHERE document_id = ?",
                (str(document_id),),
            )
    finally:
        connection.close()
