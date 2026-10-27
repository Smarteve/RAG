"""Durable deletion intent, shared by reads and all job producers."""

import sqlite3
from pathlib import Path
from uuid import UUID

from fastapi.concurrency import run_in_threadpool


def deletion_status(connection: sqlite3.Connection, document_id: UUID) -> str | None:
    row = connection.execute(
        "SELECT status FROM document_deletions WHERE document_id = ?",
        (str(document_id),),
    ).fetchone()
    return row[0] if row else None


async def is_document_deleted(document_id: UUID, database_path: Path) -> bool:
    """Hide documents as soon as deletion starts, including incomplete cleanup."""
    return await run_in_threadpool(_is_document_deleted, document_id, database_path)


def _is_document_deleted(document_id: UUID, database_path: Path) -> bool:
    connection = sqlite3.connect(database_path)
    try:
        return deletion_status(connection, document_id) is not None
    finally:
        connection.close()
