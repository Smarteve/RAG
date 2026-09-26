import logging
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool

from nexusrag.constants import CHUNK_SIZE_BYTES

logger = logging.getLogger(__name__)


async def save_document(
    file: UploadFile,
    storage_directory: Path,
) -> tuple[UUID, Path]:
    """Save a validated upload from the start; the caller owns closing it.

    Returns the generated ID and path. Filesystem failures propagate as OSError.
    """
    # Both reads and writes are synchronous inside this worker thread.
    return await run_in_threadpool(_save_document, file.file, storage_directory)


def _save_document(
    source: BinaryIO,
    storage_directory: Path,
) -> tuple[UUID, Path]:
    document_id = uuid4()
    destination = storage_directory / f"{document_id}.pdf"
    source.seek(0)
    storage_directory.mkdir(parents=True, exist_ok=True)
    created = False

    try:
        with destination.open("xb") as output:
            created = True
            while True:
                chunk = source.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                output.write(chunk)

    except OSError:
        # If exclusive creation failed, an existing file must remain untouched.
        if created:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                # Preserve the original copy error even if cleanup also fails.
                logger.exception("Could not remove partial document %s", destination)
        raise

    return document_id, destination
