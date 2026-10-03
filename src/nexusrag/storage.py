import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO
from uuid import UUID, uuid4

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool

from nexusrag.constants import CHUNK_SIZE_BYTES
from nexusrag.schemas.extraction import DocumentExtraction

logger = logging.getLogger(__name__)


async def save_document(
    file: UploadFile,
    storage_directory: Path,
) -> tuple[UUID, Path]:
    """Save a validated upload from the start; the caller owns closing it.

    Returns the generated ID and path. The final path appears only after copying
    finishes, without replacing an existing PDF. Requires hard-link support on
    the storage filesystem. Filesystem failures propagate as OSError.
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
    temporary_path = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=storage_directory,
            prefix=f".{document_id}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            while True:
                chunk = source.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                output.write(chunk)

        # A hard link publishes the completed bytes without copying them again.
        # Unlike replace(), it fails if the destination already exists. Checking
        # exists() before replace() would leave a race between check and write.
        destination.hardlink_to(temporary_path)
    finally:
        if temporary_path is not None:
            _remove_temporary_file(temporary_path, "document")

    return document_id, destination


async def save_extraction(
    extraction: DocumentExtraction, extraction_directory: Path
) -> Path:
    """Save UTF-8 JSON in a worker thread, replacing only with a complete result.

    An existing result for this ID is replaced only after the new file closes.
    Filesystem errors propagate as OSError; the original PDF is never modified.
    """
    return await run_in_threadpool(_save_extraction, extraction, extraction_directory)


def _save_extraction(
    extraction: DocumentExtraction, extraction_directory: Path
) -> Path:
    destination = extraction_directory / f"{extraction.document_id}.json"
    json_text = extraction.model_dump_json(indent=2)
    extraction_directory.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        # Keep the temporary file on the same filesystem as the final file.
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=extraction_directory,
            prefix=f".{extraction.document_id}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(json_text)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            _remove_temporary_file(temporary_path, "extraction")

    return destination


def _remove_temporary_file(path: Path, kind: str) -> None:
    """Best-effort cleanup: preserve the original error or a successful save."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.exception("Could not remove temporary %s %s", kind, path)


async def load_extraction(
    document_id: UUID, extraction_directory: Path
) -> DocumentExtraction:
    return await run_in_threadpool(_load_extraction, document_id, extraction_directory)


def _load_extraction(
    document_id: UUID, extraction_directory: Path
) -> DocumentExtraction:

    source = extraction_directory / f"{document_id}.json"
    json_text = source.read_text(encoding="utf-8")
    extraction = DocumentExtraction.model_validate_json(json_text)
    if extraction.document_id != document_id:
        raise ValueError("Stored extraction document ID does not match requested ID")

    return extraction
