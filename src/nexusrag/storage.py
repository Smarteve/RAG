import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO
from uuid import UUID, uuid4

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool

from nexusrag.constants import CHUNK_SIZE_BYTES
from nexusrag.schemas.chunk import DocumentChunks
from nexusrag.schemas.embedding import DocumentEmbeddings
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
    return _save_json(extraction, extraction_directory, "extraction")


async def save_chunks(record: DocumentChunks, chunk_directory: Path) -> Path:
    """Atomically publish chunk JSON in a worker thread; propagate storage errors."""
    return await run_in_threadpool(_save_chunks, record, chunk_directory)


def _save_chunks(record: DocumentChunks, chunk_directory: Path) -> Path:
    return _save_json(record, chunk_directory, "chunks")


async def save_embeddings(
    record: DocumentEmbeddings, embedding_directory: Path
) -> Path:
    """Publish complete vector JSON in a worker thread; propagate storage errors.

    A failed write leaves any prior record intact. This does not modify source
    PDFs, extracted pages, or chunks, and does not load an embedding model.
    """
    return await run_in_threadpool(_save_embeddings, record, embedding_directory)


def _save_embeddings(record: DocumentEmbeddings, embedding_directory: Path) -> Path:
    return _save_json(record, embedding_directory, "embeddings")


def _save_json(
    record: DocumentExtraction | DocumentChunks | DocumentEmbeddings,
    directory: Path,
    kind: str,
) -> Path:
    destination = directory / f"{record.document_id}.json"
    json_text = record.model_dump_json(indent=2)
    directory.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        # Keep the temporary file on the same filesystem as the final file.
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{record.document_id}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(json_text)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            _remove_temporary_file(temporary_path, kind)

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


async def load_chunks(document_id: UUID, chunk_directory: Path) -> DocumentChunks:
    """Load and validate saved chunks in a worker thread without changing storage."""
    return await run_in_threadpool(_load_chunks, document_id, chunk_directory)


def _load_chunks(document_id: UUID, chunk_directory: Path) -> DocumentChunks:
    source = chunk_directory / f"{document_id}.json"
    record = DocumentChunks.model_validate_json(source.read_text(encoding="utf-8"))
    if record.document_id != document_id:
        raise ValueError("Stored chunks document ID does not match requested ID")
    return record


async def load_embeddings(
    document_id: UUID, embedding_directory: Path
) -> DocumentEmbeddings:
    """Load and validate vector JSON and its document ID in a worker thread.

    Missing files, filesystem errors, corrupt records, and ID mismatches
    propagate. Model compatibility and source freshness belong to the caller.
    """
    return await run_in_threadpool(_load_embeddings, document_id, embedding_directory)


def _load_embeddings(
    document_id: UUID, embedding_directory: Path
) -> DocumentEmbeddings:
    source = embedding_directory / f"{document_id}.json"
    record = DocumentEmbeddings.model_validate_json(source.read_text(encoding="utf-8"))
    if record.document_id != document_id:
        raise ValueError("Stored embeddings document ID does not match requested ID")
    return record
