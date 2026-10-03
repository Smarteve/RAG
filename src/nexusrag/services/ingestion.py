"""Coordinate document ingestion without choosing HTTP response codes."""

import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from nexusrag.constants import CHUNK_SIZE_BYTES
from nexusrag.schemas.document import MAX_DOCUMENT_SIZE_BYTES, DocumentMetadata
from nexusrag.schemas.extraction import DocumentExtraction
from nexusrag.services.extraction import DocumentExtractionError, extract_pages
from nexusrag.storage import save_document, save_extraction

logger = logging.getLogger(__name__)
PDF_SIGNATURE = b"%PDF-"


class UnsupportedDocumentError(Exception):
    """The upload's media type or signature is unsupported."""


class InvalidDocumentMetadataError(Exception):
    """The upload has missing or invalid metadata."""


class UploadTooLargeError(Exception):
    """The upload exceeds the configured size limit."""


class DocumentStorageError(Exception):
    """The original upload could not be saved."""


class ExtractionStorageError(Exception):
    """Extraction JSON could not be saved, but the original PDF was retained."""

    def __init__(self, document_id: UUID) -> None:
        super().__init__("Could not save extraction result")
        self.document_id = document_id


@dataclass(frozen=True)
class IngestionResult:
    """Internal workflow result; the API builds its own response from this."""

    metadata: DocumentMetadata
    extraction: DocumentExtraction


async def ingest_document(
    file: UploadFile,
    storage_directory: Path,
    extraction_directory: Path,
) -> IngestionResult:
    """Validate, store, extract, and persist a result; the caller closes the upload."""
    metadata = await _validate_pdf_upload(file)

    try:
        # Storage rewinds after validation and finishes copying before returning.
        document_id, pdf_path = await save_document(file, storage_directory)
    except OSError as error:
        logger.exception("Failed to save uploaded document")
        raise DocumentStorageError("Could not save document") from error

    extraction = await _extract_document(document_id, pdf_path)

    try:
        # Persist known extraction failures too, without deleting the original.
        await save_extraction(extraction, extraction_directory)
    except OSError as error:
        logger.exception("Failed to save extraction for %s", document_id)
        raise ExtractionStorageError(document_id) from error

    return IngestionResult(metadata=metadata, extraction=extraction)


async def _validate_pdf_upload(file: UploadFile) -> DocumentMetadata:
    """Validate without saving or closing the upload; leave the cursor at EOF."""
    if file.content_type != "application/pdf":
        raise UnsupportedDocumentError("Only PDF uploads are supported")

    if not file.filename:
        raise InvalidDocumentMetadataError("A filename is required")

    header = await file.read(len(PDF_SIGNATURE))
    if header != PDF_SIGNATURE:
        raise UnsupportedDocumentError("Uploaded content is not a valid PDF")

    # Count the entire file, including the header just read.
    await file.seek(0)
    total_size_bytes = 0
    while True:
        chunk = await file.read(CHUNK_SIZE_BYTES)
        if not chunk:
            break
        total_size_bytes += len(chunk)
        if total_size_bytes > MAX_DOCUMENT_SIZE_BYTES:
            raise UploadTooLargeError("File must not exceed 25 MiB")

    try:
        return DocumentMetadata(
            filename=file.filename,
            media_type=file.content_type,
            size_bytes=total_size_bytes,
        )
    except ValidationError as error:
        raise InvalidDocumentMetadataError("Invalid document metadata") from error


async def _extract_document(document_id: UUID, pdf_path: Path) -> DocumentExtraction:
    """Return successful pages or a record of a known extraction failure."""
    try:
        pages = await run_in_threadpool(extract_pages, pdf_path)
        return DocumentExtraction(document_id=document_id, pages=pages)
    except DocumentExtractionError:
        logger.exception("Could not parse document %s", document_id)
        return DocumentExtraction(
            document_id=document_id,
            pages=[],
            status="failed",
            error="pdf_read_failed",
        )
    except OSError:
        logger.exception("Could not read stored document %s", document_id)
        return DocumentExtraction(
            document_id=document_id,
            pages=[],
            status="failed",
            error="source_read_failed",
        )
