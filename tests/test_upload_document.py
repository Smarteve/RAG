import asyncio
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
from starlette.datastructures import Headers

from nexusrag import config, main, storage
from nexusrag.constants import CHUNK_SIZE_BYTES
from nexusrag.main import app
from nexusrag.schemas.document import MAX_DOCUMENT_SIZE_BYTES
from nexusrag.schemas.extraction import DocumentExtraction

client = TestClient(app)
SAMPLE_PDF = Path(__file__).resolve().parents[1] / "examples/employee-handbook.pdf"


@pytest.fixture(autouse=True)
def storage_directory(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / "documents"
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", directory)
    return directory


@pytest.fixture(autouse=True)
def extraction_directory(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / "extracted"
    monkeypatch.setattr(config, "EXTRACTION_DIRECTORY", directory)
    return directory


@pytest.mark.parametrize("padding", [0, CHUNK_SIZE_BYTES])
def test_pdf_upload_returns_document_metadata(
    padding: int, storage_directory: Path, extraction_directory: Path
) -> None:
    # Use a real PDF; a signature alone no longer represents successful parsing.
    with SAMPLE_PDF.open("rb") as source, BytesIO() as output, PdfWriter() as writer:
        for page in PdfReader(source).pages:
            writer.add_page(page)
        writer.add_metadata({"/Subject": "x" * padding})
        writer.write(output)
        file_contents = output.getvalue()

    response = client.post(
        "/documents",
        files={"file": ("invoice.pdf", file_contents, "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    document_id = UUID(body["document_id"])
    assert body["metadata"]["filename"] == "invoice.pdf"
    assert body["metadata"]["media_type"] == "application/pdf"
    assert body["metadata"]["size_bytes"] == len(file_contents)
    assert (storage_directory / f"{document_id}.pdf").read_bytes() == file_contents
    assert body["extraction_status"] == "completed"
    assert body["page_count"] == 2
    assert body["extraction_error"] is None
    assert set(body) == {
        "document_id",
        "metadata",
        "extraction_status",
        "page_count",
        "extraction_error",
    }
    assert set(body["metadata"]) == {
        "filename",
        "media_type",
        "size_bytes",
        "uploaded_at",
        "source_created_at",
    }
    result = DocumentExtraction.model_validate_json(
        (extraction_directory / f"{document_id}.json").read_text(encoding="utf-8")
    )
    assert result.document_id == document_id
    assert result.status == "completed"
    assert result.error is None
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "Annual leave" in result.pages[0].text
    assert "Remote work" in result.pages[1].text


def test_non_pdf_upload_is_rejected() -> None:
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"notes", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Only PDF uploads are supported"


def test_upload_larger_than_limit_is_rejected() -> None:
    response = client.post(
        "/documents",
        files={
            "file": (
                "large.pdf",
                b"%PDF-" + b"x" * (MAX_DOCUMENT_SIZE_BYTES - 4),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "File must not exceed 25 MiB"


@pytest.mark.parametrize("contents", [b"not really a PDF", b"", b"%PDF"])
def test_invalid_pdf_signature_is_rejected(contents: bytes) -> None:
    response = client.post(
        "/documents",
        files={"file": ("fake.pdf", contents, "application/pdf")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Uploaded content is not a valid PDF"


def test_invalid_filename_returns_validation_error(storage_directory: Path) -> None:
    response = client.post(
        "/documents",
        files={"file": ("../invoice.pdf", b"%PDF-1.7\ncontents", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid document metadata"
    assert not storage_directory.exists()


def test_storage_failure_returns_server_error(storage_directory: Path) -> None:
    # A file in place of a directory makes mkdir fail on every platform.
    storage_directory.write_bytes(b"existing data")
    response = client.post(
        "/documents",
        files={"file": ("invoice.pdf", b"%PDF-1.7\ncontents", "application/pdf")},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not save document"}
    assert storage_directory.read_bytes() == b"existing data"


def test_parse_failure_keeps_pdf_and_persists_failed_status(
    storage_directory: Path, extraction_directory: Path
) -> None:
    contents = b"%PDF-1.4\nbroken PDF\n"
    response = client.post(
        "/documents", files={"file": ("broken.pdf", contents, "application/pdf")}
    )

    assert response.status_code == 201
    body = response.json()
    document_id = UUID(body["document_id"])
    assert body["extraction_status"] == "failed"
    assert body["page_count"] is None
    assert body["extraction_error"] == "pdf_read_failed"
    assert (storage_directory / f"{document_id}.pdf").read_bytes() == contents
    result = DocumentExtraction.model_validate_json(
        (extraction_directory / f"{document_id}.json").read_text(encoding="utf-8")
    )
    assert result.status == "failed"
    assert result.error == "pdf_read_failed"
    assert result.pages == []


def test_unsupported_pdf_filter_keeps_pdf_and_persists_failed_status(
    storage_directory: Path, extraction_directory: Path
) -> None:
    with SAMPLE_PDF.open("rb") as source, BytesIO() as output, PdfWriter() as writer:
        # Preserve font resources so pypdf actually attempts text extraction.
        # A blank page without fonts can return empty text before decoding.
        page = writer.add_page(PdfReader(source).pages[0])
        stream = DecodedStreamObject()
        stream.set_data(b"BT ET")
        stream[NameObject("/Filter")] = NameObject("/UnsupportedDecode")
        page[NameObject("/Contents")] = writer._add_object(stream)
        writer.write(output)
        contents = output.getvalue()

    response = client.post(
        "/documents", files={"file": ("unsupported.pdf", contents, "application/pdf")}
    )

    assert response.status_code == 201
    body = response.json()
    document_id = UUID(body["document_id"])
    assert body["extraction_status"] == "failed"
    assert body["extraction_error"] == "pdf_read_failed"
    assert body["page_count"] is None
    assert "UnsupportedDecode" not in response.text
    assert (storage_directory / f"{document_id}.pdf").read_bytes() == contents
    result = DocumentExtraction.model_validate_json(
        (extraction_directory / f"{document_id}.json").read_text(encoding="utf-8")
    )
    assert result.document_id == document_id
    assert result.status == "failed"
    assert result.error == "pdf_read_failed"
    assert result.pages == []


def test_source_read_failure_is_recorded_without_exposing_internal_error(
    storage_directory: Path, extraction_directory: Path, monkeypatch
) -> None:
    def fail_read(path: Path):
        raise PermissionError("Private server path /internal/storage")

    monkeypatch.setattr(main, "extract_pages", fail_read)
    contents = SAMPLE_PDF.read_bytes()
    response = client.post(
        "/documents", files={"file": ("sample.pdf", contents, "application/pdf")}
    )

    assert response.status_code == 201
    body = response.json()
    document_id = UUID(body["document_id"])
    assert body["extraction_status"] == "failed"
    assert body["extraction_error"] == "source_read_failed"
    assert body["page_count"] is None
    assert "Private server path" not in response.text
    assert (storage_directory / f"{document_id}.pdf").read_bytes() == contents
    result = DocumentExtraction.model_validate_json(
        (extraction_directory / f"{document_id}.json").read_text(encoding="utf-8")
    )
    assert result.status == "failed"
    assert result.error == "source_read_failed"


@pytest.mark.parametrize("valid_pdf", [True, False])
def test_extraction_storage_failure_returns_id_and_keeps_pdf(
    storage_directory: Path, extraction_directory: Path, valid_pdf: bool
) -> None:
    extraction_directory.write_bytes(b"existing data")
    contents = SAMPLE_PDF.read_bytes() if valid_pdf else b"%PDF-1.4\nbroken\n"
    response = client.post(
        "/documents", files={"file": ("sample.pdf", contents, "application/pdf")}
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["message"] == "Could not save extraction result"
    document_id = UUID(detail["document_id"])
    assert (storage_directory / f"{document_id}.pdf").read_bytes() == contents
    assert extraction_directory.read_bytes() == b"existing data"


def test_blank_pdf_is_successful_extraction() -> None:
    with BytesIO() as output, PdfWriter() as writer:
        writer.add_blank_page(width=595, height=842)
        writer.write(output)
        contents = output.getvalue()
    response = client.post(
        "/documents", files={"file": ("blank.pdf", contents, "application/pdf")}
    )

    assert response.status_code == 201
    assert response.json()["extraction_status"] == "completed"
    assert response.json()["page_count"] == 1


def test_extraction_and_both_storage_helpers_run_outside_event_loop(
    monkeypatch,
) -> None:
    original_extract = main.extract_pages
    original_save_document = storage._save_document
    original_save = storage._save_extraction
    calls = []

    def save_pdf_in_worker(source, directory):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        calls.append("pdf")
        return original_save_document(source, directory)

    def extract_in_worker(path):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        calls.append("extract")
        return original_extract(path)

    def save_in_worker(extraction, directory):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        calls.append("save")
        return original_save(extraction, directory)

    monkeypatch.setattr(main, "extract_pages", extract_in_worker)
    monkeypatch.setattr(storage, "_save_document", save_pdf_in_worker)
    monkeypatch.setattr(storage, "_save_extraction", save_in_worker)
    response = client.post(
        "/documents",
        files={"file": ("sample.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 201
    assert calls == ["pdf", "extract", "save"]


@pytest.mark.parametrize(
    ("case", "expected_status"),
    [
        ("success", None),
        ("invalid_media_type", 415),
        ("missing_filename", 422),
        ("invalid_metadata", 422),
        ("pdf_storage_failure", 500),
        ("json_storage_failure", 500),
        ("parse_failure", None),
    ],
)
def test_endpoint_closes_upload_on_success_and_handled_failures(
    case: str,
    expected_status: int | None,
    storage_directory: Path,
    extraction_directory: Path,
) -> None:
    filename = "sample.pdf"
    media_type = "application/pdf"
    contents = SAMPLE_PDF.read_bytes()
    if case == "invalid_media_type":
        media_type = "text/plain"
    elif case == "missing_filename":
        filename = None
    elif case == "invalid_metadata":
        filename = "../sample.pdf"
    elif case == "pdf_storage_failure":
        storage_directory.write_bytes(b"not a directory")
    elif case == "json_storage_failure":
        extraction_directory.write_bytes(b"not a directory")
    elif case == "parse_failure":
        contents = b"%PDF-1.4\nbroken\n"

    source = BytesIO(contents)
    upload = UploadFile(
        file=source,
        filename=filename,
        headers=Headers({"content-type": media_type}),
    )
    # Call the handler directly: this proves its finally block closes the file,
    # independently of FastAPI's own multipart cleanup.
    if expected_status is None:
        result = asyncio.run(main.upload_document(upload))
        expected_extraction = "failed" if case == "parse_failure" else "completed"
        assert result.extraction_status == expected_extraction
    else:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(main.upload_document(upload))
        assert exc.value.status_code == expected_status

    assert source.closed
    if case in {"invalid_media_type", "missing_filename", "invalid_metadata"}:
        assert not storage_directory.exists()
        assert not extraction_directory.exists()


def test_endpoint_closes_upload_when_unexpected_error_propagates(monkeypatch) -> None:
    error = RuntimeError("Unexpected failure")

    def fail_extraction(path):
        raise error

    monkeypatch.setattr(main, "extract_pages", fail_extraction)
    source = BytesIO(SAMPLE_PDF.read_bytes())
    upload = UploadFile(
        file=source,
        filename="sample.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(main.upload_document(upload))

    assert exc.value is error
    assert source.closed
