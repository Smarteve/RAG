from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from nexusrag import config
from nexusrag.constants import CHUNK_SIZE_BYTES
from nexusrag.main import app
from nexusrag.schemas.document import MAX_DOCUMENT_SIZE_BYTES

client = TestClient(app)


@pytest.fixture(autouse=True)
def storage_directory(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / "documents"
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", directory)
    return directory


@pytest.mark.parametrize("size_bytes", [27, CHUNK_SIZE_BYTES + 7])
def test_pdf_upload_returns_document_metadata(
    size_bytes: int, storage_directory: Path
) -> None:
    file_contents = b"%PDF-" + b"x" * (size_bytes - 5)

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
