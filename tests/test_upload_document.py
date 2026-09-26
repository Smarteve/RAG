import pytest
from fastapi.testclient import TestClient

from nexusrag.main import app
from nexusrag.schemas.document import MAX_DOCUMENT_SIZE_BYTES

client = TestClient(app)


def test_pdf_upload_returns_document_metadata() -> None:
    file_contents = b"%PDF-1.7\nexample PDF contents"

    response = client.post(
        "/documents",
        files={"file": ("invoice.pdf", file_contents, "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "invoice.pdf"
    assert response.json()["media_type"] == "application/pdf"
    assert response.json()["size_bytes"] == len(file_contents)


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
