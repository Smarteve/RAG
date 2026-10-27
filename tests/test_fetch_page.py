import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from nexusrag import config
from nexusrag.main import app
from nexusrag.schemas.extraction import DocumentExtraction, ExtractedPage
from nexusrag.storage import save_extraction

client = TestClient(app)
pytestmark = pytest.mark.usefixtures("initialized_job_database")
SAMPLE_PDF = Path(__file__).resolve().parents[1] / "examples/employee-handbook.pdf"


@pytest.fixture(autouse=True)
def directories(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", tmp_path / "documents")
    monkeypatch.setattr(config, "EXTRACTION_DIRECTORY", tmp_path / "extracted")


@pytest.fixture
def document() -> DocumentExtraction:
    result = DocumentExtraction(
        document_id=uuid4(),
        pages=[
            ExtractedPage(page_number=1, text="First page — 中文"),
            ExtractedPage(page_number=2, text=""),
            ExtractedPage(page_number=3, text="Last page"),
        ],
    )
    asyncio.run(save_extraction(result, config.EXTRACTION_DIRECTORY))
    return result


@pytest.mark.parametrize("page_number", [1, 2, 3])
def test_fetch_returns_correct_page_including_blank_page(
    document: DocumentExtraction, page_number: int
) -> None:
    response = client.get(f"/documents/{document.document_id}/pages/{page_number}")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": str(document.document_id),
        "page": document.pages[page_number - 1].model_dump(),
    }


def test_fetch_unknown_document_returns_not_found() -> None:
    response = client.get(f"/documents/{uuid4()}/pages/1")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}
    assert not config.STORAGE_DIRECTORY.exists()
    assert not config.EXTRACTION_DIRECTORY.exists()


def test_fetch_missing_page_returns_not_found(document: DocumentExtraction) -> None:
    response = client.get(f"/documents/{document.document_id}/pages/4")

    assert response.status_code == 404
    assert response.json() == {"detail": "Requested page does not exist"}


@pytest.mark.parametrize(
    ("document_id", "page_number"),
    [
        ("not-a-uuid", "1"),
        (str(uuid4()), "0"),
        (str(uuid4()), "-1"),
        (str(uuid4()), "abc"),
    ],
)
def test_fetch_validates_path_parameters(document_id: str, page_number: str) -> None:
    response = client.get(f"/documents/{document_id}/pages/{page_number}")

    assert response.status_code == 422
    assert not config.STORAGE_DIRECTORY.exists()
    assert not config.EXTRACTION_DIRECTORY.exists()


def test_fetch_failed_extraction_returns_conflict() -> None:
    failed = DocumentExtraction(
        document_id=uuid4(), pages=[], status="failed", error="pdf_read_failed"
    )
    asyncio.run(save_extraction(failed, config.EXTRACTION_DIRECTORY))

    response = client.get(f"/documents/{failed.document_id}/pages/1")

    assert response.status_code == 409
    assert response.json() == {"detail": "Document extraction failed"}


def test_fetch_saved_pdf_without_result_returns_conflict() -> None:
    document_id = uuid4()
    config.STORAGE_DIRECTORY.mkdir()
    pdf_path = config.STORAGE_DIRECTORY / f"{document_id}.pdf"
    original = SAMPLE_PDF.read_bytes()
    pdf_path.write_bytes(original)

    response = client.get(f"/documents/{document_id}/pages/1")

    assert response.status_code == 409
    assert response.json() == {"detail": "Document extraction result is not available"}
    assert pdf_path.read_bytes() == original
    assert not config.EXTRACTION_DIRECTORY.exists()


@pytest.mark.parametrize(
    "case", ["invalid_json", "invalid_page", "wrong_id", "invalid_utf8"]
)
def test_fetch_corrupt_result_returns_server_error(case: str) -> None:
    document_id = uuid4()
    if case == "invalid_json":
        contents = b"not JSON"
    elif case == "invalid_utf8":
        contents = b"\xff\xfe"
    elif case == "invalid_page":
        contents = json.dumps(
            {"document_id": str(document_id), "pages": [None]}
        ).encode("utf-8")
    else:
        contents = (
            DocumentExtraction(document_id=uuid4(), pages=[])
            .model_dump_json()
            .encode("utf-8")
        )
    config.EXTRACTION_DIRECTORY.mkdir()
    source = config.EXTRACTION_DIRECTORY / f"{document_id}.json"
    source.write_bytes(contents)

    response = client.get(f"/documents/{document_id}/pages/1")

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load document"}
    assert source.read_bytes() == contents


def test_fetch_permission_error_returns_safe_server_error(monkeypatch) -> None:
    def fail_read(self, *args, **kwargs):
        raise PermissionError("Access denied to /private/server/path")

    monkeypatch.setattr(Path, "read_text", fail_read)

    response = client.get(f"/documents/{uuid4()}/pages/1")

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load document"}


def test_pdf_existence_check_runs_in_worker_and_handles_storage_error(
    monkeypatch,
) -> None:
    def fail_stat(self):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        raise PermissionError("Cannot inspect storage")

    monkeypatch.setattr(Path, "is_file", fail_stat)

    response = client.get(f"/documents/{uuid4()}/pages/1")

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load document"}


def test_fetch_uses_stored_page_number_when_list_order_differs() -> None:
    result = DocumentExtraction(
        document_id=uuid4(),
        pages=[
            ExtractedPage(page_number=3, text="Page three"),
            ExtractedPage(page_number=1, text="Page one"),
        ],
    )
    asyncio.run(save_extraction(result, config.EXTRACTION_DIRECTORY))

    response = client.get(f"/documents/{result.document_id}/pages/3")

    assert response.status_code == 200
    assert response.json()["page"] == {"page_number": 3, "text": "Page three"}


def test_upload_then_fetch_page(prepare_document) -> None:
    document_id = prepare_document(SAMPLE_PDF.read_bytes())

    response = client.get(f"/documents/{document_id}/pages/1")

    assert response.status_code == 200
    assert response.json()["document_id"] == str(document_id)
    assert response.json()["page"]["page_number"] == 1
    assert "Annual leave" in response.json()["page"]["text"]
