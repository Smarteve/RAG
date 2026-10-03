import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag import storage
from nexusrag.schemas.extraction import DocumentExtraction, ExtractedPage


@pytest.fixture
def extraction() -> DocumentExtraction:
    return DocumentExtraction(
        document_id=uuid4(),
        pages=[
            ExtractedPage(page_number=1, text='Café — 中文\n"quoted" \\ path'),
            ExtractedPage(page_number=2, text=""),
        ],
    )


def test_save_extraction_round_trips_unicode_and_creates_directory(
    tmp_path: Path, extraction: DocumentExtraction
) -> None:
    directory = tmp_path / "nested" / "extracted"

    destination = asyncio.run(storage.save_extraction(extraction, directory))

    assert destination == directory / f"{extraction.document_id}.json"
    assert (
        DocumentExtraction.model_validate_json(destination.read_text(encoding="utf-8"))
        == extraction
    )
    assert list(directory.iterdir()) == [destination]


def test_complete_result_replaces_previous_result(
    tmp_path: Path, extraction: DocumentExtraction
) -> None:
    failed = DocumentExtraction(
        document_id=extraction.document_id,
        pages=[],
        status="failed",
        error="pdf_read_failed",
    )
    destination = asyncio.run(storage.save_extraction(failed, tmp_path))

    assert asyncio.run(storage.save_extraction(extraction, tmp_path)) == destination
    assert (
        DocumentExtraction.model_validate_json(destination.read_bytes()) == extraction
    )
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("existing_result", [False, True])
def test_failed_write_removes_temporary_file_and_preserves_previous_result(
    tmp_path: Path, extraction: DocumentExtraction, monkeypatch, existing_result: bool
) -> None:
    destination = tmp_path / f"{extraction.document_id}.json"
    if existing_result:
        destination.write_text('{"previous": true}', encoding="utf-8")
    original_temporary_file = storage.NamedTemporaryFile
    error = OSError("Simulated disk full")

    def temporary_file_with_failed_write(**kwargs):
        output = original_temporary_file(**kwargs)
        original_write = output.write

        def fail_write(text):
            original_write(text[:5])
            raise error

        output.write = fail_write
        return output

    monkeypatch.setattr(storage, "NamedTemporaryFile", temporary_file_with_failed_write)

    with pytest.raises(OSError, match="Simulated disk full") as exc:
        asyncio.run(storage.save_extraction(extraction, tmp_path))

    assert exc.value is error
    assert list(tmp_path.iterdir()) == ([destination] if existing_result else [])
    if existing_result:
        assert destination.read_text(encoding="utf-8") == '{"previous": true}'


def test_failed_replace_preserves_previous_result_and_cleans_temporary_file(
    tmp_path: Path, extraction: DocumentExtraction, monkeypatch
) -> None:
    destination = tmp_path / f"{extraction.document_id}.json"
    destination.write_text('{"previous": true}', encoding="utf-8")

    def fail_replace(self, target):
        raise PermissionError("Cannot replace result")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(PermissionError, match="Cannot replace result"):
        asyncio.run(storage.save_extraction(extraction, tmp_path))

    assert destination.read_text(encoding="utf-8") == '{"previous": true}'
    assert list(tmp_path.iterdir()) == [destination]


def test_cleanup_failure_does_not_mask_original_error(
    tmp_path: Path, extraction: DocumentExtraction, monkeypatch, caplog
) -> None:
    def fail_replace(self, target):
        raise OSError("Original replacement failure")

    def fail_unlink(self, missing_ok=False):
        raise PermissionError("Cleanup failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(OSError, match="Original replacement failure"):
        asyncio.run(storage.save_extraction(extraction, tmp_path))

    assert "Could not remove temporary extraction" in caplog.text
    assert not (tmp_path / f"{extraction.document_id}.json").exists()


def test_load_reconstructs_nested_pages_and_preserves_file(
    tmp_path: Path, extraction: DocumentExtraction
) -> None:
    destination = asyncio.run(storage.save_extraction(extraction, tmp_path))
    original_bytes = destination.read_bytes()

    loaded = asyncio.run(storage.load_extraction(extraction.document_id, tmp_path))

    assert loaded == extraction
    assert isinstance(loaded, DocumentExtraction)
    assert all(isinstance(page, ExtractedPage) for page in loaded.pages)
    assert loaded.pages[0].text == 'Café — 中文\n"quoted" \\ path'
    assert loaded.pages[1].text == ""
    assert destination.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [destination]


def test_load_returns_a_saved_failed_extraction(tmp_path: Path) -> None:
    failed = DocumentExtraction(
        document_id=uuid4(), pages=[], status="failed", error="pdf_read_failed"
    )
    asyncio.run(storage.save_extraction(failed, tmp_path))

    loaded = asyncio.run(storage.load_extraction(failed.document_id, tmp_path))

    # Loading a failure record succeeds; the API will decide how to present it.
    assert loaded == failed


def test_load_rejects_mismatched_document_id(
    tmp_path: Path, extraction: DocumentExtraction
) -> None:
    requested_id = uuid4()
    source = tmp_path / f"{requested_id}.json"
    original_json = extraction.model_dump_json()
    source.write_text(original_json, encoding="utf-8")

    with pytest.raises(
        ValueError, match="Stored extraction document ID does not match requested ID"
    ) as exc:
        asyncio.run(storage.load_extraction(requested_id, tmp_path))

    # Distinguish our integrity check from Pydantic's ValidationError subclass.
    assert type(exc.value) is ValueError
    assert source.read_text(encoding="utf-8") == original_json


@pytest.mark.parametrize("directory_exists", [True, False])
def test_load_missing_file_raises_without_creating_storage(
    tmp_path: Path, directory_exists: bool
) -> None:
    directory = tmp_path / "extracted"
    if directory_exists:
        directory.mkdir()

    with pytest.raises(FileNotFoundError):
        asyncio.run(storage.load_extraction(uuid4(), directory))

    assert directory.exists() == directory_exists
    if directory_exists:
        assert list(directory.iterdir()) == []


@pytest.mark.parametrize("case", ["invalid_json", "missing_pages", "invalid_page"])
def test_load_rejects_corrupt_json_or_invalid_fields(tmp_path: Path, case: str) -> None:
    document_id = uuid4()
    if case == "invalid_json":
        contents = "not JSON"
    elif case == "missing_pages":
        contents = json.dumps({"document_id": str(document_id)})
    else:
        contents = json.dumps(
            {
                "document_id": str(document_id),
                "pages": [{"page_number": 0, "text": "invalid page number"}],
            }
        )
    source = tmp_path / f"{document_id}.json"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ValidationError):
        asyncio.run(storage.load_extraction(document_id, tmp_path))

    assert source.read_text(encoding="utf-8") == contents


def test_load_rejects_invalid_utf8(tmp_path: Path) -> None:
    document_id = uuid4()
    source = tmp_path / f"{document_id}.json"
    source.write_bytes(b"\xff\xfe")

    with pytest.raises(UnicodeDecodeError):
        asyncio.run(storage.load_extraction(document_id, tmp_path))


def test_load_permission_error_propagates_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    error = PermissionError("Access denied")

    def fail_read(self, *args, **kwargs):
        raise error

    # Simulate the error rather than rely on OS permission bits or user identity.
    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(PermissionError) as exc:
        asyncio.run(storage.load_extraction(uuid4(), tmp_path))

    assert exc.value is error


def test_load_reads_in_worker_thread(
    tmp_path: Path, extraction: DocumentExtraction, monkeypatch
) -> None:
    source = asyncio.run(storage.save_extraction(extraction, tmp_path))
    original_read = Path.read_text
    reads = []

    def read_in_worker(self, *args, **kwargs):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        reads.append(self)
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_in_worker)

    loaded = asyncio.run(storage.load_extraction(extraction.document_id, tmp_path))

    assert loaded == extraction
    assert reads == [source]
