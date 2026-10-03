import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

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
