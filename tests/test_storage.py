import asyncio
import errno
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import UploadFile

from nexusrag import storage
from nexusrag.constants import CHUNK_SIZE_BYTES


@pytest.mark.parametrize("size_bytes", [27, CHUNK_SIZE_BYTES + 7])
def test_save_preserves_bytes_and_leaves_upload_open(
    tmp_path: Path, size_bytes: int
) -> None:
    contents = b"%PDF-" + b"x" * (size_bytes - 5)
    with BytesIO(contents) as source:
        # Simulate the endpoint having already read the entire upload.
        source.read()
        upload = UploadFile(file=source, filename="original.pdf")
        document_id, destination = asyncio.run(
            storage.save_document(upload, tmp_path / "documents")
        )

        assert isinstance(document_id, UUID)
        assert destination == tmp_path / "documents" / f"{document_id}.pdf"
        assert destination.read_bytes() == contents
        assert not source.closed
        assert list(destination.parent.iterdir()) == [destination]


class FailingReader(BytesIO):
    def read(self, size: int = -1) -> bytes:
        if self.tell() > 0:
            raise OSError("Simulated read failure")
        return super().read(size)


def test_read_failure_removes_partial_file(tmp_path: Path) -> None:
    with FailingReader(b"%PDF-partial data") as source:
        upload = UploadFile(file=source, filename="original.pdf")
        with pytest.raises(OSError, match="Simulated read failure"):
            asyncio.run(storage.save_document(upload, tmp_path))

        assert list(tmp_path.iterdir()) == []
        assert not source.closed


def test_collision_preserves_existing_file(tmp_path: Path, monkeypatch) -> None:
    document_id = UUID("00000000-0000-4000-8000-000000000001")
    monkeypatch.setattr(storage, "uuid4", lambda: document_id)
    existing = tmp_path / f"{document_id}.pdf"
    existing.write_bytes(b"existing document")

    with BytesIO(b"%PDF-new document") as source:
        upload = UploadFile(file=source, filename="original.pdf")
        with pytest.raises(FileExistsError):
            asyncio.run(storage.save_document(upload, tmp_path))

    assert existing.read_bytes() == b"existing document"
    assert list(tmp_path.iterdir()) == [existing]


def test_cleanup_failure_preserves_original_error(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    def fail_unlink(self, missing_ok=False):
        raise PermissionError("Simulated cleanup failure")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with FailingReader(b"%PDF-partial data") as source:
        upload = UploadFile(file=source, filename="original.pdf")
        with pytest.raises(OSError, match="Simulated read failure"):
            asyncio.run(storage.save_document(upload, tmp_path))

    assert "Could not remove temporary document" in caplog.text


def test_final_pdf_is_not_visible_while_copying(tmp_path: Path) -> None:
    class InspectingReader(BytesIO):
        def read(self, size=-1):
            assert list(tmp_path.glob("*.pdf")) == []
            return super().read(size)

    contents = b"%PDF-" + b"x" * (CHUNK_SIZE_BYTES + 7)
    with InspectingReader(contents) as source:
        upload = UploadFile(file=source, filename="sample.pdf")
        _, destination = asyncio.run(storage.save_document(upload, tmp_path))

    assert destination.read_bytes() == contents
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_pdf_write_cleans_temporary_file(tmp_path: Path, monkeypatch) -> None:
    original_temporary_file = storage.NamedTemporaryFile

    def temporary_file_with_failed_write(**kwargs):
        output = original_temporary_file(**kwargs)
        original_write = output.write

        def fail_write(chunk):
            original_write(chunk[:5])
            raise OSError("Disk full")

        output.write = fail_write
        return output

    monkeypatch.setattr(storage, "NamedTemporaryFile", temporary_file_with_failed_write)
    with BytesIO(b"%PDF-content") as source:
        upload = UploadFile(file=source, filename="sample.pdf")
        with pytest.raises(OSError, match="Disk full"):
            asyncio.run(storage.save_document(upload, tmp_path))
        assert not source.closed

    assert list(tmp_path.iterdir()) == []


def test_unsupported_hard_links_fail_without_publishing_pdf(
    tmp_path: Path, monkeypatch
) -> None:
    error = OSError(errno.EOPNOTSUPP, "Hard links unsupported")

    def fail_link(self, target):
        raise error

    monkeypatch.setattr(Path, "hardlink_to", fail_link)
    with BytesIO(b"%PDF-content") as source:
        upload = UploadFile(file=source, filename="sample.pdf")
        with pytest.raises(OSError) as exc:
            asyncio.run(storage.save_document(upload, tmp_path))

    assert exc.value is error
    assert list(tmp_path.iterdir()) == []


def test_collision_at_publication_preserves_competing_file(
    tmp_path: Path, monkeypatch
) -> None:
    original_link = Path.hardlink_to

    def create_competing_file_then_link(self, target):
        self.write_bytes(b"another upload won")
        original_link(self, target)

    monkeypatch.setattr(Path, "hardlink_to", create_competing_file_then_link)
    with BytesIO(b"%PDF-content") as source:
        upload = UploadFile(file=source, filename="sample.pdf")
        with pytest.raises(FileExistsError):
            asyncio.run(storage.save_document(upload, tmp_path))

    remaining = list(tmp_path.iterdir())
    assert len(remaining) == 1
    assert remaining[0].suffix == ".pdf"
    assert remaining[0].read_bytes() == b"another upload won"


def test_cleanup_failure_after_publication_still_returns_saved_pdf(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    def fail_unlink(self, missing_ok=False):
        raise PermissionError("Cleanup denied")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with BytesIO(b"%PDF-content") as source:
        upload = UploadFile(file=source, filename="sample.pdf")
        _, destination = asyncio.run(storage.save_document(upload, tmp_path))

    assert destination.read_bytes() == b"%PDF-content"
    temporary_files = list(tmp_path.glob(".*.tmp"))
    assert len(temporary_files) == 1
    assert temporary_files[0].samefile(destination)
    assert "Could not remove temporary document" in caplog.text
