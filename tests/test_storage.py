import asyncio
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

    assert "Could not remove partial document" in caplog.text
