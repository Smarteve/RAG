import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag import storage
from nexusrag.schemas.chunk import DocumentChunk, DocumentChunks
from nexusrag.schemas.extraction import DocumentExtraction, ExtractedPage
from nexusrag.services.chunking import chunk_document


@pytest.fixture
def record() -> DocumentChunks:
    extraction = DocumentExtraction(
        document_id=uuid4(),
        pages=[ExtractedPage(page_number=1, text='Café — 中文\n"quoted" \\ path')],
    )
    return DocumentChunks(
        document_id=extraction.document_id,
        chunk_size=10,
        overlap=2,
        chunks=chunk_document(extraction, chunk_size=10, overlap=2),
    )


@pytest.mark.parametrize("empty", [False, True])
def test_save_and_load_chunks_round_trip(
    tmp_path: Path, record: DocumentChunks, empty: bool
) -> None:
    if empty:
        record = DocumentChunks(
            document_id=record.document_id, chunk_size=10, overlap=2, chunks=[]
        )
    directory = tmp_path / "nested" / "chunks"

    destination = asyncio.run(storage.save_chunks(record, directory))
    original = destination.read_bytes()
    loaded = asyncio.run(storage.load_chunks(record.document_id, directory))

    assert destination == directory / f"{record.document_id}.json"
    assert loaded == record
    assert all(isinstance(chunk, DocumentChunk) for chunk in loaded.chunks)
    assert destination.read_bytes() == original
    assert list(directory.iterdir()) == [destination]


def test_complete_chunk_record_replaces_previous_result(
    tmp_path: Path, record: DocumentChunks
) -> None:
    empty = DocumentChunks(
        document_id=record.document_id, chunk_size=10, overlap=2, chunks=[]
    )
    destination = asyncio.run(storage.save_chunks(empty, tmp_path))

    assert asyncio.run(storage.save_chunks(record, tmp_path)) == destination
    assert asyncio.run(storage.load_chunks(record.document_id, tmp_path)) == record
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("stage", ["write", "publish"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_save_cleans_temp_and_preserves_previous_bytes(
    tmp_path: Path,
    record: DocumentChunks,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    existing: bool,
) -> None:
    destination = tmp_path / f"{record.document_id}.json"
    if existing:
        destination.write_bytes(b"previous result")
    error = OSError("Simulated chunk storage failure")
    if stage == "publish":

        def fail_replace(self, target):
            raise error

        monkeypatch.setattr(Path, "replace", fail_replace)
    else:
        original = storage.NamedTemporaryFile

        def failed_temporary_file(**kwargs):
            output = original(**kwargs)
            write = output.write

            def fail_write(text):
                write(text[:5])
                raise error

            output.write = fail_write
            return output

        monkeypatch.setattr(storage, "NamedTemporaryFile", failed_temporary_file)

    with pytest.raises(OSError) as exc:
        asyncio.run(storage.save_chunks(record, tmp_path))

    assert exc.value is error
    assert list(tmp_path.iterdir()) == ([destination] if existing else [])
    if existing:
        assert destination.read_bytes() == b"previous result"


def test_load_missing_chunks_does_not_create_directory(tmp_path: Path) -> None:
    directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        asyncio.run(storage.load_chunks(uuid4(), directory))

    assert not directory.exists()


@pytest.mark.parametrize("contents", [b"not JSON", b"{}", b"\xff\xfe"])
def test_load_rejects_corrupt_chunk_json(tmp_path: Path, contents: bytes) -> None:
    document_id = uuid4()
    source = tmp_path / f"{document_id}.json"
    source.write_bytes(contents)

    with pytest.raises((ValidationError, UnicodeDecodeError)):
        asyncio.run(storage.load_chunks(document_id, tmp_path))

    assert source.read_bytes() == contents


def test_load_rejects_mismatched_record_id(
    tmp_path: Path, record: DocumentChunks
) -> None:
    requested_id = uuid4()
    source = tmp_path / f"{requested_id}.json"
    source.write_text(record.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="Stored chunks document ID") as exc:
        asyncio.run(storage.load_chunks(requested_id, tmp_path))

    assert type(exc.value) is ValueError


def test_load_permission_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = PermissionError("Cannot read chunks")

    def fail_read(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(PermissionError) as exc:
        asyncio.run(storage.load_chunks(uuid4(), tmp_path))

    assert exc.value is error


def test_save_and_load_run_in_worker_threads(
    tmp_path: Path, record: DocumentChunks, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_save = storage._save_chunks
    original_load = storage._load_chunks
    calls = []

    def save_in_worker(record, directory):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        calls.append("save")
        return original_save(record, directory)

    def load_in_worker(document_id, directory):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        calls.append("load")
        return original_load(document_id, directory)

    monkeypatch.setattr(storage, "_save_chunks", save_in_worker)
    monkeypatch.setattr(storage, "_load_chunks", load_in_worker)

    asyncio.run(storage.save_chunks(record, tmp_path))
    assert asyncio.run(storage.load_chunks(record.document_id, tmp_path)) == record
    assert calls == ["save", "load"]
