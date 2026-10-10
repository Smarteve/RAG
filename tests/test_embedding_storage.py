import asyncio
import json
from pathlib import Path
from runpy import run_path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag import config, storage
from nexusrag.schemas.embedding import ChunkEmbedding, DocumentEmbeddings


@pytest.fixture
def record() -> DocumentEmbeddings:
    return DocumentEmbeddings(
        document_id=uuid4(),
        model_id="test-encoder-Café-中文",
        model_revision="test-revision",
        dimensions=3,
        source_digest="a" * 64,
        embeddings=[
            ChunkEmbedding(chunk_index=0, vector=[0.25, -0.75, 0.0]),
            ChunkEmbedding(chunk_index=1, vector=[1.0, 0.0, 0.0]),
        ],
    )


@pytest.mark.parametrize("empty", [False, True])
def test_embedding_storage_round_trip(
    tmp_path: Path, record: DocumentEmbeddings, empty: bool
) -> None:
    if empty:
        record = record.model_copy(update={"embeddings": []})
    directory = tmp_path / "nested" / "embeddings"

    destination = asyncio.run(storage.save_embeddings(record, directory))
    original = destination.read_bytes()
    restored = asyncio.run(storage.load_embeddings(record.document_id, directory))

    assert destination == directory / f"{record.document_id}.json"
    assert restored == record
    assert all(isinstance(entry, ChunkEmbedding) for entry in restored.embeddings)
    assert destination.read_bytes() == original
    assert list(directory.iterdir()) == [destination]


def test_complete_embedding_record_replaces_previous_result(
    tmp_path: Path, record: DocumentEmbeddings
) -> None:
    previous = record.model_copy(update={"embeddings": [], "source_digest": "b" * 64})
    destination = asyncio.run(storage.save_embeddings(previous, tmp_path))

    assert asyncio.run(storage.save_embeddings(record, tmp_path)) == destination
    assert asyncio.run(storage.load_embeddings(record.document_id, tmp_path)) == record
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("stage", ["write", "publish"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_embedding_save_preserves_previous_record_and_cleans_temp(
    tmp_path: Path,
    record: DocumentEmbeddings,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    existing: bool,
) -> None:
    destination = tmp_path / f"{record.document_id}.json"
    original_bytes = None
    if existing:
        previous = record.model_copy(update={"embeddings": []})
        asyncio.run(storage.save_embeddings(previous, tmp_path))
        original_bytes = destination.read_bytes()
    error = OSError("Simulated embedding storage failure")
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

    with pytest.raises(OSError) as raised:
        asyncio.run(storage.save_embeddings(record, tmp_path))

    assert raised.value is error
    assert list(tmp_path.iterdir()) == ([destination] if existing else [])
    if existing:
        assert destination.read_bytes() == original_bytes
        assert (
            asyncio.run(storage.load_embeddings(record.document_id, tmp_path))
            == previous
        )


def test_load_missing_embeddings_does_not_create_directory(tmp_path: Path) -> None:
    directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        asyncio.run(storage.load_embeddings(uuid4(), directory))

    assert not directory.exists()


@pytest.mark.parametrize("contents", [b"not JSON", b"{}", b"\xff\xfe"])
def test_load_rejects_corrupt_embedding_json(tmp_path: Path, contents: bytes) -> None:
    document_id = uuid4()
    source = tmp_path / f"{document_id}.json"
    source.write_bytes(contents)

    with pytest.raises((ValidationError, UnicodeDecodeError)):
        asyncio.run(storage.load_embeddings(document_id, tmp_path))

    assert source.read_bytes() == contents


@pytest.mark.parametrize("case", ["dimensions", "index", "zero", "nan"])
def test_load_validates_saved_vectors(
    tmp_path: Path, record: DocumentEmbeddings, case: str
) -> None:
    fields = record.model_dump(mode="json")
    if case == "dimensions":
        fields["dimensions"] = 4
    elif case == "index":
        fields["embeddings"][1]["chunk_index"] = 0
    elif case == "zero":
        fields["embeddings"][0]["vector"] = [0.0, 0.0, 0.0]
    else:
        fields["embeddings"][0]["vector"] = [float("nan"), 1.0, 1.0]
    source = tmp_path / f"{record.document_id}.json"
    source.write_text(json.dumps(fields), encoding="utf-8")
    original = source.read_bytes()

    with pytest.raises(ValidationError):
        asyncio.run(storage.load_embeddings(record.document_id, tmp_path))

    assert source.read_bytes() == original


def test_load_rejects_mismatched_embedding_document_id(
    tmp_path: Path, record: DocumentEmbeddings
) -> None:
    requested_id = uuid4()
    source = tmp_path / f"{requested_id}.json"
    source.write_text(record.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="Stored embeddings document ID") as raised:
        asyncio.run(storage.load_embeddings(requested_id, tmp_path))

    assert type(raised.value) is ValueError


def test_embedding_read_permission_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = PermissionError("Cannot read embeddings")

    def fail_read(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(PermissionError) as raised:
        asyncio.run(storage.load_embeddings(uuid4(), tmp_path))

    assert raised.value is error


def test_embedding_save_and_load_run_in_worker_threads(
    tmp_path: Path, record: DocumentEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_save = storage._save_embeddings
    original_load = storage._load_embeddings
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

    monkeypatch.setattr(storage, "_save_embeddings", save_in_worker)
    monkeypatch.setattr(storage, "_load_embeddings", load_in_worker)

    asyncio.run(storage.save_embeddings(record, tmp_path))
    assert asyncio.run(storage.load_embeddings(record.document_id, tmp_path)) == record
    assert calls == ["save", "load"]


def test_embedding_directory_defaults_to_resolved_data_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEXUSRAG_EMBEDDING_DIR", raising=False)

    settings = run_path(config.__file__)

    assert settings["EMBEDDING_DIRECTORY"] == tmp_path / "data" / "embeddings"
    assert not settings["EMBEDDING_DIRECTORY"].exists()


@pytest.mark.parametrize("relative", [False, True])
def test_embedding_directory_environment_override_is_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = "vectors" if relative else str(tmp_path / "custom-vectors")
    monkeypatch.setenv("NEXUSRAG_EMBEDDING_DIR", directory)

    settings = run_path(config.__file__)

    assert settings["EMBEDDING_DIRECTORY"] == Path(directory).resolve()
    assert not settings["EMBEDDING_DIRECTORY"].exists()
