import asyncio
from hashlib import sha256
from pathlib import Path
from threading import get_ident
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag import storage
from nexusrag.schemas.chunk import DocumentChunk, DocumentChunks
from nexusrag.services import indexing
from nexusrag.services.embedding import embed_chunks
from nexusrag.services.embedding_model import MiniEmbedder


def make_chunks(count: int = 2) -> DocumentChunks:
    document_id = uuid4()
    texts = [f"Passage {index}: café" for index in range(count)]
    return DocumentChunks(
        document_id=document_id,
        chunk_size=100,
        overlap=0,
        chunks=[
            DocumentChunk(
                document_id=document_id,
                page_number=index + 1,
                chunk_index=index,
                start_char=0,
                end_char=len(text),
                text=text,
            )
            for index, text in enumerate(texts)
        ],
    )


def make_embedder(revision: str = "test-revision") -> Mock:
    embedder = Mock(spec=MiniEmbedder, model_id="test-encoder", model_revision=revision)
    embedder.get_dimensions.return_value = 3
    embedder.embed_documents.side_effect = lambda texts: [
        [float(len(text)), 1.0, 0.0] for text in texts
    ]
    return embedder


@pytest.mark.parametrize("count", [0, 2, 17])
def test_build_saves_complete_index_and_preserves_chunks(
    tmp_path: Path, count: int
) -> None:
    chunks = make_chunks(count)
    chunk_directory = tmp_path / "chunks"
    embedding_directory = tmp_path / "embeddings"
    source = asyncio.run(storage.save_chunks(chunks, chunk_directory))
    original = source.read_bytes()
    embedder = make_embedder()

    result = asyncio.run(
        indexing.build_semantic_index(
            chunks.document_id, embedder, chunk_directory, embedding_directory
        )
    )

    assert result == asyncio.run(
        storage.load_embeddings(chunks.document_id, embedding_directory)
    )
    assert result.document_id == chunks.document_id
    assert result.model_id == embedder.model_id
    assert result.model_revision == embedder.model_revision
    assert result.dimensions == 3
    assert (
        result.source_digest
        == sha256(chunks.model_dump_json().encode("utf-8")).hexdigest()
    )
    assert [entry.chunk_index for entry in result.embeddings] == list(range(count))
    assert [entry.vector for entry in result.embeddings] == [
        [float(len(chunk.text)), 1.0, 0.0] for chunk in chunks.chunks
    ]
    assert embedder.embed_documents.call_count == (count + 7) // 8
    assert source.read_bytes() == original
    assert list(embedding_directory.iterdir()) == [
        embedding_directory / f"{chunks.document_id}.json"
    ]


def test_successful_rebuild_replaces_previous_complete_index(tmp_path: Path) -> None:
    chunks = make_chunks()
    chunk_directory = tmp_path / "chunks"
    embedding_directory = tmp_path / "embeddings"
    asyncio.run(storage.save_chunks(chunks, chunk_directory))
    previous = embed_chunks(chunks, make_embedder("old-revision"))
    asyncio.run(storage.save_embeddings(previous, embedding_directory))

    result = asyncio.run(
        indexing.build_semantic_index(
            chunks.document_id, make_embedder(), chunk_directory, embedding_directory
        )
    )

    assert result != previous
    assert result.model_revision == "test-revision"
    assert (
        asyncio.run(storage.load_embeddings(chunks.document_id, embedding_directory))
        == result
    )


@pytest.mark.parametrize("existing", [False, True])
def test_third_batch_failure_never_saves_partial_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    chunks = make_chunks(17)
    chunk_directory = tmp_path / "chunks"
    embedding_directory = tmp_path / "embeddings"
    source = asyncio.run(storage.save_chunks(chunks, chunk_directory))
    source_bytes = source.read_bytes()
    destination = embedding_directory / f"{chunks.document_id}.json"
    previous_bytes = None
    if existing:
        previous = embed_chunks(chunks, make_embedder("old-revision"))
        asyncio.run(storage.save_embeddings(previous, embedding_directory))
        previous_bytes = destination.read_bytes()
    failure = RuntimeError("third batch failed")
    embedder = make_embedder()
    embedder.embed_documents.side_effect = [
        [[1.0, 1.0, 0.0]] * 8,
        [[1.0, 1.0, 0.0]] * 8,
        failure,
    ]
    save = AsyncMock()
    monkeypatch.setattr(indexing, "save_embeddings", save)

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(
            indexing.build_semantic_index(
                chunks.document_id, embedder, chunk_directory, embedding_directory
            )
        )

    assert raised.value is failure
    assert embedder.embed_documents.call_count == 3
    save.assert_not_awaited()
    assert source.read_bytes() == source_bytes
    if existing:
        assert destination.read_bytes() == previous_bytes
    else:
        assert not embedding_directory.exists()


@pytest.mark.parametrize("vector", [[0.0, 0.0, 0.0], [1.0, 0.0]])
def test_invalid_model_output_is_not_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vector: list[float]
) -> None:
    chunks = make_chunks(1)
    chunk_directory = tmp_path / "chunks"
    asyncio.run(storage.save_chunks(chunks, chunk_directory))
    embedder = make_embedder()
    embedder.embed_documents.side_effect = lambda texts: [vector]
    save = AsyncMock()
    monkeypatch.setattr(indexing, "save_embeddings", save)

    with pytest.raises(ValidationError):
        asyncio.run(
            indexing.build_semantic_index(
                chunks.document_id, embedder, chunk_directory, tmp_path / "embeddings"
            )
        )

    save.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing chunks"),
        ValueError("corrupt chunks"),
        OSError("unreadable chunks"),
    ],
)
def test_load_failure_stops_before_encoding_or_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    document_id = uuid4()
    embedder = make_embedder()
    load = AsyncMock(side_effect=error)
    save = AsyncMock()
    monkeypatch.setattr(indexing, "load_chunks", load)
    monkeypatch.setattr(indexing, "save_embeddings", save)

    with pytest.raises(type(error)) as raised:
        asyncio.run(
            indexing.build_semantic_index(
                document_id, embedder, tmp_path / "chunks", tmp_path / "embeddings"
            )
        )

    assert raised.value is error
    load.assert_awaited_once_with(document_id, tmp_path / "chunks")
    embedder.get_dimensions.assert_not_called()
    embedder.embed_documents.assert_not_called()
    save.assert_not_awaited()


def test_failed_publish_preserves_previous_index_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = make_chunks()
    chunk_directory = tmp_path / "chunks"
    embedding_directory = tmp_path / "embeddings"
    source = asyncio.run(storage.save_chunks(chunks, chunk_directory))
    source_bytes = source.read_bytes()
    previous = embed_chunks(chunks, make_embedder("old-revision"))
    destination = asyncio.run(storage.save_embeddings(previous, embedding_directory))
    previous_bytes = destination.read_bytes()
    failure = OSError("could not publish index")

    def fail_replace(self, target):
        raise failure

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError) as raised:
        asyncio.run(
            indexing.build_semantic_index(
                chunks.document_id,
                make_embedder(),
                chunk_directory,
                embedding_directory,
            )
        )

    assert raised.value is failure
    assert destination.read_bytes() == previous_bytes
    assert source.read_bytes() == source_bytes
    assert list(embedding_directory.iterdir()) == [destination]


def test_model_work_runs_in_worker_thread(tmp_path: Path) -> None:
    chunks = make_chunks()
    chunk_directory = tmp_path / "chunks"
    asyncio.run(storage.save_chunks(chunks, chunk_directory))
    embedder = make_embedder()
    event_loop_thread = get_ident()
    worker_threads = []

    def get_dimensions():
        worker_threads.append(get_ident())
        return 3

    def encode(texts):
        worker_threads.append(get_ident())
        return [[1.0, 0.0, 0.0] for text in texts]

    embedder.get_dimensions.side_effect = get_dimensions
    embedder.embed_documents.side_effect = encode

    asyncio.run(
        indexing.build_semantic_index(
            chunks.document_id, embedder, chunk_directory, tmp_path / "embeddings"
        )
    )

    assert len(worker_threads) == 2
    assert all(thread != event_loop_thread for thread in worker_threads)
