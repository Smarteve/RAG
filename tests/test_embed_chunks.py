from hashlib import sha256
from unittest.mock import Mock, call
from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from nexusrag.schemas.chunk import DocumentChunk, DocumentChunks
from nexusrag.schemas.embedding import DocumentEmbeddings
from nexusrag.services.embedding import embed_chunks


def make_chunks(
    texts: list[str], *, document_id=None, chunk_size: int = 100
) -> DocumentChunks:
    document_id = document_id if document_id is not None else uuid4()
    return DocumentChunks(
        document_id=document_id,
        chunk_size=chunk_size,
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


def make_embedder(dimensions: int = 3) -> Mock:
    embedder = Mock(model_id="test-encoder", model_revision="test-revision")
    embedder.get_dimensions.return_value = dimensions
    embedder.embed_documents.side_effect = lambda texts: [
        [float(len(text))] + [1.0] * (dimensions - 1) for text in texts
    ]
    return embedder


def test_embed_chunks_returns_aligned_record_with_source_and_model_identity() -> None:
    chunks = make_chunks(["annual leave", "remote work"])
    original = chunks.model_dump_json()
    embedder = make_embedder()

    result = embed_chunks(chunks, embedder)

    assert result.document_id == chunks.document_id
    assert result.model_id == "test-encoder"
    assert result.model_revision == "test-revision"
    assert result.dimensions == 3
    assert result.source_digest == sha256(original.encode("utf-8")).hexdigest()
    assert [entry.chunk_index for entry in result.embeddings] == [0, 1]
    assert [entry.vector for entry in result.embeddings] == [
        [12.0, 1.0, 1.0],
        [11.0, 1.0, 1.0],
    ]
    assert DocumentEmbeddings.model_validate_json(result.model_dump_json()) == result
    assert chunks.model_dump_json() == original


def test_empty_chunks_return_empty_record_without_encoding() -> None:
    chunks = make_chunks([])
    embedder = make_embedder()

    result = embed_chunks(chunks, embedder)

    assert result.document_id == chunks.document_id
    assert result.dimensions == 3
    assert result.embeddings == []
    assert (
        result.source_digest
        == sha256(chunks.model_dump_json().encode("utf-8")).hexdigest()
    )
    embedder.embed_documents.assert_not_called()


def test_embed_chunks_batches_texts_and_keeps_document_wide_indices() -> None:
    texts = [f"passage {index}" for index in range(17)]
    chunks = make_chunks(texts)
    embedder = make_embedder()

    result = embed_chunks(chunks, embedder)

    assert embedder.embed_documents.call_args_list == [
        call(texts[:8]),
        call(texts[8:16]),
        call(texts[16:]),
    ]
    assert [entry.chunk_index for entry in result.embeddings] == list(range(17))
    assert [entry.vector for entry in result.embeddings] == [
        [float(len(text)), 1.0, 1.0] for text in texts
    ]


@pytest.mark.parametrize("vectors", [[], [[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]] * 3])
def test_embed_chunks_rejects_vector_count_mismatch(vectors) -> None:
    embedder = make_embedder()
    embedder.embed_documents.side_effect = None
    embedder.embed_documents.return_value = vectors

    with pytest.raises(ValueError, match="zip"):
        embed_chunks(make_chunks(["one", "two"]), embedder)


@pytest.mark.parametrize(
    "vector",
    [[0.0, 0.0, 0.0], [float("nan"), 1.0, 1.0], [float("inf"), 1.0, 1.0], [1.0, 2.0]],
    ids=["zero", "nan", "infinity", "wrong-dimensions"],
)
def test_embed_chunks_rejects_invalid_provider_vectors(vector: list[float]) -> None:
    embedder = make_embedder()
    embedder.embed_documents.side_effect = None
    embedder.embed_documents.return_value = [vector]

    with pytest.raises(ValidationError):
        embed_chunks(make_chunks(["one"]), embedder)


def test_embed_chunks_propagates_later_batch_failure_and_preserves_source() -> None:
    chunks = make_chunks([f"passage {index}" for index in range(9)])
    original = chunks.model_dump_json()
    failure = RuntimeError("encoding failed")
    embedder = make_embedder()
    embedder.embed_documents.side_effect = [[[1.0, 1.0, 1.0]] * 8, failure]

    with pytest.raises(RuntimeError) as raised:
        embed_chunks(chunks, embedder)

    assert raised.value is failure
    assert embedder.embed_documents.call_count == 2
    assert chunks.model_dump_json() == original


def test_source_digest_changes_with_source_text_or_chunk_settings() -> None:
    document_id = uuid4()
    records = [
        make_chunks(["one"], document_id=document_id),
        make_chunks(["two"], document_id=document_id),
        make_chunks(["one"], document_id=document_id, chunk_size=101),
    ]

    results = [embed_chunks(record, make_embedder()) for record in records]

    assert len({result.source_digest for result in results}) == 3
    assert embed_chunks(records[0], make_embedder()) == results[0]


@given(
    texts=st.lists(st.text(alphabet="abcde", min_size=1, max_size=30), max_size=25),
    dimensions=st.integers(min_value=1, max_value=8),
)
def test_generated_chunk_records_preserve_order_dimensions_and_bounded_batches(
    texts: list[str], dimensions: int
) -> None:
    chunks = make_chunks(texts)
    original = chunks.model_dump_json()
    embedder = make_embedder(dimensions)

    result = embed_chunks(chunks, embedder)

    assert [entry.chunk_index for entry in result.embeddings] == list(range(len(texts)))
    assert [entry.vector for entry in result.embeddings] == [
        [float(len(text))] + [1.0] * (dimensions - 1) for text in texts
    ]
    assert result.dimensions == dimensions
    assert embedder.embed_documents.call_args_list == [
        call(texts[start : start + 8]) for start in range(0, len(texts), 8)
    ]
    assert result.source_digest == sha256(original.encode("utf-8")).hexdigest()
    assert chunks.model_dump_json() == original
