from uuid import UUID, uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from nexusrag.schemas.embedding import ChunkEmbedding, DocumentEmbeddings


def make_record(**changes) -> DocumentEmbeddings:
    fields = {
        "document_id": uuid4(),
        "model_id": "test-encoder",
        "model_revision": "test-revision",
        "dimensions": 3,
        "source_digest": "a" * 64,
        "embeddings": [
            {"chunk_index": 0, "vector": [0.2, -0.3, 0.0]},
            {"chunk_index": 1, "vector": [-0.5, 0.0, 0.1]},
        ],
    }
    return DocumentEmbeddings(**(fields | changes))


def test_valid_chunk_embedding_preserves_negative_values_and_zeros() -> None:
    vector = [0.2, -0.3, 0.0]

    embedding = ChunkEmbedding(chunk_index=0, vector=vector)

    assert embedding.chunk_index == 0
    assert embedding.vector == vector
    assert vector == [0.2, -0.3, 0.0]


@pytest.mark.parametrize(
    "vector",
    [
        [],
        [0.0, -0.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [-float("inf"), 1.0],
        ["not a number", 1.0],
    ],
    ids=[
        "empty",
        "zero",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "nonnumeric",
    ],
)
def test_invalid_vector_is_rejected(vector) -> None:
    with pytest.raises(ValidationError):
        ChunkEmbedding(chunk_index=0, vector=vector)


def test_negative_chunk_index_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChunkEmbedding(chunk_index=-1, vector=[1.0])


def test_valid_document_embeddings_round_trip_through_json() -> None:
    record = make_record()

    restored = DocumentEmbeddings.model_validate_json(record.model_dump_json())

    assert restored == record
    assert isinstance(restored.document_id, UUID)
    assert all(isinstance(entry, ChunkEmbedding) for entry in restored.embeddings)
    assert [entry.chunk_index for entry in restored.embeddings] == [0, 1]


def test_empty_document_embeddings_are_valid() -> None:
    assert make_record(embeddings=[]).embeddings == []


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("vector", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
def test_every_vector_must_match_dimensions(index: int, vector: list[float]) -> None:
    entries = [
        {"chunk_index": 0, "vector": [0.2, -0.3, 0.0]},
        {"chunk_index": 1, "vector": [-0.5, 0.0, 0.1]},
    ]
    entries[index]["vector"] = vector

    with pytest.raises(ValidationError, match="vector length must match dimensions"):
        make_record(embeddings=entries)


@pytest.mark.parametrize(
    "indices",
    [[1], [0, 2], [0, 0], [1, 0]],
    ids=["nonzero-start", "gap", "duplicate", "out-of-order"],
)
def test_chunk_indices_must_be_sequential(indices: list[int]) -> None:
    entries = [{"chunk_index": index, "vector": [1.0, 0.0, 0.0]} for index in indices]

    with pytest.raises(ValidationError, match="sequential starting at zero"):
        make_record(embeddings=entries)


@pytest.mark.parametrize(
    "changes",
    [
        {"document_id": "not-a-uuid"},
        {"model_id": ""},
        {"model_revision": ""},
        {"dimensions": 0},
        {"dimensions": -1},
        {"dimensions": 1.5},
        {"source_digest": "a" * 63},
        {"source_digest": "a" * 65},
        {"source_digest": "g" * 64},
        {"source_digest": "A" * 64},
        {"source_digest": "a" * 64 + "\n"},
    ],
    ids=[
        "invalid-document-id",
        "empty-model-id",
        "empty-model-revision",
        "zero-dimensions",
        "negative-dimensions",
        "fractional-dimensions",
        "short-digest",
        "long-digest",
        "nonhex-digest",
        "uppercase-digest",
        "digest-with-newline",
    ],
)
def test_invalid_metadata_is_rejected_even_without_embeddings(changes: dict) -> None:
    with pytest.raises(ValidationError):
        make_record(embeddings=[], **changes)


def test_nested_invalid_vector_is_rejected_by_parent() -> None:
    with pytest.raises(ValidationError, match="nonzero"):
        make_record(embeddings=[{"chunk_index": 0, "vector": [0.0, 0.0, 0.0]}])


@st.composite
def embedding_entries(draw, min_entries: int = 0):
    dimensions = draw(st.integers(min_value=1, max_value=12))
    vector = st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        min_size=dimensions,
        max_size=dimensions,
    ).filter(any)
    vectors = draw(st.lists(vector, min_size=min_entries, max_size=8))
    entries = [
        {"chunk_index": index, "vector": values} for index, values in enumerate(vectors)
    ]
    return dimensions, entries


@given(case=embedding_entries())
def test_generated_embedding_records_preserve_alignment_and_round_trip(case) -> None:
    dimensions, entries = case

    record = make_record(dimensions=dimensions, embeddings=entries)

    assert DocumentEmbeddings.model_validate_json(record.model_dump_json()) == record
    assert [entry.chunk_index for entry in record.embeddings] == list(
        range(len(entries))
    )
    assert [entry.vector for entry in record.embeddings] == [
        entry["vector"] for entry in entries
    ]


@given(case=embedding_entries(min_entries=1))
def test_generated_dimension_and_index_mismatches_are_rejected(case) -> None:
    dimensions, entries = case

    with pytest.raises(ValidationError, match="vector length must match dimensions"):
        make_record(dimensions=dimensions + 1, embeddings=entries)

    invalid_entries = entries[:-1] + [{**entries[-1], "chunk_index": len(entries)}]
    with pytest.raises(ValidationError, match="sequential starting at zero"):
        make_record(dimensions=dimensions, embeddings=invalid_entries)
