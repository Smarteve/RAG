from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from nexusrag.schemas.chunk import DocumentChunk
from nexusrag.schemas.retrieval import SearchHit, SemanticSearchHit


@pytest.fixture
def chunk() -> DocumentChunk:
    return DocumentChunk(
        document_id=uuid4(),
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=5,
        text="Hello",
    )


def test_search_hit_round_trips_with_nested_chunk(chunk: DocumentChunk) -> None:
    hit = SearchHit(chunk=chunk, score=2)

    restored = SearchHit.model_validate_json(hit.model_dump_json())

    assert restored == hit
    assert isinstance(restored.chunk, DocumentChunk)
    assert restored.chunk == chunk
    assert restored.score == 2


@pytest.mark.parametrize("score", [0, -1, 1.5])
def test_invalid_score_is_rejected(chunk: DocumentChunk, score: float) -> None:
    with pytest.raises(ValidationError):
        SearchHit(chunk=chunk, score=score)


def test_search_hit_requires_a_chunk() -> None:
    with pytest.raises(ValidationError):
        SearchHit(score=1)


def test_search_hit_validates_nested_chunk(chunk: DocumentChunk) -> None:
    invalid_chunk = chunk.model_dump() | {"end_char": 6}

    with pytest.raises(ValidationError):
        SearchHit.model_validate({"chunk": invalid_chunk, "score": 1})


@pytest.mark.parametrize("similarity", [-1.0, -0.5, 0.0, 0.8, 1.0])
def test_semantic_hit_accepts_cosine_scores(
    chunk: DocumentChunk, similarity: float
) -> None:
    hit = SemanticSearchHit(chunk=chunk, similarity=similarity)

    assert hit.chunk == chunk
    assert hit.similarity == similarity


@pytest.mark.parametrize(
    "similarity",
    [-1.000001, 1.000001, float("nan"), float("inf"), -float("inf"), "not a number"],
    ids=[
        "below-minimum",
        "above-maximum",
        "nan",
        "infinity",
        "negative-infinity",
        "nonnumeric",
    ],
)
def test_semantic_hit_rejects_invalid_similarity(
    chunk: DocumentChunk, similarity
) -> None:
    with pytest.raises(ValidationError):
        SemanticSearchHit(chunk=chunk, similarity=similarity)


def test_semantic_hit_round_trips_with_nested_chunk(chunk: DocumentChunk) -> None:
    hit = SemanticSearchHit(chunk=chunk, similarity=-0.25)

    restored = SemanticSearchHit.model_validate_json(hit.model_dump_json())

    assert restored == hit
    assert isinstance(restored.chunk, DocumentChunk)
    assert restored.chunk == chunk


@pytest.mark.parametrize("missing_field", ["chunk", "similarity"])
def test_semantic_hit_requires_both_fields(
    chunk: DocumentChunk, missing_field: str
) -> None:
    fields = {"chunk": chunk, "similarity": 0.5}
    del fields[missing_field]

    with pytest.raises(ValidationError):
        SemanticSearchHit.model_validate(fields)


def test_semantic_hit_validates_nested_chunk(chunk: DocumentChunk) -> None:
    invalid_chunk = chunk.model_dump() | {"end_char": 6}

    with pytest.raises(ValidationError):
        SemanticSearchHit.model_validate({"chunk": invalid_chunk, "similarity": 0.5})


@given(similarity=st.floats(min_value=-1, max_value=1, allow_nan=False))
def test_generated_semantic_scores_round_trip_without_loss(similarity: float) -> None:
    # Construct here rather than sharing a function-scoped fixture between examples.
    source_chunk = DocumentChunk(
        document_id=uuid4(),
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=5,
        text="Hello",
    )
    hit = SemanticSearchHit(chunk=source_chunk, similarity=similarity)

    restored = SemanticSearchHit.model_validate_json(hit.model_dump_json())

    assert restored == hit
    assert restored.similarity == similarity
