from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag.schemas.chunk import DocumentChunk
from nexusrag.schemas.retrieval import SearchHit


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
