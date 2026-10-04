from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag.schemas.chunk import DocumentChunk, DocumentChunks


def test_valid_chunk_preserves_text_and_offsets() -> None:
    chunk = DocumentChunk(
        document_id=uuid4(),
        page_number=2,
        chunk_index=0,
        start_char=3,
        end_char=8,
        text=" Hi! ",
    )

    assert chunk.text == " Hi! "
    assert (chunk.start_char, chunk.end_char) == (3, 8)


@pytest.mark.parametrize(
    "changes",
    [
        {"page_number": 0},
        {"chunk_index": -1},
        {"start_char": -1},
        {"end_char": 0},
        {"start_char": 5},
        {"start_char": 6},
        {"end_char": 6},
        {"text": ""},
    ],
    ids=[
        "invalid-page-number",
        "negative-chunk-index",
        "negative-start",
        "nonpositive-end",
        "equal-offsets",
        "reversed-offsets",
        "length-mismatch",
        "empty-text",
    ],
)
def test_invalid_chunk_is_rejected(changes: dict[str, int | str]) -> None:
    fields = {
        "document_id": uuid4(),
        "page_number": 1,
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 5,
        "text": "Hello",
    }

    with pytest.raises(ValidationError):
        DocumentChunk(**(fields | changes))


def test_empty_chunk_record_is_valid() -> None:
    record = DocumentChunks(document_id=uuid4(), chunk_size=5, overlap=2, chunks=[])

    assert record.chunks == []


@pytest.mark.parametrize(
    "case",
    [
        "zero_size",
        "negative_overlap",
        "equal_overlap",
        "wrong_id",
        "gap",
        "duplicate",
        "out_of_order",
        "too_long",
    ],
)
def test_invalid_chunk_record_is_rejected(case: str) -> None:
    document_id = uuid4()
    fields = {
        "document_id": document_id,
        "chunk_size": 5,
        "overlap": 2,
        "chunks": [
            {
                "document_id": document_id,
                "page_number": 1,
                "chunk_index": index,
                "start_char": index * 3,
                "end_char": index * 3 + 5,
                "text": "Hello",
            }
            for index in range(2)
        ],
    }
    if case == "zero_size":
        fields["chunk_size"] = 0
    elif case == "negative_overlap":
        fields["overlap"] = -1
    elif case == "equal_overlap":
        fields["overlap"] = 5
    elif case == "wrong_id":
        fields["chunks"][0]["document_id"] = uuid4()
    elif case == "gap":
        fields["chunks"][1]["chunk_index"] = 2
    elif case == "duplicate":
        fields["chunks"][1]["chunk_index"] = 0
    elif case == "out_of_order":
        fields["chunks"].reverse()
    else:
        fields["chunk_size"] = 4

    with pytest.raises(ValidationError):
        DocumentChunks(**fields)
