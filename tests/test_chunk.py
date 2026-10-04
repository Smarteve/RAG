from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag.schemas.chunk import DocumentChunk


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
