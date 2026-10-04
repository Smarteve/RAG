from unittest.mock import Mock
from uuid import uuid4

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from pydantic import ValidationError

from nexusrag.schemas.chunk import DocumentChunk
from nexusrag.schemas.extraction import DocumentExtraction, ExtractedPage
from nexusrag.services import chunking
from nexusrag.services.chunking import chunk_document


def make_extraction(*texts: str) -> DocumentExtraction:
    return DocumentExtraction(
        document_id=uuid4(),
        pages=[
            ExtractedPage(page_number=index, text=text)
            for index, text in enumerate(texts, start=1)
        ],
    )


@pytest.mark.parametrize(
    "text, chunk_size, overlap, expected",
    [
        ("ABCDE", 5, 2, [(0, 5, "ABCDE")]),
        ("ABCDEF", 5, 2, [(0, 5, "ABCDE"), (3, 6, "DEF")]),
        (
            "ABCDEFGHIJK",
            5,
            2,
            [(0, 5, "ABCDE"), (3, 8, "DEFGH"), (6, 11, "GHIJK")],
        ),
    ],
    ids=[
        "exact-size",
        "short-final-window",
        "overlapping",
    ],
)
def test_chunk_boundaries(
    text: str,
    chunk_size: int,
    overlap: int,
    expected: list[tuple[int, int, str]],
) -> None:
    extraction = make_extraction(text)

    chunks = chunk_document(extraction, chunk_size, overlap)

    assert [
        (chunk.start_char, chunk.end_char, chunk.text) for chunk in chunks
    ] == expected
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.document_id == extraction.document_id for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_default_settings_accept_short_page() -> None:
    chunks = chunk_document(make_extraction("Hello"))

    assert [chunk.text for chunk in chunks] == ["Hello"]


def test_pages_reset_offsets_and_preserve_page_numbers() -> None:
    extraction = DocumentExtraction(
        document_id=uuid4(),
        pages=[
            ExtractedPage(page_number=2, text="ABCDEFG"),
            ExtractedPage(page_number=4, text=" \t\n"),
            ExtractedPage(page_number=7, text=""),
            ExtractedPage(page_number=9, text="XYZ"),
        ],
    )

    chunks = chunk_document(extraction, chunk_size=5, overlap=2)

    assert [chunk.text for chunk in chunks] == ["ABCDE", "DEFG", "XYZ"]
    assert [chunk.page_number for chunk in chunks] == [2, 2, 9]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.start_char for chunk in chunks] == [0, 3, 0]


def test_failed_extraction_is_rejected() -> None:
    extraction = DocumentExtraction(
        document_id=uuid4(), pages=[], status="failed", error="pdf_read_failed"
    )

    with pytest.raises(ValueError, match="Cannot chunk a failed extraction"):
        chunk_document(extraction)


@pytest.mark.parametrize(
    "chunk_size, overlap", [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)]
)
def test_invalid_settings_are_rejected_even_without_pages(
    chunk_size: int, overlap: int
) -> None:
    with pytest.raises(ValueError, match="Require chunk_size"):
        chunk_document(make_extraction(), chunk_size, overlap)


def test_chunk_validation_error_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError) as exc:
        DocumentChunk(
            document_id=uuid4(),
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=5,
            text="wrong length",
        )
    error = exc.value
    monkeypatch.setattr(chunking, "DocumentChunk", Mock(side_effect=error))

    with pytest.raises(ValidationError) as propagated:
        chunk_document(make_extraction("Hello"))

    assert propagated.value is error


@st.composite
def valid_windows(draw: st.DrawFn) -> tuple[int, int]:
    chunk_size = draw(st.integers(min_value=1, max_value=80))
    overlap = draw(st.integers(min_value=0, max_value=chunk_size - 1))
    return chunk_size, overlap


@given(
    texts=st.lists(st.text(max_size=150), max_size=5),
    window=valid_windows(),
)
# These cases always run, alongside the generated examples.
@example(texts=[], window=(1, 0))
@example(texts=["", " \t\n", "\u2003"], window=(5, 2))
@example(texts=["Hi"], window=(5, 2))
@example(texts=["ABCDEFGHIJK"], window=(5, 0))
@example(texts=["ABC"], window=(1, 0))
@example(texts=["ABCDEF"], window=(5, 4))
@example(texts=["  Hi  "], window=(10, 2))
@example(texts=["A\U0001f680\u00e9\u4e2dB"], window=(3, 1))
def test_generated_chunks_preserve_source_coverage_and_input(
    texts: list[str], window: tuple[int, int]
) -> None:
    chunk_size, overlap = window
    stride = chunk_size - overlap
    extraction = make_extraction(*texts)
    original = extraction.model_dump()

    chunks = chunk_document(extraction, chunk_size, overlap)

    assert extraction.model_dump() == original
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.document_id == extraction.document_id for chunk in chunks)
    assert [chunk.page_number for chunk in chunks] == sorted(
        chunk.page_number for chunk in chunks
    )
    for page in extraction.pages:
        page_chunks = [
            chunk for chunk in chunks if chunk.page_number == page.page_number
        ]
        if not page.text.strip():
            assert page_chunks == []
            continue

        # There is one initial window, then enough strides to reach the end.
        remaining = max(0, len(page.text) - chunk_size)
        expected_count = 1 + (remaining + stride - 1) // stride
        assert len(page_chunks) == expected_count
        assert page_chunks[0].start_char == 0
        assert page_chunks[-1].end_char == len(page.text)
        reconstructed = page_chunks[0].text
        for index, chunk in enumerate(page_chunks):
            assert chunk.start_char == index * stride
            assert chunk.end_char == min(chunk.start_char + chunk_size, len(page.text))
            assert 0 < len(chunk.text) <= chunk_size
            assert chunk.text == page.text[chunk.start_char : chunk.end_char]
            if index:
                previous = page_chunks[index - 1]
                assert previous.end_char - chunk.start_char == overlap
                reconstructed += chunk.text[overlap:]
        assert reconstructed == page.text
