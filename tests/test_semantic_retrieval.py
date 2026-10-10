from hashlib import sha256
from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nexusrag.schemas.chunk import DocumentChunk, DocumentChunks
from nexusrag.schemas.embedding import ChunkEmbedding, DocumentEmbeddings
from nexusrag.services.retrieval import search_semantic_chunks
from nexusrag.services.similarity import cosine_similarity


def make_records(
    vectors: list[list[float]], dimensions: int = 2
) -> tuple[DocumentChunks, DocumentEmbeddings]:
    document_id = uuid4()
    texts = [f"Passage {index}: café" for index in range(len(vectors))]
    chunks = DocumentChunks(
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
    embeddings = DocumentEmbeddings(
        document_id=document_id,
        model_id="test-encoder",
        model_revision="test-revision",
        dimensions=dimensions,
        source_digest=sha256(chunks.model_dump_json().encode("utf-8")).hexdigest(),
        embeddings=[
            ChunkEmbedding(chunk_index=index, vector=vector)
            for index, vector in enumerate(vectors)
        ],
    )
    return chunks, embeddings


def test_search_ranks_positive_zero_and_negative_scores() -> None:
    chunks, embeddings = make_records([[-1, 0], [0, 1], [1, 0]])

    hits = search_semantic_chunks(chunks, embeddings, [1, 0])

    assert [hit.chunk.chunk_index for hit in hits] == [2, 1, 0]
    assert [hit.similarity for hit in hits] == pytest.approx([1, 0, -1])
    assert hits[0].chunk == chunks.chunks[2]
    assert hits[0].chunk.page_number == 3
    assert hits[0].chunk.text == "Passage 2: café"


def test_ties_use_earlier_chunk_index_and_default_limit_is_five() -> None:
    chunks, embeddings = make_records([[1, 0]] * 7)

    hits = search_semantic_chunks(chunks, embeddings, [1, 0])

    assert [hit.chunk.chunk_index for hit in hits] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("limit, expected", [(1, [2]), (2, [2, 1]), (10, [2, 1, 0])])
def test_limit_keeps_the_highest_scores(limit: int, expected: list[int]) -> None:
    chunks, embeddings = make_records([[-1, 0], [0, 1], [1, 0]])

    hits = search_semantic_chunks(chunks, embeddings, [1, 0], limit)

    assert [hit.chunk.chunk_index for hit in hits] == expected


def test_empty_document_returns_no_hits_for_valid_inputs() -> None:
    chunks, embeddings = make_records([])

    assert search_semantic_chunks(chunks, embeddings, [1, 0]) == []


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_limit_is_rejected(empty: bool, limit: int) -> None:
    chunks, embeddings = make_records([] if empty else [[1, 0]])

    with pytest.raises(ValueError, match="limit must be positive"):
        search_semantic_chunks(chunks, embeddings, [1, 0], limit)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "query, message",
    [
        ([], "must not be empty"),
        ([0, 0], "nonzero"),
        ([float("nan"), 1], "finite"),
        ([float("inf"), 1], "finite"),
        ([-float("inf"), 1], "finite"),
        ([1], "match embedding dimensions"),
        ([1, 0, 0], "match embedding dimensions"),
    ],
)
def test_invalid_queries_are_rejected(
    empty: bool, query: list[float], message: str
) -> None:
    chunks, embeddings = make_records([] if empty else [[1, 0]])

    with pytest.raises(ValueError, match=message):
        search_semantic_chunks(chunks, embeddings, query)


@pytest.mark.parametrize("empty", [False, True])
def test_different_document_ids_are_rejected(empty: bool) -> None:
    chunks, embeddings = make_records([] if empty else [[1, 0]])
    embeddings.document_id = uuid4()

    with pytest.raises(ValueError, match="same document"):
        search_semantic_chunks(chunks, embeddings, [1, 0])


@pytest.mark.parametrize("vector_count", [0, 2])
def test_missing_or_extra_embeddings_are_rejected(vector_count: int) -> None:
    chunks, _ = make_records([[1, 0]])
    _, embeddings = make_records([[1, 0]] * vector_count)
    embeddings.document_id = chunks.document_id
    embeddings.source_digest = sha256(
        chunks.model_dump_json().encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="exactly one embedding"):
        search_semantic_chunks(chunks, embeddings, [1, 0])


@pytest.mark.parametrize("change", ["text", "settings", "digest"])
def test_stale_embeddings_are_rejected(change: str) -> None:
    chunks, embeddings = make_records([[1, 0]])
    if change == "text":
        chunks.chunks[0].text = chunks.chunks[0].text.replace("café", "tea!")
    elif change == "settings":
        chunks.overlap = 1
    else:
        embeddings.source_digest = "0" * 64

    with pytest.raises(ValueError, match="source digest"):
        search_semantic_chunks(chunks, embeddings, [1, 0])


def test_mismatched_indices_after_mutation_are_rejected() -> None:
    chunks, embeddings = make_records([[1, 0], [0, 1]])
    # Normal construction rejects this; mutable Pydantic models can change later.
    embeddings.embeddings.reverse()

    with pytest.raises(ValueError, match="indices must match"):
        search_semantic_chunks(chunks, embeddings, [1, 0])


@pytest.mark.parametrize(
    "vector, message",
    [([0, 0], "nonzero"), ([float("nan"), 1], "finite"), ([1], "equal dimensions")],
)
def test_invalid_stored_vector_after_mutation_is_rejected(
    vector: list[float], message: str
) -> None:
    chunks, embeddings = make_records([[1, 0]])
    embeddings.embeddings[0].vector = vector

    with pytest.raises(ValueError, match=message):
        search_semantic_chunks(chunks, embeddings, [1, 0])


@st.composite
def search_cases(draw):
    dimensions = draw(st.integers(min_value=1, max_value=6))
    vector = st.lists(
        st.integers(min_value=-5, max_value=5),
        min_size=dimensions,
        max_size=dimensions,
    ).filter(any)
    vectors = draw(st.lists(vector, max_size=15))
    query = draw(vector)
    limit = draw(st.integers(min_value=1, max_value=20))
    return dimensions, vectors, query, limit


@given(case=search_cases())
def test_generated_heap_ranking_matches_full_sort_and_preserves_inputs(case) -> None:
    dimensions, vectors, query, limit = case
    chunks, embeddings = make_records(vectors, dimensions)
    before_chunks = chunks.model_dump_json()
    before_embeddings = embeddings.model_dump_json()
    before_query = query.copy()
    # Full sort is an independent reference for heap selection. The cosine
    # calculation itself has separate mathematical/property tests.
    expected = sorted(
        [
            (index, cosine_similarity(query, vector))
            for index, vector in enumerate(vectors)
        ],
        key=lambda item: (-item[1], item[0]),
    )[:limit]

    hits = search_semantic_chunks(chunks, embeddings, query, limit)

    assert [hit.chunk.chunk_index for hit in hits] == [index for index, _ in expected]
    assert [hit.similarity for hit in hits] == pytest.approx(
        [score for _, score in expected]
    )
    assert all(hit.chunk == chunks.chunks[hit.chunk.chunk_index] for hit in hits)
    assert chunks.model_dump_json() == before_chunks
    assert embeddings.model_dump_json() == before_embeddings
    assert query == before_query
