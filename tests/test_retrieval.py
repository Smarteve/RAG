from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nexusrag.schemas.chunk import DocumentChunk
from nexusrag.services.retrieval import _tokenize, search_chunks


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Annual LEAVE, annual!", {"annual", "leave"}),
        ("Café STRASSE Straße 中文", {"café", "strasse", "中文"}),
        ("leave_annual carry-over 25", {"leave", "annual", "carry", "over", "25"}),
        ("", set()),
        (" \t\n", set()),
        ("--?!___🚀", set()),
    ],
    ids=[
        "case-and-duplicates",
        "unicode",
        "separators-and-numbers",
        "empty",
        "whitespace",
        "symbols",
    ],
)
def test_tokenize(text: str, expected: set[str]) -> None:
    assert _tokenize(text) == expected


@given(text=st.text(max_size=300))
def test_generated_tokens_are_normalized_and_repetition_does_not_change_them(
    text: str,
) -> None:
    tokens = _tokenize(text)

    assert all(
        token and token.isalnum() and token == token.casefold() for token in tokens
    )
    assert _tokenize(" ".join(sorted(tokens))) == tokens
    assert _tokenize(f"{text} {text}") == tokens


def make_chunks(*texts: str) -> list[DocumentChunk]:
    document_id = uuid4()
    return [
        DocumentChunk(
            document_id=document_id,
            page_number=1,
            chunk_index=index,
            start_char=0,
            end_char=len(text),
            text=text,
        )
        for index, text in enumerate(texts)
    ]


def test_search_ranks_by_score_then_index_and_limits_results() -> None:
    chunks = make_chunks("annual leave", "annual", "leave", "remote work")

    hits = search_chunks(list(reversed(chunks)), "annual leave", limit=2)

    assert [(hit.chunk.chunk_index, hit.score) for hit in hits] == [(0, 2), (1, 1)]
    assert [hit.chunk for hit in hits] == chunks[:2]


def test_search_counts_distinct_words_not_repeated_occurrences() -> None:
    chunks = make_chunks("annual annual annual", "annual leave leave")

    hits = search_chunks(chunks, "ANNUAL annual leave")

    assert [(hit.chunk.chunk_index, hit.score) for hit in hits] == [(1, 2), (0, 1)]


def test_search_without_whole_word_matches_returns_no_hits() -> None:
    chunks = make_chunks("remote work", "cleave", " \t\n")

    assert search_chunks(chunks, "leave") == []


def test_search_empty_chunks_with_valid_query_returns_no_hits() -> None:
    assert search_chunks([], "annual leave") == []


def test_search_defaults_to_at_most_five_hits() -> None:
    chunks = make_chunks(*(["annual leave"] * 6))

    hits = search_chunks(chunks, "leave")

    assert [hit.chunk.chunk_index for hit in hits] == [0, 1, 2, 3, 4]


def test_search_normalizes_unicode_words() -> None:
    chunks = make_chunks("Café", "Straße", "remote work")

    hits = search_chunks(chunks, "CAFÉ STRASSE")

    assert [(hit.chunk.chunk_index, hit.score) for hit in hits] == [(0, 1), (1, 1)]


@pytest.mark.parametrize("query", ["", " \t\n", "?!___🚀"])
def test_search_rejects_queries_without_words_even_without_chunks(query: str) -> None:
    with pytest.raises(ValueError, match="query must contain"):
        search_chunks([], query)


@pytest.mark.parametrize("limit", [0, -1])
def test_search_rejects_nonpositive_limits_even_without_chunks(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        search_chunks([], "annual leave", limit)


search_words = st.sampled_from(
    [
        "annual",
        "ANNUAL",
        "leave",
        "LEAVE",
        "remote",
        "work",
        "Café",
        "CAFÉ",
        "Straße",
        "STRASSE",
        "中文",
        "cleave",
    ]
)


@given(
    chunk_words=st.lists(st.lists(search_words, min_size=1, max_size=8), max_size=15),
    query_words=st.lists(search_words, min_size=1, max_size=6),
    limit=st.integers(min_value=1, max_value=20),
)
def test_generated_search_matches_reference_ranking_and_preserves_input(
    chunk_words: list[list[str]], query_words: list[str], limit: int
) -> None:
    chunks = make_chunks(*(" ".join(words) for words in chunk_words))
    # Deliberately scramble document order to exercise the index tie-breaker.
    input_chunks = list(reversed(chunks))
    original = [chunk.model_dump() for chunk in input_chunks]
    query = " ".join(query_words)

    hits = search_chunks(input_chunks, query, limit)

    # A plain full sort is an independent reference for the heap selection.
    query_terms = {word.casefold() for word in query_words}
    expected = []
    for index, words in enumerate(chunk_words):
        score = len(query_terms & {word.casefold() for word in words})
        if score:
            expected.append((index, score))
    expected = sorted(expected, key=lambda item: (-item[1], item[0]))[:limit]

    assert [(hit.chunk.chunk_index, hit.score) for hit in hits] == expected
    assert len(hits) <= limit
    assert all(
        hit.score > 0 and hit.chunk == chunks[hit.chunk.chunk_index] for hit in hits
    )
    assert [chunk.model_dump() for chunk in input_chunks] == original
    assert search_chunks(input_chunks, query, limit) == hits
