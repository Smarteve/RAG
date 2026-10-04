import re
from heapq import nsmallest

from nexusrag.schemas.chunk import DocumentChunk
from nexusrag.schemas.retrieval import SearchHit

_WORD_PATTERN = re.compile(r"[^\W_]+")


class InvalidSearchQueryError(ValueError):
    """The user query contains no searchable words, not an internal model error."""


def _tokenize(query: str) -> set[str]:
    """Return distinct case-normalized words for simple keyword matching.

    Unicode letters and numbers are kept; punctuation and underscores separate
    tokens. Empty or punctuation-only input produces an empty set. This is not
    a linguistic tokenizer: it does not stem words or remove stop words.
    The same normalization can be used on both queries and chunk text.
    """
    return set(_WORD_PATTERN.findall(query.casefold()))


def validate_query(query: str) -> set[str]:
    """Validate before storage access and return the normalized query words."""
    tokens = _tokenize(query)
    if not tokens:
        raise InvalidSearchQueryError("query must contain at least one searchable word")
    return tokens


def search_chunks(
    chunks: list[DocumentChunk],
    query: str,
    limit: int = 5,
) -> list[SearchHit]:
    """Rank matching chunks by distinct query-word count, then chunk index.

    Return at most limit hits. Reject queries without searchable words and
    nonpositive limits, even when chunks is empty. Input chunks are not mutated.
    """
    query_tokens = validate_query(query)
    if limit <= 0:
        raise ValueError("limit must be positive")
    hits: list[SearchHit] = []

    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.text)
        score = sum(1 for token in query_tokens if token in chunk_tokens)
        if score > 0:
            hits.append(SearchHit(chunk=chunk, score=score))

    return nsmallest(limit, hits, key=lambda hit: (-hit.score, hit.chunk.chunk_index))
