import re
from hashlib import sha256
from heapq import nlargest, nsmallest

from nexusrag.schemas.chunk import DocumentChunk, DocumentChunks
from nexusrag.schemas.embedding import DocumentEmbeddings
from nexusrag.schemas.retrieval import SearchHit, SemanticSearchHit
from nexusrag.services.similarity import cosine_similarity

_WORD_PATTERN = re.compile(r"[^\W_]+")


class InvalidSearchQueryError(ValueError):
    """The user query contains no searchable words, not an internal model error."""


class StaleSemanticIndexError(ValueError):
    """The saved semantic index needs rebuilding for current chunks/model."""


def validate_semantic_index(
    chunks: DocumentChunks, embeddings: DocumentEmbeddings
) -> None:
    """Check source alignment before inference; leave model compatibility to caller."""
    if chunks.document_id != embeddings.document_id:
        raise ValueError("chunks and embeddings must belong to the same document")
    if len(chunks.chunks) != len(embeddings.embeddings):
        raise StaleSemanticIndexError("every chunk must have exactly one embedding")
    source_digest = sha256(chunks.model_dump_json().encode("utf-8")).hexdigest()
    if source_digest != embeddings.source_digest:
        raise StaleSemanticIndexError(
            "embeddings are stale: source digest does not match chunks"
        )
    for chunk, embedding in zip(chunks.chunks, embeddings.embeddings, strict=True):
        if chunk.chunk_index != embedding.chunk_index:
            raise ValueError("chunk and embedding indices must match")


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


def search_semantic_chunks(
    chunks: DocumentChunks,
    embeddings: DocumentEmbeddings,
    query_vector: list[float],
    limit: int = 5,
) -> list[SemanticSearchHit]:
    """Rank aligned chunk vectors by cosine similarity, then chunk index.

    Chunks supply the original text and page references; embeddings supply the
    vectors used for scoring. Reject mismatched or stale records with ValueError.
    Validate inputs even for an empty document, and keep zero/negative scores.
    The caller must ensure the query uses the same model and revision as the
    stored embeddings: a bare query vector carries no model identity.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    # Reuse the mathematical validation, even when there are no chunks to score.
    cosine_similarity(query_vector, query_vector)
    if len(query_vector) != embeddings.dimensions:
        raise ValueError("query vector length must match embedding dimensions")
    validate_semantic_index(chunks, embeddings)

    semantic_hits: list[SemanticSearchHit] = []

    for chunk, embedding in zip(chunks.chunks, embeddings.embeddings, strict=True):
        similarity = cosine_similarity(query_vector, embedding.vector)
        semantic_hits.append(SemanticSearchHit(chunk=chunk, similarity=similarity))

    return nlargest(
        limit, semantic_hits, key=lambda hit: (hit.similarity, -hit.chunk.chunk_index)
    )
