from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from nexusrag.schemas.answer import AnswerSource
from nexusrag.schemas.retrieval import SemanticSearchHit


@dataclass
class PreparedContext:
    """Assembled source text and the lookup used to resolve LLM citations."""

    text: str
    sources: dict[str, AnswerSource]


def build_context(
    hits: list[SemanticSearchHit],
    *,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> PreparedContext:
    """Include whole source blocks in retrieval order within a token budget.

    The caller supplies the generation model's token counter and a source-only
    budget, after reserving room for instructions, the question, and output.
    Oversized blocks are skipped, not truncated; smaller later hits can still
    fit. Repeated document/chunk identities are included at most once.
    No model is loaded here, and inputs are not mutated.
    """
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")

    blocks: list[str] = []
    sources: dict[str, AnswerSource] = {}
    included_chunks: set[tuple[UUID, int]] = set()
    for hit in hits:
        chunk = hit.chunk
        identity = (chunk.document_id, chunk.chunk_index)
        if identity in included_chunks:
            continue

        source_id = f"S{len(sources) + 1}"
        block = f"[{source_id}]\nPage: {chunk.page_number}\nText:\n{chunk.text}"
        candidate_text = "\n\n".join([*blocks, block])
        candidate_tokens = count_tokens(candidate_text)
        if (
            isinstance(candidate_tokens, bool)
            or not isinstance(candidate_tokens, int)
            or candidate_tokens < 0
        ):
            raise ValueError("count_tokens must return a nonnegative integer")
        if candidate_tokens > max_tokens:
            continue

        blocks.append(block)
        sources[source_id] = AnswerSource(
            source_id=source_id,
            document_id=chunk.document_id,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
        )
        included_chunks.add(identity)

    text = "\n\n".join(blocks)

    return PreparedContext(text=text, sources=sources)
