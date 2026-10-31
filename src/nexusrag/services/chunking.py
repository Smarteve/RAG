from collections.abc import Callable, Iterator

from nexusrag.constants import DEFAULT_CHUNK_OVERLAP_CHARS, DEFAULT_CHUNK_SIZE_CHARS
from nexusrag.schemas.chunk import DocumentChunk
from nexusrag.schemas.extraction import DocumentExtraction


def chunk_document(
    extraction: DocumentExtraction,
    chunk_size: int = DEFAULT_CHUNK_SIZE_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP_CHARS,
    *,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int | None = None,
) -> list[DocumentChunk]:
    """Split pages into character windows, optionally fitted to a token budget.

    Offsets refer to the original page text, with an exclusive end offset.
    Chunk indices are zero-based across the entire document. The injected counter
    counts the exact text including special tokens. Without a counter/budget,
    existing character-window behavior is preserved. Overlap is reduced only
    when needed to advance past a chunk shorter than the requested overlap.
    """
    if extraction.status == "failed":
        raise ValueError("Cannot chunk a failed extraction")

    if chunk_size <= 0 or not (0 <= overlap < chunk_size):
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    if (count_tokens is None) != (max_tokens is None):
        raise ValueError("Provide both count_tokens and max_tokens")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("Require max_tokens > 0")

    document_chunks: list[DocumentChunk] = []

    for page in extraction.pages:
        if not page.text.strip():
            continue
        for start_char, end_char in _split_text(
            page.text, chunk_size, overlap, count_tokens, max_tokens
        ):
            text = page.text[start_char:end_char]
            document_chunks.append(
                DocumentChunk(
                    document_id=extraction.document_id,
                    page_number=page.page_number,
                    chunk_index=len(document_chunks),
                    start_char=start_char,
                    end_char=end_char,
                    text=text,
                )
            )
    return document_chunks


def _split_text(
    text: str,
    chunk_size: int,
    overlap: int,
    count_tokens: Callable[[str], int] | None,
    max_tokens: int | None,
) -> Iterator[tuple[int, int]]:
    """Yield source spans using settings validated by chunk_document."""
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if count_tokens is not None and max_tokens is not None:
            # Blank-only windows need no embedding. Retain whitespace inside
            # nonblank chunks so their text still matches the original offsets.
            if not text[start:end].strip():
                start = end
                continue
            end = _fit_end(text, start, end, count_tokens, max_tokens)
            if not text[start:end].strip():
                start = end
                continue
        yield start, end
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def _fit_end(
    text: str,
    start: int,
    maximum_end: int,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> int:
    """Find a fitting prefix; it need not be the longest possible prefix."""
    end = maximum_end
    while end > start:
        tokens = count_tokens(text[start:end])
        if tokens <= max_tokens:
            return end
        # Estimate a smaller window, then count that exact substring again.
        width = end - start
        end = start + min(width - 1, max(1, width * max_tokens // tokens))

    # Token counts can decrease when extra characters complete a token. If the
    # estimates found no fit, inspect skipped prefixes before rejecting text.
    for end in range(maximum_end - 1, start, -1):
        if count_tokens(text[start:end]) <= max_tokens:
            return end
    raise ValueError(f"Text at character {start} cannot fit the token budget")
