from nexusrag.schemas.chunk import DocumentChunk
from nexusrag.schemas.extraction import DocumentExtraction


def chunk_document(
    extraction: DocumentExtraction,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Split successful extraction pages into overlapping character windows.

    Offsets refer to the original page text, with an exclusive end offset.
    Chunk indices are zero-based across the entire document.
    """
    if extraction.status == "failed":
        raise ValueError("Cannot chunk a failed extraction")

    if chunk_size <= 0 or not (0 <= overlap < chunk_size):
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    document_chunks: list[DocumentChunk] = []

    for page in extraction.pages:
        if not page.text.strip():
            continue
        start_char = 0
        while start_char < len(page.text):
            end_char = min(start_char + chunk_size, len(page.text))
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
            # Emit the final window before stopping, even when it is short.
            if end_char == len(page.text):
                break

            start_char += chunk_size - overlap

    return document_chunks
