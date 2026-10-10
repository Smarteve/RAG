from hashlib import sha256

from nexusrag.constants import DEFAULT_EMBEDDING_BATCH_SIZE
from nexusrag.schemas.chunk import DocumentChunks
from nexusrag.schemas.embedding import ChunkEmbedding, DocumentEmbeddings
from nexusrag.services.embedding_model import MiniEmbedder


def embed_chunks(chunks: DocumentChunks, embedder: MiniEmbedder) -> DocumentEmbeddings:
    """Build a complete vector record without loading models or writing files.

    Each batch preserves chunk order and indices. Invalid provider output or
    model failures propagate; no partial record is returned or persisted.
    """
    dimensions = embedder.get_dimensions()
    source_digest = sha256(chunks.model_dump_json().encode("utf-8")).hexdigest()
    embeddings: list[ChunkEmbedding] = []

    for start in range(0, len(chunks.chunks), DEFAULT_EMBEDDING_BATCH_SIZE):
        batch = chunks.chunks[start : start + DEFAULT_EMBEDDING_BATCH_SIZE]
        vectors = embedder.embed_documents([chunk.text for chunk in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            embeddings.append(ChunkEmbedding(chunk_index=chunk.index, vector=vector))

    return DocumentEmbeddings(
        document_id=chunks.document_id,
        model_id=embedder.model_id,
        model_revision=embedder.model_revision,
        dimensions=dimensions,
        source_digest=source_digest,
        embeddings=embeddings,
    )
