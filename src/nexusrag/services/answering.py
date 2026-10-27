"""Coordinate saved semantic retrieval, bounded context, and cited answers."""

from pathlib import Path
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from nexusrag import config, storage
from nexusrag.schemas.answer import AskDocumentRequest, AskDocumentResponse
from nexusrag.services import context, generation, search
from nexusrag.services.embedding_runtime import EmbeddingRuntime
from nexusrag.services.generation_runtime import AnswerGenerator


class DocumentNotFoundError(RuntimeError):
    """No saved chunks or original PDF were found for the requested document."""


class DocumentChunksUnavailableError(RuntimeError):
    """The PDF exists, but its chunk record is unavailable."""


class SemanticIndexUnavailableError(RuntimeError):
    """The document needs an explicit semantic-index build before asking."""


async def answer_document(
    document_id: UUID,
    question: str,
    runtime: EmbeddingRuntime,
    generator: AnswerGenerator,
    *,
    storage_directory: Path,
    chunk_directory: Path,
    embedding_directory: Path,
    backend: search.VectorSearchBackend,
) -> AskDocumentResponse:
    """Use existing source/index snapshots; never save or rebuild during /ask."""
    question = AskDocumentRequest(question=question).question
    try:
        chunks = await storage.load_chunks(document_id, chunk_directory)
    except FileNotFoundError as error:
        pdf_path = storage_directory / f"{document_id}.pdf"
        if await run_in_threadpool(pdf_path.is_file):
            raise DocumentChunksUnavailableError(
                "Document chunks are not available"
            ) from error
        raise DocumentNotFoundError("Document not found") from error
    try:
        embeddings = await storage.load_embeddings(document_id, embedding_directory)
    except FileNotFoundError as error:
        raise SemanticIndexUnavailableError(
            "Semantic index is not available; build it before asking"
        ) from error

    hits = await search.semantic_search(
        chunks, embeddings, question, runtime, backend=backend, limit=5
    )
    prepared = await run_in_threadpool(
        context.build_context,
        hits,
        max_tokens=config.GENERATION_CONTEXT_TOKEN_BUDGET,
        count_tokens=generation.count_source_token_upper_bound,
    )

    answer = await generator.generate(question, prepared)

    if set(answer.citation_ids) - prepared.sources.keys():
        raise generation.InvalidGeneratedAnswerError(
            "The provider cited sources outside the supplied context"
        )
    return AskDocumentResponse(
        document_id=document_id,
        question=question,
        answer=answer.answer,
        status=answer.status,
        sources=[prepared.sources[label] for label in answer.citation_ids],
    )
