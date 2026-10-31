"""Read-only retrieval evaluation with source-grounded relevance labels."""

from hashlib import sha256
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from nexusrag import config
from nexusrag.schemas.evaluation import (
    RetrievalCase,
    RetrievalCaseResult,
    RetrievalDataset,
    RetrievalEvaluationReport,
)
from nexusrag.schemas.extraction import DocumentExtraction
from nexusrag.schemas.retrieval import SemanticSearchHit
from nexusrag.services.document_state import is_document_deleted
from nexusrag.services.embedding_runtime import EmbeddingRuntime
from nexusrag.services.job_storage import load_document_job
from nexusrag.services.processing_lock import document_lock
from nexusrag.services.search import VectorSearchBackend, semantic_search
from nexusrag.storage import load_chunks, load_embeddings, load_extraction


class EvaluationInputError(ValueError):
    """The document or labels are not suitable for this evaluation."""


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def score_case(
    case: RetrievalCase, hits: list[SemanticSearchHit], top_k: int
) -> RetrievalCaseResult:
    """A relevant hit contains an expected phrase on its expected page.

    Multiple expected passages are alternatives. Unsupported cases are retained
    for inspection, with no success/failure assigned by retrieval alone.
    """
    selected = hits[:top_k]
    rank = None
    if case.answerable:
        rank = next(
            (
                rank
                for rank, hit in enumerate(selected, 1)
                if any(
                    expected.page_number == hit.chunk.page_number
                    and _normalized(expected.text) in _normalized(hit.chunk.text)
                    for expected in case.expected_passages
                )
            ),
            None,
        )
    return RetrievalCaseResult(case=case, first_relevant_rank=rank, hits=selected)


def retrieval_metrics(
    results: list[RetrievalCaseResult],
) -> dict[str, int | float | None]:
    ranks = [result.first_relevant_rank for result in results if result.case.answerable]
    count = len(ranks)
    hits = sum(rank is not None for rank in ranks)
    reciprocal_sum = sum(1 / rank if rank else 0 for rank in ranks)
    return {
        "answerable_cases": count,
        "unsupported_cases": len(results) - count,
        "hit_rate_at_k": hits / count if count else None,
        "mrr_at_k": reciprocal_sum / count if count else None,
    }


def validate_expected_passages(
    dataset: RetrievalDataset, extraction: DocumentExtraction
) -> None:
    if extraction.status != "completed":
        raise EvaluationInputError("Document extraction must be completed")
    pages = {page.page_number: _normalized(page.text) for page in extraction.pages}
    for case in dataset.cases:
        for expected in case.expected_passages:
            if _normalized(expected.text) not in pages.get(expected.page_number, ""):
                raise EvaluationInputError(
                    f"Expected passage for case {case.case_id} is absent from its labeled page"
                )


async def evaluate_retrieval(
    document_id: UUID,
    dataset: RetrievalDataset,
    runtime: EmbeddingRuntime,
    *,
    backend: VectorSearchBackend,
    top_k: int = 5,
) -> RetrievalEvaluationReport:
    """Evaluate an existing index without changing artifacts, jobs, or vectors.

    Hold document ownership to keep deletion/reindexing from changing the source
    snapshot during the run. The caller owns the model/backend lifecycle. Any
    operational failure aborts the report rather than dropping a difficult case.
    """
    if not 1 <= top_k <= 20:
        raise ValueError("Require 1 <= top_k <= 20")
    guard = document_lock(document_id, config.JOB_DATABASE_PATH)
    await run_in_threadpool(guard.__enter__)
    try:
        if await is_document_deleted(document_id, config.JOB_DATABASE_PATH):
            raise EvaluationInputError("Document has been deleted")
        job = await load_document_job(document_id, config.JOB_DATABASE_PATH)
        if job is not None and job.status != "succeeded":
            raise EvaluationInputError(
                "Document must finish processing before evaluation"
            )
        path = config.STORAGE_DIRECTORY / f"{document_id}.pdf"
        digest = await run_in_threadpool(lambda: sha256(path.read_bytes()).hexdigest())
        if digest != dataset.document_sha256:
            raise EvaluationInputError(
                "Document PDF does not match the evaluation dataset"
            )
        extraction = await load_extraction(document_id, config.EXTRACTION_DIRECTORY)
        validate_expected_passages(dataset, extraction)
        chunks = await load_chunks(document_id, config.CHUNK_DIRECTORY)
        embeddings = await load_embeddings(document_id, config.EMBEDDING_DIRECTORY)
        results = []
        for case in dataset.cases:
            hits = await semantic_search(
                chunks, embeddings, case.question, runtime, backend=backend, limit=top_k
            )
            results.append(score_case(case, hits, top_k))
        return RetrievalEvaluationReport(
            dataset_id=dataset.dataset_id,
            document_id=document_id,
            document_sha256=digest,
            source_digest=embeddings.source_digest,
            index_generation=embeddings.index_generation,
            model_id=embeddings.model_id,
            model_revision=embeddings.model_revision,
            chunk_size=chunks.chunk_size,
            overlap=chunks.overlap,
            chunk_count=len(chunks.chunks),
            top_k=top_k,
            results=results,
            **retrieval_metrics(results),
        )
    finally:
        await run_in_threadpool(guard.__exit__, None, None, None)
