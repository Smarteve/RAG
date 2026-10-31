import asyncio
from hashlib import sha256
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag import config, storage
from nexusrag.database import initialize_database
from nexusrag.evaluation_cases import handbook_cases
from nexusrag.schemas.chunk import DocumentChunk, DocumentChunks
from nexusrag.schemas.embedding import ChunkEmbedding, DocumentEmbeddings
from nexusrag.schemas.evaluation import ExpectedPassage, RetrievalCase, RetrievalDataset
from nexusrag.schemas.extraction import DocumentExtraction, ExtractedPage
from nexusrag.schemas.retrieval import SemanticSearchHit
from nexusrag.services.embedding_model import MiniEmbedder
from nexusrag.services.embedding_runtime import EmbeddingRuntime
from nexusrag.services.evaluation import (
    EvaluationInputError,
    evaluate_retrieval,
    retrieval_metrics,
    score_case,
    validate_expected_passages,
)
from nexusrag.services.extraction import extract_pages
from nexusrag.services.processing_lock import DocumentBusyError, document_lock
from nexusrag.services.retrieval import StaleSemanticIndexError


def hit(text="25 days", page_number=1):
    return SemanticSearchHit(
        chunk=DocumentChunk(
            document_id=uuid4(),
            page_number=page_number,
            chunk_index=0,
            start_char=0,
            end_char=len(text),
            text=text,
        ),
        similarity=0.5,
    )


def question(answerable=True):
    return RetrievalCase(
        case_id="annual-leave",
        question="How much leave?",
        answerable=answerable,
        expected_passages=[ExpectedPassage(page_number=1, text="25 days")]
        if answerable
        else [],
    )


def test_score_matches_page_and_whitespace_normalized_phrase_with_top_k_cutoff():
    hits = [
        hit(page_number=2),
        hit("Other policy"),
        hit("There are 25\n DAYS of leave"),
    ]
    assert score_case(question(), hits, 2).first_relevant_rank is None
    result = score_case(question(), hits, 3)
    assert result.first_relevant_rank == 3
    assert result.hits == hits
    assert len(hits) == 3


def test_metrics_include_misses_and_exclude_unsupported_questions():
    results = [
        score_case(question(), [hit()], 5),
        score_case(question(), [hit("unrelated"), hit()], 5),
        score_case(question(), [], 5),
        score_case(question(False), [hit("nearby passage")], 5),
    ]
    metrics = retrieval_metrics(results)
    assert metrics["answerable_cases"] == 3
    assert metrics["unsupported_cases"] == 1
    assert metrics["hit_rate_at_k"] == pytest.approx(2 / 3)
    assert metrics["mrr_at_k"] == pytest.approx(0.5)
    assert results[-1].first_relevant_rank is None
    assert results[-1].hits


@pytest.mark.parametrize("results", [[], [score_case(question(False), [], 5)]])
def test_no_answerable_cases_has_no_quality_score(results):
    metrics = retrieval_metrics(results)
    assert metrics["hit_rate_at_k"] is None
    assert metrics["mrr_at_k"] is None


def test_case_labels_cannot_silently_omit_answerable_ground_truth():
    with pytest.raises(ValidationError):
        RetrievalCase(case_id="missing", question="Question?")
    with pytest.raises(ValidationError):
        RetrievalCase(
            case_id="contradictory",
            question="Question?",
            answerable=False,
            expected_passages=[{"page_number": 1, "text": "answer"}],
        )
    with pytest.raises(ValidationError):
        RetrievalDataset(
            dataset_id="duplicate",
            document_sha256="a" * 64,
            cases=[question(), question()],
        )


def test_bundled_labels_match_the_actual_sample_pdf():
    path = Path(__file__).resolve().parents[1] / "examples/employee-handbook.pdf"
    dataset = handbook_cases()
    assert sha256(path.read_bytes()).hexdigest() == dataset.document_sha256
    extraction = DocumentExtraction(document_id=uuid4(), pages=extract_pages(path))
    validate_expected_passages(dataset, extraction)
    assert len(dataset.cases) == 8
    assert sum(case.answerable for case in dataset.cases) == 7


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    initialize_database(config.JOB_DATABASE_PATH)
    for name, directory in [
        ("STORAGE_DIRECTORY", "documents"),
        ("EXTRACTION_DIRECTORY", "extracted"),
        ("EMBEDDING_DIRECTORY", "embeddings"),
    ]:
        path = tmp_path / directory
        path.mkdir()
        monkeypatch.setattr(config, name, path)
    document_id = uuid4()
    source = b"%PDF-evaluation-fixture"
    (config.STORAGE_DIRECTORY / f"{document_id}.pdf").write_bytes(source)
    texts = ["Annual leave is 25 days.", "Training budget is GBP 600."]
    extraction = DocumentExtraction(
        document_id=document_id,
        pages=[
            ExtractedPage(page_number=i + 1, text=text) for i, text in enumerate(texts)
        ],
    )
    chunks = DocumentChunks(
        document_id=document_id,
        chunk_size=100,
        overlap=0,
        chunks=[
            DocumentChunk(
                document_id=document_id,
                page_number=i + 1,
                chunk_index=i,
                start_char=0,
                end_char=len(text),
                text=text,
            )
            for i, text in enumerate(texts)
        ],
    )
    embeddings = DocumentEmbeddings(
        document_id=document_id,
        model_id=config.EMBEDDING_MODEL_ID,
        model_revision=config.EMBEDDING_MODEL_REVISION,
        dimensions=3,
        index_generation=uuid4(),
        source_digest=sha256(chunks.model_dump_json().encode()).hexdigest(),
        embeddings=[ChunkEmbedding(chunk_index=i, vector=[1, 0, 0]) for i in range(2)],
    )
    asyncio.run(storage.save_extraction(extraction, config.EXTRACTION_DIRECTORY))
    asyncio.run(storage.save_chunks(chunks, config.CHUNK_DIRECTORY))
    asyncio.run(storage.save_embeddings(embeddings, config.EMBEDDING_DIRECTORY))
    dataset = RetrievalDataset(
        dataset_id="test",
        document_sha256=sha256(source).hexdigest(),
        cases=[
            question(),
            RetrievalCase(
                case_id="training",
                question="Training money?",
                expected_passages=[ExpectedPassage(page_number=2, text="GBP 600")],
            ),
            question(False).model_copy(update={"case_id": "unsupported"}),
        ],
    )
    embedder = Mock(
        spec=MiniEmbedder,
        model_id=config.EMBEDDING_MODEL_ID,
        model_revision=config.EMBEDDING_MODEL_REVISION,
    )
    embedder.get_dimensions.return_value = 3
    embedder.embed_query.return_value = [1, 0, 0]
    loader = Mock(return_value=embedder)
    backend = Mock(
        search=AsyncMock(
            return_value=[
                SemanticSearchHit(chunk=chunk, similarity=0.5)
                for chunk in chunks.chunks
            ]
        )
    )
    return document_id, dataset, EmbeddingRuntime(loader), backend, loader


def run(prepared, **kwargs):
    document_id, dataset, runtime, backend, _ = prepared
    return asyncio.run(
        evaluate_retrieval(document_id, dataset, runtime, backend=backend, **kwargs)
    )


def test_evaluation_uses_real_search_orchestration_and_leaves_artifacts_unchanged(
    prepared,
):
    document_id, dataset, _, backend, loader = prepared
    paths = [config.STORAGE_DIRECTORY / f"{document_id}.pdf"] + [
        directory / f"{document_id}.json"
        for directory in (
            config.EXTRACTION_DIRECTORY,
            config.CHUNK_DIRECTORY,
            config.EMBEDDING_DIRECTORY,
        )
    ]
    before = [path.read_bytes() for path in paths]

    async def search(*args, **kwargs):
        with (
            pytest.raises(DocumentBusyError),
            document_lock(document_id, config.JOB_DATABASE_PATH),
        ):
            pass
        return [
            SemanticSearchHit(chunk=chunk, similarity=0.5) for chunk in args[0].chunks
        ]

    backend.search.side_effect = search
    report = run(prepared)
    assert report.hit_rate_at_k == 1.0
    assert report.mrr_at_k == 0.75
    assert [result.first_relevant_rank for result in report.results] == [1, 2, None]
    assert report.chunk_count == 2
    assert report.document_sha256 == dataset.document_sha256
    assert [path.read_bytes() for path in paths] == before
    assert backend.search.await_count == 3
    loader.assert_called_once_with()
    assert loader.return_value.embed_query.call_count == 3
    with document_lock(document_id, config.JOB_DATABASE_PATH):
        pass


def test_wrong_document_is_rejected_before_query_embedding(prepared):
    _, dataset, _, backend, loader = prepared
    dataset.document_sha256 = "0" * 64
    with pytest.raises(EvaluationInputError, match="does not match"):
        run(prepared)
    backend.search.assert_not_awaited()
    loader.assert_not_called()


def test_bad_ground_truth_is_an_input_error_not_a_retrieval_miss(prepared):
    _, dataset, _, backend, loader = prepared
    dataset.cases[0].expected_passages[0].page_number = 8
    with pytest.raises(EvaluationInputError, match="absent"):
        run(prepared)
    backend.search.assert_not_awaited()
    loader.assert_not_called()


def test_backend_failure_aborts_instead_of_silently_dropping_a_case(prepared):
    document_id, _, _, backend, _ = prepared
    backend.search.side_effect = [[], ConnectionError("private provider detail")]
    with pytest.raises(ConnectionError):
        run(prepared)
    assert backend.search.await_count == 2
    with document_lock(document_id, config.JOB_DATABASE_PATH):
        pass


def test_stale_index_is_not_evaluated_as_poor_retrieval(prepared):
    document_id, _, _, backend, loader = prepared
    record = asyncio.run(
        storage.load_embeddings(document_id, config.EMBEDDING_DIRECTORY)
    )
    record.source_digest = "0" * 64
    asyncio.run(storage.save_embeddings(record, config.EMBEDDING_DIRECTORY))
    with pytest.raises(StaleSemanticIndexError):
        run(prepared)
    backend.search.assert_not_awaited()
    loader.assert_not_called()


def test_live_document_lock_rejects_evaluation(prepared):
    with (
        document_lock(prepared[0], config.JOB_DATABASE_PATH),
        pytest.raises(DocumentBusyError),
    ):
        run(prepared)


@pytest.mark.parametrize("top_k", [0, -1, 21])
def test_invalid_limit_is_rejected_before_model_work(prepared, top_k):
    with pytest.raises(ValueError, match="top_k"):
        run(prepared, top_k=top_k)
    prepared[-1].assert_not_called()


@pytest.mark.parametrize("status", ["deleting", "deleted"])
def test_deleted_document_cannot_be_evaluated(prepared, status):
    import sqlite3

    with sqlite3.connect(config.JOB_DATABASE_PATH) as connection:
        connection.execute(
            "INSERT INTO document_deletions VALUES (?, ?)", (str(prepared[0]), status)
        )
    with pytest.raises(EvaluationInputError, match="deleted"):
        run(prepared)
    prepared[-1].assert_not_called()


@pytest.mark.parametrize("status", ["queued", "running", "failed"])
def test_unfinished_or_failed_processing_is_rejected(prepared, status):
    from datetime import UTC, datetime

    from nexusrag.schemas.job import ProcessingJob
    from nexusrag.services.job_storage import create_job, transition_job

    now = datetime.now(UTC)
    job = ProcessingJob(
        job_id=uuid4(),
        document_id=prepared[0],
        status="queued",
        created_at=now,
        updated_at=now,
    )
    asyncio.run(create_job(job, config.JOB_DATABASE_PATH))
    if status != "queued":
        asyncio.run(transition_job(job.job_id, "running", config.JOB_DATABASE_PATH))
    if status == "failed":
        asyncio.run(
            transition_job(
                job.job_id,
                "failed",
                config.JOB_DATABASE_PATH,
                "Document processing failed",
            )
        )
    with pytest.raises(EvaluationInputError, match="finish processing"):
        run(prepared)
    prepared[-1].assert_not_called()
