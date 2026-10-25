import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from nexusrag import tasks
from nexusrag.schemas.job import ProcessingJob


@pytest.fixture
def workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ProcessingJob, AsyncMock, AsyncMock]:
    now = datetime.now(UTC)
    job = ProcessingJob(
        job_id=uuid4(),
        document_id=uuid4(),
        status="queued",
        created_at=now,
        updated_at=now,
    )
    monkeypatch.setattr(tasks, "load_job", AsyncMock(return_value=job))
    transition = AsyncMock()
    monkeypatch.setattr(tasks, "transition_job", transition)
    process = AsyncMock(return_value=[])
    monkeypatch.setattr(tasks, "run_in_threadpool", process)
    monkeypatch.setattr(tasks, "save_extraction", AsyncMock())
    monkeypatch.setattr(tasks, "save_chunks", AsyncMock())
    runtime = Mock()
    runtime.use_embedder.return_value.__aenter__ = AsyncMock(return_value=Mock())
    runtime.use_embedder.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(tasks, "EmbeddingRuntime", Mock(return_value=runtime))
    monkeypatch.setattr(tasks, "build_semantic_index", AsyncMock())
    return job, transition, process


def test_success_marks_ready_after_indexing(workflow) -> None:
    job, transition, _ = workflow
    asyncio.run(tasks._process_document(job.job_id))
    assert [call.args[1] for call in transition.await_args_list] == [
        "running",
        "succeeded",
    ]
    tasks.save_extraction.assert_awaited_once()
    tasks.save_chunks.assert_awaited_once()
    tasks.build_semantic_index.assert_awaited_once()
    assert tasks.build_semantic_index.await_args.args[0] == job.document_id


def test_claim_failure_does_not_process_or_fail_job(workflow) -> None:
    job, transition, process = workflow
    transition.side_effect = ValueError("Invalid job transition")
    with pytest.raises(
        tasks.JobNotClaimedError, match="Could not claim processing job"
    ):
        asyncio.run(tasks._process_document(job.job_id))
    assert transition.await_count == 1
    process.assert_not_awaited()


@pytest.mark.parametrize(
    "stage",
    ["run_in_threadpool", "save_extraction", "save_chunks", "build_semantic_index"],
)
def test_processing_failure_is_recorded_and_reraised(workflow, stage: str) -> None:
    job, transition, _ = workflow
    error = RuntimeError("private provider/path detail")
    getattr(tasks, stage).side_effect = error
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(tasks._process_document(job.job_id))
    assert caught.value is error
    assert [call.args[1] for call in transition.await_args_list] == [
        "running",
        "failed",
    ]
    assert transition.await_args.kwargs["error"] == "Document processing failed"


def test_success_write_failure_does_not_mark_processing_failed(workflow) -> None:
    job, transition, _ = workflow
    transition.side_effect = [None, OSError("Database unavailable")]
    with pytest.raises(OSError, match="Database unavailable"):
        asyncio.run(tasks._process_document(job.job_id))
    assert [call.args[1] for call in transition.await_args_list] == [
        "running",
        "succeeded",
    ]


def test_task_entry_point_holds_lock_through_work(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from uuid import uuid4

    from nexusrag.services.processing_lock import DocumentBusyError, document_lock

    job_id, document_id = uuid4(), uuid4()
    path = tmp_path / "jobs.sqlite3"
    monkeypatch.setattr(tasks.config, "JOB_DATABASE_PATH", path)
    monkeypatch.setattr(
        tasks, "_load_job", Mock(return_value=SimpleNamespace(document_id=document_id))
    )

    async def process(identifier):
        assert identifier == job_id
        with pytest.raises(DocumentBusyError), document_lock(document_id, path):
            pytest.fail("Task did not hold lock")

    monkeypatch.setattr(tasks, "_process_document", process)
    tasks.process_document(str(job_id))
