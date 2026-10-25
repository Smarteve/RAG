from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag.schemas.job import JobStatus, ProcessingJob
from nexusrag.services.jobs import job_transition

STATUSES: tuple[JobStatus, ...] = ("queued", "running", "succeeded", "failed")
CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def make_job(status: JobStatus = "queued", **overrides) -> ProcessingJob:
    return ProcessingJob(
        **(
            {
                "job_id": uuid4(),
                "document_id": uuid4(),
                "status": status,
                "created_at": CREATED_AT,
                "updated_at": CREATED_AT,
                "error": "Processing failed" if status == "failed" else None,
            }
            | overrides
        )
    )


@pytest.mark.parametrize("status", STATUSES)
def test_valid_job_round_trips_through_json(status: JobStatus) -> None:
    job = make_job(status)
    assert ProcessingJob.model_validate_json(job.model_dump_json()) == job


def test_failed_job_trims_error() -> None:
    assert (
        make_job("failed", error="  Processing failed\n").error == "Processing failed"
    )


@pytest.mark.parametrize("error", [None, "", " ", "\n\t"])
def test_failed_job_requires_nonblank_error(error: str | None) -> None:
    with pytest.raises(ValidationError, match="nonblank error"):
        make_job("failed", error=error)


@pytest.mark.parametrize("status", ["queued", "running", "succeeded"])
@pytest.mark.parametrize("error", ["Processing failed", "", " "])
def test_nonfailed_job_rejects_error(status: JobStatus, error: str) -> None:
    with pytest.raises(ValidationError, match="Only failed jobs"):
        make_job(status, error=error)


@pytest.mark.parametrize("field", ["created_at", "updated_at"])
def test_job_rejects_naive_timestamp(field: str) -> None:
    with pytest.raises(ValidationError, match="timezone_aware"):
        make_job(**{field: CREATED_AT.replace(tzinfo=None)})


def test_job_rejects_reversed_timestamps() -> None:
    with pytest.raises(ValidationError, match="must not be earlier"):
        make_job(updated_at=CREATED_AT - timedelta(microseconds=1))


def test_job_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        make_job(status="unknown")


@pytest.mark.parametrize(
    ("status", "new_status", "error"),
    [
        ("queued", "running", None),
        ("running", "succeeded", None),
        ("running", "failed", "  Processing failed  "),
    ],
)
def test_valid_transition_preserves_identity_and_original(
    status: JobStatus, new_status: JobStatus, error: str | None
) -> None:
    original = make_job(status)
    snapshot = original.model_dump()
    before = datetime.now(UTC)

    result = job_transition(original, new_status, error)

    after = datetime.now(UTC)
    assert result is not original
    assert original.model_dump() == snapshot
    assert result.job_id == original.job_id
    assert result.document_id == original.document_id
    assert result.created_at == original.created_at
    assert before <= result.updated_at <= after
    assert result.updated_at > original.updated_at
    assert result.updated_at.utcoffset() == timedelta(0)
    assert result.status == new_status
    assert result.error == (error.strip() if error is not None else None)


@pytest.mark.parametrize(
    ("status", "new_status"),
    [
        (status, new_status)
        for status in STATUSES
        for new_status in STATUSES
        if (status, new_status)
        not in {("queued", "running"), ("running", "succeeded"), ("running", "failed")}
    ],
)
def test_invalid_transition_leaves_original_unchanged(
    status: JobStatus, new_status: JobStatus
) -> None:
    original = make_job(status)
    snapshot = original.model_dump()

    with pytest.raises(
        ValueError, match=f"Invalid job transition: {status} -> {new_status}"
    ):
        job_transition(original, new_status)

    assert original.model_dump() == snapshot


@pytest.mark.parametrize(
    ("status", "new_status", "error"),
    [
        ("running", "failed", None),
        ("running", "failed", " \t"),
        ("queued", "running", "Processing failed"),
        ("running", "succeeded", "Processing failed"),
    ],
)
def test_transition_validates_error_without_mutating_original(
    status: JobStatus, new_status: JobStatus, error: str | None
) -> None:
    original = make_job(status)
    snapshot = original.model_dump()

    with pytest.raises(ValidationError):
        job_transition(original, new_status, error)

    assert original.model_dump() == snapshot


def test_original_and_legacy_jobs_default_to_first_attempt() -> None:
    job = make_job()
    assert job.root_job_id == job.job_id
    assert job.previous_job_id is None
    assert job.attempt_number == 1
    legacy = job.model_dump(
        exclude={"root_job_id", "previous_job_id", "attempt_number"}
    )
    assert ProcessingJob.model_validate(legacy) == job


@pytest.mark.parametrize("status", STATUSES)
def test_retry_metadata_survives_json_round_trip(status: JobStatus) -> None:
    root = uuid4()
    job = make_job(status, root_job_id=root, previous_job_id=root, attempt_number=2)
    assert ProcessingJob.model_validate_json(job.model_dump_json()) == job


def test_transition_preserves_retry_metadata() -> None:
    root = uuid4()
    job = make_job(root_job_id=root, previous_job_id=root, attempt_number=2)
    updated = job_transition(job, "running")
    assert updated.root_job_id == root
    assert updated.previous_job_id == root
    assert updated.attempt_number == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"attempt_number": 0},
        {"attempt_number": -1},
        {"attempt_number": True},
        {"attempt_number": 1.5},
        {"attempt_number": "2"},
        {"root_job_id": uuid4()},
        {"previous_job_id": uuid4()},
        {"attempt_number": 2},
        {"attempt_number": 2, "root_job_id": uuid4()},
        {"root_job_id": "invalid"},
        {"previous_job_id": "invalid"},
    ],
)
def test_invalid_retry_metadata_is_rejected(overrides) -> None:
    with pytest.raises(ValidationError):
        make_job(**overrides)


def test_retry_cannot_reference_itself_as_previous_job() -> None:
    job_id = uuid4()
    with pytest.raises(ValidationError, match="different previous job"):
        make_job(
            job_id=job_id, root_job_id=uuid4(), previous_job_id=job_id, attempt_number=2
        )
