"""Status records for background document processing."""

from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, Field, model_validator

JobStatus = Literal["queued", "running", "succeeded", "failed"]


class ProcessingJob(BaseModel):
    """A processing snapshot; errors must be safe for public responses."""

    job_id: UUID
    document_id: UUID
    status: JobStatus
    created_at: AwareDatetime
    updated_at: AwareDatetime
    error: str | None = None
    root_job_id: UUID = Field(
        default_factory=lambda data: data.get("job_id"), validate_default=True
    )
    previous_job_id: UUID | None = None
    attempt_number: int = Field(default=1, ge=1, strict=True)

    @model_validator(mode="after")
    def validate_job(self) -> Self:
        if self.attempt_number == 1:
            if self.root_job_id != self.job_id or self.previous_job_id is not None:
                raise ValueError(
                    "First attempts must be their own root with no previous job"
                )
        else:
            if self.root_job_id == self.job_id:
                raise ValueError("Retries must reference a different root job")
            if self.previous_job_id is None or self.previous_job_id == self.job_id:
                raise ValueError("Retries must reference a different previous job")

        if self.status == "failed":
            if self.error is None or not self.error.strip():
                raise ValueError("A failed job must have a nonblank error")
            self.error = self.error.strip()
        elif self.error is not None:
            raise ValueError("Only failed jobs may have an error")

        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")

        return self
