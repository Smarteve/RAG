"""Status records for background document processing."""

from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, model_validator

JobStatus = Literal["queued", "running", "succeeded", "failed"]


class ProcessingJob(BaseModel):
    """A processing snapshot; errors must be safe for public responses."""

    job_id: UUID
    document_id: UUID
    status: JobStatus
    created_at: AwareDatetime
    updated_at: AwareDatetime
    error: str | None = None

    @model_validator(mode="after")
    def validate_job(self) -> Self:
        if self.status == "failed":
            if self.error is None or not self.error.strip():
                raise ValueError("A failed job must have a nonblank error")
            self.error = self.error.strip()
        elif self.error is not None:
            raise ValueError("Only failed jobs may have an error")

        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")

        return self
