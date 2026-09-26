from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_DOCUMENT_SIZE_BYTES = 25 * 1024 * 1024


class DocumentMetadata(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    media_type: Literal["application/pdf"]
    size_bytes: int = Field(gt=0, le=MAX_DOCUMENT_SIZE_BYTES)
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_created_at: datetime | None = None

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("filename cannot be blank")

        if "/" in cleaned or "\\" in cleaned:
            raise ValueError("filename must not contain a path")

        return cleaned
