from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

ExtractionStatus = Literal["completed", "failed"]
ExtractionErrorCode = Literal["pdf_read_failed", "source_read_failed"]


class ExtractedPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str


class DocumentExtraction(BaseModel):
    document_id: UUID
    pages: list[ExtractedPage]
    status: ExtractionStatus = "completed"
    error: ExtractionErrorCode | None = None
