from uuid import UUID

from pydantic import BaseModel, Field

from nexusrag.schemas.chunk import DocumentChunk


class SearchHit(BaseModel):
    """A matching chunk and its count of distinct matching query words."""

    chunk: DocumentChunk
    score: int = Field(gt=0)


class SearchResponse(BaseModel):
    document_id: UUID
    query: str
    hits: list[SearchHit]
