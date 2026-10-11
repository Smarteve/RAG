from uuid import UUID

from pydantic import BaseModel, Field

from nexusrag.schemas.chunk import DocumentChunk


class SearchHit(BaseModel):
    """A matching chunk and its count of distinct matching query words."""

    chunk: DocumentChunk
    score: int = Field(gt=0)


class SemanticSearchHit(BaseModel):
    """A chunk and its cosine similarity, not a probability or keyword count."""

    chunk: DocumentChunk
    similarity: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)


class SearchResponse(BaseModel):
    document_id: UUID
    query: str
    hits: list[SearchHit]


class SemanticSearchResponse(BaseModel):
    document_id: UUID
    query: str
    hits: list[SemanticSearchHit]
