from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class DocumentChunk(BaseModel):
    document_id: UUID
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        if self.start_char >= self.end_char:
            raise ValueError("start index must be smaller than end index")
        if len(self.text) != self.end_char - self.start_char:
            raise ValueError("text length must match the character range")
        return self


class DocumentChunks(BaseModel):
    """Persisted chunks and the character-window settings that produced them."""

    document_id: UUID
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    chunks: list[DocumentChunk]

    @model_validator(mode="after")
    def validate_chunks(self) -> Self:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        for index, chunk in enumerate(self.chunks):
            if chunk.document_id != self.document_id:
                raise ValueError("Every chunk must belong to the parent document")
            if chunk.chunk_index != index:
                raise ValueError("Chunk indices must be sequential starting at zero")
            if len(chunk.text) > self.chunk_size:
                raise ValueError("Chunk text must not exceed chunk_size")
        return self


class FetchChunksResponse(BaseModel):
    document_id: UUID
    total_chunks: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    chunks: list[DocumentChunk]
