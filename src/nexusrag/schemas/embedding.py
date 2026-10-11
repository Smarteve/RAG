from math import isfinite
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class ChunkEmbedding(BaseModel):
    """A finite, nonzero vector paired with its source chunk index."""

    chunk_index: int = Field(ge=0)
    vector: list[float] = Field(min_length=1)

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, vector: list[float]) -> list[float]:
        if not all(isfinite(num) for num in vector):
            raise ValueError("vector values must be finite")

        if not any(vector):
            raise ValueError("vector must contain at least one nonzero value")

        return vector


class DocumentEmbeddings(BaseModel):
    """Document vectors and the model/source identity that produced them."""

    document_id: UUID
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    embeddings: list[ChunkEmbedding]

    @model_validator(mode="after")
    def validate_embeddings(self) -> Self:
        for index, embedding in enumerate(self.embeddings):
            if len(embedding.vector) != self.dimensions:
                raise ValueError("Every vector length must match dimensions")
            if embedding.chunk_index != index:
                raise ValueError("Chunk indices must be sequential starting at zero")

        return self


class SemanticIndexResponse(BaseModel):
    """Compact metadata for a successfully created or rebuilt semantic index."""

    document_id: UUID
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
    embedding_count: int = Field(ge=0)
