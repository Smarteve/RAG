"""Validation at the vector adapter's input and external-result boundaries."""

from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class VectorQuery(BaseModel):
    query_vector: list[FiniteNumber] = Field(min_length=1)
    dimensions: int = Field(gt=0, strict=True)
    limit: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_vector(self) -> Self:
        if len(self.query_vector) != self.dimensions or not any(self.query_vector):
            raise ValueError("Query vector must be nonzero and match index dimensions")
        return self


class VectorPointPayload(BaseModel):
    """External metadata; extra fields such as remote text are ignored."""

    document_id: UUID
    index_generation: UUID
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    chunk_index: int = Field(ge=0, strict=True)
    page_number: int = Field(ge=1, strict=True)


class VectorSearchPoint(BaseModel):
    id: UUID
    payload: VectorPointPayload
    score: Annotated[
        float,
        Field(strict=True, allow_inf_nan=False, ge=-1.000001, le=1.000001),
    ]

    @field_validator("score")
    @classmethod
    def normalize_cosine_rounding(cls, score: float) -> float:
        # Qdrant float32 arithmetic can slightly exceed cosine's endpoints.
        return max(-1.0, min(1.0, score))
