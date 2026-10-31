from datetime import UTC, datetime
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints, model_validator

from nexusrag.schemas.retrieval import SemanticSearchHit

NonblankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ExpectedPassage(BaseModel):
    page_number: int = Field(ge=1)
    text: NonblankText


class RetrievalCase(BaseModel):
    case_id: NonblankText
    question: Annotated[NonblankText, Field(max_length=500)]
    answerable: bool = True
    expected_passages: list[ExpectedPassage] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expectations(self) -> Self:
        if self.answerable != bool(self.expected_passages):
            raise ValueError(
                "Answerable cases need passages; unsupported cases need none"
            )
        return self


class RetrievalDataset(BaseModel):
    dataset_id: NonblankText
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: list[RetrievalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self) -> Self:
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Evaluation case IDs must be unique")
        return self


class RetrievalCaseResult(BaseModel):
    case: RetrievalCase
    first_relevant_rank: int | None
    hits: list[SemanticSearchHit]


class RetrievalEvaluationReport(BaseModel):
    dataset_id: str
    document_id: UUID
    document_sha256: str
    source_digest: str
    index_generation: UUID
    model_id: str
    model_revision: str
    chunk_size: int
    overlap: int
    chunk_count: int
    top_k: int
    answerable_cases: int
    unsupported_cases: int
    hit_rate_at_k: float | None
    mrr_at_k: float | None
    results: list[RetrievalCaseResult]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
