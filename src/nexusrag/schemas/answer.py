"""Contracts for questions, generated answers, and server-resolved citations."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

AnswerStatus = Literal["answered", "insufficient_evidence"]
_NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
QuestionText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
]


def _validate_citations(status: AnswerStatus, source_ids: list[str]) -> None:
    if status == "answered" and not source_ids:
        raise ValueError("An answered result must cite at least one source")
    if status == "insufficient_evidence" and source_ids:
        raise ValueError("An insufficient-evidence result must not cite sources")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Source IDs must be unique")


class AskDocumentRequest(BaseModel):
    """A question about one document; the document ID belongs in the route."""

    model_config = ConfigDict(extra="forbid")

    question: QuestionText


class AnswerSource(BaseModel):
    """Source metadata resolved by our application, not invented by the LLM."""

    model_config = ConfigDict(extra="forbid")

    source_id: _NonBlankText
    document_id: UUID
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)


class GeneratedAnswer(BaseModel):
    """LLM output; the generation service must check IDs against its context."""

    model_config = ConfigDict(extra="forbid")

    answer: _NonBlankText
    status: AnswerStatus
    citation_ids: list[_NonBlankText]

    @model_validator(mode="after")
    def validate_citations(self) -> Self:
        _validate_citations(self.status, self.citation_ids)
        return self


class AskDocumentResponse(BaseModel):
    """Public answer with traceable sources, not a correctness guarantee."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    question: QuestionText
    answer: _NonBlankText
    status: AnswerStatus
    sources: list[AnswerSource]

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        _validate_citations(self.status, [source.source_id for source in self.sources])
        for source in self.sources:
            if source.document_id != self.document_id:
                raise ValueError("Every source must belong to the requested document")
        indices = [source.chunk_index for source in self.sources]
        if len(indices) != len(set(indices)):
            raise ValueError("Each source chunk must appear only once")
        return self
