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
