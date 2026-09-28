from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str
