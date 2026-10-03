from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from nexusrag.schemas.extraction import ExtractedPage


class DocumentExtractionError(Exception):
    """Raised when a document cannot be parsed into text."""


def extract_pages(pdf_path: Path) -> list[ExtractedPage]:
    result: list[ExtractedPage] = []

    try:
        with pdf_path.open("rb") as source:
            reader = PdfReader(source)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                result.append(ExtractedPage(page_number=page_number, text=text))
    except (PdfReadError, NotImplementedError) as error:
        # pypdf uses NotImplementedError for unsupported PDF stream filters.
        # Keep this translation at the parser boundary, not around the endpoint.
        raise DocumentExtractionError("Could not read PDF content") from error
    return result
