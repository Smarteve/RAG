from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

from nexusrag.services import extraction
from nexusrag.services.extraction import DocumentExtractionError, extract_pages

SAMPLE_PDF = Path(__file__).resolve().parents[1] / "examples/employee-handbook.pdf"


def test_extracts_sample_text_on_the_correct_pages() -> None:
    pages = extract_pages(SAMPLE_PDF)

    assert [page.page_number for page in pages] == [1, 2]
    assert "Annual leave" in pages[0].text
    assert "31 March" in pages[0].text
    assert "Remote work" in pages[1].text
    assert "GBP 600" in pages[1].text


def test_blank_page_does_not_shift_later_page_numbers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "mixed.pdf"
    with SAMPLE_PDF.open("rb") as source, PdfWriter() as writer:
        reader = PdfReader(source)
        writer.add_page(reader.pages[0])
        writer.add_blank_page(width=595, height=842)
        writer.add_page(reader.pages[1])
        with pdf_path.open("wb") as destination:
            writer.write(destination)

    pages = extract_pages(pdf_path)

    assert [page.page_number for page in pages] == [1, 2, 3]
    assert "Annual leave" in pages[0].text
    assert pages[1].text == ""
    assert "Remote work" in pages[2].text


@settings(max_examples=30)
@given(
    page_count=st.integers(min_value=1, max_value=12),
    width=st.integers(min_value=72, max_value=1440),
    height=st.integers(min_value=72, max_value=1440),
)
@example(page_count=1, width=72, height=72)
@example(page_count=12, width=1440, height=1440)
def test_blank_pdfs_preserve_page_count_numbering_and_empty_text(
    page_count: int, width: int, height: int
) -> None:
    # A fresh directory per generated example avoids sharing pytest's tmp_path
    # across all Hypothesis examples in this test invocation.
    with TemporaryDirectory(prefix="nexusrag-extraction-") as directory:
        pdf_path = Path(directory) / "blank.pdf"
        with PdfWriter() as writer:
            for _ in range(page_count):
                writer.add_blank_page(width=width, height=height)
            with pdf_path.open("wb") as destination:
                writer.write(destination)

        pages = extract_pages(pdf_path)

        assert len(pages) == page_count
        assert [page.page_number for page in pages] == list(range(1, page_count + 1))
        assert all(page.text == "" for page in pages)


@pytest.mark.parametrize(
    "contents",
    [b"", b"not a PDF", b"%PDF-1.4\ntruncated document\n"],
    ids=["empty", "not-pdf", "truncated"],
)
def test_unreadable_pdf_preserves_original_cause(
    tmp_path: Path, contents: bytes
) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(contents)

    with pytest.raises(
        DocumentExtractionError, match="Could not read PDF content"
    ) as exc:
        extract_pages(pdf_path)

    assert isinstance(exc.value.__cause__, PdfReadError)


def test_missing_file_remains_a_filesystem_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_pages(tmp_path / "missing.pdf")


def test_permission_error_is_not_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    error = PermissionError("Access denied")
    # Simulate permissions so the test also works when run as root in CI.
    monkeypatch.setattr(Path, "open", Mock(side_effect=error))

    with pytest.raises(PermissionError) as exc:
        extract_pages(Path("restricted.pdf"))

    assert exc.value is error


@pytest.mark.parametrize("failure_stage", ["reader", "page"])
@pytest.mark.parametrize("error_type", [PdfReadError, NotImplementedError])
def test_parser_failure_closes_source_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str, error_type: type[Exception]
) -> None:
    source = BytesIO(b"placeholder: the reader is mocked in this test")
    monkeypatch.setattr(Path, "open", Mock(return_value=source))
    error = error_type("Simulated parser failure")
    reader_factory = Mock()
    if failure_stage == "reader":
        reader_factory.side_effect = error
    else:
        first_page = Mock()
        first_page.extract_text.return_value = "First page succeeds"
        broken_page = Mock()
        broken_page.extract_text.side_effect = error
        reader_factory.return_value.pages = [first_page, broken_page]
    monkeypatch.setattr(extraction, "PdfReader", reader_factory)

    with pytest.raises(
        DocumentExtractionError, match="Could not read PDF content"
    ) as exc:
        extract_pages(Path("broken.pdf"))

    assert exc.value.__cause__ is error
    assert source.closed


def test_unexpected_error_is_not_disguised_as_invalid_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = BytesIO(b"placeholder")
    error = RuntimeError("Unexpected programming failure")
    monkeypatch.setattr(Path, "open", Mock(return_value=source))
    monkeypatch.setattr(extraction, "PdfReader", Mock(side_effect=error))

    with pytest.raises(RuntimeError) as exc:
        extract_pages(Path("sample.pdf"))

    assert exc.value is error
    assert source.closed


def test_successful_extraction_closes_source(monkeypatch: pytest.MonkeyPatch) -> None:
    source = BytesIO(SAMPLE_PDF.read_bytes())
    monkeypatch.setattr(Path, "open", Mock(return_value=source))

    pages = extract_pages(SAMPLE_PDF)

    assert len(pages) == 2
    assert source.closed
