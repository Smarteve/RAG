import asyncio
from importlib import import_module
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pypdf import PdfWriter

from nexusrag import config, storage
from nexusrag.main import app
from nexusrag.schemas.chunk import DocumentChunks
from nexusrag.schemas.extraction import DocumentExtraction, ExtractedPage
from nexusrag.schemas.retrieval import SearchHit, SearchResponse
from nexusrag.services import retrieval
from nexusrag.services.chunking import chunk_document

main = import_module("nexusrag.main")
client = TestClient(app)
pytestmark = pytest.mark.usefixtures("initialized_job_database")
SAMPLE_PDF = Path(__file__).resolve().parents[1] / "examples/employee-handbook.pdf"


@pytest.fixture(autouse=True)
def directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", tmp_path / "documents")
    monkeypatch.setattr(config, "EXTRACTION_DIRECTORY", tmp_path / "extracted")


@pytest.fixture
def record() -> DocumentChunks:
    extraction = DocumentExtraction(
        document_id=uuid4(),
        pages=[
            ExtractedPage(page_number=index, text=text)
            for index, text in enumerate(
                ["annual leave", "annual", "leave", "remote work"], start=1
            )
        ],
    )
    result = DocumentChunks(
        document_id=extraction.document_id,
        chunk_size=100,
        overlap=20,
        chunks=chunk_document(extraction, chunk_size=100, overlap=20),
    )
    asyncio.run(storage.save_chunks(result, config.CHUNK_DIRECTORY))
    return result


def test_search_returns_ranked_chunks_with_page_references(
    record: DocumentChunks,
) -> None:
    response = client.get(
        f"/documents/{record.document_id}/search",
        params={"query": "annual leave", "limit": 2},
    )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": str(record.document_id),
        "query": "annual leave",
        "hits": [
            {"chunk": record.chunks[0].model_dump(mode="json"), "score": 2},
            {"chunk": record.chunks[1].model_dump(mode="json"), "score": 1},
        ],
    }
    assert isinstance(SearchResponse.model_validate(response.json()).hits[0], SearchHit)


def test_search_defaults_to_five_and_returns_all_available_matches(
    record: DocumentChunks,
) -> None:
    response = client.get(
        f"/documents/{record.document_id}/search",
        params={"query": "ANNUAL annual leave"},
    )

    assert response.status_code == 200
    assert [hit["score"] for hit in response.json()["hits"]] == [2, 1, 1]
    assert [hit["chunk"]["chunk_index"] for hit in response.json()["hits"]] == [0, 1, 2]


def test_valid_query_without_matches_returns_empty_hits(record: DocumentChunks) -> None:
    response = client.get(
        f"/documents/{record.document_id}/search", params={"query": "unicorn"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": str(record.document_id),
        "query": "unicorn",
        "hits": [],
    }


@pytest.mark.parametrize("limit", [1, 20])
def test_search_accepts_both_limit_boundaries(
    record: DocumentChunks, limit: int
) -> None:
    response = client.get(
        f"/documents/{record.document_id}/search",
        params={"query": "annual leave", "limit": limit},
    )

    assert response.status_code == 200
    assert len(response.json()["hits"]) == min(limit, 3)


def test_search_accepts_query_at_maximum_length(record: DocumentChunks) -> None:
    query = "leave" + " " * 495
    response = client.get(
        f"/documents/{record.document_id}/search", params={"query": query}
    )

    assert response.status_code == 200
    assert response.json()["query"] == query
    assert len(response.json()["hits"]) == 2


def test_default_search_limit_caps_more_than_five_matches() -> None:
    extraction = DocumentExtraction(
        document_id=uuid4(),
        pages=[
            ExtractedPage(page_number=index, text="annual leave")
            for index in range(1, 7)
        ],
    )
    record = DocumentChunks(
        document_id=extraction.document_id,
        chunk_size=100,
        overlap=20,
        chunks=chunk_document(extraction, chunk_size=100, overlap=20),
    )
    asyncio.run(storage.save_chunks(record, config.CHUNK_DIRECTORY))

    response = client.get(
        f"/documents/{record.document_id}/search", params={"query": "leave"}
    )

    assert response.status_code == 200
    assert [hit["chunk"]["chunk_index"] for hit in response.json()["hits"]] == [
        0,
        1,
        2,
        3,
        4,
    ]


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"query": ""},
        {"query": " \t\n"},
        {"query": "?!___🚀"},
        {"query": "x" * 501},
        {"query": "leave", "limit": 0},
        {"query": "leave", "limit": -1},
        {"query": "leave", "limit": 21},
        {"query": "leave", "limit": "abc"},
    ],
)
def test_invalid_request_is_rejected_before_reading_storage(
    params: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = AsyncMock(side_effect=AssertionError("Storage must not be read"))
    monkeypatch.setattr(main, "load_chunks", loader)

    response = client.get(f"/documents/{uuid4()}/search", params=params)

    assert response.status_code == 422
    loader.assert_not_awaited()
    assert not config.CHUNK_DIRECTORY.exists()


def test_invalid_document_id_returns_validation_error() -> None:
    response = client.get("/documents/not-a-uuid/search", params={"query": "leave"})

    assert response.status_code == 422


def test_unknown_document_returns_not_found() -> None:
    response = client.get(f"/documents/{uuid4()}/search", params={"query": "leave"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}
    assert not config.STORAGE_DIRECTORY.exists()
    assert not config.CHUNK_DIRECTORY.exists()


def test_saved_pdf_without_chunks_returns_conflict() -> None:
    document_id = uuid4()
    config.STORAGE_DIRECTORY.mkdir()
    source = config.STORAGE_DIRECTORY / f"{document_id}.pdf"
    source.write_bytes(b"saved original")

    response = client.get(f"/documents/{document_id}/search", params={"query": "leave"})

    assert response.status_code == 409
    assert response.json() == {"detail": "Document chunks are not available"}
    assert source.read_bytes() == b"saved original"


@pytest.mark.parametrize(
    "case", ["invalid_json", "invalid_utf8", "wrong_id", "invalid_schema"]
)
def test_corrupt_stored_chunks_return_server_error_not_input_error(case: str) -> None:
    document_id = uuid4()
    if case == "invalid_json":
        contents = b"private corrupt JSON"
    elif case == "invalid_utf8":
        contents = b"\xff\xfe"
    elif case == "invalid_schema":
        contents = b"{}"
    else:
        contents = (
            DocumentChunks(document_id=uuid4(), chunk_size=100, overlap=20, chunks=[])
            .model_dump_json()
            .encode("utf-8")
        )
    config.CHUNK_DIRECTORY.mkdir()
    source = config.CHUNK_DIRECTORY / f"{document_id}.json"
    source.write_bytes(contents)

    response = client.get(f"/documents/{document_id}/search", params={"query": "leave"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load chunks"}
    assert source.read_bytes() == contents


def test_read_permission_error_returns_safe_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_read(self, *args, **kwargs):
        raise PermissionError("Private server path")

    monkeypatch.setattr(Path, "read_text", fail_read)

    response = client.get(f"/documents/{uuid4()}/search", params={"query": "leave"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load chunks"}


def test_missing_chunk_existence_check_is_offloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_stat(self):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        raise PermissionError("Private server path")

    monkeypatch.setattr(Path, "is_file", fail_stat)

    response = client.get(f"/documents/{uuid4()}/search", params={"query": "leave"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not load chunks"}


def test_loading_and_ranking_run_outside_event_loop(
    record: DocumentChunks, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_load = storage._load_chunks
    original_search = retrieval.search_chunks
    calls = []

    def load_in_worker(document_id, directory):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        calls.append("load")
        return original_load(document_id, directory)

    def rank_in_worker(chunks, query, limit):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        calls.append("rank")
        return original_search(chunks, query, limit)

    monkeypatch.setattr(storage, "_load_chunks", load_in_worker)
    monkeypatch.setattr(retrieval, "search_chunks", rank_in_worker)

    response = client.get(
        f"/documents/{record.document_id}/search", params={"query": "leave"}
    )

    assert response.status_code == 200
    assert calls == ["load", "rank"]


@pytest.mark.parametrize("failure", ["model_validation", "unexpected"])
def test_internal_ranking_errors_are_logged_and_return_server_errors(
    record: DocumentChunks,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    if failure == "model_validation":
        with pytest.raises(ValidationError) as invalid:
            SearchHit(chunk=record.chunks[0], score=0)
        error = invalid.value
    else:
        error = RuntimeError("Private internal ranking error")
    monkeypatch.setattr(retrieval, "search_chunks", Mock(side_effect=error))

    response = client.get(
        f"/documents/{record.document_id}/search", params={"query": "leave"}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not search document"}
    assert "Could not search document" in caplog.text


def test_blank_upload_is_searchable_but_has_no_hits(prepare_document) -> None:
    with BytesIO() as output, PdfWriter() as writer:
        writer.add_blank_page(width=595, height=842)
        writer.write(output)
        contents = output.getvalue()
    document_id = prepare_document(contents)

    response = client.get(f"/documents/{document_id}/search", params={"query": "leave"})

    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_failed_extraction_is_not_searchable(prepare_document) -> None:
    document_id = prepare_document(b"%PDF-1.4\nbroken")

    response = client.get(f"/documents/{document_id}/search", params={"query": "leave"})

    assert response.status_code == 409


def test_upload_then_search_returns_exact_source_passages(prepare_document) -> None:
    document_id = prepare_document(SAMPLE_PDF.read_bytes())

    response = client.get(
        f"/documents/{document_id}/search", params={"query": "annual leave", "limit": 5}
    )

    assert response.status_code == 200
    hits = response.json()["hits"]
    assert 0 < len(hits) <= 5
    assert any(
        hit["chunk"]["page_number"] == 1 and "Annual leave" in hit["chunk"]["text"]
        for hit in hits
    )
    extraction = asyncio.run(
        storage.load_extraction(document_id, config.EXTRACTION_DIRECTORY)
    )
    for hit in hits:
        chunk = hit["chunk"]
        assert chunk["document_id"] == str(document_id)
        page = extraction.pages[chunk["page_number"] - 1]
        assert chunk["text"] == page.text[chunk["start_char"] : chunk["end_char"]]
        assert hit["score"] == len(
            retrieval.validate_query("annual leave")
            & retrieval._tokenize(chunk["text"])
        )
