from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from nexusrag.schemas.answer import (
    AnswerSource,
    AskDocumentRequest,
    AskDocumentResponse,
    GeneratedAnswer,
)

DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_DOCUMENT_ID = UUID("22222222-2222-4222-8222-222222222222")
QUESTION = "How many vacation days do employees get?"
ANSWER = "Employees receive 25 days of annual leave each year."


def source_fields(**overrides) -> dict:
    return {
        "source_id": "S1",
        "document_id": DOCUMENT_ID,
        "page_number": 1,
        "chunk_index": 0,
    } | overrides


def response_fields(**overrides) -> dict:
    return {
        "document_id": DOCUMENT_ID,
        "question": QUESTION,
        "answer": ANSWER,
        "status": "answered",
        "sources": [source_fields()],
    } | overrides


def generated_fields(**overrides) -> dict:
    return {
        "answer": ANSWER,
        "status": "answered",
        "citation_ids": ["S1"],
    } | overrides


def test_request_normalizes_question_and_round_trips() -> None:
    request = AskDocumentRequest(question=f" \n{QUESTION}\t ")

    assert request.question == QUESTION
    assert AskDocumentRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize("question", ["", " ", "\t\n", "x" * 501, None, 123])
def test_request_rejects_invalid_questions(question) -> None:
    with pytest.raises(ValidationError):
        AskDocumentRequest(question=question)


@pytest.mark.parametrize("question", ["x", "x" * 500, "你好？", "???"])
def test_request_accepts_nonblank_questions_at_boundaries(question: str) -> None:
    assert AskDocumentRequest(question=question).question == question


def test_question_limit_applies_after_whitespace_normalization() -> None:
    assert AskDocumentRequest(question=" " + "x" * 500 + " ").question == "x" * 500


def test_request_requires_question_and_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AskDocumentRequest.model_validate({})
    with pytest.raises(ValidationError):
        AskDocumentRequest.model_validate({"question": QUESTION, "limit": 5})


def test_answer_source_normalizes_id_and_round_trips() -> None:
    source = AnswerSource.model_validate(source_fields(source_id=" S1 "))

    assert source.source_id == "S1"
    assert source.document_id == DOCUMENT_ID
    assert source.page_number == 1
    assert source.chunk_index == 0
    assert AnswerSource.model_validate_json(source.model_dump_json()) == source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", ""),
        ("source_id", " \t\n"),
        ("document_id", "not-a-uuid"),
        ("page_number", 0),
        ("page_number", -1),
        ("page_number", 1.5),
        ("chunk_index", -1),
        ("chunk_index", 0.5),
    ],
)
def test_answer_source_rejects_invalid_fields(field: str, value) -> None:
    with pytest.raises(ValidationError):
        AnswerSource.model_validate(source_fields(**{field: value}))


@pytest.mark.parametrize(
    "field", ["source_id", "document_id", "page_number", "chunk_index"]
)
def test_answer_source_requires_all_fields(field: str) -> None:
    fields = source_fields()
    del fields[field]

    with pytest.raises(ValidationError):
        AnswerSource.model_validate(fields)


def test_generated_answer_normalizes_text_and_ids_and_round_trips() -> None:
    generated = GeneratedAnswer.model_validate(
        generated_fields(answer=f" {ANSWER}\n", citation_ids=[" S1 ", "S2"])
    )

    assert generated.answer == ANSWER
    assert generated.citation_ids == ["S1", "S2"]
    assert GeneratedAnswer.model_validate_json(generated.model_dump_json()) == generated
    assert set(generated.model_dump()) == {"answer", "status", "citation_ids"}


def test_generated_answer_accepts_explicit_insufficient_evidence() -> None:
    generated = GeneratedAnswer(
        answer="The supplied sources do not explain parking fees.",
        status="insufficient_evidence",
        citation_ids=[],
    )

    assert generated.status == "insufficient_evidence"
    assert generated.citation_ids == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"answer": ""},
        {"answer": " \n\t"},
        {"status": "unknown"},
        {"status": None},
        {"citation_ids": []},
        {"citation_ids": [""]},
        {"citation_ids": [" \t"]},
        {"citation_ids": ["S1", "S1"]},
        {"citation_ids": ["S1", " S1 "]},
        {"status": "insufficient_evidence", "citation_ids": ["S1"]},
        {"citation_ids": "S1"},
        {"citation_ids": None},
        {"document_id": str(DOCUMENT_ID)},
        {"page_number": 1},
    ],
)
def test_generated_answer_rejects_invalid_output(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        GeneratedAnswer.model_validate(generated_fields(**overrides))


@pytest.mark.parametrize("field", ["answer", "status", "citation_ids"])
def test_generated_answer_requires_explicit_fields(field: str) -> None:
    fields = generated_fields()
    del fields[field]

    with pytest.raises(ValidationError):
        GeneratedAnswer.model_validate(fields)


def test_schema_cannot_check_citation_membership_without_context() -> None:
    # The generation service must reject S999 if it was not in the supplied context.
    generated = GeneratedAnswer.model_validate(generated_fields(citation_ids=["S999"]))

    assert generated.citation_ids == ["S999"]


def test_response_round_trips_with_nested_source_metadata() -> None:
    response = AskDocumentResponse.model_validate(
        response_fields(question=f" {QUESTION} ", answer=f" {ANSWER} ")
    )
    restored = AskDocumentResponse.model_validate_json(response.model_dump_json())

    assert restored == response
    assert restored.question == QUESTION
    assert restored.answer == ANSWER
    assert isinstance(restored.sources[0], AnswerSource)
    assert restored.sources[0].document_id == DOCUMENT_ID
    assert restored.sources[0].page_number == 1


def test_response_accepts_multiple_distinct_chunks_from_the_same_page() -> None:
    response = AskDocumentResponse.model_validate(
        response_fields(
            sources=[source_fields(), source_fields(source_id="S2", chunk_index=1)]
        )
    )

    assert [source.page_number for source in response.sources] == [1, 1]
    assert [source.chunk_index for source in response.sources] == [0, 1]


def test_response_accepts_insufficient_evidence_without_sources() -> None:
    response = AskDocumentResponse.model_validate(
        response_fields(
            answer="The document does not explain parking fees.",
            status="insufficient_evidence",
            sources=[],
        )
    )

    assert response.sources == []
    assert response.status == "insufficient_evidence"


@pytest.mark.parametrize(
    "overrides",
    [
        {"document_id": "invalid"},
        {"question": " \n"},
        {"question": "x" * 501},
        {"answer": " \t"},
        {"status": "unknown"},
        {"sources": []},
        {"sources": None},
        {"status": "insufficient_evidence"},
        {"sources": [source_fields(document_id=OTHER_DOCUMENT_ID)]},
        {"sources": [source_fields(), source_fields(chunk_index=1)]},
        {"sources": [source_fields(), source_fields(source_id=" S1 ", chunk_index=1)]},
        {"sources": [source_fields(), source_fields(source_id="S2")]},
        {"sources": [source_fields(), source_fields(source_id="S2", page_number=2)]},
        {"sources": [source_fields(page_number=0)]},
        {"sources": [source_fields(path="data/documents/private.pdf")]},
    ],
)
def test_response_rejects_invalid_or_inconsistent_data(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        AskDocumentResponse.model_validate(response_fields(**overrides))


@pytest.mark.parametrize(
    "field", ["document_id", "question", "answer", "status", "sources"]
)
def test_response_requires_all_fields(field: str) -> None:
    fields = response_fields()
    del fields[field]

    with pytest.raises(ValidationError):
        AskDocumentResponse.model_validate(fields)


def test_response_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AskDocumentResponse.model_validate(response_fields(confidence=0.99))


@given(
    question=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        min_size=1,
        max_size=500,
    )
)
def test_nonblank_questions_round_trip(question: str) -> None:
    request = AskDocumentRequest(question=question)

    assert AskDocumentRequest.model_validate_json(request.model_dump_json()) == request
    assert request.question == question


@given(
    indices=st.lists(
        st.integers(min_value=0, max_value=1000), min_size=1, max_size=10, unique=True
    )
)
def test_distinct_source_chunks_round_trip_and_preserve_order(
    indices: list[int],
) -> None:
    sources = [
        source_fields(source_id=f"S{rank}", chunk_index=index, page_number=index + 1)
        for rank, index in enumerate(indices, start=1)
    ]
    response = AskDocumentResponse.model_validate(response_fields(sources=sources))
    generated = GeneratedAnswer.model_validate(
        generated_fields(citation_ids=[source["source_id"] for source in sources])
    )

    assert (
        AskDocumentResponse.model_validate_json(response.model_dump_json()) == response
    )
    assert GeneratedAnswer.model_validate_json(generated.model_dump_json()) == generated
    assert [source.chunk_index for source in response.sources] == indices
    assert [source.source_id for source in response.sources] == generated.citation_ids


@given(source_id=st.integers(min_value=1, max_value=1000))
def test_duplicate_generated_citations_are_rejected(source_id: int) -> None:
    label = f"S{source_id}"

    with pytest.raises(ValidationError, match="Source IDs must be unique"):
        GeneratedAnswer.model_validate(generated_fields(citation_ids=[label, label]))
