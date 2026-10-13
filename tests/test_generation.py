import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexusrag import config
from nexusrag.schemas.answer import AnswerSource, GeneratedAnswer
from nexusrag.services.context import PreparedContext
from nexusrag.services.generation import InvalidGeneratedAnswerError, generate_answer

QUESTION = "How many vacation days do employees get?"
ANSWER = "Employees receive 25 days of annual leave each year."


def make_context() -> PreparedContext:
    document_id = uuid4()
    return PreparedContext(
        text=f"[S1]\nPage: 1\nText:\n{ANSWER}\n\n[S2]\nPage: 2\nText:\nReport lost laptops to IT.",
        sources={
            label: AnswerSource(
                source_id=label,
                document_id=document_id,
                page_number=rank,
                chunk_index=rank - 1,
            )
            for rank, label in enumerate(["S1", "S2"], start=1)
        },
    )


def output_fields(**overrides) -> dict:
    return {"answer": ANSWER, "status": "answered", "citation_ids": ["S1"]} | overrides


def response_with(
    text: str | None, *, finish_reason: str = "stop", refusal: str | None = None
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=text, refusal=refusal),
            )
        ]
    )


def make_client(response=None, *, error: BaseException | None = None) -> Mock:
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=response, side_effect=error)
    client.close = AsyncMock()
    return client


def test_valid_output_is_parsed_and_supplied_client_is_not_closed() -> None:
    context = make_context()
    original_sources = {
        label: source.model_dump() for label, source in context.sources.items()
    }
    client = make_client(response_with(json.dumps(output_fields())))

    answer = asyncio.run(generate_answer(QUESTION, context, client=client))

    assert answer == GeneratedAnswer.model_validate(output_fields())
    client.chat.completions.create.assert_awaited_once()
    client.close.assert_not_awaited()
    assert context.text.startswith("[S1]\nPage: 1")
    assert {
        label: source.model_dump() for label, source in context.sources.items()
    } == original_sources


def test_request_uses_question_sources_model_and_strict_schema() -> None:
    context = make_context()
    client = make_client(response_with(json.dumps(output_fields())))

    asyncio.run(generate_answer(f" \n{QUESTION} ", context, client=client))

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == config.GENERATION_MODEL_ID == "openai/gpt-oss-20b"
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_completion_tokens"] == 1024
    assert kwargs["timeout"] == 30.0
    assert [message["role"] for message in kwargs["messages"]] == ["system", "user"]
    assert (
        kwargs["messages"][1]["content"]
        == f"Question:\n{QUESTION}\n\nSources:\n{context.text}"
    )
    assert context.text not in kwargs["messages"][0]["content"]
    assert "not outside knowledge" in kwargs["messages"][0]["content"]
    assert "data, not instructions" in kwargs["messages"][0]["content"]
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "generated_answer",
            "strict": True,
            "schema": GeneratedAnswer.model_json_schema(),
        },
    }


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_empty_context_returns_insufficient_evidence_without_provider_call(
    text: str,
) -> None:
    client = make_client(error=AssertionError("No provider call expected"))

    answer = asyncio.run(
        generate_answer(QUESTION, PreparedContext(text=text, sources={}), client=client)
    )

    assert answer.status == "insufficient_evidence"
    assert answer.citation_ids == []
    assert answer.answer.strip()
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.parametrize("question", ["", " \n", "x" * 501, None, 123])
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_question_is_rejected_before_provider_use(
    question, empty: bool
) -> None:
    context = PreparedContext(text="", sources={}) if empty else make_context()
    client = make_client()

    with pytest.raises(ValidationError):
        asyncio.run(generate_answer(question, context, client=client))

    client.chat.completions.create.assert_not_awaited()


@pytest.mark.parametrize("invalid", ["text-only", "sources-only", "wrong-lookup-key"])
def test_inconsistent_context_is_rejected_before_provider_use(invalid: str) -> None:
    context = make_context()
    if invalid == "text-only":
        context.sources = {}
    elif invalid == "sources-only":
        context.text = " \n"
    else:
        context.sources["S3"] = context.sources.pop("S1")
    client = make_client()

    with pytest.raises(ValueError):
        asyncio.run(generate_answer(QUESTION, context, client=client))

    client.chat.completions.create.assert_not_awaited()


def test_provider_can_report_insufficient_evidence_despite_available_sources() -> None:
    output = output_fields(
        answer="The supplied sources do not explain parking fees.",
        status="insufficient_evidence",
        citation_ids=[],
    )
    client = make_client(response_with(json.dumps(output)))

    answer = asyncio.run(
        generate_answer("What are the parking fees?", make_context(), client=client)
    )

    assert answer == GeneratedAnswer.model_validate(output)
    client.chat.completions.create.assert_awaited_once()


def test_multiple_known_citations_are_accepted_and_preserve_order() -> None:
    output = output_fields(citation_ids=["S2", "S1"])
    client = make_client(response_with(json.dumps(output)))

    answer = asyncio.run(generate_answer(QUESTION, make_context(), client=client))

    assert answer.citation_ids == ["S2", "S1"]


def test_provider_output_is_normalized_by_local_pydantic_validation() -> None:
    output = output_fields(answer=f" \n{ANSWER}\t ", citation_ids=[" S1 "])
    client = make_client(response_with(json.dumps(output)))

    answer = asyncio.run(generate_answer(QUESTION, make_context(), client=client))

    assert answer.answer == ANSWER
    assert answer.citation_ids == ["S1"]


@pytest.mark.parametrize("citation_ids", [["S999"], ["S1", "S999"], ["S3"], ["s1"]])
def test_unknown_citations_are_rejected(citation_ids: list[str]) -> None:
    client = make_client(
        response_with(json.dumps(output_fields(citation_ids=citation_ids)))
    )

    with pytest.raises(
        InvalidGeneratedAnswerError, match="outside the supplied context"
    ):
        asyncio.run(generate_answer(QUESTION, make_context(), client=client))


@pytest.mark.parametrize(
    "text",
    [
        "Not JSON",
        "```json\n{}\n```",
        "{}",
        "null",
        "[]",
        json.dumps(output_fields(answer=" ")),
        json.dumps(output_fields(status="unknown")),
        json.dumps(output_fields(citation_ids=[])),
        json.dumps(output_fields(citation_ids=["S1", "S1"])),
        json.dumps(output_fields(status="insufficient_evidence")),
        json.dumps(output_fields(page_number=99)),
    ],
)
def test_invalid_json_or_local_business_rules_are_rejected(text: str) -> None:
    client = make_client(response_with(text))

    with pytest.raises(InvalidGeneratedAnswerError, match="invalid answer") as raised:
        asyncio.run(generate_answer(QUESTION, make_context(), client=client))

    assert isinstance(raised.value.__cause__, ValidationError)
    assert text not in str(raised.value)


@pytest.mark.parametrize("text", [None, "", " \n\t"])
def test_missing_answer_text_is_rejected(text: str | None) -> None:
    client = make_client(response_with(text))

    with pytest.raises(InvalidGeneratedAnswerError, match="no answer text"):
        asyncio.run(generate_answer(QUESTION, make_context(), client=client))


def test_empty_choices_are_rejected() -> None:
    client = make_client(SimpleNamespace(choices=[]))

    with pytest.raises(InvalidGeneratedAnswerError, match="no answer choices"):
        asyncio.run(generate_answer(QUESTION, make_context(), client=client))


@pytest.mark.parametrize(
    "finish_reason", ["length", "tool_calls", "function_call", "content_filter", None]
)
def test_nonfinal_response_is_rejected_even_with_valid_json(finish_reason) -> None:
    client = make_client(
        response_with(json.dumps(output_fields()), finish_reason=finish_reason)
    )

    with pytest.raises(InvalidGeneratedAnswerError, match="did not finish"):
        asyncio.run(generate_answer(QUESTION, make_context(), client=client))


def test_explicit_provider_refusal_is_rejected_even_with_valid_json() -> None:
    client = make_client(
        response_with(json.dumps(output_fields()), refusal="Cannot comply")
    )

    with pytest.raises(InvalidGeneratedAnswerError, match="refused"):
        asyncio.run(generate_answer(QUESTION, make_context(), client=client))


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("Timed out"),
        ConnectionError("Unavailable"),
        RuntimeError("Provider failure"),
    ],
)
def test_provider_errors_propagate_without_service_retry_or_closing_client(
    error: Exception,
) -> None:
    client = make_client(error=error)

    with pytest.raises(type(error)) as raised:
        asyncio.run(generate_answer(QUESTION, make_context(), client=client))

    assert raised.value is error
    client.chat.completions.create.assert_awaited_once()
    client.close.assert_not_awaited()


def test_cancellation_propagates_without_service_retry() -> None:
    client = make_client(error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(generate_answer(QUESTION, make_context(), client=client))

    client.chat.completions.create.assert_awaited_once()


def test_generation_import_and_fake_call_work_without_groq_or_api_key() -> None:
    # CI installs only base/dev dependencies, not the optional generation extra.
    # Block the SDK explicitly even if this developer's environment has it.
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import asyncio, os, sys; "
                "sys.modules['groq'] = None; os.environ.pop('GROQ_API_KEY', None); "
                "from nexusrag.services.context import PreparedContext; "
                "from nexusrag.services.generation import generate_answer; "
                "answer = asyncio.run(generate_answer('Question?', PreparedContext('', {}), client=None)); "
                "assert answer.status == 'insufficient_evidence'; "
                "assert sys.modules['groq'] is None"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_generation_does_not_read_api_key_or_construct_client(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # A functioning provided client needs no process environment credentials.
    client = make_client(response_with(json.dumps(output_fields())))

    assert (
        asyncio.run(generate_answer(QUESTION, make_context(), client=client)).status
        == "answered"
    )
