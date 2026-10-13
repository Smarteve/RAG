import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, Request
from hypothesis import given
from hypothesis import strategies as st

from nexusrag.dependencies import get_answer_generator
from nexusrag.schemas.answer import AnswerSource
from nexusrag.services import generation_runtime
from nexusrag.services.context import PreparedContext
from nexusrag.services.generation import (
    InvalidGeneratedAnswerError,
    count_source_token_upper_bound,
)
from nexusrag.services.generation_runtime import (
    GenerationProviderError,
    GenerationTimeoutError,
    GenerationUnavailableError,
    GroqAnswerGenerator,
)


def make_context() -> PreparedContext:
    return PreparedContext(
        "[S1]\nPage: 1\nText:\nEmployees receive 25 days of annual leave.",
        {
            "S1": AnswerSource(
                source_id="S1", document_id=uuid4(), page_number=1, chunk_index=0
            )
        },
    )


@pytest.fixture
def sdk(monkeypatch):
    class APIError(Exception):
        pass

    errors = {
        name: type(name, (APIError,), {})
        for name in [
            "APITimeoutError",
            "AuthenticationError",
            "NotFoundError",
            "PermissionDeniedError",
            "RateLimitError",
        ]
    }
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "answer": "25 days.",
                                "status": "answered",
                                "citation_ids": ["S1"],
                            }
                        ),
                        refusal=None,
                    ),
                )
            ]
        )
    )
    module = SimpleNamespace(
        APIError=APIError, AsyncGroq=Mock(return_value=client), **errors
    )
    # Replace the lazy binding without resolving it or caching a real SDK.
    monkeypatch.setitem(generation_runtime.__dict__, "groq", module)
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")
    return module, client


def test_provider_uses_request_owned_client_and_closes_it(sdk) -> None:
    module, client = sdk

    answer = asyncio.run(
        GroqAnswerGenerator().generate("Vacation days?", make_context())
    )

    assert answer.answer == "25 days."
    module.AsyncGroq.assert_called_once_with(
        api_key="fake-test-key",
        base_url="https://api.groq.com",
        timeout=30.0,
        max_retries=0,
    )
    client.__aenter__.assert_awaited_once()
    client.__aexit__.assert_awaited_once()
    client.chat.completions.create.assert_awaited_once()


def test_each_request_has_a_separate_client_lifecycle(sdk) -> None:
    module, client = sdk
    generator = GroqAnswerGenerator()

    async def scenario():
        await generator.generate("Question one?", make_context())
        await generator.generate("Question two?", make_context())

    asyncio.run(scenario())

    assert module.AsyncGroq.call_count == 2
    assert client.__aexit__.await_count == 2


@pytest.mark.parametrize("key", [None, "", " \n"])
def test_missing_or_blank_key_is_unavailable_without_client_construction(
    sdk, monkeypatch, key
) -> None:
    module, _ = sdk
    if key is None:
        monkeypatch.delenv("GROQ_API_KEY")
    else:
        monkeypatch.setenv("GROQ_API_KEY", key)

    with pytest.raises(GenerationUnavailableError):
        asyncio.run(GroqAnswerGenerator().generate("Question?", make_context()))

    module.AsyncGroq.assert_not_called()


def test_missing_sdk_is_unavailable_without_network_access(monkeypatch) -> None:
    class MissingSDK:
        def __getattr__(self, name):
            raise ModuleNotFoundError("No module named 'groq'")

    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")
    monkeypatch.setitem(generation_runtime.__dict__, "groq", MissingSDK())

    with pytest.raises(GenerationUnavailableError, match="dependencies"):
        asyncio.run(GroqAnswerGenerator().generate("Question?", make_context()))


def test_empty_context_needs_neither_key_nor_sdk(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "groq", None)

    answer = asyncio.run(
        GroqAnswerGenerator().generate("Question?", PreparedContext("", {}))
    )

    assert answer.status == "insufficient_evidence"
    assert answer.citation_ids == []


@pytest.mark.parametrize(
    "name, expected",
    [
        ("APITimeoutError", GenerationTimeoutError),
        ("AuthenticationError", GenerationUnavailableError),
        ("NotFoundError", GenerationUnavailableError),
        ("PermissionDeniedError", GenerationUnavailableError),
        ("RateLimitError", GenerationUnavailableError),
        ("APIError", GenerationProviderError),
    ],
)
def test_sdk_errors_are_translated_and_client_is_closed(sdk, name, expected) -> None:
    module, client = sdk
    error = getattr(module, name)("secret upstream details")
    client.chat.completions.create.side_effect = error

    with pytest.raises(expected) as raised:
        asyncio.run(GroqAnswerGenerator().generate("Question?", make_context()))

    assert raised.value.__cause__ is error
    assert "secret" not in str(raised.value)
    client.__aexit__.assert_awaited_once()
    client.chat.completions.create.assert_awaited_once()


def test_invalid_provider_output_remains_a_validation_error_and_closes_client(
    sdk,
) -> None:
    _, client = sdk
    client.chat.completions.create.return_value.choices[0].message.content = "not JSON"

    with pytest.raises(InvalidGeneratedAnswerError):
        asyncio.run(GroqAnswerGenerator().generate("Question?", make_context()))

    client.__aexit__.assert_awaited_once()


def test_cancellation_is_not_swallowed_and_closes_client(sdk) -> None:
    _, client = sdk
    client.chat.completions.create.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(GroqAnswerGenerator().generate("Question?", make_context()))

    client.__aexit__.assert_awaited_once()


def test_dependency_reuses_lightweight_adapter_without_reading_key_or_importing_sdk(
    monkeypatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "groq", None)
    app = FastAPI()
    request = Request({"type": "http", "app": app})

    async def scenario():
        first = await get_answer_generator(request)
        second = await get_answer_generator(request)
        assert first is second is app.state.answer_generator
        assert isinstance(first, GroqAnswerGenerator)

    asyncio.run(scenario())


def test_application_import_without_optional_libraries_or_key() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import os, sys; os.environ.pop('GROQ_API_KEY', None); sys.modules['groq'] = None; "
                "import nexusrag.main; assert sys.modules['groq'] is None; "
                "assert 'sentence_transformers' not in sys.modules; assert 'torch' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "text, expected", [("", 0), ("Hello", 5), ("café", 5), ("你好", 6), ("🚀", 4)]
)
def test_source_budget_counts_utf8_bytes_not_words_or_characters(
    text: str, expected: int
) -> None:
    assert count_source_token_upper_bound(text) == expected


@given(text=st.text(alphabet=st.characters(codec="utf-8"), max_size=200))
def test_source_upper_bound_matches_encoded_bytes_without_downloads(text: str) -> None:
    assert count_source_token_upper_bound(text) == len(text.encode("utf-8"))


@pytest.mark.parametrize("mode", ["success", "quota", "upstream", "timeout"])
def test_installed_sdk_uses_mock_http_transport_and_never_retries(
    monkeypatch, mode
) -> None:
    # Optional integration test: CI without the generation extra skips this.
    # Even with the actual SDK, HTTP requests cannot leave this mock transport.
    groq = pytest.importorskip("groq")
    monkeypatch.setitem(generation_runtime.__dict__, "groq", groq)
    original_constructor = groq.AsyncGroq
    requests = []
    clients = []
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer fake-test-key"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["response_format"]["json_schema"]["strict"] is True
        if mode == "timeout":
            raise httpx.ReadTimeout("Mock timeout", request=request)
        if mode != "success":
            return httpx.Response(
                429 if mode == "quota" else 500,
                json={"error": {"message": "mock upstream failure"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "mock-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "openai/gpt-oss-20b",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "answer": "25 days.",
                                    "status": "answered",
                                    "citation_ids": ["S1"],
                                }
                            ),
                        },
                    }
                ],
            },
        )

    def construct(**kwargs):
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        clients.append(http_client)
        return original_constructor(**kwargs, http_client=http_client)

    monkeypatch.setattr(groq, "AsyncGroq", construct)

    if mode == "success":
        answer = asyncio.run(
            GroqAnswerGenerator().generate("Question?", make_context())
        )
        assert answer.answer == "25 days."
    else:
        expected = {
            "quota": GenerationUnavailableError,
            "upstream": GenerationProviderError,
            "timeout": GenerationTimeoutError,
        }[mode]
        with pytest.raises(expected):
            asyncio.run(GroqAnswerGenerator().generate("Question?", make_context()))
    assert len(requests) == 1
    assert len(clients) == 1
    assert clients[0].is_closed


def test_lazy_sdk_import_fails_only_on_first_use_in_fresh_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            """
import asyncio
import os
import sys
from uuid import uuid4

sys.modules['groq'] = None
os.environ['GROQ_API_KEY'] = 'fake-test-key'
from nexusrag.schemas.answer import AnswerSource
from nexusrag.services.context import PreparedContext
from nexusrag.services.generation_runtime import GroqAnswerGenerator, GenerationUnavailableError

assert sys.modules['groq'] is None
generator = GroqAnswerGenerator()
empty = asyncio.run(generator.generate('Question?', PreparedContext('', {})))
assert empty.status == 'insufficient_evidence'
context = PreparedContext('[S1] Evidence', {
    'S1': AnswerSource(source_id='S1', document_id=uuid4(), page_number=1, chunk_index=0)
})
try:
    asyncio.run(generator.generate('Question?', context))
except GenerationUnavailableError as error:
    assert isinstance(error.__cause__, ModuleNotFoundError)
else:
    raise AssertionError('First use must fail when the optional SDK is absent')
""",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
