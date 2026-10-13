"""Lazy Groq access with request-owned clients and safe application errors."""

import os
from typing import Protocol

lazy import groq

from nexusrag.schemas.answer import GeneratedAnswer
from nexusrag.services.context import PreparedContext
from nexusrag.services.generation import generate_answer


class GenerationUnavailableError(RuntimeError):
    """Generation configuration, credentials, model, or quota is unavailable."""


class GenerationTimeoutError(RuntimeError):
    """The hosted generation request timed out."""


class GenerationProviderError(RuntimeError):
    """The hosted provider could not complete the generation request."""


class AnswerGenerator(Protocol):
    async def generate(
        self, question: str, context: PreparedContext
    ) -> GeneratedAnswer: ...


class GroqAnswerGenerator:
    """No credentials, SDK imports, or connections are needed at app startup.

    Each nonempty request owns and closes its async client. No shared event-loop
    client or embedding lock is held during hosted inference. No automatic retries
    or provider/model fallback can consume extra quota or enable paid inference.
    """

    async def generate(
        self, question: str, context: PreparedContext
    ) -> GeneratedAnswer:
        if not context.sources:
            return await generate_answer(question, context, client=None)

        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise GenerationUnavailableError("Generation API key is not configured")
        try:
            # First use resolves the optional SDK, not application startup.
            client_type = groq.AsyncGroq
        except ImportError as error:
            raise GenerationUnavailableError(
                "Generation dependencies are unavailable"
            ) from error

        try:
            async with client_type(
                api_key=key,
                base_url="https://api.groq.com",
                timeout=30.0,
                max_retries=0,
            ) as client:
                return await generate_answer(question, context, client=client)
        except groq.APITimeoutError as error:
            raise GenerationTimeoutError("Generation request timed out") from error
        except (
            groq.AuthenticationError,
            groq.PermissionDeniedError,
            groq.NotFoundError,
            groq.RateLimitError,
        ) as error:
            raise GenerationUnavailableError(
                "Generation provider is unavailable"
            ) from error
        except groq.APIError as error:
            raise GenerationProviderError(
                "Generation provider request failed"
            ) from error
