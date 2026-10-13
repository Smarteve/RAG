from fastapi import Request

from nexusrag.services.embedding_runtime import EmbeddingRuntime
from nexusrag.services.generation_runtime import AnswerGenerator, GroqAnswerGenerator


async def get_embedding_runtime(request: Request) -> EmbeddingRuntime:
    """Keep one lazy model runtime on this app; tests can override the dependency."""
    runtime = getattr(request.app.state, "embedding_runtime", None)
    if runtime is None:
        # No await between lookup and assignment: requests cannot interleave here.
        runtime = EmbeddingRuntime()
        request.app.state.embedding_runtime = runtime
    return runtime


async def get_answer_generator(request: Request) -> AnswerGenerator:
    """Reuse a lightweight adapter, not credentials or a live network client."""
    generator = getattr(request.app.state, "answer_generator", None)
    if generator is None:
        generator = GroqAnswerGenerator()
        request.app.state.answer_generator = generator
    return generator
