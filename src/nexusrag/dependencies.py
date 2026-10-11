from fastapi import Request

from nexusrag.services.embedding_runtime import EmbeddingRuntime


async def get_embedding_runtime(request: Request) -> EmbeddingRuntime:
    """Keep one lazy model runtime on this app; tests can override the dependency."""
    runtime = getattr(request.app.state, "embedding_runtime", None)
    if runtime is None:
        # No await between lookup and assignment: requests cannot interleave here.
        runtime = EmbeddingRuntime()
        request.app.state.embedding_runtime = runtime
    return runtime
