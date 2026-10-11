import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi.concurrency import run_in_threadpool

from nexusrag import config
from nexusrag.services.embedding_model import MiniEmbedder


class EmbeddingModelUnavailableError(RuntimeError):
    """The optional, local embedding model could not be loaded."""


def load_minilm_embedder() -> MiniEmbedder:
    """Load the pinned CPU model from cache only; never download during a request."""
    # Keep optional, heavyweight imports out of application startup and normal CI.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config.EMBEDDING_MODEL_ID,
        revision=config.EMBEDDING_MODEL_REVISION,
        device="cpu",
        cache_folder=str(config.EMBEDDING_MODEL_CACHE_DIRECTORY),
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )
    embedder = MiniEmbedder(
        model,
        model_id=config.EMBEDDING_MODEL_ID,
        model_revision=config.EMBEDDING_MODEL_REVISION,
    )
    embedder.get_dimensions()
    return embedder


class EmbeddingRuntime:
    """Lazily reuse one model and serialize its use in one application process.

    Create this runtime in the application's event loop. Separate server
    processes own separate models/locks; this is not a durable background queue.
    """

    def __init__(self, loader: Callable[[], MiniEmbedder] | None = None):
        self._loader = loader if loader is not None else load_minilm_embedder
        self._embedder: MiniEmbedder | None = None
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def use_embedder(self) -> AsyncIterator[MiniEmbedder]:
        # Hold the lock through inference, not only through initial model loading.
        async with self._lock:
            if self._embedder is None:
                try:
                    self._embedder = await run_in_threadpool(self._loader)
                except Exception as error:
                    # Failed loads are not cached, so fixing local setup allows retry.
                    raise EmbeddingModelUnavailableError(
                        "Local embedding model is not available"
                    ) from error
            yield self._embedder
