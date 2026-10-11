import asyncio
import subprocess
import sys
from pathlib import Path
from runpy import run_path
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request

from nexusrag import config
from nexusrag.dependencies import get_embedding_runtime
from nexusrag.services.embedding_model import MiniEmbedder
from nexusrag.services.embedding_runtime import (
    EmbeddingModelUnavailableError,
    EmbeddingRuntime,
    load_minilm_embedder,
)


def test_loader_uses_pinned_cached_cpu_model_without_downloads(monkeypatch) -> None:
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 384
    constructor = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=constructor),
    )

    embedder = load_minilm_embedder()

    constructor.assert_called_once_with(
        config.EMBEDDING_MODEL_ID,
        revision=config.EMBEDDING_MODEL_REVISION,
        device="cpu",
        cache_folder=str(config.EMBEDDING_MODEL_CACHE_DIRECTORY),
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )
    assert embedder.model is model
    assert embedder.model_id == config.EMBEDDING_MODEL_ID
    assert embedder.model_revision == config.EMBEDDING_MODEL_REVISION
    model.encode_document.assert_not_called()


def test_model_is_loaded_once_in_worker_thread_and_reused() -> None:
    event_loop_thread = get_ident()
    worker_threads = []
    embedder = Mock(spec=MiniEmbedder)

    def load():
        worker_threads.append(get_ident())
        return embedder

    loader = Mock(side_effect=load)

    async def scenario():
        runtime = EmbeddingRuntime(loader)
        loader.assert_not_called()
        async with runtime.use_embedder() as first:
            assert first is embedder
        async with runtime.use_embedder() as second:
            assert second is first

    asyncio.run(scenario())

    loader.assert_called_once_with()
    assert len(worker_threads) == 1
    assert worker_threads[0] != event_loop_thread


def test_concurrent_users_share_one_model_and_never_overlap() -> None:
    loader = Mock(return_value=Mock(spec=MiniEmbedder))

    async def scenario():
        runtime = EmbeddingRuntime(loader)
        active = 0
        highest_active = 0

        async def use():
            nonlocal active, highest_active
            async with runtime.use_embedder() as embedder:
                assert embedder is loader.return_value
                active += 1
                highest_active = max(highest_active, active)
                await asyncio.sleep(0)  # Let another task attempt to acquire the lock.
                active -= 1

        await asyncio.gather(*(use() for _ in range(5)))
        assert highest_active == 1

    asyncio.run(scenario())

    loader.assert_called_once_with()


@pytest.mark.parametrize("failure", [ImportError("missing extra"), OSError("no cache")])
def test_failed_load_can_be_retried_after_fixing_setup(failure: Exception) -> None:
    embedder = Mock(spec=MiniEmbedder)
    loader = Mock(side_effect=[failure, embedder])

    async def scenario():
        runtime = EmbeddingRuntime(loader)
        with pytest.raises(EmbeddingModelUnavailableError) as raised:
            async with runtime.use_embedder():
                pytest.fail("A failed model load must not enter the context")
        assert raised.value.__cause__ is failure
        async with runtime.use_embedder() as result:
            assert result is embedder

    asyncio.run(scenario())

    assert loader.call_count == 2


def test_inference_failure_releases_lock_without_reloading_model() -> None:
    loader = Mock(return_value=Mock(spec=MiniEmbedder))
    failure = RuntimeError("encoding failed")

    async def scenario():
        runtime = EmbeddingRuntime(loader)
        with pytest.raises(RuntimeError) as raised:
            async with runtime.use_embedder():
                raise failure
        assert raised.value is failure
        async with runtime.use_embedder() as embedder:
            assert embedder is loader.return_value

    asyncio.run(scenario())

    loader.assert_called_once_with()


def test_application_import_does_not_import_optional_model_libraries() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys; import nexusrag.main; "
                "assert 'sentence_transformers' not in sys.modules; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_dependency_reuses_runtime_on_the_same_app_without_loading_model(
    monkeypatch,
) -> None:
    loader = Mock(side_effect=AssertionError("Dependency must not load the model"))
    monkeypatch.setattr(
        "nexusrag.services.embedding_runtime.load_minilm_embedder", loader
    )
    app = FastAPI()
    request = Request({"type": "http", "app": app})

    async def scenario():
        first = await get_embedding_runtime(request)
        second = await get_embedding_runtime(request)
        assert first is second
        assert app.state.embedding_runtime is first

    asyncio.run(scenario())
    loader.assert_not_called()


@pytest.mark.parametrize("override", [None, "custom-models"])
def test_model_cache_configuration_resolves_without_creating_directories(
    tmp_path: Path, monkeypatch, override: str | None
) -> None:
    monkeypatch.chdir(tmp_path)
    if override is None:
        monkeypatch.delenv("NEXUSRAG_MODEL_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("NEXUSRAG_MODEL_CACHE_DIR", override)

    settings = run_path(config.__file__)

    expected = tmp_path / (override if override is not None else "data/models")
    assert settings["EMBEDDING_MODEL_CACHE_DIRECTORY"] == expected
    assert not expected.exists()
