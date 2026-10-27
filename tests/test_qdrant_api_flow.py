import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from nexusrag import config, storage
from nexusrag.dependencies import (
    get_answer_generator,
    get_embedding_runtime,
    get_vector_backend,
)
from nexusrag.main import app
from nexusrag.schemas.answer import GeneratedAnswer
from nexusrag.schemas.chunk import DocumentChunk, DocumentChunks
from nexusrag.services import retrieval
from nexusrag.services.embedding_model import MiniEmbedder
from nexusrag.services.embedding_runtime import EmbeddingRuntime
from nexusrag.services.vector_backend import QdrantBackend


def test_index_search_and_ask_use_qdrant_without_local_cosine(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", tmp_path / "documents")
    monkeypatch.setattr(config, "EMBEDDING_DIRECTORY", tmp_path / "embeddings")
    monkeypatch.setattr(config, "EMBEDDING_MODEL_ID", "test-encoder")
    monkeypatch.setattr(config, "EMBEDDING_MODEL_REVISION", "test-v1")
    monkeypatch.setattr(
        retrieval,
        "search_semantic_chunks",
        Mock(side_effect=AssertionError("Local cosine must not run")),
    )
    document_id = uuid4()
    texts = ["Annual leave is 25 days.", "Remote work needs approval."]
    chunks = DocumentChunks(
        document_id=document_id,
        chunk_size=100,
        overlap=0,
        chunks=[
            DocumentChunk(
                document_id=document_id,
                page_number=i + 1,
                chunk_index=i,
                start_char=0,
                end_char=len(text),
                text=text,
            )
            for i, text in enumerate(texts)
        ],
    )
    asyncio.run(storage.save_chunks(chunks, config.CHUNK_DIRECTORY))
    embedder = Mock(
        spec=MiniEmbedder, model_id="test-encoder", model_revision="test-v1"
    )
    embedder.get_dimensions.return_value = 3
    embedder.embed_documents.return_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    embedder.embed_query.return_value = [1.0, 0.0, 0.0]
    runtime_holder = []

    async def runtime_dependency():
        if not runtime_holder:
            runtime_holder.append(EmbeddingRuntime(lambda: embedder))
        return runtime_holder[0]

    native_client = AsyncQdrantClient(":memory:")
    backend = QdrantBackend(native_client, collection="api-test")
    accesses = []

    async def backend_dependency():
        accesses.append(True)
        return backend

    generator = SimpleNamespace(
        generate=AsyncMock(
            return_value=GeneratedAnswer(
                answer="Employees receive 25 days.",
                status="answered",
                citation_ids=["S1"],
            )
        )
    )

    async def generator_dependency():
        return generator

    previous = app.dependency_overrides.copy()
    app.dependency_overrides.update(
        {
            get_embedding_runtime: runtime_dependency,
            get_vector_backend: backend_dependency,
            get_answer_generator: generator_dependency,
        }
    )
    try:
        with TestClient(app) as client:
            indexed = client.post(f"/documents/{document_id}/semantic-index")
            assert indexed.status_code == 200, indexed.text
            record = asyncio.run(
                storage.load_embeddings(document_id, config.EMBEDDING_DIRECTORY)
            )
            assert record.index_generation is not None
            response = client.get(
                f"/documents/{document_id}/semantic-search",
                params={"query": "Annual leave"},
            )
            assert response.status_code == 200, response.text
            assert [hit["chunk"]["text"] for hit in response.json()["hits"]] == texts
            answer = client.post(
                f"/documents/{document_id}/ask",
                json={"question": "How much annual leave?"},
            )
            assert answer.status_code == 200, answer.text
            assert answer.json()["sources"][0]["page_number"] == 1
            assert answer.json()["sources"][0]["document_id"] == str(document_id)
            assert len(accesses) == 3
            # Keyword search has no vector-backend dependency.
            keyword = client.get(
                f"/documents/{document_id}/search", params={"query": "leave"}
            )
            assert keyword.status_code == 200, keyword.text
            assert keyword.json()["hits"][0]["score"] == 1
            assert len(accesses) == 3
        embedder.embed_documents.assert_called_once()
        assert embedder.embed_query.call_count == 2
        generator.generate.assert_awaited_once()
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)
        asyncio.run(native_client.close())
