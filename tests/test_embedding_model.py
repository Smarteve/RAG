from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nexusrag.services.embedding_model import MiniEmbedder


def make_model(token_counts: list[int], vectors: list[list[float]]) -> Mock:
    model = Mock(max_seq_length=256)
    model.tokenizer.return_value = {
        "input_ids": [[1] * count for count in token_counts]
    }
    model.encode_document.return_value.tolist.return_value = vectors
    return model


def make_embedder(model: Mock) -> MiniEmbedder:
    return MiniEmbedder(model, model_id="test-encoder", model_revision="test-revision")


def test_embed_documents_preserves_input_order_and_uses_normalized_batches() -> None:
    texts = ["third passage", "first passage", "third passage"]
    vectors = [[0.3, 0.4], [0.1, 0.2], [0.3, 0.4]]
    model = make_model([5, 5, 5], vectors)

    assert make_embedder(model).embed_documents(texts) == vectors
    assert texts == ["third passage", "first passage", "third passage"]
    model.tokenizer.assert_called_once_with(
        texts, add_special_tokens=True, truncation=False
    )
    model.encode_document.assert_called_once_with(
        texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False
    )
    model.encode_document.return_value.tolist.assert_called_once_with()
    model.encode_query.assert_not_called()


def test_embed_documents_empty_input_does_not_call_model() -> None:
    model = make_model([], [])

    assert make_embedder(model).embed_documents([]) == []
    model.tokenizer.assert_not_called()
    model.encode_document.assert_not_called()


def test_embed_documents_single_passage_returns_a_list_of_vectors() -> None:
    model = make_model([4], [[0.1, -0.2, 0.3]])

    assert make_embedder(model).embed_documents(["one passage"]) == [[0.1, -0.2, 0.3]]


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
@pytest.mark.parametrize("index", [0, 1])
def test_embed_documents_rejects_blank_passages_before_model_work(
    blank: str, index: int
) -> None:
    texts = ["first passage", "second passage"]
    texts[index] = blank
    model = make_model([4, 4], [[0.1], [0.2]])

    with pytest.raises(ValueError, match=f"document {index} must not be blank"):
        make_embedder(model).embed_documents(texts)

    model.tokenizer.assert_not_called()
    model.encode_document.assert_not_called()


def test_embed_documents_accepts_token_count_exactly_at_limit() -> None:
    model = make_model([256], [[0.1, 0.2]])

    assert make_embedder(model).embed_documents(["passage"]) == [[0.1, 0.2]]


@pytest.mark.parametrize("index", [0, 1])
def test_embed_documents_rejects_any_oversized_passage_before_encoding(
    index: int,
) -> None:
    token_counts = [10, 10]
    token_counts[index] = 257
    model = make_model(token_counts, [[0.1], [0.2]])

    with pytest.raises(
        ValueError, match=f"document {index} has 257 tokens; limit is 256"
    ):
        make_embedder(model).embed_documents(["first passage", "second passage"])

    model.encode_document.assert_not_called()


@pytest.mark.parametrize("returned_vectors", [[], [[0.1]], [[0.1], [0.2], [0.3]]])
def test_embed_documents_rejects_output_count_mismatch(returned_vectors) -> None:
    model = make_model([4, 4], returned_vectors)

    with pytest.raises(ValueError, match="one vector per document"):
        make_embedder(model).embed_documents(["first passage", "second passage"])


@pytest.mark.parametrize("failing_step", ["tokenizer", "encode_document"])
def test_embed_documents_propagates_model_failures(failing_step: str) -> None:
    model = make_model([4], [[0.1]])
    failure = RuntimeError("model failed")
    getattr(model, failing_step).side_effect = failure

    with pytest.raises(RuntimeError) as raised:
        make_embedder(model).embed_documents(["passage"])

    assert raised.value is failure
    if failing_step == "tokenizer":
        model.encode_document.assert_not_called()


@given(
    texts=st.lists(
        st.text(alphabet="abc ", min_size=1, max_size=64).filter(
            lambda text: bool(text.strip())
        ),
        min_size=1,
        max_size=20,
    )
)
def test_generated_document_lists_preserve_count_order_and_input(
    texts: list[str],
) -> None:
    original = texts.copy()
    vectors = [[float(index), float(len(text))] for index, text in enumerate(texts)]
    # These fake token counts are test data, not MiniLM tokenization estimates.
    model = make_model([len(text) + 2 for text in texts], vectors)

    result = make_embedder(model).embed_documents(texts)

    assert result == vectors
    assert len(result) == len(texts)
    assert texts == original


def test_embedder_keeps_loaded_model_and_identity_without_running_model() -> None:
    model = make_model([], [])

    embedder = make_embedder(model)

    assert embedder.model is model
    assert embedder.model_id == "test-encoder"
    assert embedder.model_revision == "test-revision"
    assert model.mock_calls == []


def test_dimensions_are_read_from_loaded_model() -> None:
    model = make_model([], [])
    model.get_sentence_embedding_dimension.return_value = 384

    assert make_embedder(model).get_dimensions() == 384
    model.get_sentence_embedding_dimension.assert_called_once_with()


@pytest.mark.parametrize("dimensions", [None, 0, -1, 1.5, "384", True])
def test_invalid_model_dimensions_are_rejected(dimensions) -> None:
    model = make_model([], [])
    model.get_sentence_embedding_dimension.return_value = dimensions

    with pytest.raises(ValueError, match="positive integer embedding dimension"):
        make_embedder(model).get_dimensions()
