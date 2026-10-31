from nexusrag.constants import DEFAULT_EMBEDDING_BATCH_SIZE


class EmbeddingInputError(ValueError):
    """Text cannot be encoded without accepting blank input or truncation."""


class MiniEmbedder:
    """Wrap an already-loaded model; construction never downloads model files."""

    def __init__(self, model, *, model_id: str, model_revision: str):
        self.model = model
        self.model_id = model_id
        self.model_revision = model_revision

    def get_dimensions(self) -> int:
        dimensions = self.model.get_sentence_embedding_dimension()
        if (
            not isinstance(dimensions, int)
            or isinstance(dimensions, bool)
            or dimensions <= 0
        ):
            raise ValueError("model must report a positive integer embedding dimension")
        return dimensions

    def get_token_limit(self) -> int:
        limit = self.model.max_seq_length
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("model must report a positive integer token limit")
        return limit

    def count_tokens(self, text: str) -> int:
        """Count an exact source substring with the embedding tokenizer."""
        encoded = self.model.tokenizer(text, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise EmbeddingInputError("query must not be blank")

        token_count = self.count_tokens(query)
        token_limit = self.get_token_limit()
        if token_count > token_limit:
            raise EmbeddingInputError(
                f"query has {token_count} tokens; limit is {token_limit}"
            )

        query_vector = self.model.encode_query(
            query, normalize_embeddings=True
        ).tolist()

        return query_vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode passages in input order without accepting silent truncation.

        Empty input needs no model work. Validation and model errors propagate
        to the caller rather than becoming empty or fabricated vectors.
        """
        if not texts:
            return []

        for index, text in enumerate(texts):
            if not text.strip():
                raise EmbeddingInputError(f"document {index} must not be blank")

        encoded = self.model.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
        )
        token_limit = self.get_token_limit()
        for index, token_ids in enumerate(encoded["input_ids"]):
            token_count = len(token_ids)
            if token_count > token_limit:
                raise EmbeddingInputError(
                    f"document {index} has {token_count} tokens; limit is {token_limit}"
                )

        vectors = self.model.encode_document(
            texts,
            batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        if len(vectors) != len(texts):
            raise ValueError("model must return one vector per document")

        return vectors
