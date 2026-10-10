from nexusrag.constants import DEFAULT_EMBEDDING_BATCH_SIZE


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

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("query must not be blank")

        encoded = self.model.tokenizer(
            query,
            add_special_tokens=True,
            truncation=False,
        )

        token_count = len(encoded["input_ids"])
        if token_count > self.model.max_seq_length:
            raise ValueError(
                f"query has {token_count} tokens; limit is {self.model.max_seq_length}"
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
                raise ValueError(f"document {index} must not be blank")

        encoded = self.model.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
        )
        for index, token_ids in enumerate(encoded["input_ids"]):
            token_count = len(token_ids)
            if token_count > self.model.max_seq_length:
                raise ValueError(
                    f"document {index} has {token_count} tokens; "
                    f"limit is {self.model.max_seq_length}"
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
