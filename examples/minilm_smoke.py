"""A manual, local embedding experiment; never run or download models in CI."""

from pathlib import Path
from typing import Protocol

from nexusrag.config import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION
from nexusrag.services.embedding_model import MiniEmbedder
from nexusrag.services.similarity import cosine_similarity

MODEL_ID = EMBEDDING_MODEL_ID
MODEL_REVISION = EMBEDDING_MODEL_REVISION
MODEL_CACHE = Path(__file__).resolve().parents[1] / "data" / "models"


class Tokenizer(Protocol):
    def __call__(
        self, texts: list[str], *, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[list[int]]]: ...


def check_token_lengths(
    tokenizer: Tokenizer, texts: list[str], max_tokens: int
) -> list[int]:
    """Include special tokens in the budget; refuse silently truncated inputs."""
    inputs = tokenizer(texts, add_special_tokens=True, truncation=False)
    lengths = [len(tokens) for tokens in inputs["input_ids"]]
    for index, length in enumerate(lengths):
        if length > max_tokens:
            raise ValueError(
                f"Input {index} has {length} tokens; model limit is {max_tokens}. "
                "Shorten or rechunk the text before embedding it."
            )
    return lengths


def main() -> None:
    # Optional/heavy imports and model downloads happen only when running the demo.
    from sentence_transformers import SentenceTransformer

    print(f"Loading {MODEL_ID} on CPU (first run downloads model files)...", flush=True)
    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        cache_folder=str(MODEL_CACHE),
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )
    embedder = MiniEmbedder(model, model_id=MODEL_ID, model_revision=MODEL_REVISION)
    query = "How many vacation days do employees get?"
    passages = [
        "Employees receive 25 days of annual leave each year.",
        "Report a lost laptop to the IT team immediately.",
    ]
    lengths = check_token_lengths(
        model.tokenizer, [query, *passages], model.max_seq_length
    )

    # MiniLM requires no query/passage prefixes. Use the retrieval-specific methods.
    query_vector = embedder.embed_query(query)
    passage_vectors = embedder.embed_documents(passages)

    print(f"Revision: {MODEL_REVISION}")
    print(f"Dimensions: {len(query_vector)}")
    print(f"Token limit: {model.max_seq_length}; input token counts: {lengths}")
    print(f"Model cache: {MODEL_CACHE}")
    print(f"Query: {query}")
    for passage, vector in zip(passages, passage_vectors, strict=True):
        score = cosine_similarity(query_vector, vector)
        print(f"{score:.4f}  {passage}")


if __name__ == "__main__":
    main()
