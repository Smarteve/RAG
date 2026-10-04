"""Shared constants for document processing."""

CHUNK_SIZE_BYTES = 1024 * 1024

# Character windows for extracted text, not byte counts or LLM tokens.
DEFAULT_CHUNK_SIZE_CHARS = 1000
DEFAULT_CHUNK_OVERLAP_CHARS = 200
