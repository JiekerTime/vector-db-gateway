"""Backend factories."""

from vector_gateway.backends.embedding_local import LocalEmbeddingBackend
from vector_gateway.backends.qdrant_store import QdrantStore

__all__ = ["LocalEmbeddingBackend", "QdrantStore"]
