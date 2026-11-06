"""Text embeddings: turning text into vectors so that similar meaning = nearby vectors.

Concepts:
- An embedding model maps a text to a fixed-size vector (here 384 numbers).
- Cosine similarity = dot product of two vectors after normalizing them to length 1.
  1.0 = same direction (same meaning), 0 = unrelated.
- BGE models are "asymmetric": queries get a short instruction prefix, documents don't.
  fastembed does this for us with query_embed() vs passage_embed().
- fastembed runs the model with ONNX Runtime on CPU, so we don't need PyTorch or a GPU.
"""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Anything with these two methods can be used as an embedder (duck typing).
    This lets tests use a tiny fake embedder instead of downloading a real model."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so dot product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class FastEmbedder:
    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed import TextEmbedding  # imported here so tests never load it

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # Small batches keep memory low (Render's free plan has only 512 MB RAM).
        vectors = list(self.model.passage_embed(texts, batch_size=16))
        return normalize(np.array(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(np.array(list(self.model.query_embed(text))[0], dtype=np.float32))