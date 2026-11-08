"""Build the search index once and save it to disk (an "embedding cache").

Why: embedding every chunk takes time, especially on a small free server (Render's free
plan has 0.1 CPU). So we embed everything ONCE while building the Docker image and save
the vectors as a .npy file. At startup the app just loads that file, which takes milliseconds.

The file name contains a fingerprint (hash) of the model, chunk size and chunk ids.
If the data or the settings change, the fingerprint changes and the index is rebuilt.
This is called "cache invalidation".

Build it by hand:  python -m rag_lab.index
"""

import hashlib
import logging
from pathlib import Path

import numpy as np

from rag_lab.config import PAPERS_PATH, settings
from rag_lab.data import Chunk, load_jsonl, make_chunks

logger = logging.getLogger("rag_lab")


def fingerprint(chunks: list[Chunk], model_name: str) -> str:
    text = model_name + "|" + "|".join(c.chunk_id for c in chunks)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_index(
    papers_path: Path = PAPERS_PATH,
    chunk_size: int | None = None,
    cache_dir: str | None = None,
    embedder=None,
):
    """Return (chunks, embedder, matrix). Loads the matrix from disk if it exists."""
    chunk_size = chunk_size or settings.chunk_size
    cache_dir = Path(cache_dir or settings.embed_cache_dir)
    if embedder is None:
        from rag_lab.embeddings import FastEmbedder

        embedder = FastEmbedder(settings.embed_model, str(cache_dir))

    chunks = make_chunks(load_jsonl(papers_path), chunk_size=chunk_size)
    path = cache_dir / f"index_{chunk_size}_{fingerprint(chunks, settings.embed_model)}.npy"
    if path.exists():
        logger.info("Loading cached index %s", path.name)
        matrix = np.load(path)
    else:
        logger.info("Embedding %d chunks (one time) ...", len(chunks))
        matrix = embedder.embed_documents([c.text for c in chunks])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
    return chunks, embedder, matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks, _, matrix = load_index()
    print(f"Index ready: {len(chunks)} chunks, matrix shape {matrix.shape}")