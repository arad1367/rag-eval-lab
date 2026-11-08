import numpy as np
import pytest

from rag_lab.data import make_chunks, save_jsonl
from rag_lab.index import load_index
from rag_lab.retrievers import DenseRetriever


class CountingEmbedder:
    """Wraps the fake embedder and counts how often documents get embedded."""

    def __init__(self, inner):
        self.inner = inner
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)


def test_index_is_built_once_then_loaded_from_cache(tmp_path, papers, embedder):
    papers_path = tmp_path / "papers.jsonl"
    save_jsonl(papers, papers_path)
    counting = CountingEmbedder(embedder)

    chunks1, _, m1 = load_index(papers_path, 300, str(tmp_path / "cache"), counting)
    chunks2, _, m2 = load_index(papers_path, 300, str(tmp_path / "cache"), counting)

    assert counting.document_calls == 1  # second call used the saved file
    assert len(list((tmp_path / "cache").glob("index_300_*.npy"))) == 1
    assert np.allclose(m1, m2)
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_new_data_gets_a_new_index_file(tmp_path, papers, embedder):
    papers_path = tmp_path / "papers.jsonl"
    save_jsonl(papers, papers_path)
    load_index(papers_path, 300, str(tmp_path / "cache"), embedder)
    save_jsonl(papers[:2], papers_path)  # the data changed
    load_index(papers_path, 300, str(tmp_path / "cache"), embedder)
    assert len(list((tmp_path / "cache").glob("index_300_*.npy"))) == 2


def test_dense_retriever_rejects_wrong_matrix(papers, embedder):
    chunks = make_chunks(papers)
    with pytest.raises(ValueError):
        DenseRetriever(chunks, embedder, matrix=np.zeros((1, 4)))
