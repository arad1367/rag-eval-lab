import pytest

from rag_lab.data import make_chunks
from rag_lab.retrievers import build_retrievers, unique_papers


@pytest.fixture
def retrievers(papers, embedder):
    return build_retrievers(make_chunks(papers, chunk_size=300), embedder)


@pytest.mark.parametrize("name", ["bm25", "dense", "hybrid"])
def test_each_retriever_finds_the_obvious_paper(retrievers, name):
    results = retrievers[name].search("speech recognition for Swahili", k=2)
    assert results[0].chunk.paper_id == "0002"


@pytest.mark.parametrize("name", ["bm25", "dense", "hybrid"])
def test_results_are_sorted_and_limited(retrievers, name):
    results = retrievers[name].search("transformer translation", k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_unique_papers_keeps_rank_order(papers, embedder):
    chunks = make_chunks(papers, chunk_size=5, overlap=1)  # many chunks per paper
    results = build_retrievers(chunks, embedder)["bm25"].search("hallucination retrieval", k=10)
    ids = unique_papers(results)
    assert ids[0] == "0001"
    assert len(ids) == len(set(ids))
