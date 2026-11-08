import pytest
from fastapi.testclient import TestClient

from rag_lab.api import create_app
from rag_lab.data import make_chunks
from rag_lab.rag import RAGPipeline, build_prompt
from rag_lab.retrievers import build_retrievers


@pytest.fixture
def client(papers, embedder, fake_llm):
    pipeline = RAGPipeline(build_retrievers(make_chunks(papers), embedder), fake_llm)
    with TestClient(create_app(pipeline=pipeline)) as c:  # "with" runs the lifespan
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG Eval Lab" in response.text


def test_ask_returns_answer_and_sources(client):
    response = client.post("/ask", json={"question": "speech recognition Swahili", "k": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Fake answer [1]"
    assert data["sources"][0]["paper_id"] == "0002"
    assert data["sources"][0]["url"] == "https://arxiv.org/abs/0002"
    assert len(data["sources"]) == 2
    assert data["usage"]["prompt_tokens"] == 100


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},  # too short
        {"question": "valid question", "k": 0},  # k must be >= 1
        {"question": "valid question", "retriever": "magic"},  # unknown retriever
    ],
)
def test_ask_rejects_bad_input(client, body):
    assert client.post("/ask", json=body).status_code == 422


def test_prompt_contains_numbered_context(papers, embedder):
    results = build_retrievers(make_chunks(papers), embedder)["bm25"].search("BERT reviews", 2)
    prompt = build_prompt("What model?", results)
    assert "[1]" in prompt and "[2]" in prompt
    assert prompt.endswith("Question: What model?\nAnswer:")
