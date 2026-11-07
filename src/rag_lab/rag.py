"""The RAG pipeline: Retrieve -> Augment (build the prompt) -> Generate.

Concepts:
- The LLM only sees what we put in the prompt. Retrieval decides that.
- "Grounding": we tell the model to answer ONLY from the given context and to cite
  sources like [1]. This reduces hallucination and makes answers checkable.
- Telling the model it may say "I don't know" is an important safety valve.
"""

import time

from rag_lab.retrievers import SearchResult

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the numbered context "
    "passages. Cite the passages you used like [1] or [2]. If the context does not contain "
    "the answer, say exactly: I don't know based on the provided papers. "
    "Keep the answer under 120 words."
)


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(f"[{i}] {r.chunk.text}" for i, r in enumerate(results, start=1))
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


class RAGPipeline:
    def __init__(self, retrievers: dict, llm, default_retriever: str = "hybrid"):
        self.retrievers = retrievers
        self.llm = llm
        self.default_retriever = default_retriever

    def answer(self, question: str, k: int = 4, retriever: str | None = None) -> dict:
        name = retriever or self.default_retriever
        if name not in self.retrievers:
            raise ValueError(f"Unknown retriever '{name}'. Use one of {list(self.retrievers)}")

        start = time.perf_counter()
        results = self.retrievers[name].search(question, k)
        retrieval_ms = (time.perf_counter() - start) * 1000

        response = self.llm.complete(SYSTEM_PROMPT, build_prompt(question, results))
        total_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": response.text,
            "retriever": name,
            "sources": [
                {
                    "rank": i,
                    "paper_id": r.chunk.paper_id,
                    "title": r.chunk.title,
                    "url": f"https://arxiv.org/abs/{r.chunk.paper_id}",
                    "score": round(r.score, 4),
                    "text": r.chunk.text,
                }
                for i, r in enumerate(results, start=1)
            ],
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            },
            "latency_ms": {"retrieval": round(retrieval_ms, 1), "total": round(total_ms, 1)},
        }