"""LLM-as-a-judge: score the GENERATED answers, not just the retrieval.

Usage:  python scripts/judge_answers.py --n 15 --retriever hybrid

Concepts:
- Retrieval metrics say "did we find the right paper?". They don't say "is the answer
  faithful to the papers?". For that we ask a second LLM call to grade each answer.
- Faithfulness (groundedness): is every claim in the answer supported by the context?
- Known judge biases: prefers longer answers, prefers its own style, position bias,
  and it can be wrong. Mitigations: a strict rubric, temperature 0, asking for a short
  reason, and checking a sample by hand (human agreement).
"""

import argparse
import json
import re
import statistics

from rag_lab.config import EVAL_PATH, PAPERS_PATH, RESULTS_DIR, settings
from rag_lab.data import load_jsonl, make_chunks
from rag_lab.embeddings import FastEmbedder
from rag_lab.llm import LLMClient
from rag_lab.rag import RAGPipeline
from rag_lab.retrievers import build_retrievers

JUDGE_SYSTEM = """You are a strict grader. Rate how FAITHFUL the answer is to the context.
5 = every claim is supported by the context
4 = almost all supported, one minor unsupported detail
3 = mix of supported and unsupported claims
2 = mostly unsupported
1 = contradicts the context or is made up
If the answer says it does not know and the context really lacks the answer, give 5.
Reply in exactly this format:
SCORE: <1-5>
REASON: <one sentence>"""

SCORE_RE = re.compile(r"SCORE:\s*([1-5])")


def parse_score(text: str) -> int | None:
    match = SCORE_RE.search(text)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--retriever", default=settings.retriever)
    args = parser.parse_args()

    papers = load_jsonl(PAPERS_PATH)
    questions = load_jsonl(EVAL_PATH)[: args.n]
    chunks = make_chunks(papers, chunk_size=settings.chunk_size)
    retrievers = build_retrievers(
        chunks, FastEmbedder(settings.embed_model, settings.embed_cache_dir)
    )
    llm = LLMClient()
    pipeline = RAGPipeline(retrievers, llm, default_retriever=args.retriever)

    rows = []
    for i, q in enumerate(questions, start=1):
        result = pipeline.answer(q["question"])
        # the judge sees exactly the same passages the answering LLM saw
        context = "\n\n".join(f"[{s['rank']}] {s['text']}" for s in result["sources"])
        verdict = llm.complete(
            JUDGE_SYSTEM,
            f"Context:\n{context}\n\nQuestion: {q['question']}\n\nAnswer: {result['answer']}",
            max_tokens=1024,
        ).text
        rows.append(
            {
                "question": q["question"],
                "answer": result["answer"],
                "gold_in_sources": q["paper_id"] in [s["paper_id"] for s in result["sources"]],
                "score": parse_score(verdict),
                "verdict": verdict,
            }
        )
        print(f"[{i}/{len(questions)}] score={rows[-1]['score']}")

    scored = [r["score"] for r in rows if r["score"] is not None]
    hit_rate = sum(r["gold_in_sources"] for r in rows) / len(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "judge_results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = (
        f"# LLM-as-judge results\n\n"
        f"- Retriever: `{args.retriever}` | Questions: {len(rows)}\n"
        f"- Judge model: `{settings.llm_model}`\n"
        f"- Mean faithfulness: {statistics.mean(scored):.2f} / 5 ({len(scored)} parsed)\n"
        f"- Correct paper among the sources: {hit_rate:.0%}\n\n"
        f"Read judge_results.jsonl and check at least 5 verdicts by hand.\n"
    )
    (RESULTS_DIR / "judge_results.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
