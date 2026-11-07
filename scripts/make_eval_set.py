"""Create a test set: one question per sampled paper, written by the LLM.

Usage:  python scripts/make_eval_set.py --n 50

Concepts:
- To measure retrieval we need (question, correct paper) pairs = "gold labels".
- Writing them by hand is best but slow, so we generate them ("synthetic eval data").
- Known risk: the LLM tends to copy words from the abstract, which makes keyword search
  (BM25) look better than it is. We ask it to paraphrase, and we discuss this bias
  in results.md. Always read a sample of generated questions yourself!
"""

import argparse
import random

from rag_lab.config import EVAL_PATH, PAPERS_PATH
from rag_lab.data import load_jsonl, save_jsonl
from rag_lab.llm import LLMClient

SYSTEM = (
    "You write evaluation questions for a search engine over research papers. "
    "Given one paper, write ONE specific question that this paper answers. "
    "Paraphrase: do not copy the title and avoid copying exact phrases from the abstract. "
    "Do not mention 'this paper' or 'the authors'. Output only the question."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    papers = load_jsonl(PAPERS_PATH)
    random.seed(args.seed)  # same sample of papers every time you run it
    sample = random.sample(papers, min(args.n, len(papers)))

    llm = LLMClient()
    rows = []
    for i, paper in enumerate(sample, start=1):
        prompt = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
        question = llm.complete(SYSTEM, prompt, max_tokens=1024).text.strip().strip('"')
        if not question:
            print(f"[{i}/{len(sample)}] empty answer, skipped")
            continue
        rows.append({"question": question, "paper_id": paper["id"]})
        print(f"[{i}/{len(sample)}] {question}")

    save_jsonl(rows, EVAL_PATH)
    print(f"Saved {len(rows)} questions to {EVAL_PATH}")


if __name__ == "__main__":
    main()