"""The research part: which retriever and chunk size find the right paper most often?

Usage:  python scripts/run_experiment.py

Experiment design:
- Independent variables: retriever (bm25, dense, hybrid) x chunk size (50, 300 words)
- Dependent variables: Recall@1, Recall@5, MRR (paper level)
- Same questions for every configuration -> paired comparison
- 95% bootstrap confidence intervals, and a paired bootstrap test hybrid vs dense
Outputs: results/retrieval_results.csv, results/retrieval_results.svg, results/results.md
"""

import matplotlib.pyplot as plt
import pandas as pd

from rag_lab.config import EVAL_PATH, PAPERS_PATH, RESULTS_DIR, settings
from rag_lab.data import load_jsonl, make_chunks
from rag_lab.metrics import bootstrap_ci, paired_bootstrap, recall_at_k, reciprocal_rank
from rag_lab.retrievers import build_retrievers, unique_papers

CHUNK_SIZES = [50, 300]
SEARCH_DEPTH = 20  # retrieve 20 chunks, then dedupe to papers


def per_question_scores(papers, questions, embedder, chunk_sizes=CHUNK_SIZES) -> pd.DataFrame:
    """One row per (chunk_size, retriever, question) with that question's scores."""
    rows = []
    for size in chunk_sizes:
        chunks = make_chunks(papers, chunk_size=size)
        retrievers = build_retrievers(chunks, embedder)
        for name, retriever in retrievers.items():
            for qi, q in enumerate(questions):
                ranked = unique_papers(retriever.search(q["question"], SEARCH_DEPTH))
                rows.append(
                    {
                        "chunk_size": size,
                        "retriever": name,
                        "qid": qi,
                        "recall@1": recall_at_k(ranked, q["paper_id"], 1),
                        "recall@5": recall_at_k(ranked, q["paper_id"], 5),
                        "mrr": reciprocal_rank(ranked, q["paper_id"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (size, name), group in scores.groupby(["chunk_size", "retriever"]):
        row = {"chunk_size": size, "retriever": name}
        for metric in ["recall@1", "recall@5", "mrr"]:
            mean, low, high = bootstrap_ci(group[metric])
            row[metric] = round(mean, 3)
            row[f"{metric}_ci"] = f"[{low:.2f}, {high:.2f}]"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mrr", ascending=False)


def compare(scores: pd.DataFrame, a: str, b: str, size: int, metric: str = "mrr") -> str:
    def pick(name):
        sub = scores[(scores.retriever == name) & (scores.chunk_size == size)]
        return sub.sort_values("qid")[metric].to_numpy()

    diff, low, high, p = paired_bootstrap(pick(a), pick(b))
    return (
        f"{a} vs {b} (chunk {size}, {metric}): difference = {diff:+.3f}, "
        f"95% CI [{low:+.3f}, {high:+.3f}], one-sided p = {p:.3f}"
    )


def to_markdown(df: pd.DataFrame) -> str:
    """Small markdown table writer (avoids an extra dependency)."""
    header = "| " + " | ".join(df.columns) + " |"
    divider = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def plot(summary: pd.DataFrame, path) -> None:
    labels = [f"{r.retriever}\n{r.chunk_size}w" for r in summary.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(summary))
    ax.bar([i - 0.2 for i in x], summary["recall@5"], width=0.4, label="Recall@5")
    ax.bar([i + 0.2 for i in x], summary["mrr"], width=0.4, label="MRR")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Retrieval quality by retriever and chunk size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)  # SVG = text file: renders on GitHub and is accepted by HF Spaces
    plt.close(fig)


def run(papers, questions, embedder, out_dir=RESULTS_DIR) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = per_question_scores(papers, questions, embedder)
    summary = summarize(scores)
    summary.to_csv(out_dir / "retrieval_results.csv", index=False)
    plot(summary, out_dir / "retrieval_results.svg")

    tests = [compare(scores, "hybrid", "dense", s) for s in CHUNK_SIZES]
    tests += [compare(scores, "hybrid", "bm25", s) for s in CHUNK_SIZES]
    report = [
        "# Retrieval experiment results",
        "",
        f"- Papers: {len(papers)} | Questions: {len(questions)} | "
        f"Embedding model: `{settings.embed_model}`",
        "- Metrics are at paper level (duplicate chunks of the same paper are merged).",
        "- Brackets show 95% bootstrap confidence intervals.",
        "",
        to_markdown(summary),
        "",
        "## Paired bootstrap tests",
        "",
        *[f"- {t}" for t in tests],
        "",
        "![chart](retrieval_results.svg)",
        "",
    ]
    (out_dir / "results.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main() -> None:
    from rag_lab.embeddings import FastEmbedder

    papers = load_jsonl(PAPERS_PATH)
    questions = load_jsonl(EVAL_PATH)
    embedder = FastEmbedder(settings.embed_model, settings.embed_cache_dir)
    summary = run(papers, questions, embedder)
    print(summary.to_string(index=False))
    print(f"\nFull report: {RESULTS_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
