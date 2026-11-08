"""Evaluation metrics and statistics.

Concepts:
- Recall@k: did the correct paper appear in the top k results? (1 or 0 per question,
  averaged over questions). Each question here has exactly one correct paper, so this
  is also called "hit rate@k".
- MRR (Mean Reciprocal Rank): 1/rank of the correct paper (1st = 1.0, 2nd = 0.5,
  3rd = 0.33, not found = 0). Rewards putting the right answer near the top.
- Bootstrap confidence interval: resample the questions WITH replacement many times,
  recompute the mean each time, and take the 2.5% and 97.5% percentiles. It tells you
  how much the score could move with a different sample of questions.
- Paired bootstrap: to compare two systems, resample the SAME questions for both and look
  at the difference. Paired comparison is more sensitive, because both systems face
  exactly the same easy and hard questions.
"""

import numpy as np


def recall_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_id: str) -> float:
    for rank, pid in enumerate(ranked_ids, start=1):
        if pid == gold_id:
            return 1.0 / rank
    return 0.0


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    """Return (mean, low, high) for the mean of `values`."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)  # fixed seed = same result every run
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(low), float(high)


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 42):
    """Compare system A vs B on the same questions.
    Returns (mean difference A-B, CI low, CI high, p_value).
    p_value = share of resamples where A is NOT better than B (one-sided)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, low, high = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    p_value = float((diff[idx].mean(axis=1) <= 0).mean())
    return mean, low, high, p_value
