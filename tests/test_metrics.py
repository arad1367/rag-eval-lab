import pytest

from rag_lab.metrics import bootstrap_ci, paired_bootstrap, recall_at_k, reciprocal_rank


def test_recall_at_k():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, "b", 1) == 0.0
    assert recall_at_k(ranked, "b", 2) == 1.0
    assert recall_at_k(ranked, "z", 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], "a") == 1.0
    assert reciprocal_rank(["a", "b", "c"], "c") == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b", "c"], "z") == 0.0


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    values = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    mean, low, high = bootstrap_ci(values)
    assert mean == pytest.approx(0.7)
    assert low <= mean <= high
    assert bootstrap_ci(values) == (mean, low, high)  # same seed -> same numbers


def test_paired_bootstrap_detects_clear_winner():
    better = [1.0] * 30
    worse = [0.0] * 15 + [1.0] * 15
    diff, low, high, p = paired_bootstrap(better, worse)
    assert diff == pytest.approx(0.5)
    assert low > 0  # the whole confidence interval is above zero
    assert p < 0.05
