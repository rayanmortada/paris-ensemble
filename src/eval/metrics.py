"""Fairness metrics and statistics helpers."""
import numpy as np


def gini(s) -> float:
    s = np.asarray(s, float)
    if s.mean() == 0:
        return 0.0
    return float(np.abs(s[:, None] - s[None, :]).sum() / (2 * len(s) ** 2 * s.mean()))


def sacrificed(daily: np.ndarray) -> bool:
    """daily: members x days satisfaction. True if the same member is lowest on every day."""
    lowest = daily.argmin(axis=0)
    return daily.shape[0] > 1 and bool((lowest == lowest[0]).all())


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    out, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def bootstrap_ci(x, n: int = 10_000, seed: int = 7) -> tuple[float, float]:
    """95% interval for the mean, resampling groups."""
    x = np.asarray(x, float)
    means = np.random.default_rng(seed).choice(x, (n, len(x))).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
