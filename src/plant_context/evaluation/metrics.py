"""Prediction quality metrics (TDD Section 9.1)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_r(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(stats.pearsonr(y_true, y_pred)[0])


def spearman_r(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred)[0])


def topk_selection_gain(y_true, y_pred, fraction: float = 0.2) -> float:
    """Mean true value among the top ``fraction`` predicted, minus the
    population mean true value: the realized gain from selecting on
    predictions rather than at random (TDD Section 9.1, "Top-5/10/20%
    selection gain").
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(round(n * fraction)))
    top_idx = np.argsort(-y_pred)[:k]
    return float(y_true[top_idx].mean() - y_true.mean())
