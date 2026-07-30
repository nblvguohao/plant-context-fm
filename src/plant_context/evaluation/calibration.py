"""Uncertainty calibration diagnostics for regression predictions.

The models in this project (G×E, community, pretraining) can optionally
produce uncertainty estimates (predictive standard deviations). This module
measures whether those uncertainties are calibrated: a 95% predictive
interval should cover the true value ~95% of the time, and the probability
integral transform of the residuals should be uniform.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats


def prediction_interval_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Fraction of true values inside the ``confidence`` predictive interval.

    Interval is ``[pred - z*std, pred + z*std]`` with z from the standard
    normal. For a perfectly calibrated Gaussian, coverage equals
    ``confidence``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    z = stats.norm.ppf(0.5 + confidence / 2)
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def expected_calibration_error_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    confidence_levels: Optional[np.ndarray] = None,
) -> float:
    """Regression ECE: average absolute deviation between expected and observed
    coverage across a set of confidence levels.

    Parameters
    ----------
    confidence_levels :
        Confidence levels at which to evaluate coverage. Defaults to
        [0.1, 0.2, ..., 0.9].
    """
    if confidence_levels is None:
        confidence_levels = np.arange(0.1, 1.0, 0.1)

    coverages = np.array(
        [prediction_interval_coverage(y_true, y_pred, y_std, c) for c in confidence_levels]
    )
    return float(np.mean(np.abs(coverages - confidence_levels)))


def probability_integral_transform(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> np.ndarray:
    """PIT values assuming a Gaussian predictive distribution N(pred, std^2).

    Under a perfectly calibrated model, the PIT values are i.i.d. Uniform(0,1).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    y_std = np.clip(y_std, 1e-12, None)
    z = (y_true - y_pred) / y_std
    return stats.norm.cdf(z)


def pit_uniformity_test(pit_values: np.ndarray) -> dict:
    """Kolmogorov-Smirnov test of PIT values against Uniform(0,1).

    Returns statistic, p_value, and a qualitative flag. A well-calibrated
    model should have a non-significant p-value (i.e. we cannot reject
    uniformity).
    """
    pit_values = np.asarray(pit_values, dtype=float)
    pit_values = pit_values[~np.isnan(pit_values)]
    if len(pit_values) == 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": 0}

    statistic, p_value = stats.kstest(pit_values, "uniform", args=(0, 1))
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n": len(pit_values),
    }


def reliability_diagram_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
    confidence: float = 0.95,
) -> dict:
    """Bin predictions by their predicted confidence and report observed coverage.

    Returns bin_centers, predicted_confidence, observed_coverage, and counts.
    A calibrated model has observed_coverage ≈ predicted_confidence within
    each bin.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    z = stats.norm.ppf(0.5 + confidence / 2)
    half_width = z * y_std
    relative_width = half_width / (np.abs(y_pred) + 1e-8)

    # Bin by predicted relative interval width (a proxy for predicted confidence)
    bin_edges = np.quantile(relative_width, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-8  # ensure max value lands in last bin

    bin_centers = []
    pred_conf = []
    obs_cov = []
    counts = []

    for i in range(n_bins):
        mask = (relative_width >= bin_edges[i]) & (relative_width < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        pred_conf.append(confidence)
        lower = y_pred[mask] - half_width[mask]
        upper = y_pred[mask] + half_width[mask]
        obs_cov.append(float(np.mean((y_true[mask] >= lower) & (y_true[mask] <= upper))))
        counts.append(int(mask.sum()))

    return {
        "bin_centers": np.array(bin_centers),
        "predicted_confidence": np.array(pred_conf),
        "observed_coverage": np.array(obs_cov),
        "counts": np.array(counts),
    }


def calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> dict:
    """One-stop calibration summary."""
    ece = expected_calibration_error_regression(y_true, y_pred, y_std)
    pit = probability_integral_transform(y_true, y_pred, y_std)
    pit_test = pit_uniformity_test(pit)
    cov_95 = prediction_interval_coverage(y_true, y_pred, y_std, confidence=0.95)
    cov_68 = prediction_interval_coverage(y_true, y_pred, y_std, confidence=0.68)

    return {
        "expected_calibration_error": ece,
        "pit_uniformity_statistic": pit_test["statistic"],
        "pit_uniformity_p_value": pit_test["p_value"],
        "coverage_95": cov_95,
        "coverage_68": cov_68,
        "n": pit_test["n"],
    }
