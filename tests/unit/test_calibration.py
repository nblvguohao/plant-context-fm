"""Unit tests for regression uncertainty calibration (TDD Section 9)."""

import numpy as np
import pytest

from plant_context.evaluation.calibration import (
    calibration_report,
    expected_calibration_error_regression,
    prediction_interval_coverage,
    probability_integral_transform,
    pit_uniformity_test,
    reliability_diagram_data,
)


def test_perfectly_calibrated_gaussian_has_correct_coverage():
    rng = np.random.default_rng(42)
    n = 5000
    y_pred = np.zeros(n)
    y_std = np.ones(n)
    y_true = rng.normal(loc=0.0, scale=1.0, size=n)

    cov_95 = prediction_interval_coverage(y_true, y_pred, y_std, confidence=0.95)
    cov_68 = prediction_interval_coverage(y_true, y_pred, y_std, confidence=0.68)

    assert cov_95 == pytest.approx(0.95, abs=0.02)
    assert cov_68 == pytest.approx(0.68, abs=0.02)


def test_overconfident_predictions_have_low_coverage():
    rng = np.random.default_rng(43)
    n = 5000
    y_pred = np.zeros(n)
    y_std = np.full(n, 0.5)  # true std is 1.0 -> undercovers
    y_true = rng.normal(loc=0.0, scale=1.0, size=n)

    cov_95 = prediction_interval_coverage(y_true, y_pred, y_std, confidence=0.95)
    assert cov_95 < 0.90


def test_ece_is_low_for_calibrated_model():
    rng = np.random.default_rng(44)
    n = 5000
    y_pred = np.zeros(n)
    y_std = np.ones(n)
    y_true = rng.normal(loc=0.0, scale=1.0, size=n)

    ece = expected_calibration_error_regression(y_true, y_pred, y_std)
    assert ece < 0.05


def test_ece_is_high_for_miscalibrated_model():
    rng = np.random.default_rng(45)
    n = 5000
    y_pred = np.zeros(n)
    y_std = np.full(n, 0.3)  # much too confident
    y_true = rng.normal(loc=0.0, scale=1.0, size=n)

    ece = expected_calibration_error_regression(y_true, y_pred, y_std)
    assert ece > 0.15


def test_pit_is_uniform_for_calibrated_model():
    rng = np.random.default_rng(46)
    n = 5000
    y_pred = np.zeros(n)
    y_std = np.ones(n)
    y_true = rng.normal(loc=0.0, scale=1.0, size=n)

    pit = probability_integral_transform(y_true, y_pred, y_std)
    test = pit_uniformity_test(pit)
    assert test["p_value"] > 0.01


def test_pit_is_nonuniform_for_overconfident_model():
    rng = np.random.default_rng(47)
    n = 5000
    y_pred = np.zeros(n)
    y_std = np.full(n, 0.3)
    y_true = rng.normal(loc=0.0, scale=1.0, size=n)

    pit = probability_integral_transform(y_true, y_pred, y_std)
    test = pit_uniformity_test(pit)
    assert test["p_value"] < 0.01


def test_reliability_diagram_returns_non_empty_bins():
    rng = np.random.default_rng(48)
    n = 1000
    y_pred = rng.normal(size=n)
    y_std = np.abs(rng.normal(size=n)) + 0.1
    y_true = y_pred + rng.normal(scale=y_std, size=n)

    data = reliability_diagram_data(y_true, y_pred, y_std, n_bins=5)
    assert len(data["bin_centers"]) > 0
    assert len(data["observed_coverage"]) == len(data["bin_centers"])
    assert np.all(data["counts"] > 0)


def test_calibration_report_contains_expected_keys():
    rng = np.random.default_rng(49)
    n = 1000
    y_pred = np.zeros(n)
    y_std = np.ones(n)
    y_true = rng.normal(loc=0.0, scale=1.0, size=n)

    report = calibration_report(y_true, y_pred, y_std)
    expected_keys = {
        "expected_calibration_error",
        "pit_uniformity_statistic",
        "pit_uniformity_p_value",
        "coverage_95",
        "coverage_68",
        "n",
    }
    assert set(report.keys()) == expected_keys
    assert report["n"] == n
