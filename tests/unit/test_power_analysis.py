"""Unit tests for statistical power analysis utilities (TDD Section 9).

Covers: paired MDE, Nadeau-Bengio corrected t-test, seed-count
recommendation, and the results-table power report.
"""

import numpy as np
import pandas as pd
import pytest

from plant_context.evaluation.power_analysis import (
    nadeau_bengio_corrected_ttest,
    paired_mde,
    power_report_from_results,
    recommend_seed_count,
)


def test_paired_mde_with_zero_variance_requires_more_seeds():
    diffs = [0.0, 0.0, 0.0]
    assert paired_mde(diffs) == 0.0


def test_paired_mde_is_non_negative():
    rng = np.random.default_rng(42)
    diffs = rng.normal(loc=0.1, scale=0.05, size=10)
    assert paired_mde(diffs) > 0


def test_paired_mde_decreases_with_more_seeds():
    rng = np.random.default_rng(7)
    diffs_small = rng.normal(loc=0.1, scale=0.05, size=5)
    diffs_large = np.tile(diffs_small, 4)  # 20 draws from same distribution

    mde_small = paired_mde(diffs_small)
    mde_large = paired_mde(diffs_large)
    assert mde_large < mde_small


def test_paired_mde_returns_nan_for_insufficient_data():
    assert np.isnan(paired_mde([1.0]))
    assert np.isnan(paired_mde([]))


def test_nadeau_bengio_p_value_for_identical_scores():
    scores = [0.5, 0.5, 0.5, 0.5]
    result = nadeau_bengio_corrected_ttest(
        scores, scores, n_train=1000, n_test=200, alpha=0.05
    )
    assert result["mean_diff"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_nadeau_bengio_detects_consistent_difference():
    a = [1.0, 1.1, 0.9, 1.05]
    b = [1.5, 1.6, 1.55, 1.52]
    result = nadeau_bengio_corrected_ttest(
        a, b, n_train=1000, n_test=200, alpha=0.05
    )
    assert result["mean_diff"] < 0  # a is better (lower metric)
    assert result["p_value"] < 0.05


def test_nadeau_bengio_correction_increases_mde():
    a = [1.0, 1.1, 0.9, 1.05, 1.0]
    b = [1.2, 1.3, 1.15, 1.25, 1.18]

    result_small = nadeau_bengio_corrected_ttest(
        a, b, n_train=100_000, n_test=100, alpha=0.05
    )
    result_large = nadeau_bengio_corrected_ttest(
        a, b, n_train=1_000, n_test=500, alpha=0.05
    )

    # Higher n_test/n_train ratio inflates variance -> larger MDE.
    assert result_large["corrected_mde"] > result_small["corrected_mde"]


def test_nadeau_bengio_returns_nan_with_too_few_seeds():
    result = nadeau_bengio_corrected_ttest(
        [1.0], [1.2], n_train=100, n_test=20, alpha=0.05
    )
    assert np.isnan(result["mean_diff"])
    assert result["n_seeds"] == 1


def test_recommend_seed_count_suggests_more_seeds_for_smaller_effects():
    # Use a modest train/test ratio so that effects in the 0.1-0.5 range are
    # actually detectable within the 20-seed cap.
    rec_large = recommend_seed_count(
        expected_effect=0.5,
        score_std=0.05,
        n_test=100,
        n_train=2000,
        max_seeds=20,
    )
    rec_small = recommend_seed_count(
        expected_effect=0.1,
        score_std=0.05,
        n_test=100,
        n_train=2000,
        max_seeds=20,
    )

    assert rec_large["achievable"] is True
    assert rec_small["achievable"] is True
    assert rec_small["recommended_seeds"] >= rec_large["recommended_seeds"]


def test_recommend_seed_count_flags_undetectable_effects():
    rec = recommend_seed_count(
        expected_effect=0.001,
        score_std=0.1,
        n_test=200,
        n_train=1000,
        max_seeds=5,
    )
    assert rec["achievable"] is False
    assert rec["recommended_seeds"] == 5


def test_recommend_seed_count_rejects_invalid_inputs():
    rec = recommend_seed_count(
        expected_effect=-0.1,
        score_std=0.1,
        n_test=200,
        n_train=1000,
    )
    assert rec["achievable"] is False
    assert rec["recommended_seeds"] is None


def test_power_report_flags_underpowered_comparison():
    rng = np.random.default_rng(99)
    n_seeds = 3
    # Two models with essentially identical RMSE: difference is far below MDE.
    df = pd.DataFrame(
        {
            "outer_split_type": ["leave_environment"] * n_seeds * 2,
            "model": ["gblup"] * n_seeds + ["ours"] * n_seeds,
            "seed": list(range(n_seeds)) * 2,
            "rmse": list(rng.normal(1.0, 0.02, n_seeds))
            + list(rng.normal(1.01, 0.02, n_seeds)),
            "n_train": [800] * n_seeds * 2,
            "n_test": [200] * n_seeds * 2,
        }
    )

    report = power_report_from_results(df, baseline_model="gblup")
    assert len(report) == 1
    assert report.iloc[0]["model_a"] == "ours"
    assert report.iloc[0]["model_b"] == "gblup"
    # powered is stored as a numpy bool_; use boolean coercion, not identity.
    assert not report.iloc[0]["powered"]


def test_power_report_uses_all_pairs_without_baseline():
    df = pd.DataFrame(
        {
            "outer_split_type": ["leave_environment"] * 4,
            "model": ["a", "a", "b", "b"],
            "seed": [0, 1, 0, 1],
            "rmse": [1.0, 1.1, 1.2, 1.3],
            "n_train": [100] * 4,
            "n_test": [20] * 4,
        }
    )
    report = power_report_from_results(df)
    assert len(report) == 1
    assert set(report.iloc[0][["model_a", "model_b"]]) == {"a", "b"}


def test_power_report_returns_empty_for_empty_input():
    assert power_report_from_results(pd.DataFrame()).empty
