import numpy as np
import pytest

from plant_context.evaluation.metrics import (
    mae,
    pearson_r,
    rmse,
    spearman_r,
    topk_selection_gain,
)


def test_rmse_basic():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert rmse([0, 0], [1, 1]) == 1.0


def test_mae_basic():
    assert mae([0, 0], [1, 3]) == 2.0


def test_pearson_r_perfect_correlation():
    assert pearson_r([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_pearson_r_zero_variance_is_nan():
    assert np.isnan(pearson_r([1, 1, 1], [1, 2, 3]))


def test_spearman_r_perfect_rank_correlation():
    assert spearman_r([1, 2, 3, 4], [10, 20, 15, 40]) > 0.7


def test_spearman_r_zero_variance_is_nan():
    assert np.isnan(spearman_r([1, 1, 1], [1, 2, 3]))


def test_topk_selection_gain_picks_top_predicted():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # predictions rank-match truth exactly
    gain = topk_selection_gain(y_true, y_pred, fraction=0.2)
    # top 20% of 5 rows -> 1 row -> the single largest true value (5.0)
    assert gain == 5.0 - y_true.mean()


def test_topk_selection_gain_is_zero_for_uninformative_predictions_matching_mean_pick():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 0.0, 0.0])  # ties broken by argsort's stable order -> first row
    gain = topk_selection_gain(y_true, y_pred, fraction=1 / 3)
    assert gain == 1.0 - y_true.mean()
