import numpy as np
import pandas as pd
import pytest

from plant_context.statistics.gblup import (
    compute_vanraden_grm,
    fit_gblup,
    pivot_genotype_marker_to_wide,
    select_gblup_lambda,
)


def test_pivot_genotype_marker_to_wide_shapes_correctly():
    long_df = pd.DataFrame(
        {
            "genotype_id": ["g1", "g1", "g2", "g2"],
            "marker_id": ["m1", "m2", "m1", "m2"],
            "allele_dosage": [0.0, 1.0, 2.0, 0.5],
        }
    )
    wide = pivot_genotype_marker_to_wide(long_df)
    assert wide.shape == (2, 2)
    assert wide.loc["g1", "m1"] == 0.0
    assert wide.loc["g2", "m2"] == 0.5


def test_pivot_genotype_marker_to_wide_imputes_missing_with_column_mean():
    long_df = pd.DataFrame(
        {
            "genotype_id": ["g1", "g2", "g2"],
            "marker_id": ["m1", "m1", "m2"],  # g1 has no m2 observation
            "allele_dosage": [0.0, 2.0, 1.0],
        }
    )
    wide = pivot_genotype_marker_to_wide(long_df)
    assert wide.loc["g1", "m2"] == pytest.approx(1.0)  # column mean of m2 (only g2's value)


def _hand_computed_grm_fixture():
    # 3 genotypes x 2 markers, diploid dosage in {0,1,2}.
    wide = pd.DataFrame(
        {"m1": [0.0, 1.0, 2.0], "m2": [2.0, 1.0, 0.0]},
        index=["g1", "g2", "g3"],
    )
    return wide


def test_compute_vanraden_grm_matches_hand_calculation():
    wide = _hand_computed_grm_fixture()
    grm = compute_vanraden_grm(wide, max_dosage=2.0)

    # By hand: p_m1 = mean([0,1,2])/2 = 0.5, p_m2 = mean([2,1,0])/2 = 0.5
    # Z = M - 2*p = M - 1 (since both p=0.5) ->
    #   Z = [[-1, 1], [0, 0], [1, -1]]
    # denom = 2 * (0.5*0.5 + 0.5*0.5) = 2 * 0.5 = 1.0
    # G = Z Z^T / 1.0
    expected = np.array(
        [
            [2.0, 0.0, -2.0],
            [0.0, 0.0, 0.0],
            [-2.0, 0.0, 2.0],
        ]
    )
    np.testing.assert_allclose(grm.to_numpy(), expected, atol=1e-8)


def test_compute_vanraden_grm_is_symmetric():
    wide = _hand_computed_grm_fixture()
    grm = compute_vanraden_grm(wide, max_dosage=2.0)
    np.testing.assert_allclose(grm.to_numpy(), grm.to_numpy().T)


def test_compute_vanraden_grm_rejects_non_segregating_markers():
    # p = mean(dosage)/max_dosage = 0/2 = 0 for every genotype -> every
    # marker is fixed (non-segregating), so the denominator is exactly 0.
    wide = pd.DataFrame({"m1": [0.0, 0.0, 0.0]}, index=["g1", "g2", "g3"])
    with pytest.raises(ValueError, match="non-positive"):
        compute_vanraden_grm(wide, max_dosage=2.0)


def _relatedness_grm():
    # Two identical pairs: {g1, g2} and {g3, g4}; the pairs are unrelated
    # to each other. g1 and g3 will be the training genotypes below, so
    # ybar is non-trivial (mean of two distinct values) and g2/g4 have a
    # real, non-degenerate relative to be predicted from.
    index = ["g1", "g2", "g3", "g4"]
    values = np.array(
        [
            [2.0, 2.0, -2.0, -2.0],
            [2.0, 2.0, -2.0, -2.0],
            [-2.0, -2.0, 2.0, 2.0],
            [-2.0, -2.0, 2.0, 2.0],
        ]
    )
    return pd.DataFrame(values, index=index, columns=index)


def test_fit_gblup_predicts_related_genotype_close_to_its_relative():
    grm = _relatedness_grm()
    # g1 and g3 are trained with different phenotypes; g2 (identical to
    # g1) and g4 (identical to g3) are untrained.
    y_train = pd.Series({"g1": 10.0, "g2": np.nan, "g3": 20.0, "g4": np.nan})
    preds = fit_gblup(grm, y_train, lambda_=0.5)

    # g2, being identical to g1 (not g3), should be predicted much closer
    # to g1's phenotype than to g3's.
    assert abs(preds["g2"] - preds["g1"]) < abs(preds["g2"] - preds["g3"])


def test_fit_gblup_shrinks_to_mean_as_lambda_grows():
    grm = _relatedness_grm()
    y_train = pd.Series({"g1": 10.0, "g2": np.nan, "g3": 20.0, "g4": np.nan})
    preds_small_lambda = fit_gblup(grm, y_train, lambda_=0.01)
    preds_huge_lambda = fit_gblup(grm, y_train, lambda_=1e6)

    ybar = y_train.dropna().mean()
    # As lambda -> infinity, every prediction should collapse to the mean.
    assert preds_huge_lambda.sub(ybar).abs().max() < preds_small_lambda.sub(ybar).abs().max()


def test_fit_gblup_raises_with_no_training_genotypes():
    grm = _relatedness_grm()
    y_train = pd.Series({"g1": np.nan, "g2": np.nan, "g3": np.nan})
    with pytest.raises(ValueError):
        fit_gblup(grm, y_train, lambda_=1.0)


def test_select_gblup_lambda_returns_value_from_grid():
    grm = _relatedness_grm()
    y_train = pd.Series({"g1": 10.0, "g2": 20.0, "g3": 15.0})
    grid = [0.1, 1.0, 10.0]
    chosen = select_gblup_lambda(grm, y_train, grid, n_folds=3)
    assert chosen in grid


def test_select_gblup_lambda_rejects_more_folds_than_genotypes():
    grm = _relatedness_grm()
    y_train = pd.Series({"g1": 10.0, "g2": 20.0, "g3": 15.0})
    grid = [0.1, 1.0, 10.0]
    with pytest.raises(ValueError):
        # Only 3 genotypes have a training phenotype here; asking for more
        # folds than that should fail instead of silently producing empty
        # folds.
        select_gblup_lambda(grm, y_train, grid, n_folds=5)
