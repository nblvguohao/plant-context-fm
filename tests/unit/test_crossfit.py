"""Unit tests for the cross-fitting harness (TDD 8.3, 15 item 4).

The central property under test is that fit_predict_fn never sees a test
row's phenotype_value and never sees a test-fold row among its "train_rows"
argument -- these are exactly the leakage channels TDD 8.3 is worried
about, at the harness level rather than at any one model's level.
"""

import numpy as np
import pandas as pd
import pytest

from plant_context.evaluation.splits import make_leave_genotype_split
from plant_context.statistics.crossfit import run_crossfit


@pytest.fixture
def phenotype_and_split(synthetic_gxe_df):
    phenotype_df = synthetic_gxe_df.assign(
        sample_id=synthetic_gxe_df["sample_id"],
        phenotype_value=np.arange(len(synthetic_gxe_df), dtype=float),
    )
    split_df = make_leave_genotype_split(phenotype_df, n_folds=3, seed=1234, split_version="test")
    return phenotype_df, split_df


def test_run_crossfit_never_exposes_phenotype_value_of_eval_rows(phenotype_and_split):
    phenotype_df, split_df = phenotype_and_split

    def spy_fit_predict(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> np.ndarray:
        assert "phenotype_value" not in eval_rows.columns
        return np.zeros(len(eval_rows))

    run_crossfit(phenotype_df, split_df, spy_fit_predict)


def test_run_crossfit_train_rows_are_only_role_train(phenotype_and_split):
    phenotype_df, split_df = phenotype_and_split
    merged = split_df.merge(phenotype_df, on="sample_id")

    def spy_fit_predict(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> np.ndarray:
        assert (train_rows["role"] == "train").all()
        return np.zeros(len(eval_rows))

    run_crossfit(phenotype_df, split_df, spy_fit_predict)


def test_run_crossfit_output_has_one_row_per_evaluated_sample(phenotype_and_split):
    phenotype_df, split_df = phenotype_and_split

    def constant_fit_predict(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(eval_rows), train_rows["phenotype_value"].mean())

    result = run_crossfit(phenotype_df, split_df, constant_fit_predict)
    expected_n = (split_df["role"] == "test").sum()
    assert len(result) == expected_n
    assert set(result.columns) == {
        "sample_id", "outer_split_type", "outer_fold", "role", "y_true", "y_pred",
    }


def test_run_crossfit_raises_on_wrong_length_predictions(phenotype_and_split):
    phenotype_df, split_df = phenotype_and_split

    def bad_fit_predict(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(eval_rows) + 1)

    with pytest.raises(ValueError, match="predictions"):
        run_crossfit(phenotype_df, split_df, bad_fit_predict)


def test_run_crossfit_raises_when_no_rows_match_requested_roles(phenotype_and_split):
    phenotype_df, split_df = phenotype_and_split

    def unused_fit_predict(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(eval_rows))

    with pytest.raises(ValueError, match="no fold"):
        run_crossfit(phenotype_df, split_df, unused_fit_predict, roles_to_predict=("nonexistent",))
