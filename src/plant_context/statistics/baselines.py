"""Ready-made fit_predict_fn closures for run_crossfit (TDD 15 item 4).

Each factory below returns a function with the crossfit.FitPredictFn
signature: ``(train_rows, eval_rows) -> predictions``, so it can be passed
straight to ``run_crossfit``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from plant_context.statistics.crossfit import FitPredictFn
from plant_context.statistics.gblup import (
    compute_allele_frequencies,
    compute_vanraden_grm_with_frequencies,
    fit_gblup,
    pivot_genotype_marker_to_wide,
    select_gblup_lambda,
)
from plant_context.statistics.reaction_norm import (
    compute_environment_index_from_phenotype,
    fit_reaction_norm,
    predict_reaction_norm,
)


def environment_mean_predict_fn(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> np.ndarray:
    """Predict each row as its environment's train-mean phenotype.

    Falls back to the overall train mean for an environment never seen in
    train (e.g. every leave_environment_out test fold, by construction).
    """
    env_mean = train_rows.groupby("environment_id")["phenotype_value"].mean()
    overall_mean = train_rows["phenotype_value"].mean()
    return env_mean.reindex(eval_rows["environment_id"]).fillna(overall_mean).to_numpy()


def make_gblup_predict_fn(
    genotype_marker_df: pd.DataFrame,
    max_dosage: float,
    lambda_grid: Optional[list] = None,
    n_folds: int = 5,
    seed: int = 1234,
) -> FitPredictFn:
    """Build a GBLUP fit_predict_fn.

    The wide dosage matrix (every genotype's raw markers) is pivoted once
    up front -- that alone carries no train/test distinction. But the
    allele-frequency centering used to build the GRM *is* a train-only
    statistic (see gblup.py's module docstring), so it -- and therefore the
    GRM itself -- is recomputed fresh inside the per-fold closure from that
    fold's train genotypes only, before being applied to the full dosage
    matrix (which is what lets held-out genotypes still get a relatedness-
    based prediction).
    """
    wide_full = pivot_genotype_marker_to_wide(genotype_marker_df)
    grid = lambda_grid or [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    def _fit_predict(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> np.ndarray:
        train_genotype_ids = set(train_rows["genotype_id"])
        wide_train = wide_full.loc[wide_full.index.isin(train_genotype_ids)]
        allele_freq = compute_allele_frequencies(wide_train, max_dosage)
        grm = compute_vanraden_grm_with_frequencies(wide_full, allele_freq, max_dosage)

        y_train = train_rows.groupby("genotype_id")["phenotype_value"].mean()
        y_train = y_train.reindex(grm.index)
        lam = select_gblup_lambda(grm, y_train, grid, n_folds=n_folds, seed=seed)
        preds = fit_gblup(grm, y_train, lam)
        overall_mean = train_rows["phenotype_value"].mean()
        return preds.reindex(eval_rows["genotype_id"]).fillna(overall_mean).to_numpy()

    return _fit_predict


def reaction_norm_predict_fn(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> np.ndarray:
    """Fit/predict Finlay-Wilkinson reaction norm within one fold.

    Uses the phenotype-mean environment index -- see
    ``compute_environment_index_from_phenotype``'s docstring for when this
    is and is not leakage-safe. Only appropriate for leave_genotype_out or
    random folds, not leave_environment_out/leave_ge_out.
    """
    env_index = compute_environment_index_from_phenotype(train_rows)
    params = fit_reaction_norm(train_rows, env_index)
    overall_mean = train_rows["phenotype_value"].mean()
    preds = predict_reaction_norm(
        params, env_index, eval_rows["genotype_id"], eval_rows["environment_id"]
    )
    return np.where(np.isnan(preds), overall_mean, preds)
