"""Finlay-Wilkinson reaction norm baseline (TDD Section 6.3, 15 item 4).

y_ge = a_i + b_i * h_e + eps: each genotype gets its own intercept and its
own sensitivity slope against an environment index h_e.

This module does not decide how h_e is computed -- that choice interacts
with which split is being evaluated (see
``compute_environment_index_from_phenotype`` for the important caveat about
which splits it is and is not safe for) -- it only fits/predicts given
whatever index the caller supplies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_environment_index_from_phenotype(train_df: pd.DataFrame) -> pd.Series:
    """Classical Finlay-Wilkinson environment index: mean phenotype per
    environment, computed only from the rows passed in.

    The caller must restrict ``train_df`` to train-role rows of a single
    outer fold. This index is only safe to use for splits where the
    evaluated environments already have *other* genotypes' train-phase
    phenotypes in them -- i.e. leave_genotype_out or a random split. Do NOT
    use it for leave_environment_out or leave_ge_out folds, where the
    held-out environment's rows may be entirely absent from train: computing
    its index from its own (excluded) phenotypes there is not possible
    without leaking, and computing it from nothing is undefined. Use a
    weather/phenology-derived covariate index instead in that case (not
    implemented here).
    """
    return train_df.groupby("environment_id")["phenotype_value"].mean()


def fit_reaction_norm(train_df: pd.DataFrame, environment_index: pd.Series) -> pd.DataFrame:
    """Fit y = a_i + b_i * h_e per genotype via OLS on its train rows only.

    Returns a DataFrame indexed by genotype_id with columns ``a``
    (intercept), ``b`` (slope), ``n_obs``. A genotype observed in fewer
    than two distinct environment_index values has an unidentifiable slope:
    it gets ``b = NaN`` and ``a`` = its mean phenotype, rather than a
    fabricated slope.
    """
    df = train_df.merge(
        environment_index.rename("environment_index"),
        left_on="environment_id",
        right_index=True,
        how="inner",
    )
    records = []
    for genotype_id, group in df.groupby("genotype_id"):
        n_obs = len(group)
        if group["environment_index"].nunique() < 2:
            records.append(
                {
                    "genotype_id": genotype_id,
                    "a": float(group["phenotype_value"].mean()),
                    "b": float("nan"),
                    "n_obs": n_obs,
                }
            )
            continue
        slope, intercept = np.polyfit(group["environment_index"], group["phenotype_value"], deg=1)
        records.append(
            {"genotype_id": genotype_id, "a": float(intercept), "b": float(slope), "n_obs": n_obs}
        )
    return pd.DataFrame.from_records(records).set_index("genotype_id")


def predict_reaction_norm(
    params: pd.DataFrame,
    environment_index: pd.Series,
    genotype_ids: pd.Series,
    environment_ids: pd.Series,
) -> np.ndarray:
    """Predict y_hat for paired (genotype_id, environment_id) rows.

    A genotype absent from ``params`` (never seen in train) or an
    environment absent from ``environment_index`` (its index is unknown)
    both produce NaN for that row rather than a guess.
    """
    a = params["a"].reindex(genotype_ids).to_numpy()
    b = params["b"].reindex(genotype_ids).to_numpy()
    b = np.where(np.isnan(b), 0.0, b)  # unidentifiable slope -> intercept-only fallback
    h = environment_index.reindex(environment_ids).to_numpy()
    return a + b * h
