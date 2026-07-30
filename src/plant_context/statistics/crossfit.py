"""Cross-fitting harness (TDD Section 6.3, 8.3, 15 item 4).

Ties a phenotype table and a split_table together and calls a model's
fit/predict function once per (outer_split_type, outer_fold), passing it
only that fold's train rows -- it never even receives the phenotype values
of the rows it is about to predict, so a model cannot accidentally read
what it is being evaluated on.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

FitPredictFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def run_crossfit(
    phenotype_df: pd.DataFrame,
    split_df: pd.DataFrame,
    fit_predict_fn: FitPredictFn,
    roles_to_predict: Iterable[str] = ("test",),
) -> pd.DataFrame:
    """Run ``fit_predict_fn`` once per outer fold and collect predictions.

    ``fit_predict_fn(train_rows, eval_rows)`` receives the fold's train
    rows (full phenotype_df columns, including phenotype_value) and the
    rows to predict with ``phenotype_value`` already dropped -- it must
    return one prediction per row of ``eval_rows``, in that row order.

    Returns a DataFrame with sample_id, outer_split_type, outer_fold, role,
    y_true, y_pred: one row per (fold, evaluated sample).
    """
    roles_to_predict = set(roles_to_predict)
    merged = split_df.merge(phenotype_df, on="sample_id", validate="many_to_one")

    output_frames = []
    for (split_type, fold), fold_df in merged.groupby(["outer_split_type", "outer_fold"]):
        train_rows = fold_df[fold_df["role"] == "train"]
        eval_rows = fold_df[fold_df["role"].isin(roles_to_predict)]
        if eval_rows.empty:
            continue

        eval_inputs = eval_rows.drop(columns=["phenotype_value"])
        predictions = np.asarray(fit_predict_fn(train_rows, eval_inputs))
        if len(predictions) != len(eval_rows):
            raise ValueError(
                f"fit_predict_fn returned {len(predictions)} predictions for "
                f"{len(eval_rows)} eval rows in fold ({split_type}, {fold})"
            )

        output_frames.append(
            pd.DataFrame(
                {
                    "sample_id": eval_rows["sample_id"].to_numpy(),
                    "outer_split_type": split_type,
                    "outer_fold": fold,
                    "role": eval_rows["role"].to_numpy(),
                    "y_true": eval_rows["phenotype_value"].to_numpy(),
                    "y_pred": predictions,
                }
            )
        )

    if not output_frames:
        raise ValueError("no fold produced any rows to predict -- check roles_to_predict")
    return pd.concat(output_frames, ignore_index=True)
