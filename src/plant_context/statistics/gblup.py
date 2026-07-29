"""GBLUP statistical baseline (TDD Section 6.3, 15 item 4).

GBLUP predicts genotype breeding values from a genomic relationship matrix
(GRM, VanRaden method 1) and known phenotypes for a subset of genotypes.
Genotypes with no phenotype are predicted purely from their GRM relatedness
to genotypes that do have one -- this is what makes GBLUP work correctly
under leave-genotype-out: no special-casing is needed, the same formula
predicts held-in and held-out genotypes alike.

Lambda (the ridge/shrinkage penalty, equivalent to sigma_e^2/sigma_g^2) is
chosen by cross-validation restricted to whatever training genotypes are
passed in -- never anything from an outer test fold (TDD Section 8.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pivot_genotype_marker_to_wide(genotype_marker_df: pd.DataFrame) -> pd.DataFrame:
    """genotype_marker (long, TDD 4.2) -> genotype x marker allele-dosage matrix.

    Any missing (genotype, marker) cell -- there should be none for a dense
    array like G2F's, but sparser marker sets may have some -- is filled
    with that marker's column mean. This is a simple, explicit imputation
    choice, not a silent one: it is worth revisiting before treating any
    such matrix as fully "clean".
    """
    wide = genotype_marker_df.pivot_table(
        index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first"
    )
    if wide.isna().any().any():
        wide = wide.fillna(wide.mean(axis=0))
    return wide


def compute_vanraden_grm(genotype_wide: pd.DataFrame, max_dosage: float) -> pd.DataFrame:
    """VanRaden method-1 genomic relationship matrix.

    ``max_dosage`` is the ploidy scale the allele_dosage column is on (2 for
    a raw diploid 0/1/2 count, 1 if it has already been divided by ploidy).
    This is deliberately a required argument rather than a silently assumed
    default: get it wrong and every relatedness value is wrong.
    """
    M = genotype_wide.to_numpy(dtype=float)
    p = M.mean(axis=0) / max_dosage
    Z = M - max_dosage * p
    denom = max_dosage * float(np.sum(p * (1 - p)))
    if denom <= 0:
        raise ValueError(
            "VanRaden GRM denominator is non-positive -- markers may be "
            "non-segregating (all genotypes identical at every marker)"
        )
    G = (Z @ Z.T) / denom
    return pd.DataFrame(G, index=genotype_wide.index, columns=genotype_wide.index)


def fit_gblup(grm: pd.DataFrame, y_train: pd.Series, lambda_: float) -> pd.Series:
    """BLUP prediction for every genotype in ``grm``.

    ``y_train`` may be a strict subset of ``grm``'s index (its non-null
    entries are the training genotypes); genotypes absent from it -- e.g. a
    leave-genotype-out test fold -- are predicted purely from their GRM
    relatedness to the training genotypes, with no special-casing needed.
    """
    train_ids = [g for g in y_train.dropna().index if g in grm.index]
    if not train_ids:
        raise ValueError("fit_gblup received no training genotypes present in the GRM")

    y = y_train.loc[train_ids].to_numpy(dtype=float)
    ybar = float(y.mean())
    n = len(train_ids)

    G_tt = grm.loc[train_ids, train_ids].to_numpy()
    coef = np.linalg.solve(G_tt + lambda_ * np.eye(n), y - ybar)

    G_all_t = grm.loc[:, train_ids].to_numpy()
    preds = G_all_t @ coef + ybar
    return pd.Series(preds, index=grm.index)


def select_gblup_lambda(
    grm: pd.DataFrame,
    y_train: pd.Series,
    lambda_grid: list[float],
    n_folds: int = 5,
    seed: int = 1234,
) -> float:
    """Pick lambda by k-fold CV restricted to ``y_train``'s own genotypes.

    Never touches anything outside ``y_train``: the GRM used for every
    inner fold is restricted to the training genotype set before any
    fitting happens, so this cannot leak an outer test fold's genotypes
    even by accident.
    """
    train_ids = np.array(sorted(y_train.dropna().index))
    if len(train_ids) < n_folds:
        raise ValueError(
            f"need at least n_folds={n_folds} training genotypes, got {len(train_ids)}"
        )

    rng = np.random.default_rng(seed)
    shuffled = train_ids.copy()
    rng.shuffle(shuffled)
    fold_of = {g: i % n_folds for i, g in enumerate(shuffled)}

    inner_grm = grm.loc[train_ids, train_ids]

    best_lambda, best_rmse = lambda_grid[0], float("inf")
    for lam in lambda_grid:
        squared_errors = []
        for fold in range(n_folds):
            inner_test_ids = [g for g in train_ids if fold_of[g] == fold]
            inner_train_ids = [g for g in train_ids if fold_of[g] != fold]
            y_inner_train = y_train.loc[inner_train_ids].reindex(inner_grm.index)
            preds = fit_gblup(inner_grm, y_inner_train, lam)
            truth = y_train.loc[inner_test_ids].to_numpy()
            errors = truth - preds.loc[inner_test_ids].to_numpy()
            squared_errors.extend((errors**2).tolist())
        candidate_rmse = float(np.sqrt(np.mean(squared_errors)))
        if candidate_rmse < best_rmse:
            best_rmse, best_lambda = candidate_rmse, lam
    return best_lambda
