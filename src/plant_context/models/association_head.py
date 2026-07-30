"""Association Head: sparse, regularized SNP/block-level effect estimation
with structure-preserving permutation nulls (TDD Section 7.2 / Paper 4).

This module is intentionally simple — a regularized linear head over
(genotype-wide) token features — so that statistical calibration (FDR,
genomic inflation, permutation nulls) can be tested independently of
neural-network training dynamics. More elaborate nonlinear association heads
should inherit from or wrap this and keep the same calibration API.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn


def permute_within_strata(
    phenotype: np.ndarray,
    strata: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Shuffle ``phenotype`` independently within each stratum.

    This preserves the marginal distribution of ``phenotype`` within each
    stratum and the stratum frequencies, while breaking any genotype-
    phenotype association that crosses strata. The two canonical choices
    for GW×E permutation nulls are:

    - ``strata = environment_id``: preserves environment main effects and
      the genotype-LD/genetic-relationship structure (genotypes keep their
      markers); only the phenotype-to-genotype assignment is randomized.
    - ``strata = genotype_id``: preserves genotype main effects and LD
      structure; randomizes environment-to-phenotype assignment.

    Returns a copy; the input is not modified.
    """
    if rng is None:
        rng = np.random.default_rng()
    permuted = np.asarray(phenotype, dtype=float).copy()
    strata = np.asarray(strata)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if len(idx) > 1:
            permuted[idx] = rng.permutation(permuted[idx])
    return permuted


class SparseAssociationHead(nn.Module):
    """L1-regularized linear association head over genotype token features.

    Parameters
    ----------
    n_features :
        Number of input features (e.g. LD-block mean dosages).
    l1_penalty :
        Coefficient on the L1 norm of the feature weights. Set to 0.0 for
        ordinary least squares.
    """

    def __init__(self, n_features: int, l1_penalty: float = 0.0):
        super().__init__()
        self.l1_penalty = l1_penalty
        self.bias = nn.Parameter(torch.zeros(1))
        self.weight = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape ``[batch, n_features]``; returns ``[batch]``."""
        return x @ self.weight + self.bias

    def l1_loss(self) -> torch.Tensor:
        return self.l1_penalty * self.weight.abs().sum()

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> "SparseAssociationHead":
        """Fit with Adam. Returns self for chaining."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.forward(x), y) + self.l1_loss()
            loss.backward()
            optimizer.step()
        return self

    def effects(self) -> np.ndarray:
        """Return fitted feature weights as a numpy array."""
        return self.weight.detach().cpu().numpy()


def fit_sparse_association_head(
    x: np.ndarray,
    y: np.ndarray,
    l1_penalty: float = 0.0,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 1234,
) -> SparseAssociationHead:
    """Convenience wrapper that builds and fits a ``SparseAssociationHead``."""
    torch.manual_seed(seed)
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model = SparseAssociationHead(n_features=x.shape[1], l1_penalty=l1_penalty)
    return model.fit(x_t, y_t, epochs=epochs, lr=lr)


def compute_genomic_inflation(p_values: np.ndarray) -> float:
    """Genomic inflation factor lambda_GC.

    lambda_GC = median(chi2) / 0.455, where chi2 is the 1-df chi-square
    statistic from two-sided p-values. Values near 1.0 suggest no systematic
    inflation; values substantially above 1.0 suggest population stratification,
cryptic relatedness, or analytical p-values that are too liberal.

    P-values equal to 0 or 1 are clipped to (1e-300, 1-1e-300) so the chi-square
    transform remains finite.
    """
    p = np.asarray(p_values, dtype=float)
    p = np.clip(p, 1e-300, 1.0 - 1e-300)
    chi2 = stats_chi2_from_pvalue(p)
    return float(np.median(chi2) / 0.455)


def stats_chi2_from_pvalue(p_values: np.ndarray) -> np.ndarray:
    """Two-sided p-value -> 1-df chi-square statistic."""
    from scipy import stats

    p = np.asarray(p_values, dtype=float)
    p = np.clip(p, 1e-300, 1.0 - 1e-300)
    z = stats.norm.ppf(1 - p / 2)
    return z**2


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values.

    Returns an array the same shape as ``p_values`` with adjusted p-values
    (not just the reject/accept mask). NaN p-values remain NaN.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p.copy()

    # Flatten, sort, compute BH, then restore order
    flat = p.ravel()
    sorted_idx = np.argsort(flat)
    sorted_p = flat[sorted_idx]
    ranks = np.arange(1, n + 1)
    raw = sorted_p * n / ranks
    # Adjusted p-value at rank i is the minimum raw value at ranks >= i.
    bh_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    bh_sorted = np.clip(bh_sorted, 0.0, 1.0)

    result = np.empty_like(flat)
    result[sorted_idx] = bh_sorted
    result = result.reshape(p.shape)
    result[np.isnan(p)] = np.nan
    return result


def permutation_association_pvalues(
    x: np.ndarray,
    y: np.ndarray,
    strata: Optional[np.ndarray] = None,
    n_permutations: int = 100,
    l1_penalty: float = 0.0,
    two_sided: bool = True,
    seed: int = 1234,
) -> np.ndarray:
    """Feature-level permutation p-values for sparse association effects.

    The observed effect vector is compared against a null distribution
    obtained by refitting the association head on permuted phenotypes.
    Permutations are stratified by ``strata`` when provided, preserving
    environment or genotype structure as desired.

    Parameters
    ----------
    x :
        ``[n_samples, n_features]`` genotype feature matrix.
    y :
        ``[n_samples]`` phenotype vector.
    strata :
        Optional ``[n_samples]`` stratum labels for structured permutation.
    n_permutations :
        Number of permutations.
    l1_penalty :
        L1 penalty passed to ``fit_sparse_association_head``.
    two_sided :
        If True, p-value is the proportion of null effects whose absolute
        value exceeds the observed absolute effect. If False, the signed
        tail is used.
    seed :
        Random seed for reproducibility.

    Returns
    -------
    ``[n_features]`` array of permutation p-values. A feature with effect
    exactly zero gets p-value 1.0.
    """
    rng = np.random.default_rng(seed)
    observed = fit_sparse_association_head(
        x, y, l1_penalty=l1_penalty, seed=seed
    ).effects()

    null_effects = np.zeros((n_permutations, x.shape[1]))
    for i in range(n_permutations):
        y_perm = (
            permute_within_strata(y, strata, rng=rng)
            if strata is not None
            else rng.permutation(y)
        )
        null_effects[i] = fit_sparse_association_head(
            x, y_perm, l1_penalty=l1_penalty, seed=seed + i + 1
        ).effects()

    if two_sided:
        pvals = np.mean(np.abs(null_effects) >= np.abs(observed), axis=0)
    else:
        pvals = np.mean(null_effects >= observed, axis=0)

    # Ensure exact zeros / no null exceedances don't produce 0 p-values
    pvals = np.clip(pvals, 1.0 / (n_permutations + 1), 1.0)
    return pvals


def association_report(
    feature_names: list[str],
    effects: np.ndarray,
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Assemble an association report with FDR adjustment and discovery flags.

    Columns: feature_name, effect, p_value, q_value, reject_bh.
    """
    q_values = benjamini_hochberg(p_values)
    return pd.DataFrame(
        {
            "feature_name": feature_names,
            "effect": effects,
            "p_value": p_values,
            "q_value": q_values,
            "reject_bh": q_values < alpha,
        }
    )
