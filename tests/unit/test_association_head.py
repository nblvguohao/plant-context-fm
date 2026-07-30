"""Unit tests for the Association Head (TDD Section 7.2).

Covers: structure-preserving permutation, L1-regularized effect estimation,
genomic inflation, FDR adjustment, and the association report assembly.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from plant_context.models.association_head import (
    SparseAssociationHead,
    association_report,
    benjamini_hochberg,
    compute_genomic_inflation,
    fit_sparse_association_head,
    permute_within_strata,
    permutation_association_pvalues,
)


def test_permute_within_strata_preserves_stratum_sums():
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    strata = np.array(["A", "A", "A", "B", "B"])
    perm = permute_within_strata(y, strata, rng=np.random.default_rng(42))

    assert np.isclose(perm[strata == "A"].sum(), y[strata == "A"].sum())
    assert np.isclose(perm[strata == "B"].sum(), y[strata == "B"].sum())


def test_permute_within_strata_breaks_cross_stratum_ordering():
    rng = np.random.default_rng(1)
    y = np.arange(100, dtype=float)
    strata = np.array(["A"] * 50 + ["B"] * 50)
    perm = permute_within_strata(y, strata, rng=rng)

    # The permutation should change the values at some positions.
    assert not np.array_equal(perm, y)
    # But stratum-level sums are preserved.
    assert np.isclose(perm[:50].sum(), y[:50].sum())
    assert np.isclose(perm[50:].sum(), y[50:].sum())


def test_sparse_association_head_forward_shape():
    model = SparseAssociationHead(n_features=5, l1_penalty=0.1)
    x = torch.randn(10, 5)
    out = model(x)
    assert out.shape == (10,)


def test_l1_penalty_sparsifies_weights():
    rng = np.random.default_rng(7)
    n_samples, n_features = 200, 10
    # Only the first two features are causal.
    true_weights = np.array([2.0, -1.5] + [0.0] * (n_features - 2))
    x = rng.normal(size=(n_samples, n_features))
    y = x @ true_weights + rng.normal(scale=0.5, size=n_samples)

    dense = fit_sparse_association_head(x, y, l1_penalty=0.0, epochs=300, lr=0.05, seed=7)
    sparse = fit_sparse_association_head(x, y, l1_penalty=0.5, epochs=300, lr=0.05, seed=7)

    n_nonzero_dense = np.sum(np.abs(dense.effects()) > 0.1)
    n_nonzero_sparse = np.sum(np.abs(sparse.effects()) > 0.1)
    assert n_nonzero_sparse <= n_nonzero_dense


def test_genomic_inflation_near_one_for_uniform_pvalues():
    rng = np.random.default_rng(11)
    pvals = rng.uniform(0.001, 0.999, size=1000)
    lam = compute_genomic_inflation(pvals)
    assert 0.8 < lam < 1.2


def test_genomic_inflation_high_for_inflated_pvalues():
    # Many small p-values produce a lambda well above 1.
    pvals = np.concatenate([np.full(500, 1e-10), np.random.default_rng(13).uniform(0.1, 0.9, size=500)])
    lam = compute_genomic_inflation(pvals)
    assert lam > 2.0


def test_benjamini_hochberg_monotone_and_bounded():
    rng = np.random.default_rng(17)
    pvals = rng.uniform(0.001, 0.5, size=50)
    qvals = benjamini_hochberg(pvals)

    assert qvals.shape == pvals.shape
    assert np.all(qvals >= pvals)
    assert np.all(qvals <= 1.0)
    # Sorted q-values should be non-decreasing when p-values are sorted.
    assert np.all(np.diff(np.sort(qvals)) >= -1e-12)


def test_benjamini_hochberg_preserves_nan():
    pvals = np.array([0.01, 0.05, np.nan, 0.10])
    qvals = benjamini_hochberg(pvals)
    assert np.isnan(qvals[2])


def test_permutation_pvalues_uniform_under_null():
    rng = np.random.default_rng(23)
    n_samples, n_features = 80, 5
    x = rng.normal(size=(n_samples, n_features))
    # y is independent of x -> no true associations
    y = rng.normal(size=n_samples)
    strata = np.array(["E1"] * 40 + ["E2"] * 40)

    pvals = permutation_association_pvalues(
        x, y, strata=strata, n_permutations=50, l1_penalty=0.1, seed=23
    )
    assert pvals.shape == (n_features,)
    # Under the null, p-values should not be pathologically small.
    assert np.all(pvals > 0.01)


def test_permutation_pvalues_detects_signal():
    rng = np.random.default_rng(29)
    n_samples = 100
    x = rng.normal(size=(n_samples, 1))
    y = 3.0 * x.ravel() + rng.normal(scale=0.5, size=n_samples)

    pvals = permutation_association_pvalues(
        x, y, strata=None, n_permutations=50, l1_penalty=0.01, seed=29
    )
    assert pvals[0] < 0.05


def test_association_report_assembles_expected_columns():
    features = ["block_1", "block_2", "block_3"]
    effects = np.array([0.5, -0.2, 0.05])
    pvals = np.array([0.01, 0.20, 0.80])

    report = association_report(features, effects, pvals, alpha=0.05)
    assert list(report.columns) == ["feature_name", "effect", "p_value", "q_value", "reject_bh"]
    assert report["reject_bh"].tolist() == [True, False, False]
    assert np.all(report["q_value"] >= report["p_value"])
