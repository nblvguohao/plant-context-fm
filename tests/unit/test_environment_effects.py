"""Unit tests for association/environment_effects.py (TDD 15 item 10).

Covers the two-step reaction-norm-coefficient GWAS pipeline:
build_block_dosage_matrix → scan_slope_association → adjust_multiple_testing.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from plant_context.association.environment_effects import (
    adjust_multiple_testing,
    build_block_dosage_matrix,
    format_manhattan,
    scan_slope_association,
)

# ---------------------------------------------------------------------------
#  build_block_dosage_matrix
# ---------------------------------------------------------------------------


def test_build_block_dosage_matrix_basic():
    """3 genotypes × 4 markers across 2 blocks → 3×2 matrix with correct means."""
    genotype_marker_df = pd.DataFrame(
        {
            "genotype_id": ["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2", "g3", "g3", "g3", "g3"],
            "marker_id": ["m1", "m2", "m3", "m4"] * 3,
            "allele_dosage": [0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        }
    )
    block_assignment = pd.DataFrame(
        {
            "marker_id": ["m1", "m2", "m3", "m4"],
            "ld_block_id": ["block_A", "block_A", "block_B", "block_B"],
        }
    )

    mat = build_block_dosage_matrix(genotype_marker_df, block_assignment)
    assert mat.shape == (3, 2)
    assert set(mat.columns) == {"block_A", "block_B"}
    assert list(mat.index) == ["g1", "g2", "g3"]

    # g1: block_A = mean(0, 1) = 0.5, block_B = mean(0, 2) = 1.0
    assert np.isclose(mat.loc["g1", "block_A"], 0.5)
    assert np.isclose(mat.loc["g1", "block_B"], 1.0)
    # g2: both blocks = mean(1, 1) = 1.0
    assert np.isclose(mat.loc["g2", "block_A"], 1.0)
    assert np.isclose(mat.loc["g2", "block_B"], 1.0)
    # g3: block_A = mean(2, 1) = 1.5, block_B = mean(2, 0) = 1.0
    assert np.isclose(mat.loc["g3", "block_A"], 1.5)
    assert np.isclose(mat.loc["g3", "block_B"], 1.0)


def test_build_block_dosage_matrix_genotype_without_block_gets_nan():
    """Genotype present in marker data but not in a particular block gets NaN."""
    genotype_marker_df = pd.DataFrame(
        {
            "genotype_id": ["g1", "g1", "g2", "g2"],
            "marker_id": ["m1", "m2", "m1", "m3"],
            "allele_dosage": [0.0, 1.0, 1.0, 2.0],
        }
    )
    # m3 belongs to block_B; g1 has no m3
    block_assignment = pd.DataFrame(
        {
            "marker_id": ["m1", "m2", "m3"],
            "ld_block_id": ["block_A", "block_A", "block_B"],
        }
    )

    mat = build_block_dosage_matrix(genotype_marker_df, block_assignment)
    assert np.isnan(mat.loc["g1", "block_B"])
    assert np.isclose(mat.loc["g2", "block_B"], 2.0)


def test_build_block_dosage_matrix_columns_are_sorted():
    """Column order must be deterministic (alphabetical by block_id)."""
    genotype_marker_df = pd.DataFrame(
        {
            "genotype_id": ["g1", "g1"],
            "marker_id": ["m1", "m2"],
            "allele_dosage": [0.5, 1.5],
        }
    )
    block_assignment = pd.DataFrame(
        {
            "marker_id": ["m1", "m2"],
            "ld_block_id": ["Z_block", "A_block"],
        }
    )
    mat = build_block_dosage_matrix(genotype_marker_df, block_assignment)
    assert list(mat.columns) == ["A_block", "Z_block"]


# ---------------------------------------------------------------------------
#  scan_slope_association
# ---------------------------------------------------------------------------


def _make_synthetic_block_dosage(
    n_genotypes: int, n_blocks: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Random (genotype × block) dosage matrix."""
    data = rng.uniform(0.0, 2.0, size=(n_genotypes, n_blocks))
    genotypes = [f"g{i}" for i in range(n_genotypes)]
    blocks = [f"block_{j}" for j in range(n_blocks)]
    return pd.DataFrame(data, index=genotypes, columns=blocks)


def test_scan_slope_association_known_signal():
    """A block with true effect is detected (p < alpha); a null block is not."""
    rng = np.random.default_rng(1234)
    n_geno = 60
    dosage = _make_synthetic_block_dosage(n_geno, 5, rng)

    # Block 0 has a real effect: slope = 2 + 1.5 * dosage + noise
    true_beta = 1.5
    noise = rng.normal(0, 0.5, size=n_geno)
    slopes = 2.0 + true_beta * dosage.iloc[:, 0].to_numpy() + noise
    slopes_series = pd.Series(slopes, index=dosage.index, name="b")

    result = scan_slope_association(dosage, slopes_series)
    # Block 0 should be significant
    assert result.loc["block_0", "p_value"] < 0.05
    assert abs(result.loc["block_0", "beta"] - true_beta) < 0.5
    # Other blocks should not be significant at Bonferroni level
    for b in ["block_1", "block_2", "block_3", "block_4"]:
        assert result.loc[b, "p_value"] > 0.05 / 5, f"{b} spuriously significant"


def test_scan_slope_association_under_null():
    """Under global null (slopes independent of all blocks), p-values ~ U(0,1)."""
    rng = np.random.default_rng(5678)
    n_geno = 150
    dosage = _make_synthetic_block_dosage(n_geno, 10, rng)
    slopes = pd.Series(rng.normal(5, 1, size=n_geno), index=dosage.index)

    result = scan_slope_association(dosage, slopes)
    p_vals = result["p_value"].dropna().to_numpy()

    # KS test for uniformity
    ks_stat, ks_p = scipy_stats.kstest(p_vals, "uniform")
    assert ks_p > 0.01, f"p-values deviate from uniformity (KS stat={ks_stat:.4f}, p={ks_p:.6f})"


def test_scan_slope_association_nan_slopes_dropped():
    """Genotypes with NaN slope are silently excluded."""
    rng = np.random.default_rng(42)
    n_geno = 30
    dosage = _make_synthetic_block_dosage(n_geno, 3, rng)
    slopes = pd.Series(rng.normal(3, 1, size=n_geno), index=dosage.index)
    # Set some slopes to NaN
    slopes.iloc[:5] = np.nan

    result = scan_slope_association(dosage, slopes)
    assert result.loc["block_0", "n_genotypes"] == n_geno - 5
    assert np.isfinite(result.loc["block_0", "p_value"])


def test_scan_slope_association_blocks_with_zero_variance_return_nan():
    """A block where every genotype has identical dosage cannot be tested."""
    rng = np.random.default_rng(99)
    n_geno = 30
    dosage = _make_synthetic_block_dosage(n_geno, 3, rng)
    # Set block_1 to constant dosage
    dosage["block_1"] = 1.0
    slopes = pd.Series(rng.normal(3, 1, size=n_geno), index=dosage.index)

    result = scan_slope_association(dosage, slopes)
    assert np.isnan(result.loc["block_1", "p_value"])
    assert np.isnan(result.loc["block_1", "beta"])
    assert np.isfinite(result.loc["block_0", "p_value"])
    assert np.isfinite(result.loc["block_2", "p_value"])


def test_scan_slope_association_below_min_genotypes_returns_nan():
    """Block with fewer than min_genotypes valid observations returns NaN."""
    rng = np.random.default_rng(77)
    dosage = _make_synthetic_block_dosage(5, 2, rng)  # n_geno=5
    slopes = pd.Series(rng.normal(3, 1, size=5), index=dosage.index)

    result = scan_slope_association(dosage, slopes, min_genotypes=10)
    assert result["p_value"].isna().all()


def test_scan_slope_association_returns_expected_columns():
    """Output DataFrame has the correct schema."""
    rng = np.random.default_rng(2024)
    dosage = _make_synthetic_block_dosage(20, 3, rng)
    slopes = pd.Series(rng.normal(3, 1, size=20), index=dosage.index)

    result = scan_slope_association(dosage, slopes)
    expected = {"beta", "se", "t_stat", "p_value", "n_genotypes"}
    assert expected.issubset(set(result.columns))
    assert result.index.name == "ld_block_id"


# ---------------------------------------------------------------------------
#  adjust_multiple_testing
# ---------------------------------------------------------------------------


def test_adjust_multiple_testing_fdr_bh():
    """BH procedure: known small p-values should be called significant."""
    rng = np.random.default_rng(111)
    n_tests = 100
    # 5 true positives with small p-values, rest uniform noise
    p_vals = rng.uniform(0, 1, size=n_tests)
    p_vals[:5] = [0.0001, 0.0005, 0.001, 0.005, 0.01]

    df = pd.DataFrame({"p_value": p_vals})
    result = adjust_multiple_testing(df, alpha=0.05)

    assert "q_value" in result.columns
    assert "significant_0.05" in result.columns
    # At least the strongest signal should be called significant
    assert result["q_value"].iloc[:3].notna().all()
    assert result["significant_0.05"].iloc[0]  # p=0.0001


def test_adjust_multiple_testing_all_nan():
    """All-NaN p_values → all-NaN q_values, no crash."""
    df = pd.DataFrame({"p_value": [np.nan, np.nan]})
    result = adjust_multiple_testing(df)
    assert result["q_value"].isna().all()
    assert not result["significant_0.05"].any()


def test_adjust_multiple_testing_single_test():
    """Single test: q_value == p_value when there's only one test."""
    df = pd.DataFrame({"p_value": [0.01]})
    result = adjust_multiple_testing(df)
    assert np.isclose(result["q_value"].iloc[0], 0.01)
    assert result["significant_0.05"].iloc[0]  # 0.01 < 0.05


def test_adjust_multiple_testing_bh_monotonicity():
    """BH q-values are monotonic non-decreasing with respect to p-value order."""
    rng = np.random.default_rng(222)
    df = pd.DataFrame({"p_value": np.sort(rng.uniform(0, 1, size=50))})
    result = adjust_multiple_testing(df)
    q_vals = result["q_value"].to_numpy()
    # q-values should be non-decreasing for sorted p-values (ignoring NaN)
    finite_q = q_vals[np.isfinite(q_vals)]
    assert np.all(np.diff(finite_q) >= -1e-10)


# ---------------------------------------------------------------------------
#  format_manhattan
# ---------------------------------------------------------------------------


def test_format_manhattan_basic():
    """Manhattan formatting adds chromosome, position, and -log10(p)."""
    results_df = pd.DataFrame(
        {"p_value": [0.1, 0.01, 0.001]},
        index=pd.Index(["block_1", "block_2", "block_3"], name="ld_block_id"),
    )
    block_assignment = pd.DataFrame(
        {
            "ld_block_id": ["block_1", "block_1", "block_2", "block_3"],
            "marker_id": ["m1", "m2", "m3", "m4"],
            "chromosome": [1, 1, 1, 2],
            "position": [10, 20, 50, 100],
        }
    )

    manhattan = format_manhattan(results_df, block_assignment)
    assert list(manhattan.columns) == ["ld_block_id", "p_value", "chromosome", "position", "-log10_p"]
    # block_1 has mean position 15, block_2 has 50, block_3 has 100
    assert np.isclose(
        manhattan.loc[manhattan["ld_block_id"] == "block_1", "position"].iloc[0], 15.0
    )
    assert np.isclose(
        manhattan.loc[manhattan["ld_block_id"] == "block_3", "position"].iloc[0], 100.0
    )
    # Sorted by chromosome then position
    assert manhattan["position"].is_monotonic_increasing


def test_format_manhattan_drops_unmatched_blocks():
    """Blocks in results_df but not in block_assignment are dropped."""
    results_df = pd.DataFrame(
        {"p_value": [0.01]},
        index=pd.Index(["unknown_block"], name="ld_block_id"),
    )
    block_assignment = pd.DataFrame(
        {
            "ld_block_id": ["known_block"],
            "marker_id": ["m1"],
            "chromosome": [1],
            "position": [100],
        }
    )
    manhattan = format_manhattan(results_df, block_assignment)
    assert len(manhattan) == 0


# ---------------------------------------------------------------------------
#  End-to-end smoke
# ---------------------------------------------------------------------------


def test_association_pipeline_smoke():
    """Full pipeline: build_dosage → scan_slope → adjust → format."""
    rng = np.random.default_rng(333)
    n_geno = 40
    n_markers = 20
    n_blocks = 4

    genotype_ids = [f"g{i}" for i in range(n_geno)]
    marker_ids = [f"m{j}" for j in range(n_markers)]

    genotype_marker_df = pd.DataFrame(
        {
            "genotype_id": np.repeat(genotype_ids, n_markers),
            "marker_id": np.tile(marker_ids, n_geno),
            "allele_dosage": rng.uniform(0, 2, size=n_geno * n_markers),
        }
    )
    block_assignment = pd.DataFrame(
        {
            "marker_id": marker_ids,
            "ld_block_id": [f"block_{j % n_blocks}" for j in range(n_markers)],
            "chromosome": [1] * n_markers,
            "position": list(range(100, 100 + n_markers)),
        }
    )

    dosage = build_block_dosage_matrix(genotype_marker_df, block_assignment)
    # Block 0 has a real effect on slope
    true_beta = 2.0
    noise = rng.normal(0, 0.3, size=n_geno)
    slopes = 1.0 + true_beta * dosage.iloc[:, 0].to_numpy() + noise
    slopes_series = pd.Series(slopes, index=dosage.index)

    assoc = scan_slope_association(dosage, slopes_series)
    adj = adjust_multiple_testing(assoc)
    manhattan = format_manhattan(adj, block_assignment)

    assert assoc.shape[0] == n_blocks
    assert adj["q_value"].notna().sum() <= n_blocks
    assert len(manhattan) == n_blocks
    # Block 0 should be the most significant
    assert assoc["p_value"].idxmin() == f"block_{0}"
