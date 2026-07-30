"""Environment-dependent marker/block association analysis (TDD Section 6.5,
15 item 10).

Estimates how marker effects vary across environments and tests which markers
are significantly associated with environmental sensitivity. This is
deliberately separate from the predictive G×E model (gxe_model.py) --
prediction attribution and statistical association are kept in different
modules with different guarantees (TDD Section 1.1).

The base approach ("reaction norm coefficient GWAS", TDD 6.5 bullet 1) is a
two-step design:

  1. Reduce each genotype's phenotypic response to its environmental
     sensitivity slope (from ``statistics.reaction_norm.fit_reaction_norm``).
  2. For each LD block, test whether its mean allele dosage predicts the
     per-genotype slope: ``slope_i ~ mu + beta_j * dosage_ij``.

  This tells which genomic regions are associated with environmental
  sensitivity -- i.e. which markers have effects that depend on the
  environment.

All functions work with LD blocks (``ld_block_id``) as the marker unit,
matching the GenotypeTokenizer's block structure. A single-marker-level
scan is a natural extension but not implemented here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def build_block_dosage_matrix(
    genotype_marker_df: pd.DataFrame, block_assignment: pd.DataFrame
) -> pd.DataFrame:
    """Build a (genotype × block) matrix of mean allele dosage per LD block.

    Parameters
    ----------
    genotype_marker_df :
        GenotypeMarker-contract table with columns ``genotype_id``,
        ``marker_id``, ``allele_dosage``.
    block_assignment :
        Output of ``tokenizers.genotype.fit_ld_blocks`` (or any DataFrame
        with ``marker_id`` and ``ld_block_id`` columns).

    Returns
    -------
    DataFrame indexed by ``genotype_id``, columns = ``ld_block_id`` sorted
    alphabetically, values = mean allele dosage of markers in that block for
    that genotype. A genotype with no markers in a block gets NaN.
    """
    merged = genotype_marker_df[["genotype_id", "marker_id", "allele_dosage"]].merge(
        block_assignment[["marker_id", "ld_block_id"]], on="marker_id", how="inner"
    )
    block_means = (
        merged.groupby(["genotype_id", "ld_block_id"])["allele_dosage"]
        .mean()
        .unstack(level="ld_block_id")
    )
    block_means.columns = sorted(block_means.columns, key=str)
    return block_means


def scan_slope_association(
    block_dosage: pd.DataFrame,
    genotype_slopes: pd.Series,
    min_genotypes: int = 10,
) -> pd.DataFrame:
    """Test each LD block for association with per-genotype reaction-norm slope.

    For every block column in ``block_dosage``, fits a simple linear
    regression ``slope_i = mu + beta_j * dosage_ij + noise`` via OLS and
    reports the estimated coefficient, its standard error, the t-statistic,
    and the two-sided p-value under H0: beta_j = 0.

    Genotypes with NaN slope or NaN block dosage are dropped per block
    (not globally) so a genotype with one missing block still contributes
    to all other blocks.

    Parameters
    ----------
    block_dosage :
        (genotype × block) DataFrame, typically from
        ``build_block_dosage_matrix``.
    genotype_slopes :
        Series indexed by ``genotype_id`` with the reaction-norm slope b_i
        from ``statistics.reaction_norm.fit_reaction_norm``.  NaN values
        (genotypes where slope was unidentifiable) are silently dropped.
    min_genotypes :
        Minimum number of non-NaN observations required to run the
        regression.  Blocks below this threshold are reported with NaN
        results.

    Returns
    -------
    DataFrame index = ``ld_block_id`` with columns
    ``beta``, ``se``, ``t_stat``, ``p_value``, ``n_genotypes``.
    """
    common_genotypes = block_dosage.index.intersection(genotype_slopes.dropna().index)
    y = genotype_slopes.loc[common_genotypes]

    results = {}
    for block_id in block_dosage.columns:
        x = block_dosage.loc[common_genotypes, block_id]
        valid = y.notna() & x.notna()
        n = valid.sum()
        if n < min_genotypes or x.loc[valid].std() == 0:
            results[block_id] = {
                "beta": np.nan,
                "se": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "n_genotypes": n,
            }
            continue

        y_v = y.loc[valid].to_numpy(dtype=np.float64)
        x_v = x.loc[valid].to_numpy(dtype=np.float64)
        # OLS: y = a + b * x
        # polyfit with deg=1 returns [slope, intercept]
        if n == 1:
            slope, intercept = 0.0, float(y_v[0])
        else:
            slope, intercept = np.polyfit(x_v, y_v, deg=1)

        residuals = y_v - (intercept + slope * x_v)
        dof = n - 2
        if dof < 1:
            results[block_id] = {
                "beta": np.nan,
                "se": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "n_genotypes": n,
            }
            continue

        mse = (residuals**2).sum() / dof
        x_centered = x_v - x_v.mean()
        se = np.sqrt(mse / (x_centered**2).sum())
        t_stat = slope / se if se > 0 else 0.0
        # Two-sided t-test
        p_value = 2.0 * scipy_stats.t.sf(abs(t_stat), df=dof)

        results[block_id] = {
            "beta": float(slope),
            "se": float(se),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "n_genotypes": n,
        }

    return pd.DataFrame.from_dict(results, orient="index").rename_axis("ld_block_id")


def adjust_multiple_testing(
    results_df: pd.DataFrame,
    p_column: str = "p_value",
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add multiple-testing-adjusted significance columns.

    Adds ``q_value`` (minimum FDR at which this test would be called
    significant) and ``significant_{alpha}`` (bool) columns.  Requires
    ``p_value`` column in the input; tests with NaN p-value get NaN q_value
    and False significance.

    Parameters
    ----------
    results_df :
        Output from ``scan_slope_association`` or similar.  Must contain
        ``p_column``.
    method :
        Currently only ``"fdr_bh"`` (Benjamini-Hochberg) is implemented.
    alpha :
        Nominal FDR threshold for the ``significant_{alpha}`` flag.

    Returns
    -------
    DataFrame with additional ``q_value`` and ``significant_{alpha}`` columns.
    """
    df = results_df.copy()
    p_vals = df[p_column].to_numpy()
    n_tests = int(np.isfinite(p_vals).sum())
    q_vals = np.full(len(df), np.nan)
    significant = np.full(len(df), False, dtype=bool)

    if n_tests == 0:
        df["q_value"] = q_vals
        df[f"significant_{alpha}"] = significant
        return df

    if method == "fdr_bh":
        finite_mask = np.isfinite(p_vals)
        finite_idx = np.where(finite_mask)[0]
        p_finite = p_vals[finite_mask]
        order = np.argsort(p_finite)
        n = len(p_finite)
        # BH step-up
        q = np.ones(n) * np.inf
        cumulative_min = np.inf
        for i in range(n - 1, -1, -1):
            rank = i + 1
            raw_q = p_finite[order[i]] * n / rank
            cumulative_min = min(cumulative_min, raw_q)
            q[order[i]] = cumulative_min
        q_vals[finite_mask] = q
        significant[finite_mask] = q <= alpha
    else:
        raise ValueError(f"Unknown multiple-testing method: {method}")

    df["q_value"] = q_vals
    df[f"significant_{alpha}"] = significant
    return df


def format_manhattan(
    results_df: pd.DataFrame,
    block_assignment: pd.DataFrame,
    p_column: str = "p_value",
) -> pd.DataFrame:
    """Augment association results with chromosome and position for Manhattan
    plotting.

    Parameters
    ----------
    results_df :
        Output from ``scan_slope_association`` (or similar).  Index must
        be ``ld_block_id``, and must have ``p_column``.
    block_assignment :
        Output of ``tokenizers.genotype.fit_ld_blocks``.  Must contain
        ``ld_block_id``, ``chromosome``, ``position``.

    Returns
    -------
    DataFrame with columns ``ld_block_id``, ``chromosome``, ``position``
    (midpoint of the block), ``-log10_p``, and ``p_value``, sorted by
    chromosome and position.  Blocks not present in ``block_assignment``
    are dropped.
    """
    block_positions = (
        block_assignment.groupby("ld_block_id", sort=False)
        .agg(chromosome=("chromosome", "first"), position=("position", "mean"))
        .reset_index()
    )
    merged = results_df.reset_index().merge(block_positions, on="ld_block_id", how="inner")
    merged["-log10_p"] = -np.log10(merged[p_column].clip(lower=1e-300))
    return merged.sort_values(["chromosome", "position"]).reset_index(drop=True)
