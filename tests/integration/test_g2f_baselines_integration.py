"""Integration smoke test: statistical baselines end-to-end on real G2F data.

Skipped automatically if data/external/g2f isn't present -- see README.md.

Subsamples to ~300 genotypes so GBLUP's genomic relationship matrix stays
small enough to run in a few seconds; this is a plumbing/leakage check, not
the actual W5-W6 baseline benchmark run (that uses the full data and lives
under experiments/, not tests/).
"""

from pathlib import Path

import numpy as np
import pytest

from plant_context.data.g2f_adapter import load_g2f_genotype_marker, load_g2f_phenotype_plot
from plant_context.evaluation.metrics import pearson_r, rmse
from plant_context.evaluation.splits import make_leave_genotype_split
from plant_context.statistics.baselines import (
    environment_mean_predict_fn,
    make_gblup_predict_fn,
    reaction_norm_predict_fn,
)
from plant_context.statistics.crossfit import run_crossfit

G2F_ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "g2f"

pytestmark = pytest.mark.skipif(
    not (G2F_ROOT / "phenotype.parquet").exists(),
    reason="data/external/g2f is not present on this machine",
)


@pytest.fixture(scope="module")
def subsampled_g2f():
    phenotype_df = load_g2f_phenotype_plot(G2F_ROOT)
    genotype_df = load_g2f_genotype_marker(G2F_ROOT)

    if len(genotype_df) == 0:
        # Genotype VCF has not been parsed yet; phenotype-only baselines
        # can still run by subsampling phenotype genotypes directly.
        rng = np.random.default_rng(1234)
        genotype_ids = sorted(phenotype_df["genotype_id"].unique())
        subset = set(rng.choice(genotype_ids, size=min(300, len(genotype_ids)), replace=False))
        phenotype_subset = phenotype_df[phenotype_df["genotype_id"].isin(subset)].reset_index(drop=True)
        return phenotype_subset, genotype_df

    common_genotypes = sorted(set(phenotype_df["genotype_id"]) & set(genotype_df["genotype_id"]))
    rng = np.random.default_rng(1234)
    subset = set(rng.choice(common_genotypes, size=min(300, len(common_genotypes)), replace=False))

    phenotype_subset = phenotype_df[phenotype_df["genotype_id"].isin(subset)].reset_index(drop=True)
    genotype_subset = genotype_df[genotype_df["genotype_id"].isin(subset)].reset_index(drop=True)
    return phenotype_subset, genotype_subset


def test_environment_mean_baseline_produces_finite_metrics(subsampled_g2f):
    phenotype_df, _ = subsampled_g2f
    split_df = make_leave_genotype_split(phenotype_df, n_folds=3, seed=1234, split_version="smoke")

    result = run_crossfit(phenotype_df, split_df, environment_mean_predict_fn)
    assert np.isfinite(result["y_pred"]).all()
    assert np.isfinite(rmse(result["y_true"], result["y_pred"]))


def test_gblup_baseline_produces_finite_metrics(subsampled_g2f):
    phenotype_df, genotype_df = subsampled_g2f
    if len(genotype_df) == 0:
        pytest.skip("genotype.parquet is empty placeholder (raw VCF not yet parsed)")
    split_df = make_leave_genotype_split(phenotype_df, n_folds=3, seed=1234, split_version="smoke")

    gblup_fn = make_gblup_predict_fn(genotype_df, max_dosage=1.0, n_folds=3)
    result = run_crossfit(phenotype_df, split_df, gblup_fn)

    assert np.isfinite(result["y_pred"]).all()
    assert np.isfinite(rmse(result["y_true"], result["y_pred"]))
    assert np.isfinite(pearson_r(result["y_true"], result["y_pred"]))


def test_reaction_norm_baseline_produces_finite_metrics_on_leave_genotype_split(subsampled_g2f):
    phenotype_df, _ = subsampled_g2f
    # reaction_norm_predict_fn's phenotype-mean environment index is only
    # leakage-safe for leave_genotype_out/random splits -- see its docstring.
    split_df = make_leave_genotype_split(phenotype_df, n_folds=3, seed=1234, split_version="smoke")

    result = run_crossfit(phenotype_df, split_df, reaction_norm_predict_fn)
    assert np.isfinite(result["y_pred"]).all()
    assert np.isfinite(rmse(result["y_true"], result["y_pred"]))
