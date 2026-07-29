"""Integration test: the low-rank G-E model end-to-end on real G2F data.

Ties together GenotypeTokenizer, EnvironmentTokenizer, the model, and
run_crossfit -- the same pieces built in the last several steps -- on a
genotype subsample for test speed (this is a plumbing/leakage smoke test,
not the actual W5-W6-scale benchmark run, which belongs under experiments/).

Skipped automatically if data/external/g2f isn't present -- see README.md.
"""

from pathlib import Path

import numpy as np
import pytest

from plant_context.data.g2f_adapter import (
    load_g2f_environment_daily,
    load_g2f_genotype_marker,
    load_g2f_phenotype_plot,
)
from plant_context.evaluation.metrics import rmse
from plant_context.evaluation.splits import make_leave_genotype_split
from plant_context.models.gxe_model import (
    make_low_rank_gxe_predict_fn,
    pivot_environment_tokens_wide,
    pivot_genotype_tokens_wide,
)
from plant_context.statistics.crossfit import run_crossfit
from plant_context.tokenizers.environment import tokenize_environment_stages
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks

G2F_ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "g2f"

pytestmark = pytest.mark.skipif(
    not (G2F_ROOT / "phenotype.parquet").exists(),
    reason="data/external/g2f is not present on this machine",
)


@pytest.fixture(scope="module")
def g2f_gxe_fixture():
    phenotype_df = load_g2f_phenotype_plot(G2F_ROOT)
    genotype_df = load_g2f_genotype_marker(G2F_ROOT)
    environment_daily_df = load_g2f_environment_daily(G2F_ROOT)

    rng = np.random.default_rng(1234)
    all_genotypes = sorted(set(phenotype_df["genotype_id"]) & set(genotype_df["genotype_id"]))
    subset_genotypes = set(rng.choice(all_genotypes, size=min(150, len(all_genotypes)), replace=False))

    phenotype_subset = phenotype_df[phenotype_df["genotype_id"].isin(subset_genotypes)].reset_index(
        drop=True
    )
    genotype_subset = genotype_df[genotype_df["genotype_id"].isin(subset_genotypes)].reset_index(
        drop=True
    )

    split_df = make_leave_genotype_split(phenotype_subset, n_folds=3, seed=1234, split_version="smoke")
    fold0 = split_df[(split_df["outer_split_type"] == "leave_genotype") & (split_df["outer_fold"] == 0)]
    train_sample_ids = set(fold0.loc[fold0["role"] == "train", "sample_id"])
    train_genotype_ids = set(
        phenotype_subset.loc[phenotype_subset["sample_id"].isin(train_sample_ids), "genotype_id"]
    )

    blocks = fit_ld_blocks(genotype_subset, train_genotype_ids, r2_threshold=0.7, max_block_size=50)
    genotype_tokens = tokenize_genotype_blocks(genotype_subset, blocks)
    genotype_features = pivot_genotype_tokens_wide(genotype_tokens)

    relevant_environments = set(phenotype_subset["environment_id"])
    environment_daily_subset = environment_daily_df[
        environment_daily_df["environment_id"].isin(relevant_environments)
    ]
    environment_tokens = tokenize_environment_stages(environment_daily_subset)
    environment_features = pivot_environment_tokens_wide(environment_tokens)

    # Restrict phenotype rows to genotypes/environments that actually ended
    # up with features (e.g. an environment with no weather coverage).
    phenotype_subset = phenotype_subset[
        phenotype_subset["genotype_id"].isin(genotype_features.index)
        & phenotype_subset["environment_id"].isin(environment_features.index)
    ].reset_index(drop=True)
    split_df = make_leave_genotype_split(phenotype_subset, n_folds=3, seed=1234, split_version="smoke")

    return phenotype_subset, split_df, genotype_features, environment_features


def test_low_rank_gxe_model_runs_end_to_end_on_real_g2f_subsample(g2f_gxe_fixture):
    phenotype_subset, split_df, genotype_features, environment_features = g2f_gxe_fixture

    fit_predict = make_low_rank_gxe_predict_fn(
        genotype_features, environment_features, epochs=150, rank=4, hidden=16, seed=1234
    )
    result = run_crossfit(phenotype_subset, split_df, fit_predict)

    assert np.isfinite(result["y_pred"]).all()
    assert np.isfinite(rmse(result["y_true"], result["y_pred"]))
