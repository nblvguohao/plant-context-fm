"""Integration test: masked-reconstruction pretraining on real G2F tokens.

Ties GenotypeTokenizer / EnvironmentTokenizer, masking.py, and
models/pretraining.py together end-to-end on subsamples of real G2F data
(a plumbing/leakage-adjacent smoke test, not the actual research-scale
pretraining run, which belongs under experiments/, not tests/).

Skipped automatically if data/external/g2f isn't present -- see README.md.
"""

import functools
from pathlib import Path

import numpy as np
import pytest

from plant_context.data.g2f_adapter import load_g2f_environment_daily, load_g2f_genotype_marker
from plant_context.models.pretraining import pretrain_masked_reconstruction
from plant_context.tokenizers.environment import STAGE_ORDER, tokenize_environment_stages
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks
from plant_context.tokenizers.masking import mask_contiguous_run

G2F_ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "g2f"

pytestmark = pytest.mark.skipif(
    not (G2F_ROOT / "phenotype.parquet").exists(),
    reason="data/external/g2f is not present on this machine",
)


def _ordered_per_sample_dict(long_tokens, id_col, order_col, canonical_order, feature_columns):
    """long_tokens (one row per id_col x order_col) -> {id: DataFrame
    indexed by order_col, rows reindexed to canonical_order's subsequence
    actually present for that id} -- so contiguous masking operates on a
    real chromosome/chronological order, not merge/groupby incidental order.
    """
    per_sample = {}
    for sample_id, group in long_tokens.groupby(id_col):
        indexed = group.set_index(order_col)
        present_in_order = [c for c in canonical_order if c in indexed.index]
        per_sample[sample_id] = indexed.loc[present_in_order, feature_columns]
    return per_sample


def test_genotype_masked_pretraining_runs_on_real_g2f_subsample():
    genotype_df = load_g2f_genotype_marker(G2F_ROOT)
    if len(genotype_df) == 0:
        pytest.skip("genotype.parquet is empty placeholder (raw VCF not yet parsed)")
    rng = np.random.default_rng(1234)
    subset_genotypes = set(
        rng.choice(sorted(genotype_df["genotype_id"].unique()), size=80, replace=False)
    )
    genotype_subset = genotype_df[genotype_df["genotype_id"].isin(subset_genotypes)]

    blocks = fit_ld_blocks(genotype_subset, subset_genotypes, r2_threshold=0.7, max_block_size=50)
    genotype_tokens = tokenize_genotype_blocks(genotype_subset, blocks)

    block_order = list(dict.fromkeys(blocks["ld_block_id"]))
    per_sample_tokens = _ordered_per_sample_dict(
        genotype_tokens, "genotype_id", "ld_block_id", block_order, ["mean_dosage"]
    )

    mask_fn = functools.partial(mask_contiguous_run, mask_fraction=0.15)
    result = pretrain_masked_reconstruction(
        per_sample_tokens, feature_columns=["mean_dosage"], mask_fn=mask_fn, epochs=30, seed=1234
    )

    assert np.isfinite(result["final_loss"])
    assert "effective_rank" in result["collapse_diagnostics"]


def test_environment_masked_pretraining_runs_on_real_g2f_subsample():
    environment_daily_df = load_g2f_environment_daily(G2F_ROOT)
    if len(environment_daily_df) == 0:
        pytest.skip("weather_daily.parquet is empty (2021 weather missing in raw data)")
    rng = np.random.default_rng(1234)
    env_ids = sorted(environment_daily_df["environment_id"].unique())
    subset_environments = set(
        rng.choice(env_ids, size=min(40, len(env_ids)), replace=False)
    )
    environment_subset = environment_daily_df[
        environment_daily_df["environment_id"].isin(subset_environments)
    ]

    stage_tokens = tokenize_environment_stages(environment_subset)
    feature_columns = [
        "tmax_mean", "tmin_mean", "tmean_mean", "gdd_sum",
        "precipitation_sum", "solar_radiation_mean", "relative_humidity_mean", "vpd_mean",
    ]
    per_sample_tokens = _ordered_per_sample_dict(
        stage_tokens, "environment_id", "growth_stage", STAGE_ORDER, feature_columns
    )
    # Only keep environments with at least 2 stage tokens -- a single-token
    # sample has nothing to mask-and-reconstruct from.
    per_sample_tokens = {k: v for k, v in per_sample_tokens.items() if len(v) >= 2}

    mask_fn = functools.partial(mask_contiguous_run, mask_fraction=0.2)
    result = pretrain_masked_reconstruction(
        per_sample_tokens, feature_columns=feature_columns, mask_fn=mask_fn, epochs=30, seed=1234
    )

    assert np.isfinite(result["final_loss"])
    assert "effective_rank" in result["collapse_diagnostics"]
