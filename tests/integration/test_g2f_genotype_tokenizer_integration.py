"""Integration test: GenotypeTokenizer against real G2F genotype data.

Skipped automatically if data/external/g2f isn't present -- see README.md.
"""

from pathlib import Path

import pytest

from plant_context.data.g2f_adapter import load_g2f_genotype_marker, load_g2f_phenotype_plot
from plant_context.evaluation.splits import make_leave_genotype_split
from plant_context.tokenizers.genotype import (
    OOV_BLOCK_ID,
    fit_ld_blocks,
    tokenize_genotype_blocks,
)

G2F_ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "g2f"

pytestmark = pytest.mark.skipif(
    not (G2F_ROOT / "phenotype.parquet").exists(),
    reason="data/external/g2f is not present on this machine",
)


@pytest.fixture(scope="module")
def g2f_genotype_and_train_ids():
    genotype_df = load_g2f_genotype_marker(G2F_ROOT)
    phenotype_df = load_g2f_phenotype_plot(G2F_ROOT)
    split_df = make_leave_genotype_split(phenotype_df, n_folds=5, seed=1234, split_version="smoke")
    fold0 = split_df[
        (split_df["outer_split_type"] == "leave_genotype") & (split_df["outer_fold"] == 0)
    ]
    train_sample_ids = set(fold0.loc[fold0["role"] == "train", "sample_id"])
    train_genotype_ids = set(
        phenotype_df.loc[phenotype_df["sample_id"].isin(train_sample_ids), "genotype_id"]
    )
    return genotype_df, train_genotype_ids


def test_fit_ld_blocks_runs_on_real_g2f_genotypes(g2f_genotype_and_train_ids):
    genotype_df, train_genotype_ids = g2f_genotype_and_train_ids
    blocks = fit_ld_blocks(genotype_df, train_genotype_ids, r2_threshold=0.7, max_block_size=50)

    assert len(blocks) == genotype_df["marker_id"].nunique()
    for _, block_rows in blocks.groupby("ld_block_id"):
        assert block_rows["chromosome"].nunique() == 1
    block_sizes = blocks.groupby("ld_block_id").size()
    assert (block_sizes <= 50).all()


def test_tokenize_genotype_blocks_runs_on_real_g2f_genotypes(g2f_genotype_and_train_ids):
    genotype_df, train_genotype_ids = g2f_genotype_and_train_ids
    blocks = fit_ld_blocks(genotype_df, train_genotype_ids, r2_threshold=0.7, max_block_size=50)
    tokens = tokenize_genotype_blocks(genotype_df, blocks)

    assert len(tokens) > 0
    # Same marker panel was used to fit and to tokenize, so nothing should
    # fall back to OOV here.
    assert OOV_BLOCK_ID not in tokens["ld_block_id"].unique()
    # G2F's observed allele_dosage range is [0, 1] (see data manifest); any
    # within-block mean must stay inside that range.
    assert tokens["mean_dosage"].between(0, 1).all()
