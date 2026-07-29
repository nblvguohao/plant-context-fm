"""Unit tests for GenotypeTokenizer (TDD Section 5.2, 10.1).

Uses a hand-constructed fixture where adjacent-marker correlation is exact
by design (orthogonal or identical contrast vectors), so expected LD block
boundaries are known ground truth, not just "plausible".
"""

import numpy as np
import pandas as pd
import pytest

from plant_context.tokenizers.genotype import (
    OOV_BLOCK_ID,
    fit_ld_blocks,
    sort_markers,
    tokenize_genotype_blocks,
)

TRAIN_GENOTYPES = [f"g{i}" for i in range(8)]

# A and its identical copy have r^2 = 1 with each other.
# C is exactly orthogonal to A (centered dot product = 0), so r^2 = 0.
PATTERN_A = [2, 2, 2, 2, 0, 0, 0, 0]
PATTERN_C = [2, 0, 2, 0, 2, 0, 2, 0]


def _rows_for_marker(genotype_ids, marker_id, chromosome, position, dosage_pattern):
    return [
        {
            "genotype_id": g,
            "marker_id": marker_id,
            "chromosome": chromosome,
            "position": position,
            "allele_dosage": d,
        }
        for g, d in zip(genotype_ids, dosage_pattern)
    ]


def _known_ld_fixture(genotype_ids=TRAIN_GENOTYPES):
    rows = []
    # Chromosome 1: m1/m2 identical (r2=1, merge); m3/m4 identical to each
    # other but orthogonal to m2 (r2=0, must break); expect blocks {m1,m2}
    # and {m3,m4}.
    rows += _rows_for_marker(genotype_ids, "m1", "S1", 100, PATTERN_A)
    rows += _rows_for_marker(genotype_ids, "m2", "S1", 200, PATTERN_A)
    rows += _rows_for_marker(genotype_ids, "m3", "S1", 300, PATTERN_C)
    rows += _rows_for_marker(genotype_ids, "m4", "S1", 400, PATTERN_C)
    # Chromosome 2: n1/n2 identical (r2=1, would merge if same chromosome
    # as chr1, but must not merge across chromosomes regardless).
    rows += _rows_for_marker(genotype_ids, "n1", "S2", 50, PATTERN_A)
    rows += _rows_for_marker(genotype_ids, "n2", "S2", 150, PATTERN_A)
    return pd.DataFrame(rows)


def test_sort_markers_orders_by_chromosome_then_position_regardless_of_input_order():
    df = _known_ld_fixture()
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    sorted_meta = sort_markers(shuffled)
    assert sorted_meta["marker_id"].tolist() == ["m1", "m2", "m3", "m4", "n1", "n2"]


def test_fit_ld_blocks_merges_correlated_and_breaks_uncorrelated_adjacent_markers():
    df = _known_ld_fixture()
    blocks = fit_ld_blocks(df, TRAIN_GENOTYPES, r2_threshold=0.7, max_block_size=50)
    block_of = dict(zip(blocks["marker_id"], blocks["ld_block_id"]))

    assert block_of["m1"] == block_of["m2"]
    assert block_of["m3"] == block_of["m4"]
    assert block_of["m2"] != block_of["m3"]


def test_fit_ld_blocks_never_spans_chromosomes():
    df = _known_ld_fixture()
    blocks = fit_ld_blocks(df, TRAIN_GENOTYPES, r2_threshold=0.7, max_block_size=50)

    for _, block_rows in blocks.groupby("ld_block_id"):
        assert block_rows["chromosome"].nunique() == 1


def test_fit_ld_blocks_is_deterministic():
    df = _known_ld_fixture()
    first = fit_ld_blocks(df, TRAIN_GENOTYPES, r2_threshold=0.7, max_block_size=50)
    second = fit_ld_blocks(df, TRAIN_GENOTYPES, r2_threshold=0.7, max_block_size=50)
    pd.testing.assert_frame_equal(first, second)


def test_fit_ld_blocks_respects_max_block_size():
    genotype_ids = TRAIN_GENOTYPES
    rows = []
    # 7 perfectly-correlated markers on one chromosome: without a cap they
    # would all merge into a single block.
    for i, pos in enumerate(range(100, 800, 100)):
        rows += _rows_for_marker(genotype_ids, f"m{i}", "S1", pos, PATTERN_A)
    df = pd.DataFrame(rows)

    blocks = fit_ld_blocks(df, genotype_ids, r2_threshold=0.7, max_block_size=3)
    block_sizes = blocks.groupby("ld_block_id").size()
    assert (block_sizes <= 3).all()
    assert block_sizes.sum() == 7


def test_fit_ld_blocks_ignores_non_training_genotype_data():
    train_only_df = _known_ld_fixture(genotype_ids=TRAIN_GENOTYPES)

    # Extra "test" genotypes carry dosage that, if it were allowed to
    # influence fitting, would make m2 and m3 look correlated (breaking the
    # expected block boundary). They must have zero effect.
    extra_genotypes = ["test_g0", "test_g1"]
    extra_rows = []
    extra_rows += _rows_for_marker(extra_genotypes, "m1", "S1", 100, [1, 1])
    extra_rows += _rows_for_marker(extra_genotypes, "m2", "S1", 200, [2, 0])
    extra_rows += _rows_for_marker(extra_genotypes, "m3", "S1", 300, [2, 0])
    extra_rows += _rows_for_marker(extra_genotypes, "m4", "S1", 400, [1, 1])
    extra_rows += _rows_for_marker(extra_genotypes, "n1", "S2", 50, [1, 1])
    extra_rows += _rows_for_marker(extra_genotypes, "n2", "S2", 150, [1, 1])
    combined_df = pd.concat([train_only_df, pd.DataFrame(extra_rows)], ignore_index=True)

    blocks_without_extra = fit_ld_blocks(train_only_df, TRAIN_GENOTYPES, r2_threshold=0.7, max_block_size=50)
    blocks_with_extra = fit_ld_blocks(combined_df, TRAIN_GENOTYPES, r2_threshold=0.7, max_block_size=50)

    pd.testing.assert_frame_equal(blocks_without_extra, blocks_with_extra)


def test_tokenize_genotype_blocks_aggregates_dosage_per_block():
    df = _known_ld_fixture()
    blocks = fit_ld_blocks(df, TRAIN_GENOTYPES, r2_threshold=0.7, max_block_size=50)
    tokens = tokenize_genotype_blocks(df, blocks)

    g0_block_m1m2 = tokens[
        (tokens["genotype_id"] == "g0") & (tokens["ld_block_id"] == blocks.set_index("marker_id").loc["m1", "ld_block_id"])
    ].iloc[0]
    assert g0_block_m1m2["n_markers"] == 2
    assert g0_block_m1m2["mean_dosage"] == pytest.approx(PATTERN_A[0])  # m1 == m2 for g0


def test_tokenize_genotype_blocks_flags_unseen_marker_as_explicit_oov():
    df = _known_ld_fixture()
    # Fit blocks on a panel that is missing marker "m4".
    fit_df = df[df["marker_id"] != "m4"]
    blocks = fit_ld_blocks(fit_df, TRAIN_GENOTYPES, r2_threshold=0.7, max_block_size=50)

    # Tokenize the full panel, which does include "m4".
    tokens = tokenize_genotype_blocks(df, blocks)
    oov_rows = tokens[tokens["ld_block_id"] == OOV_BLOCK_ID]

    assert len(oov_rows) == len(TRAIN_GENOTYPES)  # one OOV row per genotype
    assert (oov_rows["n_markers"] == 1).all()  # only marker m4 is unseen
