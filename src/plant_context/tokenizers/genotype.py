"""GenotypeTokenizer: chromosome/position-ordered LD-block tokens (TDD
Section 5.2, 15 item 6).

Blocks are built from *adjacent* marker-pair genotypic r^2 (squared Pearson
correlation of allele dosage across genotypes -- an unphased approximation
of true haplotype-based LD, documented here rather than silently assumed to
be the phased version) walked in chromosome/position order, not from
sorting markers by some statistic like variance and chunking arbitrarily
(TDD Section 5.2 explicitly forbids treating that as a biological
sequence). A chromosome change is always a hard block boundary, and a
``max_block_size`` cap prevents a long high-LD run from becoming one
unboundedly large block.

LD block boundaries are fit using only a caller-supplied set of training
genotype IDs -- TDD Section 8.3 lists "LD block parameters" among the
things that must be fit on outer training data only, since the block
structure would otherwise depend on which genotypes end up in an outer
test fold -- and then applied to every genotype, seen or unseen, at
tokenization time.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

OOV_BLOCK_ID = "__OOV__"


def sort_markers(genotype_marker_df: pd.DataFrame) -> pd.DataFrame:
    """One row per marker_id (chromosome, position), sorted by chromosome
    then position regardless of the input row order (TDD 10.1: shuffling
    the input must not change this output).
    """
    return (
        genotype_marker_df[["marker_id", "chromosome", "position"]]
        .drop_duplicates(subset="marker_id")
        .sort_values(["chromosome", "position"], kind="mergesort")
        .reset_index(drop=True)
    )


def _pivot_train_dosage(
    genotype_marker_df: pd.DataFrame,
    train_genotype_ids: Iterable[str],
    marker_order: list,
) -> pd.DataFrame:
    train_ids = set(train_genotype_ids)
    train_df = genotype_marker_df[genotype_marker_df["genotype_id"].isin(train_ids)]
    wide = train_df.pivot_table(
        index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first"
    )
    return wide.reindex(columns=marker_order)


def _pairwise_r2(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 2:
        return float("nan")
    x_v, y_v = x[valid], y[valid]
    if x_v.std() == 0 or y_v.std() == 0:
        return float("nan")  # monomorphic in the training set -> can't correlate
    r = np.corrcoef(x_v, y_v)[0, 1]
    return float(r**2)


def fit_ld_blocks(
    genotype_marker_df: pd.DataFrame,
    train_genotype_ids: Iterable[str],
    r2_threshold: float = 0.7,
    max_block_size: int = 50,
) -> pd.DataFrame:
    """Assign each marker an ``ld_block_id``, using only ``train_genotype_ids``.

    Returns marker metadata (marker_id, chromosome, position) plus
    ``ld_block_id``. Deterministic: identical input always produces an
    identical assignment, and the assignment does not depend on any
    genotype outside ``train_genotype_ids`` -- a marker's block never
    changes because of what a held-out genotype's dosage happens to be.
    A NaN adjacent r^2 (e.g. a monomorphic marker in the training set)
    forces a block break rather than being treated as "related".
    """
    marker_meta = sort_markers(genotype_marker_df)
    wide = _pivot_train_dosage(genotype_marker_df, train_genotype_ids, marker_meta["marker_id"].tolist())

    block_ids = []
    current_block_idx = -1
    current_chromosome = None
    current_block_size = 0
    prev_marker_id = None

    for _, row in marker_meta.iterrows():
        marker_id = row["marker_id"]
        chromosome = row["chromosome"]
        start_new_block = False

        if chromosome != current_chromosome:
            start_new_block = True
        elif current_block_size >= max_block_size:
            start_new_block = True
        elif prev_marker_id is not None:
            r2 = _pairwise_r2(wide[prev_marker_id], wide[marker_id])
            if not (r2 >= r2_threshold):
                start_new_block = True

        if start_new_block:
            current_block_idx += 1
            current_chromosome = chromosome
            current_block_size = 0

        block_ids.append(f"{chromosome}_block{current_block_idx}")
        current_block_size += 1
        prev_marker_id = marker_id

    return marker_meta.assign(ld_block_id=block_ids)


def fit_ld_blocks_with_fixed_boundaries(
    genotype_marker_df: pd.DataFrame,
    train_genotype_ids: Iterable[str],
    window_size_bp: int = 100_000,
    max_block_size: int = 50,
) -> pd.DataFrame:
    """Assign each marker an ``ld_block_id`` using fixed physical windows.

    This is the boundary-stability mode required by TDD Section 5.2:
    block boundaries (e.g. fixed physical windows or gene windows) should
    be identical across all outer folds, while block-internal LD statistics
    are estimated only on ``train_genotype_ids``.

    The returned ``ld_block_id`` depends only on chromosome, physical
    position, and ``window_size_bp`` -- not on the genotypes in
    ``train_genotype_ids``. Therefore two folds with the same marker map
    will produce identical block assignments, making cross-fold comparison
    and interpretation possible.

    Parameters
    ----------
    genotype_marker_df :
        Long-format genotype/marker table (must contain marker_id, chromosome,
        position, and genotype_id columns).
    train_genotype_ids :
        Training genotype IDs. Currently used only to respect the TDD
        contract that LD-related fitting must accept a training set; the
        fixed-boundary assignment itself does not depend on these IDs.
    window_size_bp :
        Physical window size in base pairs. Each chromosome is partitioned
        into consecutive windows of this size; markers falling in the same
        window belong to the same block (subject to ``max_block_size``).
    max_block_size :
        Maximum number of markers per block. If a window contains more
        markers than this, it is split into multiple consecutive blocks.

    Returns
    -------
    DataFrame with columns marker_id, chromosome, position, ld_block_id.
    """
    # Accept training IDs to stay consistent with the fit_ld_blocks API,
    # but intentionally do not use them for boundary decisions.
    _ = set(train_genotype_ids)

    marker_meta = sort_markers(genotype_marker_df)
    if marker_meta.empty:
        return marker_meta.assign(ld_block_id=pd.Series([], dtype=str))

    block_ids = []
    current_chromosome = None
    current_window_idx = -1
    current_block_in_window = 0
    current_block_size = 0

    for _, row in marker_meta.iterrows():
        chromosome = row["chromosome"]
        position = int(row["position"])
        window_idx = position // window_size_bp

        new_chromosome = chromosome != current_chromosome
        new_window = window_idx != current_window_idx

        if new_chromosome or new_window or current_block_size >= max_block_size:
            if new_chromosome or new_window:
                current_window_idx = window_idx
                current_block_in_window = 0
            else:
                current_block_in_window += 1
            current_chromosome = chromosome
            current_block_size = 0

        block_ids.append(f"{chromosome}_w{window_idx}_b{current_block_in_window}")
        current_block_size += 1

    return marker_meta.assign(ld_block_id=block_ids)


def tokenize_genotype_blocks(
    genotype_marker_df: pd.DataFrame, block_assignment: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate marker-level dosage into one row per (genotype_id, ld_block_id).

    A marker present in ``genotype_marker_df`` but absent from
    ``block_assignment`` (unseen at fit time -- e.g. a marker panel that
    grew after the block structure was frozen) is assigned to an explicit
    OOV block (``OOV_BLOCK_ID``) rather than silently dropped or folded
    into an arbitrary real block.
    """
    merged = genotype_marker_df.merge(
        block_assignment[["marker_id", "ld_block_id"]], on="marker_id", how="left"
    )
    merged["ld_block_id"] = merged["ld_block_id"].fillna(OOV_BLOCK_ID)

    tokens = merged.groupby(["genotype_id", "ld_block_id"], sort=False).agg(
        chromosome=("chromosome", "first"),
        n_markers=("marker_id", "nunique"),
        mean_dosage=("allele_dosage", "mean"),
        min_position=("position", "min"),
        max_position=("position", "max"),
    )
    return tokens.reset_index()
