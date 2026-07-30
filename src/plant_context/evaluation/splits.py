"""Outer split protocols (TDD Section 8, implementation-order item 3).

Each ``make_*_split`` function takes a phenotype_plot-contract DataFrame
(needs at least ``sample_id``, ``genotype_id``, ``environment_id``, ``year``)
and returns a ``split_table`` (TDD Section 4.5): one row per
``(sample_id, outer_fold)``, with ``role`` in {train, validation, test}.

These functions only assign roles. Nothing here fits an imputer,
standardizer, marker selection, or hyperparameter -- TDD Section 8.3 requires
all of those to be fit on the ``train`` rows of the resulting table,
elsewhere, never here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VALIDATION_FRACTION_DEFAULT = 0.15


def _assign_group_folds(unique_values: np.ndarray, n_folds: int, seed: int) -> dict:
    """Seeded shuffle of ``unique_values`` into ``n_folds`` near-equal groups."""
    values = np.array(sorted(unique_values))
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return {v: i % n_folds for i, v in enumerate(values)}


def _group_kfold_split(
    df: pd.DataFrame,
    group_col: str,
    outer_split_type: str,
    n_folds: int,
    seed: int,
    split_version: str,
    validation_fraction: float,
) -> pd.DataFrame:
    fold_of = _assign_group_folds(df[group_col].unique(), n_folds, seed)
    rng = np.random.default_rng(seed)
    fold_frames = []

    for fold in range(n_folds):
        test_groups = {g for g, f in fold_of.items() if f == fold}
        remaining = np.array(sorted(g for g, f in fold_of.items() if f != fold))
        rng.shuffle(remaining)
        n_val = int(round(len(remaining) * validation_fraction)) if len(remaining) else 0
        n_val = min(max(n_val, 1 if len(remaining) else 0), len(remaining))
        validation_groups = set(remaining[:n_val])

        def _role(g, _test=test_groups, _val=validation_groups):
            if g in _test:
                return "test"
            if g in _val:
                return "validation"
            return "train"

        fold_frames.append(
            pd.DataFrame(
                {
                    "sample_id": df["sample_id"].to_numpy(),
                    "outer_split_type": outer_split_type,
                    "outer_fold": fold,
                    "role": df[group_col].map(_role).to_numpy(),
                    "seed": seed,
                    "group_key": df[group_col].astype(str).to_numpy(),
                    "split_version": split_version,
                }
            )
        )
    return pd.concat(fold_frames, ignore_index=True)


def make_leave_genotype_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 1234,
    split_version: str = "v1",
    validation_fraction: float = VALIDATION_FRACTION_DEFAULT,
) -> pd.DataFrame:
    """Test genotype_ids never appear in that fold's train/validation rows."""
    return _group_kfold_split(
        df, "genotype_id", "leave_genotype", n_folds, seed, split_version, validation_fraction
    )


def make_leave_environment_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 1234,
    split_version: str = "v1",
    validation_fraction: float = VALIDATION_FRACTION_DEFAULT,
) -> pd.DataFrame:
    """Test environment_ids never appear in that fold's train/validation rows."""
    return _group_kfold_split(
        df, "environment_id", "leave_environment", n_folds, seed, split_version, validation_fraction
    )


def make_forward_year_split(
    df: pd.DataFrame,
    seed: int = 1234,
    split_version: str = "v1",
) -> pd.DataFrame:
    """One fold per test year: strictly later than every train/validation year.

    Fold order follows the sorted distinct years. The earliest two years can
    never be a test year (there must be at least one training year and one
    validation year before it). For a given fold, the year immediately
    before the test year becomes validation and every earlier year becomes
    train, so ``max(train_year) < min(validation_year) < min(test_year)``
    holds by construction (TDD Section 4.3, 10.3).

    ``seed`` is recorded into the output's ``seed`` column for schema
    consistency with the other three split functions, but it does not
    change the fold assignment: chronological order has no randomness to
    seed. Do not report "n=k seeds" statistics for forward_year results
    without accounting for this -- running this function with k different
    seed values produces k byte-identical splits, not k independent
    resamples. (This is exactly the gap the SRG-GxE audit found in its own
    forward_year/leave_year splits after the fact; see
    docs/srg_gxe_audit_2026-07-29.md. Any variance observed across "seeds"
    on this split type is model-initialization noise, not
    train/test-resampling variance -- report it as such, or use a
    seed-sensitive design like a genuine leave-year-out CV if independent
    replicates across years are actually needed.)
    """
    years = sorted(df["year"].unique())
    if len(years) < 3:
        raise ValueError(
            "forward_year split needs at least 3 distinct years "
            f"(train, validation, test); got {len(years)}"
        )

    fold_frames = []
    for fold, test_year in enumerate(years[2:]):
        idx = years.index(test_year)
        validation_year = years[idx - 1]
        train_years = set(years[: idx - 1])
        in_scope = df["year"] <= test_year

        def _role(y, _train=train_years, _val=validation_year, _test=test_year):
            if y == _test:
                return "test"
            if y == _val:
                return "validation"
            return "train"  # guaranteed to be in train_years given in_scope filter

        scoped = df.loc[in_scope]
        fold_frames.append(
            pd.DataFrame(
                {
                    "sample_id": scoped["sample_id"].to_numpy(),
                    "outer_split_type": "forward_year",
                    "outer_fold": fold,
                    "role": scoped["year"].map(_role).to_numpy(),
                    "seed": seed,
                    "group_key": scoped["year"].astype(str).to_numpy(),
                    "split_version": split_version,
                }
            )
        )
    return pd.concat(fold_frames, ignore_index=True)


def make_leave_ge_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 1234,
    split_version: str = "v1",
    validation_fraction: float = VALIDATION_FRACTION_DEFAULT,
) -> pd.DataFrame:
    """Held-out (genotype, environment) combinations, each side seen elsewhere.

    TDD Section 8.1: "leave_ge_out: test G-E combination unseen, but G and E
    may each appear separately [in training]." A candidate test combo is
    only kept as test if, once removed, its genotype still appears in some
    other training combo *and* its environment still appears in some other
    training combo; otherwise it is pushed back into train so the "each side
    seen elsewhere" property always holds rather than being violated.
    """
    combo_df = df[["genotype_id", "environment_id"]].drop_duplicates().reset_index(drop=True)
    combo_key = combo_df["genotype_id"].astype(str) + "::" + combo_df["environment_id"].astype(str)
    combo_df = combo_df.assign(combo_key=combo_key)
    fold_of = _assign_group_folds(combo_df["combo_key"].unique(), n_folds, seed)

    df_combo_key = df["genotype_id"].astype(str) + "::" + df["environment_id"].astype(str)
    rng = np.random.default_rng(seed)
    fold_frames = []

    for fold in range(n_folds):
        candidate_test = {c for c, f in fold_of.items() if f == fold}

        # First decide train vs. validation among everything not a test
        # candidate, *then* check the leave_ge guarantee against train
        # specifically -- checking against "remaining" (train+validation
        # combined) would let a genotype/environment end up present only in
        # validation, which does not satisfy "appears elsewhere in training".
        non_candidate = np.array(sorted(c for c in combo_df["combo_key"] if c not in candidate_test))
        rng.shuffle(non_candidate)
        n_val = int(round(len(non_candidate) * validation_fraction)) if len(non_candidate) else 0
        n_val = min(max(n_val, 1 if len(non_candidate) else 0), len(non_candidate))
        validation_combos = set(non_candidate[:n_val])
        train_combos = set(non_candidate[n_val:])

        train_genotypes = set(combo_df.loc[combo_df["combo_key"].isin(train_combos), "genotype_id"])
        train_environments = set(
            combo_df.loc[combo_df["combo_key"].isin(train_combos), "environment_id"]
        )

        valid_test = set(
            combo_df.loc[
                combo_df["combo_key"].isin(candidate_test)
                & combo_df["genotype_id"].isin(train_genotypes)
                & combo_df["environment_id"].isin(train_environments),
                "combo_key",
            ]
        )
        # Candidates that fail the guarantee fall through to "train" in
        # _role below (they are in neither valid_test nor validation_combos),
        # so every combo in the input still gets a role.

        def _role(c, _test=valid_test, _val=validation_combos):
            if c in _test:
                return "test"
            if c in _val:
                return "validation"
            return "train"

        fold_frames.append(
            pd.DataFrame(
                {
                    "sample_id": df["sample_id"].to_numpy(),
                    "outer_split_type": "leave_ge",
                    "outer_fold": fold,
                    "role": df_combo_key.map(_role).to_numpy(),
                    "seed": seed,
                    "group_key": df_combo_key.to_numpy(),
                    "split_version": split_version,
                }
            )
        )
    return pd.concat(fold_frames, ignore_index=True)
