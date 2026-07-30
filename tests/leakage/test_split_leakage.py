"""Leakage tests for the four split protocols (TDD Section 10.3).

These check the specific anti-leakage invariants TDD 10.3 calls out by name:

- leave-genotype-out: train/test genotype intersection is empty
- leave-environment-out: train/test environment intersection is empty
- forward-year: max(train_year) < min(test_year)
- leave-ge-out: the held-out (genotype, environment) combos are absent from
  train, but each genotype and environment involved still appears in train

Two TDD 10.3 items are not covered here because nothing to check exists yet:
"repeated location/plot groups don't cross folds" needs a location-based
split (TDD's optional leave_location_out, not implemented), and "fitted
preprocessing objects only see train IDs" / "the residual model never reads
outer-test phenotypes" need a preprocessing/model implementation to test
against. Do not mark those as passing here.
"""

import pandas as pd

from plant_context.evaluation.splits import (
    make_forward_year_split,
    make_leave_environment_split,
    make_leave_genotype_split,
    make_leave_ge_split,
)


def test_leave_genotype_out_has_zero_train_test_genotype_overlap(synthetic_gxe_df):
    split_df = make_leave_genotype_split(synthetic_gxe_df, n_folds=3, seed=1234, split_version="test")
    merged = split_df.merge(synthetic_gxe_df, on="sample_id")

    for _, fold_df in merged.groupby("outer_fold"):
        train_genotypes = set(fold_df.loc[fold_df["role"] == "train", "genotype_id"])
        val_genotypes = set(fold_df.loc[fold_df["role"] == "validation", "genotype_id"])
        test_genotypes = set(fold_df.loc[fold_df["role"] == "test", "genotype_id"])

        assert train_genotypes & test_genotypes == set()
        assert val_genotypes & test_genotypes == set()
        assert train_genotypes & val_genotypes == set()


def test_leave_environment_out_has_zero_train_test_environment_overlap(synthetic_gxe_df):
    split_df = make_leave_environment_split(
        synthetic_gxe_df, n_folds=3, seed=1234, split_version="test"
    )
    merged = split_df.merge(synthetic_gxe_df, on="sample_id")

    for _, fold_df in merged.groupby("outer_fold"):
        train_envs = set(fold_df.loc[fold_df["role"] == "train", "environment_id"])
        val_envs = set(fold_df.loc[fold_df["role"] == "validation", "environment_id"])
        test_envs = set(fold_df.loc[fold_df["role"] == "test", "environment_id"])

        assert train_envs & test_envs == set()
        assert val_envs & test_envs == set()
        assert train_envs & val_envs == set()


def test_forward_year_max_train_year_below_min_test_year(synthetic_gxe_df):
    split_df = make_forward_year_split(synthetic_gxe_df, seed=1234, split_version="test")
    merged = split_df.merge(synthetic_gxe_df, on="sample_id")

    for _, fold_df in merged.groupby("outer_fold"):
        train_years = fold_df.loc[fold_df["role"] == "train", "year"]
        val_years = fold_df.loc[fold_df["role"] == "validation", "year"]
        test_years = fold_df.loc[fold_df["role"] == "test", "year"]

        assert train_years.max() < val_years.min()
        assert val_years.max() < test_years.min()


def test_leave_ge_out_combos_absent_from_train_but_each_side_present(synthetic_gxe_df):
    split_df = make_leave_ge_split(synthetic_gxe_df, n_folds=3, seed=1234, split_version="test")
    merged = split_df.merge(synthetic_gxe_df, on="sample_id")

    for _, fold_df in merged.groupby("outer_fold"):
        train = fold_df[fold_df["role"] == "train"]
        test = fold_df[fold_df["role"] == "test"]
        assert len(test) > 0

        train_combos = set(zip(train["genotype_id"], train["environment_id"]))
        test_combos = set(zip(test["genotype_id"], test["environment_id"]))
        assert train_combos & test_combos == set(), "held-out G-E combo leaked into train"

        train_genotypes = set(train["genotype_id"])
        train_environments = set(train["environment_id"])
        for genotype_id, environment_id in test_combos:
            assert genotype_id in train_genotypes, (
                f"test genotype {genotype_id!r} does not appear in any training "
                "combo -- leave_ge_out requires it to appear elsewhere"
            )
            assert environment_id in train_environments, (
                f"test environment {environment_id!r} does not appear in any "
                "training combo -- leave_ge_out requires it to appear elsewhere"
            )
