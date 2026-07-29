"""Unit tests for the four split protocols (TDD Section 8), synthetic fixture.

Leakage-specific invariants (TDD Section 10.3) live in
tests/leakage/test_split_leakage.py instead, to keep that gate separately
runnable/reportable from general correctness checks.
"""

import pandas as pd
import pytest

from plant_context.data.contracts import validate_split_table
from plant_context.evaluation.splits import (
    make_forward_year_split,
    make_leave_environment_split,
    make_leave_genotype_split,
    make_leave_ge_split,
)


def test_leave_genotype_split_satisfies_contract(synthetic_gxe_df):
    split_df = make_leave_genotype_split(synthetic_gxe_df, n_folds=3, seed=1234, split_version="test")
    assert validate_split_table(split_df) == []
    assert set(split_df["outer_fold"]) == {0, 1, 2}


def test_leave_genotype_split_every_fold_holds_something_out(synthetic_gxe_df):
    split_df = make_leave_genotype_split(synthetic_gxe_df, n_folds=3, seed=1234, split_version="test")
    test_counts = split_df[split_df["role"] == "test"].groupby("outer_fold").size()
    assert (test_counts > 0).all()
    val_counts = split_df[split_df["role"] == "validation"].groupby("outer_fold").size()
    assert (val_counts > 0).all()


def test_leave_environment_split_satisfies_contract(synthetic_gxe_df):
    split_df = make_leave_environment_split(synthetic_gxe_df, n_folds=3, seed=1234, split_version="test")
    assert validate_split_table(split_df) == []


def test_forward_year_split_satisfies_contract(synthetic_gxe_df):
    split_df = make_forward_year_split(synthetic_gxe_df, seed=1234, split_version="test")
    assert validate_split_table(split_df) == []
    # years are [2018, 2019, 2020, 2021]; only 2020 and 2021 can be test years.
    assert set(split_df["outer_fold"]) == {0, 1}


def test_forward_year_split_requires_at_least_three_years(synthetic_gxe_df):
    only_two_years = synthetic_gxe_df[synthetic_gxe_df["year"].isin([2018, 2019])]
    with pytest.raises(ValueError):
        make_forward_year_split(only_two_years, seed=1234, split_version="test")


def test_leave_ge_split_satisfies_contract(synthetic_gxe_df):
    split_df = make_leave_ge_split(synthetic_gxe_df, n_folds=3, seed=1234, split_version="test")
    assert validate_split_table(split_df) == []


def test_leave_ge_split_holds_something_out_every_fold(synthetic_gxe_df):
    split_df = make_leave_ge_split(synthetic_gxe_df, n_folds=3, seed=1234, split_version="test")
    counts = split_df[split_df["role"] == "test"].groupby("outer_fold").size()
    assert (counts > 0).all()


def test_split_functions_are_deterministic_given_same_seed(synthetic_gxe_df):
    a = make_leave_genotype_split(synthetic_gxe_df, n_folds=3, seed=42, split_version="test")
    b = make_leave_genotype_split(synthetic_gxe_df, n_folds=3, seed=42, split_version="test")
    pd.testing.assert_frame_equal(a, b)


def test_split_functions_differ_across_seeds(synthetic_gxe_df):
    a = make_leave_genotype_split(synthetic_gxe_df, n_folds=3, seed=1, split_version="test")
    b = make_leave_genotype_split(synthetic_gxe_df, n_folds=3, seed=2, split_version="test")
    assert not a["role"].equals(b["role"])
