"""Integration test: the four split protocols against the real G2F phenotype table.

Skipped automatically if data/external/g2f isn't present -- see README.md.
"""

from pathlib import Path

import pytest

from plant_context.data.contracts import validate_split_table
from plant_context.data.g2f_adapter import load_g2f_phenotype_plot
from plant_context.evaluation.splits import (
    make_forward_year_split,
    make_leave_environment_split,
    make_leave_genotype_split,
    make_leave_ge_split,
)

G2F_ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "g2f"

pytestmark = pytest.mark.skipif(
    not (G2F_ROOT / "phenotype.parquet").exists(),
    reason="data/external/g2f is not present on this machine",
)


@pytest.fixture(scope="module")
def g2f_phenotype():
    return load_g2f_phenotype_plot(G2F_ROOT)


def test_leave_genotype_split_on_real_g2f_is_leakage_safe(g2f_phenotype):
    split_df = make_leave_genotype_split(g2f_phenotype, n_folds=5, seed=1234, split_version="g2f_v1")
    assert validate_split_table(split_df) == []

    merged = split_df.merge(g2f_phenotype, on="sample_id")
    for _, fold_df in merged.groupby("outer_fold"):
        train_g = set(fold_df.loc[fold_df["role"] == "train", "genotype_id"])
        test_g = set(fold_df.loc[fold_df["role"] == "test", "genotype_id"])
        assert train_g & test_g == set()


def test_leave_environment_split_on_real_g2f_is_leakage_safe(g2f_phenotype):
    split_df = make_leave_environment_split(
        g2f_phenotype, n_folds=5, seed=1234, split_version="g2f_v1"
    )
    assert validate_split_table(split_df) == []

    merged = split_df.merge(g2f_phenotype, on="sample_id")
    for _, fold_df in merged.groupby("outer_fold"):
        train_e = set(fold_df.loc[fold_df["role"] == "train", "environment_id"])
        test_e = set(fold_df.loc[fold_df["role"] == "test", "environment_id"])
        assert train_e & test_e == set()


def test_forward_year_split_on_real_g2f_is_leakage_safe(g2f_phenotype):
    # G2F spans 2014-2023 (10 distinct years), so this should produce 8 folds
    # (years[2:]).
    split_df = make_forward_year_split(g2f_phenotype, seed=1234, split_version="g2f_v1")
    assert validate_split_table(split_df) == []
    assert g2f_phenotype["year"].nunique() - 2 == split_df["outer_fold"].nunique()

    merged = split_df.merge(g2f_phenotype, on="sample_id")
    for _, fold_df in merged.groupby("outer_fold"):
        train_years = fold_df.loc[fold_df["role"] == "train", "year"]
        test_years = fold_df.loc[fold_df["role"] == "test", "year"]
        assert train_years.max() < test_years.min()


def test_leave_ge_split_on_real_g2f_is_leakage_safe(g2f_phenotype):
    split_df = make_leave_ge_split(g2f_phenotype, n_folds=5, seed=1234, split_version="g2f_v1")
    assert validate_split_table(split_df) == []

    merged = split_df.merge(g2f_phenotype, on="sample_id")
    for _, fold_df in merged.groupby("outer_fold"):
        train = fold_df[fold_df["role"] == "train"]
        test = fold_df[fold_df["role"] == "test"]
        assert len(test) > 0

        train_combos = set(zip(train["genotype_id"], train["environment_id"]))
        test_combos = set(zip(test["genotype_id"], test["environment_id"]))
        assert train_combos & test_combos == set()

        train_genotypes = set(train["genotype_id"])
        train_environments = set(train["environment_id"])
        for genotype_id, environment_id in test_combos:
            assert genotype_id in train_genotypes
            assert environment_id in train_environments
