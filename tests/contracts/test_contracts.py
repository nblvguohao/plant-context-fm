"""Contract tests for the canonical tables (TDD Section 10.2).

Each table gets: one valid fixture that must pass with zero violations, and
one broken fixture per constraint that must be caught. These are intentionally
tiny and synthetic — no real data is read here.
"""

import math

import pandas as pd
import pytest

from plant_context.data.contracts import (
    ContractViolation,
    check,
    validate_community_plot,
    validate_environment_daily,
    validate_genotype_marker,
    validate_phenotype_plot,
    validate_split_table,
)


def test_valid_phenotype_plot_has_no_violations():
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "plot_id": ["p1", "p2"],
            "genotype_id": ["g1", "g2"],
            "environment_id": ["e1", "e1"],
            "year": [2020, 2020],
            "location_id": ["loc1", "loc1"],
            "trait": ["yield", "yield"],
            "phenotype_value": [10.5, 11.2],
            "unit": ["t/ha", "t/ha"],
        }
    )
    assert validate_phenotype_plot(df) == []


def test_phenotype_plot_rejects_duplicate_sample_trait():
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s1"],
            "plot_id": ["p1", "p2"],
            "genotype_id": ["g1", "g2"],
            "environment_id": ["e1", "e1"],
            "year": [2020, 2020],
            "location_id": ["loc1", "loc1"],
            "trait": ["yield", "yield"],
            "phenotype_value": [10.5, 11.2],
            "unit": ["t/ha", "t/ha"],
        }
    )
    violations = validate_phenotype_plot(df)
    assert any("uniqueness" in v for v in violations)


def test_phenotype_plot_rejects_non_finite_value():
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "plot_id": ["p1", "p2"],
            "genotype_id": ["g1", "g2"],
            "environment_id": ["e1", "e1"],
            "year": [2020, 2020],
            "location_id": ["loc1", "loc1"],
            "trait": ["yield", "yield"],
            "phenotype_value": [10.5, math.inf],
            "unit": ["t/ha", "t/ha"],
        }
    )
    violations = validate_phenotype_plot(df)
    assert any("non-finite" in v for v in violations)


def test_phenotype_plot_missing_column_reported():
    df = pd.DataFrame({"sample_id": ["s1"]})
    violations = validate_phenotype_plot(df)
    assert any("missing required columns" in v for v in violations)


def test_valid_genotype_marker_has_no_violations():
    df = pd.DataFrame(
        {
            "genotype_id": ["g1", "g1"],
            "marker_id": ["m1", "m2"],
            "chromosome": [1, 1],
            "position": [100, 200],
            "reference_build": ["B73v5", "B73v5"],
            "allele_dosage": [0, 2],
        }
    )
    assert validate_genotype_marker(df) == []


def test_genotype_marker_rejects_out_of_range_dosage():
    df = pd.DataFrame(
        {
            "genotype_id": ["g1"],
            "marker_id": ["m1"],
            "chromosome": [1],
            "position": [100],
            "reference_build": ["B73v5"],
            "allele_dosage": [3],
        }
    )
    violations = validate_genotype_marker(df)
    assert any("allele_dosage" in v for v in violations)


def test_genotype_marker_rejects_null_reference_build():
    df = pd.DataFrame(
        {
            "genotype_id": ["g1"],
            "marker_id": ["m1"],
            "chromosome": [1],
            "position": [100],
            "reference_build": [None],
            "allele_dosage": [1],
        }
    )
    violations = validate_genotype_marker(df)
    assert any("reference_build" in v for v in violations)


def test_valid_environment_daily_has_no_violations():
    df = pd.DataFrame(
        {
            "environment_id": ["e1", "e1"],
            "date": ["2020-05-01", "2020-05-02"],
            "days_after_planting": [0, 1],
            "growth_stage": ["emergence", "emergence"],
            "missing_flag": [False, False],
        }
    )
    assert validate_environment_daily(df) == []


def test_environment_daily_rejects_negative_dap():
    df = pd.DataFrame(
        {
            "environment_id": ["e1"],
            "date": ["2020-05-01"],
            "days_after_planting": [-1],
            "growth_stage": ["emergence"],
            "missing_flag": [False],
        }
    )
    violations = validate_environment_daily(df)
    assert any("days_after_planting" in v for v in violations)


def test_environment_daily_rejects_duplicate_environment_date():
    df = pd.DataFrame(
        {
            "environment_id": ["e1", "e1"],
            "date": ["2020-05-01", "2020-05-01"],
            "days_after_planting": [0, 0],
            "growth_stage": ["emergence", "emergence"],
            "missing_flag": [False, False],
        }
    )
    violations = validate_environment_daily(df)
    assert any("uniqueness" in v for v in violations)


def test_valid_community_plot_has_no_violations():
    df = pd.DataFrame(
        {
            "plot_id": ["p1"],
            "survey_date": ["2019-06-01"],
            "latitude": [51.0],
            "longitude": [11.0],
            "species_id": ["Festuca brachyphylla"],
            "accepted_taxon_id": ["taxon_123"],
            "abundance": [10.0],
            "abundance_scale": ["CoverPerc"],
            "dataset_id": ["sPlotOpen"],
        }
    )
    assert validate_community_plot(df) == []


def test_community_plot_rejects_unresolved_taxon():
    df = pd.DataFrame(
        {
            "plot_id": ["p1"],
            "survey_date": ["2019-06-01"],
            "latitude": [51.0],
            "longitude": [11.0],
            "species_id": ["Unresolved sp."],
            "accepted_taxon_id": [None],
            "abundance": [10.0],
            "abundance_scale": ["CoverPerc"],
            "dataset_id": ["sPlotOpen"],
        }
    )
    violations = validate_community_plot(df)
    assert any("accepted_taxon_id" in v for v in violations)


def test_community_plot_rejects_invalid_coordinates():
    df = pd.DataFrame(
        {
            "plot_id": ["p1"],
            "survey_date": ["2019-06-01"],
            "latitude": [999.0],
            "longitude": [11.0],
            "species_id": ["Festuca brachyphylla"],
            "accepted_taxon_id": ["taxon_123"],
            "abundance": [10.0],
            "abundance_scale": ["CoverPerc"],
            "dataset_id": ["sPlotOpen"],
        }
    )
    violations = validate_community_plot(df)
    assert any("latitude" in v for v in violations)


def test_valid_split_table_has_no_violations():
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "outer_split_type": ["leave_genotype", "leave_genotype"],
            "outer_fold": [0, 0],
            "role": ["train", "test"],
            "seed": [1234, 1234],
            "group_key": ["g1", "g2"],
            "split_version": ["v1", "v1"],
        }
    )
    assert validate_split_table(df) == []


def test_split_table_rejects_invalid_role():
    df = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "outer_split_type": ["leave_genotype"],
            "outer_fold": [0],
            "role": ["holdout"],
            "seed": [1234],
            "group_key": ["g1"],
            "split_version": ["v1"],
        }
    )
    violations = validate_split_table(df)
    assert any("invalid role" in v for v in violations)


def test_check_raises_contract_violation_with_messages():
    with pytest.raises(ContractViolation, match="phenotype_plot"):
        check(["something is wrong"], table="phenotype_plot")


def test_check_passes_silently_when_no_violations():
    check([], table="phenotype_plot")
