"""Data contracts for PlantContext-FM tables (TDD Section 4).

Each ``validate_*`` function checks structural and value constraints for one
canonical table and returns a list of human-readable violation strings. An
empty list means the table satisfies its contract. Callers that want a hard
failure should use :func:`check` to raise :class:`ContractViolation`.

These checks are deliberately table-local: they do not know about splits,
model fitting, or cross-table joins. Leakage checks live in
``tests/leakage`` instead (TDD Section 10.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


class ContractViolation(Exception):
    """Raised by :func:`check` when a table fails its data contract."""


def check(violations: list[str], *, table: str) -> None:
    if violations:
        joined = "\n".join(f"- {v}" for v in violations)
        raise ContractViolation(f"{table} failed its data contract:\n{joined}")


@dataclass(frozen=True)
class TableSchema:
    name: str
    required_columns: tuple[str, ...]

    def missing_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.required_columns if c not in df.columns]


PHENOTYPE_PLOT_SCHEMA = TableSchema(
    "phenotype_plot",
    (
        "sample_id", "plot_id", "genotype_id", "environment_id", "year",
        "location_id", "trait", "phenotype_value", "unit",
    ),
)

GENOTYPE_MARKER_SCHEMA = TableSchema(
    "genotype_marker",
    (
        "genotype_id", "marker_id", "chromosome", "position",
        "reference_build", "allele_dosage",
    ),
)

ENVIRONMENT_DAILY_SCHEMA = TableSchema(
    "environment_daily",
    (
        "environment_id", "date", "days_after_planting", "growth_stage",
        "missing_flag",
    ),
)

COMMUNITY_PLOT_SCHEMA = TableSchema(
    "community_plot",
    (
        "plot_id", "survey_date", "latitude", "longitude", "species_id",
        "accepted_taxon_id", "abundance", "abundance_scale", "dataset_id",
    ),
)

SPLIT_TABLE_SCHEMA = TableSchema(
    "split_table",
    (
        "sample_id", "outer_split_type", "outer_fold", "role", "seed",
        "group_key", "split_version",
    ),
)

ALLOWED_SPLIT_ROLES = frozenset({"train", "validation", "test"})


def _duplicate_key_violations(df: pd.DataFrame, key_cols: list[str], table: str) -> list[str]:
    if not all(c in df.columns for c in key_cols):
        return []
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        n = int(dup_mask.sum())
        return [f"{table}: {n} rows violate uniqueness of key {key_cols}"]
    return []


def validate_phenotype_plot(df: pd.DataFrame) -> list[str]:
    violations: list[str] = []
    missing = PHENOTYPE_PLOT_SCHEMA.missing_columns(df)
    if missing:
        return [f"phenotype_plot: missing required columns {missing}"]

    violations += _duplicate_key_violations(df, ["sample_id", "trait"], "phenotype_plot")

    non_finite = ~df["phenotype_value"].apply(
        lambda v: isinstance(v, (int, float)) and math.isfinite(v)
    )
    if non_finite.any():
        violations.append(
            f"phenotype_plot: {int(non_finite.sum())} rows have a non-finite "
            "phenotype_value (NaN/inf not allowed; use qc_flag exclusion instead)"
        )

    if df["unit"].isna().any():
        violations.append("phenotype_plot: unit must not be null")

    return violations


def validate_genotype_marker(df: pd.DataFrame) -> list[str]:
    violations: list[str] = []
    missing = GENOTYPE_MARKER_SCHEMA.missing_columns(df)
    if missing:
        return [f"genotype_marker: missing required columns {missing}"]

    if df["reference_build"].isna().any():
        violations.append("genotype_marker: reference_build must not be null")

    if df["chromosome"].isna().any() or df["position"].isna().any():
        violations.append("genotype_marker: chromosome/position must not be null")

    dosage_col = df["allele_dosage"]
    numeric_dosage = pd.to_numeric(dosage_col, errors="coerce")
    out_of_range = numeric_dosage.notna() & ((numeric_dosage < 0) | (numeric_dosage > 2))
    if out_of_range.any():
        violations.append(
            f"genotype_marker: {int(out_of_range.sum())} rows have allele_dosage "
            "outside [0, 2]"
        )

    return violations


def validate_environment_daily(df: pd.DataFrame) -> list[str]:
    violations: list[str] = []
    missing = ENVIRONMENT_DAILY_SCHEMA.missing_columns(df)
    if missing:
        return [f"environment_daily: missing required columns {missing}"]

    if df["missing_flag"].isna().any():
        violations.append("environment_daily: missing_flag must not be null")

    if (pd.to_numeric(df["days_after_planting"], errors="coerce") < 0).any():
        violations.append("environment_daily: days_after_planting must be >= 0")

    violations += _duplicate_key_violations(
        df, ["environment_id", "date"], "environment_daily"
    )

    return violations


def validate_community_plot(df: pd.DataFrame) -> list[str]:
    violations: list[str] = []
    missing = COMMUNITY_PLOT_SCHEMA.missing_columns(df)
    if missing:
        return [f"community_plot: missing required columns {missing}"]

    if df["accepted_taxon_id"].isna().any():
        n = int(df["accepted_taxon_id"].isna().sum())
        violations.append(
            f"community_plot: {n} rows have no accepted_taxon_id "
            "(species_id must be resolved against a frozen taxonomy before use)"
        )

    lat_bad = ~df["latitude"].between(-90, 90)
    lon_bad = ~df["longitude"].between(-180, 180)
    if lat_bad.any() or lon_bad.any():
        violations.append(
            f"community_plot: {int(lat_bad.sum())} rows with invalid latitude, "
            f"{int(lon_bad.sum())} rows with invalid longitude"
        )

    return violations


def validate_split_table(df: pd.DataFrame) -> list[str]:
    violations: list[str] = []
    missing = SPLIT_TABLE_SCHEMA.missing_columns(df)
    if missing:
        return [f"split_table: missing required columns {missing}"]

    bad_role = ~df["role"].isin(ALLOWED_SPLIT_ROLES)
    if bad_role.any():
        bad_values = sorted(df.loc[bad_role, "role"].unique().tolist())
        violations.append(f"split_table: invalid role values {bad_values}")

    violations += _duplicate_key_violations(
        df, ["sample_id", "outer_split_type", "outer_fold", "seed"], "split_table"
    )

    return violations
