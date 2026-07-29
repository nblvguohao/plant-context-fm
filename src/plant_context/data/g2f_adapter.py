"""G2F adapter (TDD Section 15, implementation-order item 2).

Converts the G2F tables borrowed read-only from the sibling SRG-GxE data
pipeline (``data/external/g2f``, see ``data/manifests/``) into this
project's ``phenotype_plot`` / ``genotype_marker`` / ``environment_daily``
contracts (TDD Section 4).

No field the source data does not provide is guessed. Where a contract
column has no real source (``reference_build``, ``growth_stage``, plot
``row``/``column``), it is filled with an explicit missing/placeholder value
and documented below, rather than invented, so that downstream code can tell
the difference between "measured" and "unknown".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WEATHER_COLUMNS = (
    "tmax", "tmin", "tmean", "precipitation", "solar_radiation", "relative_humidity",
)

# The source data (data/external/g2f/data_manifest.yaml) does not record
# which maize reference genome build the G2F SNP array positions are
# aligned to. Do not rely on this placeholder for any cross-build position
# comparison until a real build id is confirmed from the array documentation.
UNKNOWN_REFERENCE_BUILD = "G2F_unspecified"


def phenotype_to_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw G2F phenotype table onto the phenotype_plot contract.

    ``row``/``column`` are not present in the source table (it has
    plot/block/replicate ids instead of a plot grid position) and are left
    null rather than backfilled.
    """
    return pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "plot_id": df["plot_id"].astype(str),
            "genotype_id": df["genotype_id"],
            "environment_id": df["environment_id"],
            "year": df["year"].astype(int),
            "location_id": df["location_id"],
            "trait": df["trait_id"],
            "phenotype_value": df["phenotype_value"].astype(float),
            "unit": df["phenotype_unit"],
            "replicate": df["replicate_id"].astype(str),
            "block": df["block_id"].astype(str),
            "row": pd.Series(pd.NA, index=df.index, dtype="Int64"),
            "column": pd.Series(pd.NA, index=df.index, dtype="Int64"),
        }
    )


def genotype_to_contract(
    df: pd.DataFrame, reference_build: str = UNKNOWN_REFERENCE_BUILD
) -> pd.DataFrame:
    """Map the raw G2F genotype table onto the genotype_marker contract."""
    out = df[["genotype_id", "marker_id", "chromosome", "position", "allele_dosage"]].copy()
    out["reference_build"] = reference_build
    return out[
        ["genotype_id", "marker_id", "chromosome", "position", "reference_build", "allele_dosage"]
    ]


def environment_daily_to_contract(
    weather_df: pd.DataFrame, environment_df: pd.DataFrame
) -> pd.DataFrame:
    """Map raw G2F weather + environment tables onto the environment_daily contract.

    ``growth_stage`` is left null: the source data has no phenological stage
    boundaries, and TDD Section 5.3 requires those to come from a GDD rule
    plus an explicit estimation-uncertainty flag, not a guess made here. Rows
    before the estimated planting date are dropped so that
    ``days_after_planting >= 0`` holds everywhere, per TDD Section 4.3 (dates
    must fall within the planting-harvest window); there is no harvest date
    in the source data, so the upper bound is not enforced here.
    """
    env_meta = (
        environment_df[["environment_id", "year", "estimated_planting_doy"]]
        .dropna(subset=["estimated_planting_doy"])
        .drop_duplicates(subset=["environment_id"])
        .copy()
    )
    env_meta["planting_date"] = pd.to_datetime(
        env_meta["year"].astype(int).astype(str) + "-01-01"
    ) + pd.to_timedelta(env_meta["estimated_planting_doy"].astype(int) - 1, unit="D")

    merged = weather_df.merge(
        env_meta[["environment_id", "planting_date"]], on="environment_id", how="inner"
    )
    merged["days_after_planting"] = (merged["date"] - merged["planting_date"]).dt.days
    merged = merged[merged["days_after_planting"] >= 0].copy()
    merged["growth_stage"] = pd.Series(pd.NA, index=merged.index, dtype="object")
    merged["missing_flag"] = merged[list(WEATHER_COLUMNS)].isna().any(axis=1)

    return merged[
        ["environment_id", "date", "days_after_planting", "growth_stage", "missing_flag"]
        + list(WEATHER_COLUMNS)
    ]


def load_g2f_phenotype_plot(root: Path | str) -> pd.DataFrame:
    root = Path(root)
    df = pd.read_parquet(root / "phenotype.parquet")
    return phenotype_to_contract(df)


def load_g2f_genotype_marker(
    root: Path | str, reference_build: str = UNKNOWN_REFERENCE_BUILD
) -> pd.DataFrame:
    root = Path(root)
    df = pd.read_parquet(root / "genotype.parquet")
    return genotype_to_contract(df, reference_build=reference_build)


def load_g2f_environment_daily(root: Path | str) -> pd.DataFrame:
    root = Path(root)
    weather_df = pd.read_parquet(root / "weather_daily.parquet")
    environment_df = pd.read_parquet(root / "environment.parquet")
    return environment_daily_to_contract(weather_df, environment_df)
