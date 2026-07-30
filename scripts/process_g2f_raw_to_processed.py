"""Process raw G2F 2020-2023 tables into the processed parquet contract
expected by ``g2f_adapter.py``.

This is a first-pass adapter for the raw public G2F release files. It builds
the four parquet tables:

- ``environment.parquet``: environment_id, year, estimated_planting_doy
- ``phenotype.parquet``: sample_id, plot_id, genotype_id, environment_id,
  year, location_id, trait_id, phenotype_value, phenotype_unit,
  replicate_id, block_id
- ``weather_daily.parquet``: environment_id, date, tmax, tmin, tmean,
  precipitation, solar_radiation, relative_humidity
- ``genotype.parquet``: placeholder empty table until the 3.8 GB VCF is
  parsed separately.

Usage:
    source /f/Anaconda/etc/profile.d/conda.sh
    conda activate tree-py310
    PYTHONPATH=src python scripts/process_g2f_raw_to_processed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
OUT_ROOT = PROJECT_ROOT / "data" / "external" / "g2f"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

YEARS = [2020, 2021, 2022, 2023]
TRAITS = [
    ("Anthesis [days]", "anthesis_days", "days"),
    ("Silking [days]", "silking_days", "days"),
    ("Plant Height [cm]", "plant_height_cm", "cm"),
    ("Ear Height [cm]", "ear_height_cm", "cm"),
    ("Stand Count [# of plants]", "stand_count", "count"),
    ("Root Lodging [# of plants]", "root_lodging", "count"),
    ("Stalk Lodging [# of plants]", "stalk_lodging", "count"),
    ("Grain Moisture [%]", "grain_moisture_pct", "%"),
    ("Test Weight [lbs]", "test_weight_lbs", "lbs"),
    ("Plot Weight [lbs]", "plot_weight_lbs", "lbs"),
    ("Grain Yield (bu/A)", "grain_yield_bu_per_acre", "bu/A"),
]


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV with fallback encodings for G2F's mixed encodings."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, encoding="latin-1", errors="replace")


def build_environment_table() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        fname = RAW_ROOT / f"g2f_{year}_phenotypic_clean_data.csv"
        if not fname.exists():
            print(f"  skipping {fname} (missing)")
            continue
        df = _read_csv_robust(fname)
        df["environment_id"] = df["Field-Location"].astype(str) + "_" + df["Year"].astype(int).astype(str)
        planted = pd.to_datetime(df["Date Plot Planted [MM/DD/YY]"], format="%m/%d/%y", errors="coerce")
        env = pd.DataFrame(
            {
                "environment_id": df["environment_id"],
                "year": df["Year"].astype(int),
                "estimated_planting_doy": planted.dt.dayofyear,
            }
        ).dropna(subset=["estimated_planting_doy"]).drop_duplicates("environment_id")
        rows.append(env)
    env_df = pd.concat(rows, ignore_index=True)
    print(f"  environment.parquet: {len(env_df)} rows")
    return env_df


def build_phenotype_table() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        fname = RAW_ROOT / f"g2f_{year}_phenotypic_clean_data.csv"
        if not fname.exists():
            continue
        df = _read_csv_robust(fname)
        df["environment_id"] = df["Field-Location"].astype(str) + "_" + df["Year"].astype(int).astype(str)
        df["genotype_id"] = df["Pedigree"].astype(str)
        df["plot_id"] = (
            df["Plot_ID"].astype(str)
            + "_"
            + df["Range"].fillna("").astype(str)
            + "_"
            + df["Pass"].fillna("").astype(str)
        )
        df["replicate_id"] = df["Replicate"].astype(str)
        df["block_id"] = df["Block"].astype(str)
        df["location_id"] = df["Field-Location"].astype(str)

        for col, trait_id, unit in TRAITS:
            if col not in df.columns:
                continue
            sub = df[["environment_id", "plot_id", "genotype_id", "Year", "location_id", "replicate_id", "block_id"]].copy()
            sub["trait_id"] = trait_id
            sub["phenotype_value"] = pd.to_numeric(df[col], errors="coerce")
            sub["phenotype_unit"] = unit
            sub = sub.dropna(subset=["phenotype_value"])
            sub["sample_id"] = sub["plot_id"] + "_" + sub["trait_id"]
            sub["year"] = sub["Year"].astype(int)
            rows.append(sub[
                ["sample_id", "plot_id", "genotype_id", "environment_id", "year",
                 "location_id", "trait_id", "phenotype_value", "phenotype_unit",
                 "replicate_id", "block_id"]
            ])
    phen_df = pd.concat(rows, ignore_index=True)
    # Drop exact duplicate sample_id+trait_id pairs (should be none after
    # disambiguating plot_id, but keep as a safety net)
    phen_df = phen_df.drop_duplicates(subset=["sample_id", "trait_id"], keep="first")
    print(f"  phenotype.parquet: {len(phen_df)} rows")
    return phen_df


def _weather_file_for_year(year: int) -> Path:
    if year == 2020:
        return RAW_ROOT / "2020_weather_cleaned.csv"
    return RAW_ROOT / f"g2f_{year}_weather_cleaned.csv"


def build_weather_table(env_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        fname = _weather_file_for_year(year)
        if not fname.exists():
            print(f"  skipping {fname} (missing)")
            continue
        df = _read_csv_robust(fname)
        # Standardize environment id
        loc_col = "Field Location"
        year_series = pd.to_numeric(df["Year"], errors="coerce")
        valid = year_series.notna()
        df = df.loc[valid].copy()
        df["environment_id"] = df[loc_col].astype(str) + "_" + year_series.loc[valid].astype(int).astype(str)
        df["date"] = pd.to_datetime(
            year_series.loc[valid].astype(int).astype(str)
            + "-"
            + pd.to_numeric(df["Month"], errors="coerce").astype("Int64").astype(str)
            + "-"
            + pd.to_numeric(df["Day"], errors="coerce").astype("Int64").astype(str),
            format="%Y-%m-%d", errors="coerce",
        )
        df = df.dropna(subset=["date"])

        # Aggregate 15-min measurements to daily
        daily = df.groupby(["environment_id", "date"], as_index=False).agg(
            tmax=("Temperature [C]", "max"),
            tmin=("Temperature [C]", "min"),
            tmean=("Temperature [C]", "mean"),
            precipitation=("Rainfall [mm]", "sum"),
            solar_radiation=("Solar Radiation [W/m2]", "mean"),
            relative_humidity=("Relative Humidity [%]", "mean"),
        )
        rows.append(daily)
    weather_df = pd.concat(rows, ignore_index=True)
    # Keep only environments that also have planting metadata
    weather_df = weather_df[weather_df["environment_id"].isin(env_df["environment_id"])]
    print(f"  weather_daily.parquet: {len(weather_df)} rows")
    return weather_df


def main():
    print("Building environment.parquet ...")
    env_df = build_environment_table()
    env_df.to_parquet(OUT_ROOT / "environment.parquet", index=False)

    print("Building phenotype.parquet ...")
    phen_df = build_phenotype_table()
    phen_df.to_parquet(OUT_ROOT / "phenotype.parquet", index=False)

    print("Building weather_daily.parquet ...")
    weather_df = build_weather_table(env_df)
    weather_df.to_parquet(OUT_ROOT / "weather_daily.parquet", index=False)

    print("Building placeholder genotype.parquet ...")
    gen_df = pd.DataFrame(
        columns=["genotype_id", "marker_id", "chromosome", "position", "allele_dosage"]
    )
    gen_df.to_parquet(OUT_ROOT / "genotype.parquet", index=False)

    print(f"Processed tables written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
