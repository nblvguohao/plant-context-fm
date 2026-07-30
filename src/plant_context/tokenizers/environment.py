"""EnvironmentTokenizer: growth-stage segmentation and per-stage feature
aggregation from daily weather (TDD Section 5.3, 15 item 5).

Real phenological stage-transition dates are not available in the weather
data this project has integrated so far (G2F/FIP1 give daily weather, not
observed BBCH/VT/R-stage dates). Every stage boundary here is therefore
estimated from a cumulative growing-degree-day (GDD) rule, per TDD Section
5.3's fallback for missing phenological dates: "1. estimate via a GDD rule;
2. carry the estimate as an explicit flag; 3. sensitivity-test the
estimation error." This module implements (1) and always sets an explicit
``stage_estimation_method`` flag for (2); (3) is left to callers -- re-run
with a different ``stage_gdd_thresholds`` to see how sensitive downstream
results are, this module does not run that analysis itself.

Not computed here, because the daily weather tables integrated so far do
not carry the inputs needed to do so honestly: reference evapotranspiration
(ET0; the simplified formulas that need only temperature still need
latitude/day-of-year for extraterrestrial radiation, not wired up here),
water deficit (needs ET0 and a soil water balance), soil moisture, and
management events. VPD is computed, since it only needs mean temperature
and relative humidity, both already available.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Approximate maize growth-stage GDD (base 10C, upper cap 30C) thresholds
# drawn from general agronomic literature. These are illustrative defaults,
# not a value calibrated for any specific hybrid or company: override
# `stage_gdd_thresholds` with values appropriate to what is actually being
# modeled before treating stage assignments as more than approximate.
DEFAULT_STAGE_GDD_THRESHOLDS = {
    "emergence": 0.0,
    "vegetative": 100.0,
    "flowering": 700.0,
    "grain_filling": 1000.0,
    "maturity": 1600.0,
}

STAGE_ORDER = ("pre_planting", "emergence", "vegetative", "flowering", "grain_filling", "maturity")


def compute_gdd(
    tmax: pd.Series, tmin: pd.Series, base_temp: float = 10.0, upper_cap: float = 30.0
) -> pd.Series:
    """Modified/capped growing-degree-day formula (common for maize).

    GDD = max(0, (min(tmax, upper_cap) + max(tmin, base_temp)) / 2 - base_temp)
    """
    capped_max = tmax.clip(upper=upper_cap)
    floored_min = tmin.clip(lower=base_temp)
    gdd = (capped_max + floored_min) / 2 - base_temp
    return gdd.clip(lower=0.0)


def compute_vpd(tmean: pd.Series, relative_humidity: pd.Series) -> pd.Series:
    """Vapor pressure deficit (kPa) via the Tetens saturation-vapor-pressure formula.

    es = 0.6108 * exp(17.27 * T / (T + 237.3))  [kPa, T in Celsius]
    VPD = es * (1 - RH / 100)
    """
    es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
    return es * (1 - relative_humidity / 100.0)


def assign_growth_stage(
    env_daily_df: pd.DataFrame,
    stage_gdd_thresholds: Optional[dict] = None,
    base_temp: float = 10.0,
    upper_cap: float = 30.0,
) -> pd.DataFrame:
    """Add ``gdd``, ``cumulative_gdd``, ``growth_stage`` columns.

    ``cumulative_gdd`` accumulates within each environment_id starting from
    days_after_planting == 0. Rows with days_after_planting < 0 are labeled
    "pre_planting" and excluded from the accumulation, so a pre-planting hot
    or cold spell cannot shift post-planting stage boundaries.
    """
    thresholds = stage_gdd_thresholds or DEFAULT_STAGE_GDD_THRESHOLDS
    df = env_daily_df.sort_values(["environment_id", "days_after_planting"]).reset_index(drop=True)
    df["gdd"] = compute_gdd(df["tmax"], df["tmin"], base_temp=base_temp, upper_cap=upper_cap)

    post_planting = df["days_after_planting"] >= 0
    df["cumulative_gdd"] = np.nan
    df.loc[post_planting, "cumulative_gdd"] = df.loc[post_planting].groupby("environment_id")[
        "gdd"
    ].cumsum()

    stage_names_sorted = sorted(thresholds, key=lambda s: thresholds[s])
    stage_bounds = np.array([thresholds[s] for s in stage_names_sorted])

    df["growth_stage"] = "pre_planting"
    # A day with unknown daily GDD (upstream missing weather) would
    # otherwise get cumulative_gdd = NaN, and NaN sorts as "greater than
    # everything" for np.searchsorted -- that would bucket a single missing
    # day into a nonsensical late stage instead of near its neighbors. Use a
    # forward-filled cumulative_gdd only to choose the stage bucket; the
    # reported `cumulative_gdd` column itself is left as computed above, so
    # the missingness is still visible there.
    cum_gdd_for_staging = (
        df.loc[post_planting].groupby("environment_id")["cumulative_gdd"].ffill().fillna(0.0)
    ).to_numpy()
    stage_idx = np.searchsorted(stage_bounds, cum_gdd_for_staging, side="right") - 1
    stage_idx = np.clip(stage_idx, 0, len(stage_names_sorted) - 1)
    df.loc[post_planting, "growth_stage"] = np.array(stage_names_sorted)[stage_idx]

    df["stage_estimation_method"] = "gdd_rule"
    return df


def tokenize_environment_stages(
    env_daily_df: pd.DataFrame,
    stage_gdd_thresholds: Optional[dict] = None,
    base_temp: float = 10.0,
    upper_cap: float = 30.0,
) -> pd.DataFrame:
    """Aggregate daily weather into one row per (environment_id, growth_stage).

    Aggregation skips missing daily values (does not impute them) and
    reports the missing fraction explicitly per stage. Not imputing at all
    sidesteps TDD Section 10.1's concern about a missing flag and an
    imputed value silently drifting out of sync -- there is no imputed
    value here to drift.
    """
    staged = assign_growth_stage(
        env_daily_df,
        stage_gdd_thresholds=stage_gdd_thresholds,
        base_temp=base_temp,
        upper_cap=upper_cap,
    )
    staged["vpd"] = compute_vpd(staged["tmean"], staged["relative_humidity"])

    grouped = staged.groupby(["environment_id", "growth_stage"], sort=False)
    tokens = grouped.agg(
        n_days=("date", "count"),
        start_dap=("days_after_planting", "min"),
        end_dap=("days_after_planting", "max"),
        tmax_mean=("tmax", "mean"),
        tmax_max=("tmax", "max"),
        tmin_mean=("tmin", "mean"),
        tmin_min=("tmin", "min"),
        tmean_mean=("tmean", "mean"),
        gdd_sum=("gdd", "sum"),
        precipitation_sum=("precipitation", "sum"),
        solar_radiation_mean=("solar_radiation", "mean"),
        relative_humidity_mean=("relative_humidity", "mean"),
        vpd_mean=("vpd", "mean"),
        missing_fraction=("missing_flag", "mean"),
    ).reset_index()
    tokens["stage_estimation_method"] = "gdd_rule"
    return tokens
