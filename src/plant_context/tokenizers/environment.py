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


def audit_phenological_date_coverage(
    env_daily_df: pd.DataFrame,
    stage_gdd_thresholds: Optional[dict] = None,
    estimation_warning_threshold: float = 0.5,
) -> dict:
    """Audit how phenological stage boundaries were determined.

    TDD Section 5.3 requires reporting: the proportion of environments
    whose stage dates are observed vs estimated, the estimation method,
    the distribution of estimated boundaries, and a warning if the
    estimation rate is high enough to cause systematic stage-boundary
    shifts.

    In the current implementation all stage boundaries are estimated from
    a GDD rule (``stage_estimation_method == "gdd_rule"``). Future
    observed-date integration should add a column such as
    ``stage_boundary_source`` with values "observed" / "estimated" and
    this audit will then report the observed fraction automatically.
    """
    if env_daily_df.empty:
        return {
            "n_environments": 0,
            "stage_estimation_method": None,
            "fraction_estimated": float("nan"),
            "fraction_observed": float("nan"),
            "stage_boundary_days": [],
            "stage_boundary_gdd": [],
            "high_estimation_rate_warning": False,
        }

    n_environments = env_daily_df["environment_id"].nunique()

    # Detect source of stage boundaries. If an explicit source column is
    # present, use it; otherwise every boundary is assumed to be estimated
    # from the GDD rule (the only method implemented in this module).
    if "stage_boundary_source" in env_daily_df.columns:
        source_counts = env_daily_df.groupby("environment_id")["stage_boundary_source"].first().value_counts()
        n_observed = int(source_counts.get("observed", 0))
        n_estimated = int(source_counts.get("estimated", 0))
        method = "mixed"
    else:
        n_observed = 0
        n_estimated = n_environments
        method = "gdd_rule"

    fraction_estimated = n_estimated / n_environments if n_environments > 0 else float("nan")
    fraction_observed = n_observed / n_environments if n_environments > 0 else float("nan")

    # Compute boundary distributions using the same GDD rule the tokenizer uses.
    staged = assign_growth_stage(
        env_daily_df,
        stage_gdd_thresholds=stage_gdd_thresholds,
    )

    boundary_days = []
    boundary_gdd = []
    for env_id, env_df in staged.groupby("environment_id"):
        env_df = env_df.sort_values("days_after_planting")
        transitions = env_df["growth_stage"] != env_df["growth_stage"].shift(1)
        transition_rows = env_df[transitions & (env_df["days_after_planting"] >= 0)]
        boundary_days.extend(transition_rows["days_after_planting"].dropna().tolist())
        boundary_gdd.extend(transition_rows["cumulative_gdd"].dropna().tolist())

    return {
        "n_environments": n_environments,
        "stage_estimation_method": method,
        "fraction_estimated": fraction_estimated,
        "fraction_observed": fraction_observed,
        "stage_boundary_days": boundary_days,
        "stage_boundary_gdd": boundary_gdd,
        "high_estimation_rate_warning": fraction_estimated > estimation_warning_threshold,
    }
