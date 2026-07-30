"""Unit tests for EnvironmentTokenizer (TDD Section 5.3, 10.1).

Covers the literal 10.1 EnvironmentTokenizer bullets that apply to this
module: DAP/GDD monotonicity, deterministic stage boundaries, and missing
flag / (absence of) imputation staying in sync. Structured masking and
scaler-fit-on-test-year checks belong to later steps (masking, model
training) and are not tested here -- see the module docstring for why.
"""

import numpy as np
import pandas as pd
import pytest

from plant_context.tokenizers.environment import (
    STAGE_ORDER,
    assign_growth_stage,
    compute_gdd,
    compute_vpd,
    tokenize_environment_stages,
)


def _single_environment_series(tmax_values, tmin_values, start_dap=0):
    n = len(tmax_values)
    return pd.DataFrame(
        {
            "environment_id": ["E1"] * n,
            "date": pd.date_range("2020-05-01", periods=n, freq="D"),
            "days_after_planting": list(range(start_dap, start_dap + n)),
            "growth_stage": [None] * n,
            "missing_flag": [False] * n,
            "tmax": tmax_values,
            "tmin": tmin_values,
            "tmean": [(a + b) / 2 for a, b in zip(tmax_values, tmin_values)],
            "precipitation": [0.0] * n,
            "solar_radiation": [10.0] * n,
            "relative_humidity": [60.0] * n,
        }
    )


def test_compute_gdd_is_never_negative():
    tmax = pd.Series([5.0, 15.0, 40.0])
    tmin = pd.Series([-5.0, 5.0, 20.0])
    gdd = compute_gdd(tmax, tmin, base_temp=10.0, upper_cap=30.0)
    assert (gdd >= 0).all()


def test_gdd_and_cumulative_gdd_are_monotonic_over_days_after_planting():
    # A steadily warming profile: GDD should never be negative, so
    # cumulative GDD over increasing days_after_planting must be
    # non-decreasing.
    tmax = [20.0 + i for i in range(30)]
    tmin = [10.0 + i * 0.5 for i in range(30)]
    df = _single_environment_series(tmax, tmin)
    staged = assign_growth_stage(df)

    assert (staged["gdd"] >= 0).all()
    cumulative = staged.sort_values("days_after_planting")["cumulative_gdd"].to_numpy()
    assert np.all(np.diff(cumulative) >= 0)


def test_assign_growth_stage_is_deterministic():
    tmax = [20.0 + i for i in range(40)]
    tmin = [10.0 + i * 0.5 for i in range(40)]
    df = _single_environment_series(tmax, tmin)
    first = assign_growth_stage(df)["growth_stage"].tolist()
    second = assign_growth_stage(df)["growth_stage"].tolist()
    assert first == second


def test_assign_growth_stage_transitions_forward_only():
    tmax = [25.0 + i for i in range(60)]
    tmin = [15.0 + i * 0.3 for i in range(60)]
    df = _single_environment_series(tmax, tmin)
    staged = assign_growth_stage(df).sort_values("days_after_planting")

    seen_indices = [STAGE_ORDER.index(s) for s in staged["growth_stage"]]
    assert all(b >= a for a, b in zip(seen_indices, seen_indices[1:])), (
        "growth stage regressed backward as days_after_planting increased"
    )
    # With a steadily warming 60-day profile crossing all default
    # thresholds, expect to see more than just the first stage.
    assert len(set(staged["growth_stage"])) > 1


def test_assign_growth_stage_labels_pre_planting_and_excludes_it_from_accumulation():
    df = _single_environment_series([20.0] * 5, [10.0] * 5, start_dap=-3)
    staged = assign_growth_stage(df).sort_values("days_after_planting")

    pre_planting_rows = staged[staged["days_after_planting"] < 0]
    assert (pre_planting_rows["growth_stage"] == "pre_planting").all()
    assert pre_planting_rows["cumulative_gdd"].isna().all()

    day0_row = staged[staged["days_after_planting"] == 0].iloc[0]
    day0_gdd = staged[staged["days_after_planting"] == 0]["gdd"].iloc[0]
    assert day0_row["cumulative_gdd"] == pytest.approx(day0_gdd)


def test_missing_daily_temperature_does_not_derail_stage_assignment():
    # A single missing tmax in the middle of an otherwise-flat, cool
    # profile (comfortably within one stage) must not get shoved into some
    # unrelated later stage just because its cumulative_gdd is NaN --
    # np.searchsorted treats NaN as "greater than everything", which would
    # otherwise silently bucket it into the last stage.
    tmax = [15.0, 15.0, np.nan, 15.0, 15.0]
    tmin = [8.0, 8.0, 8.0, 8.0, 8.0]
    df = _single_environment_series(tmax, tmin)
    staged = assign_growth_stage(df).sort_values("days_after_planting")

    stages = staged["growth_stage"].tolist()
    assert len(set(stages)) == 1, f"missing day was bucketed into a different stage: {stages}"
    # The raw cumulative_gdd for the missing day is still honestly NaN --
    # only the internal stage-bucket decision was forward-filled, not the
    # reported value.
    missing_row = staged[staged["tmax"].isna()].iloc[0]
    assert np.isnan(missing_row["cumulative_gdd"])


def test_compute_vpd_matches_hand_calculation():
    tmean = pd.Series([25.0])
    rh = pd.Series([50.0])
    vpd = compute_vpd(tmean, rh)
    es = 0.6108 * np.exp(17.27 * 25.0 / (25.0 + 237.3))
    expected = es * 0.5
    assert vpd.iloc[0] == pytest.approx(expected)


def test_different_thresholds_change_stage_boundaries():
    tmax = [25.0 + i for i in range(60)]
    tmin = [15.0 + i * 0.3 for i in range(60)]
    df = _single_environment_series(tmax, tmin)

    default_stages = assign_growth_stage(df)["growth_stage"].tolist()
    shifted_thresholds = {
        "emergence": 0.0,
        "vegetative": 100.0,
        "flowering": 5000.0,  # pushed far out -> flowering should never be reached
        "grain_filling": 6000.0,
        "maturity": 7000.0,
    }
    shifted_stages = assign_growth_stage(df, stage_gdd_thresholds=shifted_thresholds)[
        "growth_stage"
    ].tolist()

    assert "flowering" not in shifted_stages
    assert default_stages != shifted_stages


def test_tokenize_environment_stages_aggregates_correctly():
    df = _single_environment_series([20.0, 20.0, 30.0, 30.0], [10.0, 10.0, 15.0, 15.0])
    tokens = tokenize_environment_stages(df)

    assert set(tokens["environment_id"]) == {"E1"}
    total_days = tokens["n_days"].sum()
    assert total_days == 4
    # Every reported stage must actually appear in STAGE_ORDER.
    assert set(tokens["growth_stage"]).issubset(set(STAGE_ORDER))


def test_tokenize_environment_stages_skips_missing_values_without_fabricating():
    df = _single_environment_series([20.0, np.nan, 20.0], [10.0, 10.0, 10.0])
    df["missing_flag"] = [False, True, False]
    tokens = tokenize_environment_stages(df)

    stage_row = tokens.iloc[0]
    # tmax_mean should be the mean of the two non-null values (20, 20), not
    # a mean that treats the missing value as 0 or otherwise fabricates it.
    assert stage_row["tmax_mean"] == pytest.approx(20.0)
    assert stage_row["missing_fraction"] == pytest.approx(1 / 3)
