"""Integration test: EnvironmentTokenizer against real G2F daily weather.

Skipped automatically if data/external/g2f isn't present -- see README.md.
"""

from pathlib import Path

import numpy as np
import pytest

from plant_context.data.g2f_adapter import load_g2f_environment_daily
from plant_context.tokenizers.environment import STAGE_ORDER, tokenize_environment_stages

G2F_ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "g2f"

pytestmark = pytest.mark.skipif(
    not (G2F_ROOT / "phenotype.parquet").exists(),
    reason="data/external/g2f is not present on this machine",
)


@pytest.fixture(scope="module")
def g2f_environment_daily():
    return load_g2f_environment_daily(G2F_ROOT)


def test_tokenize_environment_stages_runs_on_real_g2f_weather(g2f_environment_daily):
    tokens = tokenize_environment_stages(g2f_environment_daily)

    assert len(tokens) > 0
    assert set(tokens["growth_stage"]).issubset(set(STAGE_ORDER))
    assert (tokens["missing_fraction"] >= 0).all() and (tokens["missing_fraction"] <= 1).all()
    assert (tokens["gdd_sum"] >= 0).all()
    assert (tokens["n_days"] > 0).all()


def test_tokenize_environment_stages_preserves_total_day_count(g2f_environment_daily):
    tokens = tokenize_environment_stages(g2f_environment_daily)

    per_environment_total = tokens.groupby("environment_id")["n_days"].sum()
    raw_total = g2f_environment_daily.groupby("environment_id").size()
    common_envs = per_environment_total.index.intersection(raw_total.index)

    assert len(common_envs) > 0
    pd_equal = (per_environment_total.loc[common_envs] == raw_total.loc[common_envs]).all()
    assert pd_equal, "stage token day counts should sum back to the raw per-environment day count"


def test_tokenize_environment_stages_stage_order_is_chronological_per_environment(
    g2f_environment_daily,
):
    tokens = tokenize_environment_stages(g2f_environment_daily)
    order_lookup = {stage: i for i, stage in enumerate(STAGE_ORDER)}

    for _, env_tokens in tokens.groupby("environment_id"):
        stage_positions = env_tokens["growth_stage"].map(order_lookup).tolist()
        assert stage_positions == sorted(stage_positions), (
            "growth stages are not in chronological order within one environment"
        )


def test_gdd_is_finite_and_nonnegative_on_real_weather(g2f_environment_daily):
    from plant_context.tokenizers.environment import compute_gdd

    gdd = compute_gdd(g2f_environment_daily["tmax"], g2f_environment_daily["tmin"])
    finite = gdd.dropna()
    assert (finite >= 0).all()
    assert np.isfinite(finite).all()
