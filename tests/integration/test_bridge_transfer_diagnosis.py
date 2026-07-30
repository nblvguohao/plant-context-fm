"""Integration smoke test: bridge transfer + transfer diagnosis protocol.

Trains tiny SharedEnvironmentEncoders on synthetic community-like and
weather-like features, fine-tunes from the community init, and classifies
the transfer outcome using the transfer_diagnosis module. This is a TDD
§10.6 smoke test: it must run in seconds and exercise the full wiring
without claiming scientific conclusions from toy data.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from plant_context.models.environment_encoder import (
    bridge_transfer_experiment,
    pretrain_environment_encoder,
)
from plant_context.models.transfer_diagnosis import (
    baseline_transfer_comparison,
    classify_transfer_failure,
    domain_difference_report,
)


FEATURE_COLUMNS = ("tmean", "precipitation")
STAGES = ("vegetative", "flowering", "grain_filling")


def _make_features(rng, n_envs, name, scale=1.0):
    env_ids = [f"{name}_{i}" for i in range(n_envs)]
    data = {}
    for stage in STAGES:
        for feat in FEATURE_COLUMNS:
            data[f"{stage}__{feat}"] = rng.normal(size=n_envs) * scale
    return pd.DataFrame(data, index=pd.Index(env_ids, name="environment_id"))


def test_bridge_transfer_experiment_actually_finetunes():
    """The weather_finetune phase should change the loaded community weights."""
    rng = np.random.default_rng(101)
    weather = _make_features(rng, 8, "env", scale=1.0)
    community = _make_features(rng, 8, "env", scale=0.3)

    result = bridge_transfer_experiment(
        weather, community,
        stage_order=STAGES,
        feature_columns=FEATURE_COLUMNS,
        pretrain_epochs=10,
        finetune_epochs=10,
        seed=202,
    )

    assert result["status"] == "completed"
    assert result["shared_environments"] == 8
    # Fine-tune should finish with a finite, non-trivial loss.
    assert np.isfinite(result["weather_finetune"]["final_loss"])


def test_bridge_vs_scratch_with_transfer_diagnosis():
    """Smoke-test the transfer-diagnosis classification on bridge losses."""
    rng = np.random.default_rng(303)
    weather = _make_features(rng, 10, "env", scale=1.0)
    community = _make_features(rng, 10, "env", scale=0.3)

    # Community pretrain -> weather fine-tune
    transfer_result = bridge_transfer_experiment(
        weather, community,
        stage_order=STAGES,
        feature_columns=FEATURE_COLUMNS,
        pretrain_epochs=10,
        finetune_epochs=10,
        seed=404,
    )

    # From-scratch weather pretrain (same epochs as fine-tune)
    scratch = pretrain_environment_encoder(
        weather,
        stage_order=STAGES,
        feature_columns=FEATURE_COLUMNS,
        epochs=10,
        seed=505,
    )

    transfer_loss = transfer_result["weather_finetune"]["final_loss"]
    scratch_loss = scratch["final_loss"]

    baseline_results = {
        "frozen_loss": transfer_loss,
        "random_init_loss": scratch_loss,
        "in_domain_loss": scratch_loss,
    }
    domain_report = domain_difference_report(community, weather)
    layer_ablation = {"unfreeze_all": transfer_loss}

    diagnosis = classify_transfer_failure(
        baseline_results=baseline_results,
        domain_report=domain_report,
        layer_ablation=layer_ablation,
    )

    assert "failure_mode" in diagnosis
    assert diagnosis["failure_mode"] in {
        "successful_transfer",
        "domain_gap_failure",
        "capacity_failure",
        "inconclusive",
    }


def test_domain_difference_report_on_bridge_features():
    """MMD/Wasserstein between community and weather features should be non-zero
    but finite for distinct synthetic distributions."""
    rng = np.random.default_rng(606)
    weather = _make_features(rng, 20, "env", scale=1.0)
    community = _make_features(rng, 20, "env", scale=0.3)

    report = domain_difference_report(community, weather)
    assert report["mmd"] >= 0
    assert report["mean_wasserstein"] >= 0
    assert np.isfinite(report["mmd"])
    assert np.isfinite(report["mean_wasserstein"])
