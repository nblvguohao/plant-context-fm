"""Tests for transfer_diagnosis.py (TDD 6.1 cross-domain migration protocol)."""

import numpy as np
import pandas as pd
import pytest
import torch

from plant_context.models.environment_encoder import SharedEnvironmentEncoder
from plant_context.models.transfer_diagnosis import (
    baseline_transfer_comparison,
    classify_transfer_failure,
    compute_mmd,
    compute_wasserstein_distance,
    domain_difference_report,
    layer_wise_finetune_ablation,
)

STAGE_ORDER = ("pre_planting", "emergence", "vegetative", "flowering", "grain_filling", "maturity")
FEATURES = ["f0", "f1"]


def _make_features(n_envs, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    data = {}
    for stage in STAGE_ORDER:
        for feat in FEATURES:
            data[f"{stage}__{feat}"] = rng.normal(loc=0.0, scale=scale, size=n_envs)
    index = [f"env_{i}" for i in range(n_envs)]
    return pd.DataFrame(data, index=index)


def test_mmd_same_distribution_is_small():
    x = np.random.randn(100, 4)
    y = np.random.randn(100, 4)
    mmd = compute_mmd(x, y)
    assert mmd >= 0
    assert mmd < 0.2  # same Gaussian should be near zero


def test_mmd_different_distributions_is_larger():
    x = np.random.randn(100, 4)
    y = np.random.randn(100, 4) + 5.0
    mmd = compute_mmd(x, y)
    assert mmd > 0.5


def test_wasserstein_zero_for_identical():
    x = np.random.randn(50, 3)
    assert compute_wasserstein_distance(x, x) == pytest.approx(0.0, abs=1e-6)


def test_wasserstein_increases_with_shift():
    x = np.random.randn(50, 3)
    y = x + 2.0
    assert compute_wasserstein_distance(x, y) > 1.0


def test_domain_difference_report_per_feature():
    source = _make_features(20, seed=1, scale=1.0)
    target = _make_features(20, seed=2, scale=2.0)
    report = domain_difference_report(source, target)
    assert report["n_source"] == 20
    assert report["n_target"] == 20
    assert report["n_common_features"] == len(source.columns)
    assert len(report["per_feature_wasserstein"]) == len(source.columns)
    assert report["mean_wasserstein"] > 0
    assert report["mmd"] >= 0


def test_domain_difference_report_no_common_columns():
    source = pd.DataFrame({"a": [1, 2]})
    target = pd.DataFrame({"b": [3, 4]})
    report = domain_difference_report(source, target)
    assert np.isnan(report["mmd"])
    assert report["n_common_features"] == 0


def dummy_pretrain_fn(features, stage_order, feature_columns, **kwargs):
    """A tiny pretraining function that just returns an encoder."""
    encoder = SharedEnvironmentEncoder(
        n_stage_features=len(feature_columns),
        d_model=8,
        n_layers=1,
        n_heads=2,
        stage_names=stage_order,
    )
    return {"encoder": encoder}


def dummy_finetune_fn(encoder, features, stage_order, feature_columns, **kwargs):
    """Return encoder unchanged for layer ablation tests."""
    return encoder


def test_baseline_transfer_comparison_no_shared_envs():
    source = _make_features(10, seed=1)
    target = _make_features(10, seed=2)
    source.index = [f"s_{i}" for i in range(10)]
    target.index = [f"t_{i}" for i in range(10)]
    result = baseline_transfer_comparison(
        source, target, STAGE_ORDER, FEATURES,
        pretrain_fn=dummy_pretrain_fn,
        pretrain_kwargs={},
    )
    assert result["status"] == "no_shared_environments"


def test_baseline_transfer_comparison_returns_losses():
    source = _make_features(15, seed=1)
    target = _make_features(15, seed=2)
    result = baseline_transfer_comparison(
        source, target, STAGE_ORDER, FEATURES,
        pretrain_fn=dummy_pretrain_fn,
        pretrain_kwargs={},
    )
    assert result["status"] == "completed"
    assert result["n_shared_environments"] == 15
    for key in ["frozen_loss", "random_init_loss", "in_domain_loss"]:
        assert isinstance(result[key], float)


def test_layer_wise_finetune_ablation():
    source = _make_features(10, seed=1)
    encoder = dummy_pretrain_fn(source, STAGE_ORDER, FEATURES)["encoder"]
    target = _make_features(10, seed=2)
    result = layer_wise_finetune_ablation(
        encoder, target, STAGE_ORDER, FEATURES,
        finetune_fn=dummy_finetune_fn,
        finetune_kwargs={},
    )
    assert set(result.keys()) == {"frozen_all", "unfreeze_head", "unfreeze_last_layer", "unfreeze_all"}
    for v in result.values():
        assert isinstance(v, float)


def test_classify_successful_transfer():
    baseline = {"frozen_loss": 0.1, "random_init_loss": 0.5, "in_domain_loss": 0.08}
    domain = {"mmd": 0.1, "mean_wasserstein": 0.2}
    ablation = {"unfreeze_all": 0.09}
    result = classify_transfer_failure(baseline, domain, ablation)
    assert result["failure_mode"] == "successful_transfer"


def test_classify_domain_gap_failure():
    baseline = {"frozen_loss": 0.5, "random_init_loss": 0.5, "in_domain_loss": 0.1}
    domain = {"mmd": 0.8, "mean_wasserstein": 1.5}
    ablation = {"unfreeze_all": 0.45}
    result = classify_transfer_failure(baseline, domain, ablation)
    assert result["failure_mode"] == "domain_gap_failure"


def test_classify_capacity_failure():
    baseline = {"frozen_loss": 0.5, "random_init_loss": 0.5, "in_domain_loss": 0.1}
    domain = {"mmd": 0.1, "mean_wasserstein": 0.2}
    ablation = {"unfreeze_all": 0.11}
    result = classify_transfer_failure(baseline, domain, ablation)
    assert result["failure_mode"] == "capacity_failure"


def test_classify_inconclusive():
    baseline = {"frozen_loss": 0.5, "random_init_loss": 0.5, "in_domain_loss": 0.5}
    domain = {"mmd": 0.1, "mean_wasserstein": 0.2}
    ablation = {"unfreeze_all": 0.5}
    result = classify_transfer_failure(baseline, domain, ablation)
    assert result["failure_mode"] == "inconclusive"
