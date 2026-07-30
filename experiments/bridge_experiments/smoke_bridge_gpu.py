"""GPU/CPU smoke-test for the community → weather bridge experiment.

Uses synthetic community-like features and (when available) real G2F weather
data; falls back to fully synthetic weather if G2F is not present. The
synthetic community features share the same stage × feature structure as the
weather features, so bridge_transfer_experiment can actually share weights.

This is a TDD §10.6 smoke-test: tiny subsets, few epochs, fast runtime,
just verifying the full pipeline runs on the available device and produces
sane diagnostics.

Usage:
    source /f/Anaconda/etc/profile.d/conda.sh
    conda activate tree-py310
    PYTHONPATH=src python experiments/bridge_experiments/smoke_bridge_gpu.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plant_context.data.g2f_adapter import load_g2f_environment_daily
from plant_context.models.environment_encoder import (
    bridge_transfer_experiment,
    pretrain_environment_encoder,
)
from plant_context.models.transfer_diagnosis import (
    classify_transfer_failure,
    domain_difference_report,
)
from plant_context.tokenizers.environment import STAGE_ORDER, tokenize_environment_stages

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1234
rng = np.random.default_rng(SEED)

G2F_ROOT = PROJECT_ROOT / "data" / "external" / "g2f"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "bridge_experiments" / "results_smoke_bridge_gpu"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_FEATURES = [
    "tmax_mean", "tmin_mean", "tmean_mean", "gdd_sum",
    "precipitation_sum", "solar_radiation_mean", "relative_humidity_mean", "vpd_mean",
]

# Tiny smoke-test config
N_ENVIRONMENTS = 200
D_MODEL = 16
PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 20


def build_weather_features():
    print("[1] Loading G2F weather data...")
    if not G2F_ROOT.exists():
        print(f"    WARNING: G2F root not found at {G2F_ROOT}; using synthetic weather data.")
        return None
    try:
        env_daily = load_g2f_environment_daily(G2F_ROOT)
    except FileNotFoundError as e:
        print(f"    WARNING: {e}; using synthetic weather data.")
        return None
    stage_tokens = tokenize_environment_stages(env_daily)
    wide = stage_tokens.pivot_table(
        index="environment_id", columns="growth_stage",
        values=WEATHER_FEATURES, aggfunc="first",
    )
    wide.columns = [f"{stage}__{feat}" for feat, stage in wide.columns]
    ok = wide.notna().sum(axis=1) >= 2
    wide = wide.loc[ok].fillna(wide.loc[ok].mean())
    print(f"    {len(wide)} environments available")
    return wide


def build_synthetic_weather_features(n_envs):
    """Synthetic weather features for smoke-testing when real G2F is unavailable."""
    env_ids = [f"env_{i}" for i in range(n_envs)]
    data = {}
    for stage in STAGE_ORDER:
        for feat in WEATHER_FEATURES:
            data[f"{stage}__{feat}"] = rng.normal(loc=20.0, scale=5.0, size=n_envs)
    df = pd.DataFrame(data, index=pd.Index(env_ids, name="environment_id"))
    return df.fillna(df.mean())


def build_synthetic_community_features(weather_features):
    """Synthetic community-like features sharing index and columns with weather."""
    data = {}
    for stage in STAGE_ORDER:
        for feat in WEATHER_FEATURES:
            col = f"{stage}__{feat}"
            data[col] = rng.binomial(1, 0.3, size=len(weather_features)).astype(float)
    return pd.DataFrame(data, index=weather_features.index)


def main():
    print("=" * 60)
    print("Bridge smoke-test: community → weather transfer")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    t_start = time.time()

    weather_all = build_weather_features()
    if weather_all is None:
        print("    Falling back to synthetic weather features.")
        weather_all = build_synthetic_weather_features(N_ENVIRONMENTS)

    sampled_envs = rng.choice(
        weather_all.index, size=min(N_ENVIRONMENTS, len(weather_all)), replace=False
    )
    weather = weather_all.loc[sampled_envs]
    community = build_synthetic_community_features(weather)

    print(f"[2] Running bridge transfer on {len(weather)} environments...")
    t0 = time.time()
    transfer = bridge_transfer_experiment(
        weather, community,
        stage_order=list(STAGE_ORDER),
        feature_columns=WEATHER_FEATURES,
        pretrain_epochs=PRETRAIN_EPOCHS,
        finetune_epochs=FINETUNE_EPOCHS,
        seed=SEED,
        device=DEVICE,
    )
    print(f"    Bridge transfer done in {time.time() - t0:.1f}s")
    print(f"    Community pretrain loss: {transfer['community_pretrain']['final_loss']:.4f}")
    print(f"    Weather fine-tune loss:  {transfer['weather_finetune']['final_loss']:.4f}")

    print("[3] From-scratch weather baseline...")
    t0 = time.time()
    scratch = pretrain_environment_encoder(
        weather,
        stage_order=list(STAGE_ORDER),
        feature_columns=WEATHER_FEATURES,
        epochs=FINETUNE_EPOCHS,
        seed=SEED + 100,
        device=DEVICE,
    )
    print(f"    Scratch done in {time.time() - t0:.1f}s")
    print(f"    Scratch loss: {scratch['final_loss']:.4f}")

    print("[4] Transfer diagnosis...")
    domain_report = domain_difference_report(community, weather)
    baseline_results = {
        "frozen_loss": transfer["weather_finetune"]["final_loss"],
        "random_init_loss": scratch["final_loss"],
        "in_domain_loss": scratch["final_loss"],
    }
    layer_ablation = {"unfreeze_all": transfer["weather_finetune"]["final_loss"]}
    diagnosis = classify_transfer_failure(
        baseline_results=baseline_results,
        domain_report=domain_report,
        layer_ablation=layer_ablation,
    )
    print(f"    Failure mode: {diagnosis['failure_mode']}")
    print(f"    Reason: {diagnosis['reason']}")

    results = {
        "device": DEVICE,
        "n_environments": len(weather),
        "d_model": D_MODEL,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "community_pretrain_loss": transfer["community_pretrain"]["final_loss"],
        "weather_finetune_loss": transfer["weather_finetune"]["final_loss"],
        "scratch_loss": scratch["final_loss"],
        "community_eff_rank": transfer["community_pretrain"]["collapse_diagnostics"]["effective_rank"],
        "weather_eff_rank": transfer["weather_finetune"]["collapse_diagnostics"]["effective_rank"],
        "scratch_eff_rank": scratch["collapse_diagnostics"]["effective_rank"],
        "mmd": domain_report["mmd"],
        "mean_wasserstein": domain_report["mean_wasserstein"],
        "failure_mode": diagnosis["failure_mode"],
        "total_time_seconds": time.time() - t_start,
    }

    out_file = OUTPUT_DIR / "smoke_bridge_gpu_results.csv"
    pd.DataFrame([results]).to_csv(out_file, index=False)
    print(f"\n[5] Results saved to {out_file}")
    print(f"Total time: {results['total_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
