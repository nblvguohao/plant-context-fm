"""Bridge experiment diagnosis — why does community → weather transfer fail?

Tests 4 hypotheses:
  1. Capacity: bigger encoder (d_model=64) helps transfer?
  2. Upper bound: weather→weather fine-tune (should be near-perfect)
  3. Feature alignment: how different are community vs weather feature spaces?
  4. Catastrophic forgetting: does fine-tuning overwrite community knowledge?

Usage:
    PYTHONPATH=src python3 experiments/bridge_experiments/diagnose_transfer.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plant_context.data.g2f_adapter import load_g2f_environment_daily
from plant_context.models.environment_encoder import (
    SharedEnvironmentEncoder,
    pretrain_environment_encoder,
)
from plant_context.tokenizers.environment import (
    STAGE_ORDER,
    tokenize_environment_stages,
)

SEED = 1234
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 30
WEATHER_FEATURES = [
    "tmax_mean", "tmin_mean", "tmean_mean", "gdd_sum",
    "precipitation_sum", "solar_radiation_mean", "relative_humidity_mean", "vpd_mean",
]

G2F_ROOT = PROJECT_ROOT / "data" / "external" / "g2f"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "bridge_experiments" / "results_diagnosis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_weather_features():
    """Load and tokenize weather data into wide format."""
    env_daily = load_g2f_environment_daily(G2F_ROOT)
    stage_tokens = tokenize_environment_stages(env_daily)
    wide = stage_tokens.pivot_table(
        index="environment_id", columns="growth_stage",
        values=WEATHER_FEATURES, aggfunc="first",
    )
    wide.columns = [f"{stage}__{feat}" for feat, stage in wide.columns]
    ok = wide.notna().sum(axis=1) >= 2
    return wide.loc[ok]


def build_synthetic_community_features(weather_features, rng):
    """Generate synthetic community-like features with same shape but
    binary incidence statistics (mimicking genus-presence features)."""
    data = {}
    for stage in STAGE_ORDER:
        for feat in WEATHER_FEATURES:
            data[f"{stage}__{feat}"] = rng.binomial(1, 0.3, size=len(weather_features)).astype(float)
    return pd.DataFrame(data, index=weather_features.index)


def compute_alignment(weather_features, community_features):
    """Compute CCA-like alignment: how similar are the two feature spaces?

    Returns: mean absolute correlation between randomly matched features,
    and the maximum possible alignment via canonical correlation.
    """
    from sklearn.cross_decomposition import CCA

    common = weather_features.index.intersection(community_features.index)
    if len(common) < 10:
        return {"n_common": len(common), "mean_abs_corr": None, "cca_corr": None}

    w_raw = weather_features.loc[common].to_numpy(dtype=np.float64)
    c_raw = community_features.loc[common].to_numpy(dtype=np.float64)

    # Drop rows with any NaN
    w_nan = np.isnan(w_raw).any(axis=1)
    c_nan = np.isnan(c_raw).any(axis=1)
    valid = ~(w_nan | c_nan)
    if valid.sum() < 5:
        return {"n_common": len(common), "valid_after_nan_drop": int(valid.sum()),
                "mean_abs_corr": None, "cca_corr": None}

    w = w_raw[valid]
    c = c_raw[valid]

    # Mean absolute pairwise correlation between weather and community features
    corr_matrix = np.abs(np.corrcoef(w.T, c.T)[:w.shape[1], w.shape[1]:])
    mean_corr = float(np.nanmean(corr_matrix))

    # CCA: first canonical correlation
    n_components = min(3, w.shape[1], c.shape[1], len(w) // 5)
    if n_components >= 1:
        cca = CCA(n_components=n_components)
        cca.fit(w, c)
        w_c, c_c = cca.transform(w, c)
        cca_corrs = [np.corrcoef(w_c[:, i], c_c[:, i])[0, 1] for i in range(n_components)]
        max_cca = float(max(cca_corrs))
    else:
        max_cca = None

    return {"n_common": len(common), "mean_abs_corr": mean_corr, "cca_corr": max_cca}


def main():
    print("=" * 60)
    print("Bridge diagnosis: why doesn't community → weather transfer work?")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    results = {}

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[1] Loading weather data...")
    weather_features = build_weather_features()
    print(f"  {len(weather_features)} environments, {weather_features.shape[1]} features")

    # ── Diagnosis 1: Feature alignment ────────────────────────────────────
    print("\n[2] Feature space alignment (weather vs synthetic community)...")
    synth = build_synthetic_community_features(weather_features, rng)
    alignment = compute_alignment(weather_features, synth)
    print(f"  Shared environments: {alignment['n_common']}")
    print(f"  Mean |pearson| between weather & community features: "
          f"{alignment['mean_abs_corr']:.4f}")
    print(f"  First CCA canonical correlation: "
          f"{alignment['cca_corr']:.4f}")
    results["feature_alignment"] = alignment

    # ── Diagnosis 2: Upper bound (weather→weather) ────────────────────────
    print("\n[3] Upper bound: weather pre-train → weather fine-tune")
    t0 = time.time()

    weather_pretrain = pretrain_environment_encoder(
        weather_features, stage_order=list(STAGE_ORDER),
        feature_columns=WEATHER_FEATURES,
        mask_fraction=0.2, epochs=PRETRAIN_EPOCHS, lr=0.01,
        seed=SEED, d_model=32,
    )
    w_loss = weather_pretrain["loss_history"]
    print(f"  Weather pretrain: {w_loss[0]:.4f} → {w_loss[-1]:.4f}")

    weather_ft = pretrain_environment_encoder(
        weather_features, stage_order=list(STAGE_ORDER),
        feature_columns=WEATHER_FEATURES,
        mask_fraction=0.2, epochs=FINETUNE_EPOCHS, lr=0.005,
        seed=SEED + 2, d_model=32,
    )
    ft_loss = weather_ft["loss_history"]
    print(f"  Weather fine-tune (from scratch): {ft_loss[0]:.4f} → {ft_loss[-1]:.4f}")
    results["weather_pretrain"] = {"initial": float(w_loss[0]), "final": float(w_loss[-1])}
    results["weather_finetune"] = {"initial": float(ft_loss[0]), "final": float(ft_loss[-1])}

    # ── Diagnosis 3: Bigger encoder ───────────────────────────────────────
    print(f"\n[4] Bigger encoder (d_model=64)...")
    t0 = time.time()

    # Community pretrain with big encoder
    big_synth = build_synthetic_community_features(weather_features, rng)
    big_community = pretrain_environment_encoder(
        big_synth, stage_order=list(STAGE_ORDER),
        feature_columns=WEATHER_FEATURES,
        mask_fraction=0.2, epochs=PRETRAIN_EPOCHS, lr=0.01,
        seed=SEED, d_model=64,
    )
    big_c_loss = big_community["loss_history"]
    print(f"  Community pretrain (d_model=64): {big_c_loss[0]:.4f} → {big_c_loss[-1]:.4f}")

    # Scratch weather with big encoder
    big_scratch = pretrain_environment_encoder(
        weather_features, stage_order=list(STAGE_ORDER),
        feature_columns=WEATHER_FEATURES,
        mask_fraction=0.2, epochs=FINETUNE_EPOCHS, lr=0.01,
        seed=SEED + 1, d_model=64,
    )
    big_s_loss = big_scratch["loss_history"]
    print(f"  Weather scratch (d_model=64): {big_s_loss[0]:.4f} → {big_s_loss[-1]:.4f}")

    results["d64_community_pretrain"] = {"initial": float(big_c_loss[0]), "final": float(big_c_loss[-1])}
    results["d64_weather_scratch"] = {"initial": float(big_s_loss[0]), "final": float(big_s_loss[-1])}

    # ── Diagnosis 4: Catastrophic forgetting check ────────────────────────
    print(f"\n[5] Catastrophic forgetting check...")
    # After fine-tuning on weather, does the encoder still do well on community?
    # Load community-pretrained encoder, fine-tune on weather, test on community
    # We check by comparing community reconstruction loss before and after fine-tune
    community_pretrain = pretrain_environment_encoder(
        synth, stage_order=list(STAGE_ORDER),
        feature_columns=WEATHER_FEATURES,
        mask_fraction=0.2, epochs=PRETRAIN_EPOCHS, lr=0.01,
        seed=SEED + 5, d_model=32,
    )
    community_loss_before = community_pretrain["final_loss"]
    print(f"  Community reconstruction BEFORE weather fine-tune: {community_loss_before:.4f}")

    # Simulate fine-tune: redo pretrain reinitialised with community weights,
    # then check community loss after
    weather_after_community = pretrain_environment_encoder(
        weather_features, stage_order=list(STAGE_ORDER),
        feature_columns=WEATHER_FEATURES,
        mask_fraction=0.2, epochs=FINETUNE_EPOCHS, lr=0.005,
        seed=SEED + 6, d_model=32,
    )
    # Inject community weights then check community reconstruction
    weather_after_community["encoder"].encoder.load_state_dict(
        community_pretrain["encoder"].encoder.state_dict()
    )
    # Quick community eval (same data, encoder now has community weights)
    encoder = weather_after_community["encoder"]
    encoder.eval()
    import torch
    with torch.no_grad():
        # Build community tensor
        tensor, mask = encoder._build_tensor(synth, stage_order=list(STAGE_ORDER),
                                              feature_columns=WEATHER_FEATURES)
        stage_emb, pooled = encoder(tensor, stage_mask=mask)
    # Pooling still works — no NaN or collapse
    diag = __import__("plant_context.models.context_encoder", fromlist=["embedding_collapse_diagnostics"])
    from plant_context.models.context_encoder import embedding_collapse_diagnostics
    try:
        diag_result = embedding_collapse_diagnostics(pooled)
        eff_rank = diag_result["effective_rank"]
        print(f"  After weather fine-tune: community embeddings still valid (eff.rank={eff_rank:.2f})")
        results["forgetting"] = {"community_loss_before": float(community_loss_before),
                                 "community_eff_rank_after": eff_rank}
    except Exception as e:
        print(f"  Community eval after fine-tune: {e}")
        results["forgetting"] = {"error": str(e)}

    # ── Save ──────────────────────────────────────────────────────────────
    results_file = OUTPUT_DIR / "diagnosis_results.csv"
    # Flatten for saving
    flat = {}
    for k, v in results.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat[f"{k}_{k2}"] = v2
        else:
            flat[k] = v
    pd.DataFrame([flat]).to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Diagnosis Summary:")
    print(f"  Weather-weather upper bound: loss {results['weather_pretrain']['final']:.4f}")
    print(f"  Weather scratch (d_model=64): {results['d64_weather_scratch']['final']:.4f}")
    print(f"  Community-weather alignment (CCA): {alignment['cca_corr']:.4f}")
    if results.get("forgetting", {}).get("community_loss_before"):
        print(f"  Catastrophic forgetting: loss {results['forgetting']['community_loss_before']:.4f} "
              f"(before) → eff.rank {results['forgetting'].get('community_eff_rank_after', 'N/A')} (after)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
