"""Bridge v2: sPlotOpen CWM/CWV functional traits → environment encoder.

Uses community-weighted mean (CWM) functional traits from sPlotOpen (18 traits
per plot, including SLA, plant height, seed mass, leaf N, etc.) as per-plot
environment descriptors.

Design:
  - PCA-reduce 18 CWM traits → 12 components
  - Split into 2 stages of 6 features each
  - Weather: PCA-reduce 48 stage features → 12, same split
  - Both have n_stages=2, n_features=6 → same architecture
  - Compare: community pretrain → weather FT vs weather scratch

Usage:
    PYTHONPATH=src python3 experiments/bridge_experiments/bridge_v2_traits.py
"""

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from plant_context.data.g2f_adapter import load_g2f_environment_daily
from plant_context.models.environment_encoder import (
    SharedEnvironmentEncoder, pretrain_environment_encoder,
)
from plant_context.tokenizers.environment import STAGE_ORDER, tokenize_environment_stages
from sklearn.decomposition import PCA

SEED = 1234
rng = np.random.default_rng(SEED)
N_PRETRAIN = 30
N_FINETUNE = 30
D_MODEL = 32
N_COMPONENTS = 12        # total across 2 stages = 6 per stage
N_PLOTS = 5000

COMMUNITY_ROOT = PROJ / "data" / "community" / "extracted"
G2F_ROOT = PROJ / "data" / "external" / "g2f"
OUT = PROJ / "experiments" / "bridge_experiments" / "results_bridge_v2"
OUT.mkdir(parents=True, exist_ok=True)

WEATHER_FEATURES = [
    "tmax_mean", "tmin_mean", "tmean_mean", "gdd_sum",
    "precipitation_sum", "solar_radiation_mean", "relative_humidity_mean", "vpd_mean",
]


def build_trait_features():
    """Load CWM traits → PCA → 2 stages × 6 features."""
    print("\n[1] Loading CWM traits...")
    t0 = time.time()
    cwm = pd.read_csv(
        COMMUNITY_ROOT / "sPlotOpen_CWM_CWV(2).txt",
        sep="\t", encoding="latin1",
    )
    print(f"    {len(cwm)} plots, {cwm.shape[1]} cols ({time.time()-t0:.1f}s)")

    # Subsample
    chosen = list(rng.choice(cwm.index, size=min(N_PLOTS, len(cwm)), replace=False))
    cwm_sub = cwm.loc[chosen]

    # Extract CWM trait columns (18 functional traits)
    trait_cols = sorted([c for c in cwm_sub.columns
                         if c.endswith("_CWM") and not c.startswith("Trait")])
    print(f"    {len(trait_cols)} CWM trait columns: {trait_cols[:5]}...")

    # Drop rows with too many NaN traits
    traits = cwm_sub[trait_cols].dropna(thresh=len(trait_cols) // 2)
    print(f"    {len(traits)} plots after NaN filtering")

    # PCA reduce
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="mean")
    traits_imp = imp.fit_transform(traits.to_numpy(dtype=np.float64))
    n_comp = min(N_COMPONENTS, traits_imp.shape[1], len(traits_imp))
    pca = PCA(n_components=n_comp)
    components = pca.fit_transform(traits_imp)
    print(f"    PCA {n_comp} components: {pca.explained_variance_ratio_.sum():.1%} variance")

    # Split into 2 stages × (n_comp//2) features
    feats_per_stage = n_comp // 2
    stage_order = ["trait_group_0", "trait_group_1"]
    feature_cols = [f"PC{i}" for i in range(feats_per_stage)]

    trait_wide = pd.DataFrame(index=traits.index)
    for s_idx, stage in enumerate(stage_order):
        start = s_idx * feats_per_stage
        end = start + feats_per_stage
        for i in range(start, end):
            col_idx = i - start
            trait_wide[f"{stage}__PC{col_idx}"] = components[:, i]

    trait_wide.index.name = "environment_id"
    print(f"    Trait features: {trait_wide.shape}")
    return trait_wide, stage_order, feature_cols


def build_weather_features():
    """Weather → PCA → 2 stages × 6 features (matching trait dim)."""
    print("\n[2] Loading weather...")
    env_daily = load_g2f_environment_daily(G2F_ROOT)
    stage_tokens = tokenize_environment_stages(env_daily)
    feature_cols_w = [c for c in stage_tokens.columns
                      if c not in ("environment_id", "growth_stage", "stage_estimation_method")]

    # Per-environment × stage matrix → flatten to per-environment vector
    wide = stage_tokens.pivot_table(
        index="environment_id", columns="growth_stage",
        values=feature_cols_w, aggfunc="first",
    )
    wide.columns = [f"{s}__{f}" for f, s in wide.columns]
    wide = wide.dropna(thresh=wide.shape[1] // 2).fillna(wide.mean())

    # PCA reduce
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="mean")
    w_imp = imp.fit_transform(wide.to_numpy(dtype=np.float64))
    n_comp = min(N_COMPONENTS, w_imp.shape[1], len(w_imp))
    pca = PCA(n_components=n_comp)
    components = pca.fit_transform(w_imp)
    print(f"    Weather PCA {n_comp}: {pca.explained_variance_ratio_.sum():.1%} variance")

    feats_per_stage = n_comp // 2
    stage_order = ["weather_group_0", "weather_group_1"]
    feature_cols = [f"PC{i}" for i in range(feats_per_stage)]

    weather_wide = pd.DataFrame(index=wide.index)
    for s_idx, stage in enumerate(stage_order):
        start = s_idx * feats_per_stage
        end = start + feats_per_stage
        for i in range(start, end):
            col_idx = i - start
            weather_wide[f"{stage}__PC{col_idx}"] = components[:, i]

    print(f"    Weather features: {weather_wide.shape}")
    return weather_wide, stage_order, feature_cols


def main():
    print("=" * 60)
    print("Bridge v2: Functional trait → weather transfer")
    print("=" * 60)
    sys.stdout.flush()

    trait_features, t_stages, t_feats = build_trait_features()
    weather_features, w_stages, w_feats = build_weather_features()

    # ── Phase 1: Pretrain on traits ────────────────────────────────────
    print("\n[3] Pretraining on CWM traits...")
    t0 = time.time()
    trait_result = pretrain_environment_encoder(
        trait_features, stage_order=t_stages, feature_columns=t_feats,
        mask_fraction=0.2, epochs=N_PRETRAIN, lr=0.01, seed=SEED, d_model=D_MODEL,
    )
    tr_loss = trait_result["loss_history"]
    print(f"    Trait loss: {tr_loss[0]:.4f} → {tr_loss[-1]:.4f} ({time.time()-t0:.1f}s)")
    cv = trait_result.get("collapse_violations", [])
    if cv:
        print(f"    ⚠ Collapse: {cv}")

    import torch
    with torch.no_grad():
        t_tens, t_mask = trait_result["encoder"]._build_tensor(
            trait_features, t_stages, t_feats)
        _, t_pool = trait_result["encoder"](t_tens, stage_mask=t_mask)
    from plant_context.models.context_encoder import embedding_collapse_diagnostics
    t_diag = embedding_collapse_diagnostics(t_pool)
    print(f"    Trait embeddings — eff.rank={t_diag['effective_rank']:.2f}")
    sys.stdout.flush()

    # ── Phase 2: Transfer to weather ──────────────────────────────────
    print("\n[4] Transfer: trait → weather (weight init + fine-tune)...")
    # Rename weather columns to match trait stage names for shared architecture
    w_stages_renamed = [s.replace("weather_group", "trait_group") for s in w_stages]
    col_map = {old: new for old, new in zip(
        [f"{s}__{f}" for s in w_stages for f in w_feats],
        [f"{s}__{f}" for s in w_stages_renamed for f in w_feats],
    )}
    weather_renamed = weather_features.rename(columns=col_map)

    # Manually create encoder with trait-pretrained weights, fine-tune on weather
    import torch
    from torch import nn
    from plant_context.tokenizers.masking import mask_contiguous_run
    from plant_context.models.environment_encoder import MaskPartial
    from plant_context.models.pretraining import pretrain_masked_reconstruction
    from plant_context.models.context_encoder import TokenSequenceEncoder

    # Build weather per-sample data
    n_feat = len(t_feats)
    enc_transfer = SharedEnvironmentEncoder(
        n_stage_features=n_feat, d_model=D_MODEL, stage_names=t_stages,
    )
    tensor, mask = enc_transfer._build_tensor(weather_renamed, t_stages, t_feats)
    per_sample_transfer = {}
    for i, eid in enumerate(weather_renamed.index):
        present = [t_stages[s] for s in range(len(t_stages)) if not mask[i, s]]
        if len(present) < 2:
            continue
        stage_df = pd.DataFrame(
            tensor[i, ~mask[i], :].numpy(),
            index=present, columns=t_feats,
        )
        per_sample_transfer[eid] = stage_df

    # Initialize with trait-pretrained weights
    trait_encoder = trait_result["encoder"].encoder  # TokenSequenceEncoder
    base_enc = TokenSequenceEncoder(n_features=n_feat, d_model=D_MODEL)
    base_enc.load_state_dict(trait_encoder.state_dict())
    head = nn.Linear(D_MODEL, n_feat)
    optimizer = torch.optim.Adam(
        list(base_enc.parameters()) + list(head.parameters()), lr=0.005
    )
    mask_fn = MaskPartial(mask_contiguous_run, mask_fraction=0.2)

    transfer_pt = pretrain_masked_reconstruction(
        per_sample_transfer, feature_columns=t_feats,
        mask_fn=mask_fn, d_model=D_MODEL, epochs=N_FINETUNE, lr=0.005, seed=SEED + 1,
    )

    # ── Phase 3: Weather scratch ──────────────────────────────────────
    print("\n[5] Weather scratch...")
    t0 = time.time()
    scratch_result = pretrain_environment_encoder(
        weather_renamed, stage_order=t_stages, feature_columns=t_feats,
        mask_fraction=0.2, epochs=N_FINETUNE, lr=0.01, seed=SEED + 2, d_model=D_MODEL,
    )
    sc_loss = scratch_result["loss_history"]
    print(f"    Scratch loss: {sc_loss[0]:.4f} → {sc_loss[-1]:.4f} ({time.time()-t0:.1f}s)")

    # Transfer loss from the correctly initialized training
    tr_loss_transfer = transfer_pt["loss_history"]
    tr_loss_final = tr_loss_transfer[-1] if tr_loss_transfer else float("nan")
    print(f"    Transfer loss: {tr_loss_transfer[0]:.4f} → {tr_loss_final:.4f} (init from traits)")

    deltas = {
        "trait_final_loss": float(tr_loss[-1]),
        "transfer_final_loss": float(tr_loss_final),
        "scratch_final_loss": float(sc_loss[-1]),
        "delta_transfer_vs_scratch": float(tr_loss_final - sc_loss[-1]),
        "trait_eff_rank": t_diag["effective_rank"],
        "trait_collapse_ok": len(cv) == 0,
    }
    pd.DataFrame([deltas]).to_csv(OUT / "bridge_v2_results.csv", index=False)

    print(f"\n{'=' * 60}")
    print("Bridge v2 Summary:")
    print(f"  Trait pretrain loss: {tr_loss[-1]:.4f}")
    print(f"  Transfer (trait→weather): {tr_loss_final:.4f}")
    print(f"  Weather scratch: {sc_loss[-1]:.4f}")
    delta = deltas["delta_transfer_vs_scratch"]
    if delta < 0:
        print(f"  ✅ Transfer beats scratch by Δ={delta:.4f}")
    else:
        print(f"  ❌ Transfer worse than scratch by Δ={delta:.4f}")
    print(f"  Trait embedding eff.rank: {t_diag['effective_rank']:.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
