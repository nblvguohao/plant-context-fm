"""Bridge deepening: real sPlotOpen community data → SharedEnvironmentEncoder.

Replaces the synthetic binary features from the initial bridge experiment with
real sPlotOpen genus-presence data, then tests whether community-derived
pretraining helps weather reconstruction.

Key challenge: community data has no natural "stage" structure. We handle this
by grouping genera by frequency into 6 bins, each bin = one "stage", and each
stage's feature = proportion of genera in that bin present in the plot.

This gives both community and weather data the same stage × feature structure
(n_stages=6, n_features=1), enabling direct weight transfer.

Usage:
    PYTHONPATH=src python3 experiments/bridge_experiments/real_bridge.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from plant_context.data.community_adapter import load_splotopen_community_plot
from plant_context.data.g2f_adapter import load_g2f_environment_daily
from plant_context.models.environment_encoder import (
    SharedEnvironmentEncoder,
    pretrain_environment_encoder,
)
from plant_context.tokenizers.community import tokenize_community_plot
from plant_context.tokenizers.environment import STAGE_ORDER, tokenize_environment_stages

SEED = 1234
rng = np.random.default_rng(SEED)

N_PRETRAIN_EPOCHS = 30
N_FINETUNE_EPOCHS = 30
N_SPLOT_PLOTS = 3000
N_TOP_GENERA = 300
N_GROUPS = 6          # match weather's ~6 stages
D_MODEL = 32

COMMUNITY_ROOT = PROJ / "data" / "community" / "extracted"
G2F_ROOT = PROJ / "data" / "external" / "g2f"
OUT = PROJ / "experiments" / "bridge_experiments" / "results_real_bridge"
OUT.mkdir(parents=True, exist_ok=True)

WEATHER_FEATURES = [
    "tmax_mean", "tmin_mean", "tmean_mean", "gdd_sum",
    "precipitation_sum", "solar_radiation_mean", "relative_humidity_mean", "vpd_mean",
]


def build_community_multi_stage():
    """sPlotOpen → per-plot genus-group presence → multi-stage features.

    Returns (community_wide, stage_order, feature_cols) compatible with
    ``pretrain_environment_encoder``.
    """
    print("\n[1] Loading sPlotOpen...")
    t0 = time.time()
    raw = load_splotopen_community_plot(COMMUNITY_ROOT)
    print(f"    {len(raw)} rows, {raw['plot_id'].nunique()} plots ({time.time()-t0:.1f}s)")

    # Subsample
    all_plots = sorted(raw["plot_id"].unique())
    chosen = set(rng.choice(all_plots, size=min(N_SPLOT_PLOTS, len(all_plots)), replace=False))
    subset = raw[raw["plot_id"].isin(chosen)].copy()
    print(f"    Subsampled to {len(chosen)} plots")

    # Tokenize → get genus
    print("  Tokenizing...")
    t0 = time.time()
    tokens = tokenize_community_plot(subset)
    print(f"    {len(tokens)} token rows ({time.time()-t0:.1f}s)")

    # Top N genera
    genus_freq = tokens["genus"].value_counts()
    top_genera = genus_freq.head(N_TOP_GENERA).index.tolist()
    coverage = genus_freq.head(N_TOP_GENERA).sum() / genus_freq.sum()
    print(f"    Top {N_TOP_GENERA} genera: {coverage:.1%} of occurrences")

    # Assign each genus to a frequency group (round-robin)
    genus_to_group = {g: i % N_GROUPS for i, g in enumerate(top_genera)}

    # Per-plot binary presence per genus
    binary = (
        tokens[tokens["genus"].isin(top_genera)]
        .groupby(["plot_id", "genus"]).size()
        .unstack(fill_value=0)
        .clip(upper=1)
    )

    # Build multi-stage features: each freq_group → stage, prop_present → feature
    stage_order = [f"freq_group_{i}" for i in range(N_GROUPS)]
    feature_cols = ["prop_present"]

    community_wide = pd.DataFrame(index=binary.index)
    for g_idx in range(N_GROUPS):
        cols_in_group = [g for g in top_genera if genus_to_group.get(g) == g_idx
                         and g in binary.columns]
        if not cols_in_group:
            continue
        stage = f"freq_group_{g_idx}"
        community_wide[f"{stage}__prop_present"] = binary[cols_in_group].mean(axis=1)

    community_wide.index.name = "environment_id"
    print(f"    Community features: {community_wide.shape}")
    return community_wide, stage_order, feature_cols


def build_weather_multi_stage():
    """G2F daily weather → per-environment stage features.

    Returns (weather_wide, stage_order, feature_cols) using G2F's growth stages.
    """
    print("\n[2] Loading weather data...")
    env_daily = load_g2f_environment_daily(G2F_ROOT)
    print(f"    {env_daily['environment_id'].nunique()} environments")

    stage_tokens = tokenize_environment_stages(env_daily)

    # Pivot to wide format matching pretrain_environment_encoder expectations
    # Columns: {stage}__{feature}
    weather_wide = pd.DataFrame(index=stage_tokens["environment_id"].unique())
    for stage in STAGE_ORDER:
        stage_data = stage_tokens[stage_tokens["growth_stage"] == stage]
        if stage_data.empty:
            continue
        stage_data = stage_data.set_index("environment_id")
        for feat in WEATHER_FEATURES:
            if feat in stage_data.columns:
                col = f"{stage}__{feat}"
                weather_wide[col] = stage_data[feat]

    # Drop environments with too few stages
    n_stages_present = weather_wide.notna().sum(axis=1) // len(WEATHER_FEATURES)
    weather_wide = weather_wide.loc[n_stages_present >= 2]
    weather_wide = weather_wide.fillna(weather_wide.mean())

    print(f"    Weather features: {weather_wide.shape}")
    return weather_wide, list(STAGE_ORDER), WEATHER_FEATURES


def main():
    print("=" * 60)
    print("Real Bridge: sPlotOpen → G×E environment encoder transfer")
    print("=" * 60)
    sys.stdout.flush()

    # ── Build community (multi-stage genus groups) ──────────────────────
    community_wide, comm_stages, comm_features = build_community_multi_stage()

    # ── Build weather (growth stages) ───────────────────────────────────
    weather_wide, weather_stages, weather_features = build_weather_multi_stage()

    # ── Phase 1: Pretrain on community data ─────────────────────────────
    print("\n[3] Pretraining on community data...")
    t0 = time.time()
    community_pretrain = pretrain_environment_encoder(
        community_wide,
        stage_order=comm_stages,
        feature_columns=comm_features,
        mask_fraction=0.2,
        epochs=N_PRETRAIN_EPOCHS,
        lr=0.01,
        seed=SEED,
        d_model=D_MODEL,
    )
    cp_loss = community_pretrain["loss_history"]
    print(f"    Community loss: {cp_loss[0]:.4f} → {cp_loss[-1]:.4f}  "
          f"({time.time()-t0:.1f}s)")
    cv = community_pretrain.get("collapse_violations", [])
    if cv:
        print(f"    ⚠  Collapse violations: {cv}")
    sys.stdout.flush()

    # ── Phase 2: Transfer (community → weather fine-tune) ──────────────
    # Weather and community have DIFFERENT architectures (different stages
    # and features), so direct weight transfer isn't possible.
    # Instead, compare:
    #   - Community embedding quality (does genus composition produce
    #     well-structured environment representations?)
    #   - Can community embeddings predict weather patterns?

    print("\n[4] Evaluating community embeddings...")
    t0 = time.time()
    encoder = community_pretrain["encoder"]
    encoder.eval()

    import torch
    from plant_context.models.context_encoder import embedding_collapse_diagnostics

    with torch.no_grad():
        tensor, mask = encoder._build_tensor(
            community_wide, comm_stages, comm_features
        )
        _, pooled = encoder(tensor, stage_mask=mask)

    diag = embedding_collapse_diagnostics(pooled)
    print(f"    Community embeddings — eff.rank={diag['effective_rank']:.2f}, "
          f"per_dim_std={diag['per_dim_std_mean']:.4f}")
    sys.stdout.flush()

    # ── Phase 3: Independent weather pretraining (for comparison) ──────
    # Since direct weight transfer isn't possible (different dims), we
    # train a separate encoder on weather and compare embedding quality.
    print("\n[5] Independent weather pretraining...")
    t0 = time.time()
    weather_result = pretrain_environment_encoder(
        weather_wide,
        stage_order=weather_stages,
        feature_columns=weather_features,
        mask_fraction=0.2,
        epochs=N_FINETUNE_EPOCHS,
        lr=0.01,
        seed=SEED + 1,
        d_model=D_MODEL,
    )
    wr_loss = weather_result["loss_history"]
    print(f"    Weather loss: {wr_loss[0]:.4f} → {wr_loss[-1]:.4f}  "
          f"({time.time()-t0:.1f}s)")

    with torch.no_grad():
        w_tensor, w_mask = weather_result["encoder"]._build_tensor(
            weather_wide, weather_stages, weather_features
        )
        _, w_pooled = weather_result["encoder"](w_tensor, stage_mask=w_mask)
    w_diag = embedding_collapse_diagnostics(w_pooled)
    print(f"    Weather embeddings — eff.rank={w_diag['effective_rank']:.2f}, "
          f"per_dim_std={w_diag['per_dim_std_mean']:.4f}")

    # ── Phase 4: Cross-modal correlation ───────────────────────────────
    # Do community embeddings encode environment info that correlates with
    # weather? If so, the bridge concept is valid even without weight transfer.
    print("\n[6] Cross-modal correlation...")
    # Align community and weather embeddings by environment
    # (plots ≠ environments, so this is a conceptual check)
    # We test: do community embeddings show geographic structure?
    # (lat/lon clustering of nearby plots having similar community embeddings)

    # Quick check: add back lat/lon from raw data
    raw = load_splotopen_community_plot(COMMUNITY_ROOT)
    plot_latlon = raw[["plot_id", "latitude", "longitude"]].drop_duplicates("plot_id")
    plot_latlon = plot_latlon.set_index("plot_id")
    common = community_wide.index.intersection(plot_latlon.index)
    if len(common) >= 100:
        pooled_df = pd.DataFrame(
            pooled.numpy(), index=community_wide.index
        ).loc[common]
        ll = plot_latlon.loc[common]
        corr_with_lat = pooled_df.corrwith(ll["latitude"]).abs().max()
        corr_with_lon = pooled_df.corrwith(ll["longitude"]).abs().max()
        print(f"    Max |corr| with latitude:  {corr_with_lat:.3f}")
        print(f"    Max |corr| with longitude: {corr_with_lon:.3f}")
    else:
        corr_with_lat = corr_with_lon = None
        print(f"    Only {len(common)} shared IDs — skipping geo correlation")

    sys.stdout.flush()

    # ── Save ──────────────────────────────────────────────────────────
    results = {
        "n_community_plots": community_wide.shape[0],
        "n_weather_envs": weather_wide.shape[0],
        "community_stages": len(comm_stages),
        "weather_stages": len(weather_stages),
        "community_initial_loss": float(cp_loss[0]),
        "community_final_loss": float(cp_loss[-1]),
        "weather_initial_loss": float(wr_loss[0]),
        "weather_final_loss": float(wr_loss[-1]),
        "community_embed_eff_rank": diag["effective_rank"],
        "weather_embed_eff_rank": w_diag["effective_rank"],
        "community_collapse_ok": len(cv) == 0,
        "geo_corr_max_lat": corr_with_lat,
        "geo_corr_max_lon": corr_with_lon,
    }
    pd.DataFrame([results]).to_csv(OUT / "real_bridge_results.csv", index=False)

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Community:   {len(comm_stages)} stages × {len(comm_features)} feature(s)")
    print(f"  Weather:     {len(weather_stages)} stages × {len(weather_features)} feature(s)")
    print(f"  Community pretrain loss: {cp_loss[-1]:.4f}")
    print(f"  Weather pretrain loss:   {wr_loss[-1]:.4f}")
    print(f"  Community embedding eff.rank: {diag['effective_rank']:.2f}")
    print(f"  Weather embedding eff.rank:   {w_diag['effective_rank']:.2f}")
    if corr_with_lat is not None:
        print(f"  Community vs geography: lat max|r|={corr_with_lat:.3f}, "
              f"lon max|r|={corr_with_lon:.3f}")
    print(f"  Results saved to {OUT / 'real_bridge_results.csv'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
