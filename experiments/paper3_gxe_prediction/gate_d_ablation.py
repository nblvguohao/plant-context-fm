"""Gate D ablation: encoder vs MLP vs flat baseline, 5 seeds each.

Key question: Does Transformer cross-block attention add value beyond
per-block processing? Compare:
  1. encoder_scratch — TokenSequenceEncoder (self-attention)
  2. mlp_ablation   — Per-block MLP + pooling (no cross-block comm)
  3. flat_baseline  — Current wide-feature model

Also runs 5 seeds to compute mean ± std RMSE per condition.

Usage:
    PYTHONPATH=src python3 experiments/paper3_gxe_prediction/gate_d_ablation.py
"""

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from plant_context.data.g2f_adapter import (
    load_g2f_environment_daily, load_g2f_genotype_marker, load_g2f_phenotype_plot,
)
from plant_context.evaluation.metrics import rmse, pearson_r
from plant_context.evaluation.splits import make_leave_genotype_split
from plant_context.models.gxe_model import (
    make_encoder_gxe_predict_fn,
    make_mlp_gxe_predict_fn,
    make_low_rank_gxe_predict_fn,
    pivot_environment_tokens_wide,
    pivot_genotype_tokens_wide,
)
from plant_context.statistics.crossfit import run_crossfit
from plant_context.tokenizers.environment import STAGE_ORDER, tokenize_environment_stages
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks

G2F_ROOT = PROJ / "data" / "external" / "g2f"
OUT = PROJ / "experiments" / "paper3_gxe_prediction" / "results_gate_d_ablation"
OUT.mkdir(parents=True, exist_ok=True)

N_GENO = 100
N_MARKERS = 500          # same as Gate D 500-marker run
EPOCHS = 80
N_FOLDS = 3
D_MODEL = 32
DEVICE = "cuda:1"
SEEDS = [1234, 2345, 3456, 4567, 5678]


def _ordered_per_sample_dict(lt, id_col, order_col, canonical_order, fcols):
    ps = {}
    for sid, grp in lt.groupby(id_col):
        idxd = grp.set_index(order_col)
        present = [c for c in canonical_order if c in idxd.index]
        ps[sid] = idxd.loc[present, fcols]
    return ps


def run_condition(label, make_fn_kwargs, seed):
    """Run one condition with a specific seed, return RMSE and r."""
    rng = np.random.default_rng(seed)

    pheno = load_g2f_phenotype_plot(G2F_ROOT)
    geno = load_g2f_genotype_marker(G2F_ROOT)
    env = load_g2f_environment_daily(G2F_ROOT)

    all_g = sorted(set(pheno["genotype_id"]) & set(geno["genotype_id"]))
    chosen = set(rng.choice(all_g, size=min(N_GENO, len(all_g)), replace=False))
    pheno_s = pheno[pheno["genotype_id"].isin(chosen)].reset_index(drop=True)
    geno_s = geno[geno["genotype_id"].isin(chosen)].reset_index(drop=True)

    # Subsample markers
    all_m = sorted(geno_s["marker_id"].unique())
    if len(all_m) > N_MARKERS:
        cm = set(rng.choice(all_m, size=N_MARKERS, replace=False))
        geno_s = geno_s[geno_s["marker_id"].isin(cm)].reset_index(drop=True)

    # Environment features
    env_daily = env[env["environment_id"].isin(pheno_s["environment_id"])]
    env_tokens = tokenize_environment_stages(env_daily)
    env_feat = pivot_environment_tokens_wide(env_tokens)
    present_env = set(env_feat.index)
    pheno_s = pheno_s[pheno_s["environment_id"].isin(present_env)].reset_index(drop=True)

    # LD blocks
    blocks = fit_ld_blocks(geno_s, set(geno_s["genotype_id"]), r2_threshold=0.7, max_block_size=50)
    gtokens = tokenize_genotype_blocks(geno_s, blocks)
    block_order = list(dict.fromkeys(blocks["ld_block_id"]))
    wide_g = pivot_genotype_tokens_wide(gtokens)

    per_sample = _ordered_per_sample_dict(
        gtokens, "genotype_id", "ld_block_id", block_order, ["mean_dosage"]
    )

    split = make_leave_genotype_split(pheno_s, n_folds=N_FOLDS, seed=seed,
                                      split_version=f"ablation_{label}_{seed}")

    if label == "flat_baseline":
        fn = make_low_rank_gxe_predict_fn(
            wide_g, env_feat, rank=8, hidden=16, epochs=EPOCHS, lr=0.01,
            seed=seed, device=DEVICE,
        )
    elif label == "mlp_ablation":
        fn = make_mlp_gxe_predict_fn(
            per_sample, env_feat, block_feature_columns=["mean_dosage"],
            d_model=D_MODEL, epochs=EPOCHS, lr=0.001, seed=seed, device=DEVICE,
        )
    else:
        fn = make_encoder_gxe_predict_fn(
            per_sample, env_feat, block_feature_columns=["mean_dosage"],
            d_model=D_MODEL, epochs=EPOCHS, lr=0.001, seed=seed, device=DEVICE,
        )

    result = run_crossfit(pheno_s, split, fn)
    return {
        "seed": seed,
        "rmse": float(rmse(result["y_true"], result["y_pred"])),
        "pearson_r": float(pearson_r(result["y_true"], result["y_pred"])),
    }


def main():
    print("=" * 60)
    print("Gate D ablation: encoder vs MLP vs baseline (5 seeds)")
    print("=" * 60)
    sys.stdout.flush()

    labels = ["encoder_scratch", "mlp_ablation", "flat_baseline"]
    all_records = []

    for label in labels:
        print(f"\n{'─' * 50}")
        print(f"  {label} ({len(SEEDS)} seeds)")
        print(f"{'─' * 50}")
        sys.stdout.flush()

        for seed in SEEDS:
            t0 = time.time()
            rec = run_condition(label, {}, seed)
            elapsed = time.time() - t0
            print(f"    seed={seed}: RMSE={rec['rmse']:.4f}  r={rec['pearson_r']:.4f}  ({elapsed:.0f}s)")
            sys.stdout.flush()
            rec["label"] = label
            all_records.append(rec)

        # Summarize
        vals = [r for r in all_records if r["label"] == label]
        rmses = [v["rmse"] for v in vals]
        rs = [v["pearson_r"] for v in vals]
        print(f"    → {label}: RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}  "
              f"r={np.mean(rs):.4f}±{np.std(rs):.4f}")
        sys.stdout.flush()

    df = pd.DataFrame.from_records(all_records)
    csv_path = OUT / "gate_d_ablation_results.csv"
    df.to_csv(csv_path, index=False)

    # Summary table
    print(f"\n{'=' * 60}")
    print("Summary (mean ± std over 5 seeds):")
    for label in labels:
        vals = df[df["label"] == label]
        print(f"  {label:20s}: RMSE={vals['rmse'].mean():.4f}±{vals['rmse'].std():.4f}  "
              f"r={vals['pearson_r'].mean():.4f}±{vals['pearson_r'].std():.4f}")

    # Key comparisons
    enc = df[df["label"] == "encoder_scratch"]["rmse"]
    mlp = df[df["label"] == "mlp_ablation"]["rmse"]
    flat = df[df["label"] == "flat_baseline"]["rmse"]
    print(f"\n  Δ(encoder - flat): {enc.mean() - flat.mean():+.4f}±"
          f"{np.sqrt(enc.var()/5 + flat.var()/5):.4f}")
    print(f"  Δ(mlp - flat):     {mlp.mean() - flat.mean():+.4f}±"
          f"{np.sqrt(mlp.var()/5 + flat.var()/5):.4f}")
    print(f"  Δ(encoder - mlp):  {enc.mean() - mlp.mean():+.4f}±"
          f"{np.sqrt(enc.var()/5 + mlp.var()/5):.4f}")
    print(f"  Saved to {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
