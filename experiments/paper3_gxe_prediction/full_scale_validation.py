"""Full-scale validation: block-MLP vs GBLUP on ALL 4938 genotypes × 2425 markers.

Fits LD blocks on ALL data, fixes boundaries, then runs both GBLUP and
block-MLP with the SAME 3-fold leave-genotype CV splits.

This answers the reviewer's #1 question: "Does the block-level advantage
hold at full scale, or is it an artifact of the 100-geno × 500-marker subset?"

Also runs the low_rank_gxe (flat) baseline for the full chain of comparison.

Usage:
    PYTHONPATH=src python3 experiments/paper3_gxe_prediction/full_scale_validation.py
"""

import os, sys, time, math
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
    make_low_rank_gxe_predict_fn,
    make_mlp_gxe_predict_fn,
    pivot_environment_tokens_wide,
    pivot_genotype_tokens_wide,
)
from plant_context.statistics.baselines import (
    environment_mean_predict_fn,
    make_gblup_predict_fn,
)
from plant_context.statistics.crossfit import run_crossfit
from plant_context.tokenizers.environment import tokenize_environment_stages
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks

G2F_ROOT = PROJ / "data" / "external" / "g2f"
OUT = PROJ / "experiments" / "paper3_gxe_prediction" / "results_full_validation"
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 3
SEED = 1234
EPOCHS = 80
D_MODEL = 32
DEVICE = os.environ.get("GXE_DEVICE", "cuda:1")


def _ordered_per_sample_dict(lt, id_col, order_col, canonical_order, fcols):
    ps = {}
    for sid, grp in lt.groupby(id_col):
        idxd = grp.set_index(order_col)
        present = [c for c in canonical_order if c in idxd.index]
        ps[sid] = idxd.loc[present, fcols]
    return ps


def main():
    print("=" * 70)
    print("Full-scale validation: block-MLP vs GBLUP vs flat vs env_mean")
    print("ALL 4938 genotypes × 2425 markers, 3-fold leave-genotype CV")
    print("=" * 70)
    sys.stdout.flush()

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    t0 = time.time()
    pheno = load_g2f_phenotype_plot(G2F_ROOT)
    geno = load_g2f_genotype_marker(G2F_ROOT)
    env = load_g2f_environment_daily(G2F_ROOT)
    print(f"  Phenotype: {len(pheno)} rows, {pheno['genotype_id'].nunique()} genos, "
          f"{pheno['environment_id'].nunique()} envs")
    print(f"  Genotype:  {len(geno)} rows, {geno['genotype_id'].nunique()} genos, "
          f"{geno['marker_id'].nunique()} markers")
    print(f"  Weather:   {len(env)} rows, {env['environment_id'].nunique()} envs")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Common genotype / environment set ─────────────────────────────────
    common_g = sorted(set(pheno["genotype_id"]) & set(geno["genotype_id"]))
    print(f"\n[2/5] Common genotypes: {len(common_g)}")
    pheno_s = pheno[pheno["genotype_id"].isin(common_g)].reset_index(drop=True)
    geno_s = geno[geno["genotype_id"].isin(common_g)].reset_index(drop=True)

    # Environment features
    env_daily = env[env["environment_id"].isin(pheno_s["environment_id"])]
    env_tokens = tokenize_environment_stages(env_daily)
    env_feat = pivot_environment_tokens_wide(env_tokens)
    present_env = set(env_feat.index)
    pheno_s = pheno_s[pheno_s["environment_id"].isin(present_env)].reset_index(drop=True)
    print(f"  Environment features: {env_feat.shape}")

    # ── Fit LD blocks ONCE on all data ────────────────────────────────────
    print(f"\n[3/5] Fitting LD blocks on ALL {len(common_g)} genotypes × "
          f"{geno_s['marker_id'].nunique()} markers...")
    t0 = time.time()
    blocks = fit_ld_blocks(
        geno_s, set(geno_s["genotype_id"]), r2_threshold=0.7, max_block_size=50
    )
    n_blocks = blocks["ld_block_id"].nunique()
    print(f"  {n_blocks} blocks from {blocks['marker_id'].nunique()} markers "
          f"({time.time()-t0:.1f}s)")

    # Block tokens — ONCE, reused for all models
    print(f"\n[4/5] Building block tokens for {len(common_g)} genotypes...")
    t0 = time.time()
    gtokens = tokenize_genotype_blocks(geno_s, blocks)
    block_order = list(dict.fromkeys(blocks["ld_block_id"]))
    wide_g = pivot_genotype_tokens_wide(gtokens)
    print(f"  Wide genotype features: {wide_g.shape}")

    per_sample = _ordered_per_sample_dict(
        gtokens, "genotype_id", "ld_block_id", block_order, ["mean_dosage"]
    )
    print(f"  Per-sample dict: {len(per_sample)} genotypes, "
          f"~{gtokens.groupby('genotype_id').size().mean():.0f} blocks/geno")
    print(f"  Built in {time.time()-t0:.1f}s")

    # ── Run all models with the SAME split ────────────────────────────────
    print(f"\n[5/5] Running models (3-fold leave-genotype CV)...")
    sys.stdout.flush()

    split = make_leave_genotype_split(
        pheno_s, n_folds=N_FOLDS, seed=SEED,
        split_version="full_scale_validation",
    )

    models = {}

    # 1. Environment mean (null baseline, uses label) — no sampling needed
    print("\n  --- environment_mean ---")
    t1 = time.time()
    result = run_crossfit(pheno_s, split, environment_mean_predict_fn)
    models["environment_mean"] = {
        "rmse": float(rmse(result["y_true"], result["y_pred"])),
        "r": float(pearson_r(result["y_true"], result["y_pred"])),
        "elapsed_s": round(time.time() - t1, 1),
    }
    print(f"    RMSE={models['environment_mean']['rmse']:.4f}  "
          f"r={models['environment_mean']['r']:.4f}  "
          f"({models['environment_mean']['elapsed_s']}s)")
    sys.stdout.flush()

    # 2. GBLUP (SOTA reference) — on full marker set
    print("\n  --- gblup (full marker set) ---")
    t1 = time.time()
    # Use the genotype_marker_df in long format
    fn_gblup = make_gblup_predict_fn(
        geno_s, max_dosage=1.0,
        lambda_grid=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        n_folds=3, seed=SEED,
    )
    result = run_crossfit(pheno_s, split, fn_gblup)
    models["gblup"] = {
        "rmse": float(rmse(result["y_true"], result["y_pred"])),
        "r": float(pearson_r(result["y_true"], result["y_pred"])),
        "elapsed_s": round(time.time() - t1, 1),
    }
    print(f"    RMSE={models['gblup']['rmse']:.4f}  "
          f"r={models['gblup']['r']:.4f}  "
          f"({models['gblup']['elapsed_s']}s)")
    sys.stdout.flush()

    # 3. Low-rank G×E (flat baseline, with env features)
    # Need to subsample genotypes to avoid OOM — use same 100 genos for flat
    # Actually, the flat low_rank_gxe uses a linear model so memory is fine
    print("\n  --- low_rank_gxe (flat + env features) ---")
    t1 = time.time()
    fn_flat = make_low_rank_gxe_predict_fn(
        wide_g, env_feat, rank=8, hidden=16, epochs=EPOCHS, lr=0.01,
        seed=SEED, device=DEVICE,
    )
    result = run_crossfit(pheno_s, split, fn_flat)
    models["low_rank_gxe"] = {
        "rmse": float(rmse(result["y_true"], result["y_pred"])),
        "r": float(pearson_r(result["y_true"], result["y_pred"])),
        "elapsed_s": round(time.time() - t1, 1),
    }
    print(f"    RMSE={models['low_rank_gxe']['rmse']:.4f}  "
          f"r={models['low_rank_gxe']['r']:.4f}  "
          f"({models['low_rank_gxe']['elapsed_s']}s)")
    sys.stdout.flush()

    # 4. Block-MLP (our method)
    print("\n  --- block_mlp (per-block MLP + pool + env) ---")
    t1 = time.time()
    fn_mlp = make_mlp_gxe_predict_fn(
        per_sample, env_feat, block_feature_columns=["mean_dosage"],
        d_model=16, mlp_hidden=64, epochs=EPOCHS, lr=0.001, seed=SEED,
        device=DEVICE, batch_size=200,
    )
    result = run_crossfit(pheno_s, split, fn_mlp)
    models["block_mlp"] = {
        "rmse": float(rmse(result["y_true"], result["y_pred"])),
        "r": float(pearson_r(result["y_true"], result["y_pred"])),
        "elapsed_s": round(time.time() - t1, 1),
    }
    print(f"    RMSE={models['block_mlp']['rmse']:.4f}  "
          f"r={models['block_mlp']['r']:.4f}  "
          f"({models['block_mlp']['elapsed_s']}s)")
    sys.stdout.flush()

    # ── Results table ─────────────────────────────────────────────────────
    records = []
    for name, m in models.items():
        records.append({"model": name, "rmse": m["rmse"], "r": m["r"], "elapsed_s": m["elapsed_s"]})
    df = pd.DataFrame.from_records(records)
    csv_path = OUT / "full_scale_validation_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}")
    print("FULL-SCALE RESULTS")
    print(f"{'=' * 70}")
    print(f"\n{'Model':25s} {'RMSE':>10s} {'Pearson r':>10s} {'Time':>10s}")
    print('-' * 57)
    for _, row in df.iterrows():
        print(f"  {row['model']:25s} {row['rmse']:>8.4f}  {row['r']:>8.4f}  {row['elapsed_s']:>8.1f}s")

    # ── Comparison to subset results ──────────────────────────────────────
    # Quick comparison with subset
    if "gblup" in models and "block_mlp" in models:
        gb = models["gblup"]["rmse"]
        bm = models["block_mlp"]["rmse"]
        print(f"\n  Δ(block_mlp − gblup): {bm - gb:+.4f}  "
              f"({(bm - gb)/gb*100:+.1f}%)")
        gbr = models["gblup"]["r"]
        bmr = models["block_mlp"]["r"]
        print(f"  r: block_mlp={bmr:.4f} vs gblup={gbr:.4f}")

    print(f"\n  Full-scale results saved to {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
