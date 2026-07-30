"""Gate D: masked reconstruction pretraining value for G×E prediction.

Three conditions on 100 genotypes (leave-genotype-out, 3-fold):

  1. **pretrain_ft**:   Masked-reconstruction pretrain → fine-tune G×E
  2. **scratch**:        Random encoder init → train G×E
  3. **baseline**:       Current flat-feature model (no encoder, wide features)

Usage:
    PYTHONPATH=src python3 experiments/paper3_gxe_prediction/gate_d_pretraining.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from plant_context.data.g2f_adapter import (
    load_g2f_environment_daily,
    load_g2f_genotype_marker,
    load_g2f_phenotype_plot,
)
from plant_context.evaluation.metrics import rmse, pearson_r
from plant_context.evaluation.splits import make_leave_genotype_split
from plant_context.models.genotype_encoder import (
    GenotypeBlockEncoder,
    pretrain_genotype_encoder,
)
from plant_context.models.gxe_model import (
    make_encoder_gxe_predict_fn,
    make_low_rank_gxe_predict_fn,
    pivot_environment_tokens_wide,
    pivot_genotype_tokens_wide,
)
from plant_context.statistics.crossfit import run_crossfit
from plant_context.tokenizers.environment import STAGE_ORDER, tokenize_environment_stages
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks

G2F_ROOT = PROJ / "data" / "external" / "g2f"
OUT = PROJ / "experiments" / "paper3_gxe_prediction" / "results_gate_d"
OUT.mkdir(parents=True, exist_ok=True)

N_GENO = 100
N_MARKERS = 500          # subsample to keep transformer fast on GPU
EPOCHS_PRETRAIN = 80
EPOCHS_GXE = 80
N_FOLDS = 3
D_MODEL = 32
BLOCK_SIZE = 50
LD_R2 = 0.7
DEVICE = "cuda:1"
SEED = 1234
rng = np.random.default_rng(SEED)


def _ordered_per_sample_dict(long_tokens, id_col, order_col, canonical_order, feature_columns):
    """Convert long-format tokens to per-sample dict with deterministic order."""
    per_sample = {}
    for sample_id, group in long_tokens.groupby(id_col):
        indexed = group.set_index(order_col)
        present_in_order = [c for c in canonical_order if c in indexed.index]
        per_sample[sample_id] = indexed.loc[present_in_order, feature_columns]
    return per_sample


def main():
    print("=" * 60)
    print("Gate D: Pretraining value for G×E prediction")
    print("=" * 60)
    sys.stdout.flush()

    # ── [1] Load data ──────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    pheno = load_g2f_phenotype_plot(G2F_ROOT)
    geno = load_g2f_genotype_marker(G2F_ROOT)
    env = load_g2f_environment_daily(G2F_ROOT)
    print(f"  pheno: {len(pheno)} rows, geno: {len(geno)} rows")
    sys.stdout.flush()

    # Subset genotypes
    all_g = sorted(set(pheno["genotype_id"]) & set(geno["genotype_id"]))
    chosen = set(rng.choice(all_g, size=min(N_GENO, len(all_g)), replace=False))
    pheno_s = pheno[pheno["genotype_id"].isin(chosen)].reset_index(drop=True)
    geno_s = geno[geno["genotype_id"].isin(chosen)].reset_index(drop=True)
    print(f"  {len(chosen)} genotypes")
    sys.stdout.flush()

    # Subsample markers for fast transformer
    all_markers = sorted(geno_s["marker_id"].unique())
    if len(all_markers) > N_MARKERS:
        chosen_marks = set(rng.choice(all_markers, size=N_MARKERS, replace=False))
        geno_s = geno_s[geno_s["marker_id"].isin(chosen_marks)].reset_index(drop=True)
        print(f"  Subsampled to {N_MARKERS} markers")
    sys.stdout.flush()

    # ── [2] Environment features (wide format, same for all conditions) ────
    print("\n[2] Environment features...")
    env_daily = env[env["environment_id"].isin(pheno_s["environment_id"])]
    env_tokens = tokenize_environment_stages(env_daily)
    env_feat = pivot_environment_tokens_wide(env_tokens)
    # Drop phenotype rows whose environment_id is NOT in env_feat
    present_env = set(env_feat.index)
    before = len(pheno_s)
    pheno_s = pheno_s[pheno_s["environment_id"].isin(present_env)].reset_index(drop=True)
    if len(pheno_s) < before:
        print(f"  Dropped {before - len(pheno_s)} pheno rows with missing environment features")
    print(f"  env features: {env_feat.shape}, pheno after filter: {len(pheno_s)}")
    sys.stdout.flush()

    # ── [3] LD blocks and genotype tokens ──────────────────────────────────
    print("\n[3] LD blocks and genotype tokens...")
    blocks = fit_ld_blocks(geno_s, set(geno_s["genotype_id"]),
                           r2_threshold=LD_R2, max_block_size=BLOCK_SIZE)
    gtokens = tokenize_genotype_blocks(geno_s, blocks)
    n_blocks = gtokens["ld_block_id"].nunique()
    block_order = list(dict.fromkeys(blocks["ld_block_id"]))
    print(f"  {n_blocks} LD blocks, {len(gtokens)} token rows")
    sys.stdout.flush()

    # Wide features for baseline condition
    wide_g = pivot_genotype_tokens_wide(gtokens)
    print(f"  wide features: {wide_g.shape}")
    sys.stdout.flush()

    # Per-sample dict for encoder-based conditions
    block_feature_columns = ["mean_dosage"]
    per_sample_tokens = _ordered_per_sample_dict(
        gtokens, "genotype_id", "ld_block_id", block_order,
        block_feature_columns,
    )
    print(f"  {len(per_sample_tokens)} genotypes with block sequences")
    sys.stdout.flush()

    # ── [4] Split (leave-genotype-out) ────────────────────────────────────
    split = make_leave_genotype_split(pheno_s, n_folds=N_FOLDS, seed=SEED,
                                      split_version="gate_d")
    print(f"  split: {split.shape[0]} rows, {N_FOLDS} folds")
    sys.stdout.flush()

    # ── [5] Condition 1: Pretrain → fine-tune ─────────────────────────────
    print("\n[4] Condition: pretrain_ft")
    sys.stdout.flush()
    t0 = time.time()

    # Pretrain encoder
    pretrain_result = pretrain_genotype_encoder(
        gtokens, block_order,
        feature_columns=block_feature_columns,
        mask_fraction=0.15,
        d_model=D_MODEL, epochs=EPOCHS_PRETRAIN, lr=0.01, seed=SEED,
        device=DEVICE,
    )
    pretrained_encoder = pretrain_result["encoder"]
    pt_loss = pretrain_result["loss_history"]
    print(f"  pretrain loss: {pt_loss[0]:.4f} → {pt_loss[-1]:.4f}  "
          f"({time.time()-t0:.0f}s)")
    sys.stdout.flush()

    # Fine-tune G×E with pretrained encoder (lower LR for transformer stability)
    t0 = time.time()
    fn_pt = make_encoder_gxe_predict_fn(
        per_sample_tokens, env_feat,
        block_feature_columns=block_feature_columns,
        d_model=D_MODEL, epochs=EPOCHS_GXE, lr=0.001, seed=SEED,
        genotype_encoder=pretrained_encoder.encoder,
        device=DEVICE,
    )
    result_pt = run_crossfit(pheno_s, split, fn_pt)
    rmse_pt = float(rmse(result_pt["y_true"], result_pt["y_pred"]))
    r_pt = float(pearson_r(result_pt["y_true"], result_pt["y_pred"]))
    print(f"  RMSE={rmse_pt:.4f}  r={r_pt:.4f}  ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

    # ── [6] Condition 2: Scratch (random encoder) ──────────────────────────
    print("\n[5] Condition: scratch")
    sys.stdout.flush()
    t0 = time.time()

    # Don't pass an encoder → randomly initialized in make_encoder_gxe_predict_fn
    fn_sc = make_encoder_gxe_predict_fn(
        per_sample_tokens, env_feat,
        block_feature_columns=block_feature_columns,
        d_model=D_MODEL, epochs=EPOCHS_GXE, lr=0.001, seed=SEED,
        genotype_encoder=None,
        device=DEVICE,
    )
    result_sc = run_crossfit(pheno_s, split, fn_sc)
    rmse_sc = float(rmse(result_sc["y_true"], result_sc["y_pred"]))
    r_sc = float(pearson_r(result_sc["y_true"], result_sc["y_pred"]))
    print(f"  RMSE={rmse_sc:.4f}  r={r_sc:.4f}  ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

    # ── [7] Condition 3: Baseline (flat features, no encoder) ─────────────
    print("\n[6] Condition: baseline")
    sys.stdout.flush()
    t0 = time.time()

    fn_bl = make_low_rank_gxe_predict_fn(
        wide_g, env_feat,
        rank=8, hidden=16, epochs=EPOCHS_GXE, lr=0.01, seed=SEED,
        device=DEVICE,
    )
    result_bl = run_crossfit(pheno_s, split, fn_bl)
    rmse_bl = float(rmse(result_bl["y_true"], result_bl["y_pred"]))
    r_bl = float(pearson_r(result_bl["y_true"], result_bl["y_pred"]))
    print(f"  RMSE={rmse_bl:.4f}  r={r_bl:.4f}  ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

    # ── [8] Summary ────────────────────────────────────────────────────────
    records = [
        {"label": "pretrain_ft", "rmse": rmse_pt, "pearson_r": r_pt},
        {"label": "scratch", "rmse": rmse_sc, "pearson_r": r_sc},
        {"label": "baseline", "rmse": rmse_bl, "pearson_r": r_bl},
    ]
    df = pd.DataFrame.from_records(records)
    df["delta_vs_baseline"] = df["rmse"] - rmse_bl
    csv_path = OUT / "gate_d_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 60}")
    print("Gate D Results:")
    print(f"{'Condition':20s}  {'RMSE':>8s}  {'r':>8s}  {'Δvs baseline':>12s}")
    print("-" * 52)
    for rec in records:
        delta = rec["rmse"] - rmse_bl
        print(f"{rec['label']:20s}  {rec['rmse']:8.4f}  {rec['pearson_r']:8.4f}  "
              f"{delta:+10.4f}")
    print(f"\nPretrain vs scratch ΔRMSE: {rmse_pt - rmse_sc:+.4f}")
    print(f"Saved to {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
