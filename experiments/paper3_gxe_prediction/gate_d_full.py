"""Gate D full: all markers, masked reconstruction pretraining value.

Same as gate_d_pretraining.py but using ALL 2425 markers (no subsample) and
fewer epochs for speed. Only pretrain_ft vs scratch (baseline from prev run).

Usage:
    PYTHONPATH=src python3 experiments/paper3_gxe_prediction/gate_d_full.py
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
from plant_context.models.genotype_encoder import pretrain_genotype_encoder
from plant_context.models.gxe_model import make_encoder_gxe_predict_fn
from plant_context.statistics.crossfit import run_crossfit
from plant_context.tokenizers.environment import STAGE_ORDER, tokenize_environment_stages
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks
from plant_context.models.gxe_model import pivot_environment_tokens_wide

G2F_ROOT = PROJ / "data" / "external" / "g2f"
OUT = PROJ / "experiments" / "paper3_gxe_prediction" / "results_gate_d_full"
OUT.mkdir(parents=True, exist_ok=True)

N_GENO = 100
EPOCHS_PRETRAIN = 30     # reduced for speed
EPOCHS_GXE = 30          # reduced for speed
N_FOLDS = 3
D_MODEL = 32
DEVICE = "cuda:1"
SEED = 1234
rng = np.random.default_rng(SEED)

def _ordered_per_sample_dict(lt, id_col, order_col, canonical_order, feature_columns):
    ps = {}
    for sid, grp in lt.groupby(id_col):
        idxd = grp.set_index(order_col)
        present = [c for c in canonical_order if c in idxd.index]
        ps[sid] = idxd.loc[present, feature_columns]
    return ps

def main():
    print("=" * 60)
    print("Gate D Full: all markers, pretraining value")
    print("=" * 60)
    sys.stdout.flush()

    pheno = load_g2f_phenotype_plot(G2F_ROOT)
    geno = load_g2f_genotype_marker(G2F_ROOT)
    env = load_g2f_environment_daily(G2F_ROOT)

    all_g = sorted(set(pheno["genotype_id"]) & set(geno["genotype_id"]))
    chosen = set(rng.choice(all_g, size=min(N_GENO, len(all_g)), replace=False))
    pheno_s = pheno[pheno["genotype_id"].isin(chosen)].reset_index(drop=True)
    geno_s = geno[geno["genotype_id"].isin(chosen)].reset_index(drop=True)
    print(f"  {len(chosen)} genotypes, {geno_s['marker_id'].nunique()} markers")
    sys.stdout.flush()

    # Environment features
    env_daily = env[env["environment_id"].isin(pheno_s["environment_id"])]
    env_tokens = tokenize_environment_stages(env_daily)
    env_feat = pivot_environment_tokens_wide(env_tokens)
    present_env = set(env_feat.index)
    pheno_s = pheno_s[pheno_s["environment_id"].isin(present_env)].reset_index(drop=True)
    print(f"  env: {env_feat.shape}, pheno: {len(pheno_s)}")
    sys.stdout.flush()

    # LD blocks (all markers)
    blocks = fit_ld_blocks(geno_s, set(geno_s["genotype_id"]),
                           r2_threshold=0.7, max_block_size=50)
    gtokens = tokenize_genotype_blocks(geno_s, blocks)
    block_order = list(dict.fromkeys(blocks["ld_block_id"]))
    n_blocks = gtokens["ld_block_id"].nunique()
    print(f"  {n_blocks} LD blocks")
    sys.stdout.flush()

    per_sample_tokens = _ordered_per_sample_dict(
        gtokens, "genotype_id", "ld_block_id", block_order, ["mean_dosage"],
    )
    split = make_leave_genotype_split(pheno_s, n_folds=N_FOLDS, seed=SEED, split_version="gate_d_full")
    print(f"  split: {len(split)} rows, {N_FOLDS} folds")
    sys.stdout.flush()

    # ── Pretrain ──────────────────────────────────────────────
    print("\n[4] Condition: pretrain_ft")
    sys.stdout.flush()
    t0 = time.time()
    pt_res = pretrain_genotype_encoder(
        gtokens, block_order, feature_columns=["mean_dosage"],
        mask_fraction=0.15, d_model=D_MODEL, epochs=EPOCHS_PRETRAIN,
        lr=0.01, seed=SEED, device=DEVICE,
    )
    print(f"  pretrain: {pt_res['loss_history'][0]:.4f} → {pt_res['loss_history'][-1]:.4f} ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

    t0 = time.time()
    fn_pt = make_encoder_gxe_predict_fn(
        per_sample_tokens, env_feat,
        block_feature_columns=["mean_dosage"],
        d_model=D_MODEL, epochs=EPOCHS_GXE, lr=0.001, seed=SEED,
        genotype_encoder=pt_res["encoder"].encoder, device=DEVICE,
    )
    r_pt = run_crossfit(pheno_s, split, fn_pt)
    rmse_pt = float(rmse(r_pt["y_true"], r_pt["y_pred"]))
    r_pt_val = float(pearson_r(r_pt["y_true"], r_pt["y_pred"]))
    print(f"  RMSE={rmse_pt:.4f}  r={r_pt_val:.4f} ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

    # ── Scratch ────────────────────────────────────────────────
    print("\n[5] Condition: scratch")
    sys.stdout.flush()
    t0 = time.time()
    fn_sc = make_encoder_gxe_predict_fn(
        per_sample_tokens, env_feat,
        block_feature_columns=["mean_dosage"],
        d_model=D_MODEL, epochs=EPOCHS_GXE, lr=0.001, seed=SEED,
        genotype_encoder=None, device=DEVICE,
    )
    r_sc = run_crossfit(pheno_s, split, fn_sc)
    rmse_sc = float(rmse(r_sc["y_true"], r_sc["y_pred"]))
    r_sc_val = float(pearson_r(r_sc["y_true"], r_sc["y_pred"]))
    print(f"  RMSE={rmse_sc:.4f}  r={r_sc_val:.4f} ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

    # ── Results ────────────────────────────────────────────────
    records = [
        {"label": "pretrain_ft", "rmse": rmse_pt, "pearson_r": r_pt_val},
        {"label": "scratch", "rmse": rmse_sc, "pearson_r": r_sc_val},
    ]
    df = pd.DataFrame.from_records(records)
    csv_path = OUT / "gate_d_full_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 60}")
    print("Gate D Full Results:")
    for rec in records:
        print(f"  {rec['label']:15s}: RMSE={rec['rmse']:.4f}  r={rec['pearson_r']:.4f}")
    print(f"  ΔRMSE (pretrain - scratch): {rmse_pt - rmse_sc:+.4f}")
    print(f"  Saved to {csv_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
