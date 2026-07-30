"""Permutation test: block-MLP under shuffled genotype labels.

Shuffles phenotype values 200 times (destroying genotype↔yield mapping),
runs block-MLP on each shuffle, and computes the RMSE null distribution.

If the true RMSE (from unshuffled data) is below the 2.5th percentile of
the null distribution, we can reject the null hypothesis that "block-MLP
captures no real signal" at α=0.05 (one-tailed).

Uses seed=1234, 100 genos × 500 markers (same as Gate D subset).

Usage:
    PYTHONPATH=src python3 experiments/paper3_gxe_prediction/permutation_test.py
"""

import os, sys, time, json
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
    make_mlp_gxe_predict_fn,
    make_low_rank_gxe_predict_fn,
    pivot_environment_tokens_wide,
)
from plant_context.statistics.crossfit import run_crossfit
from plant_context.tokenizers.environment import tokenize_environment_stages
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks

G2F_ROOT = PROJ / "data" / "external" / "g2f"
OUT = PROJ / "experiments" / "paper3_gxe_prediction" / "results_permutation"
OUT.mkdir(parents=True, exist_ok=True)

N_GENO = 100
N_MARKERS = 500
EPOCHS = 80
N_FOLDS = 3
D_MODEL = 32
DEVICE = os.environ.get("GXE_DEVICE", "cuda:1")
SEED = 1234
N_PERMUTATIONS = 200


def _ordered_per_sample_dict(lt, id_col, order_col, canonical_order, fcols):
    ps = {}
    for sid, grp in lt.groupby(id_col):
        idxd = grp.set_index(order_col)
        present = [c for c in canonical_order if c in idxd.index]
        ps[sid] = idxd.loc[present, fcols]
    return ps


def main():
    print("=" * 70)
    print("Permutation test: block-MLP under shuffled labels")
    print(f"{N_PERMUTATIONS} permutations, seed={SEED}, "
          f"{N_GENO} genos, {N_MARKERS} markers")
    print("=" * 70)
    sys.stdout.flush()

    # ── Load and build features ONCE ──────────────────────────────────────
    rng = np.random.default_rng(SEED)
    print("\n[1] Loading data...")
    pheno = load_g2f_phenotype_plot(G2F_ROOT)
    geno = load_g2f_genotype_marker(G2F_ROOT)
    env = load_g2f_environment_daily(G2F_ROOT)

    all_g = sorted(set(pheno["genotype_id"]) & set(geno["genotype_id"]))
    chosen = set(rng.choice(all_g, size=min(N_GENO, len(all_g)), replace=False))
    pheno_s = pheno[pheno["genotype_id"].isin(chosen)].reset_index(drop=True)
    geno_s = geno[geno["genotype_id"].isin(chosen)].reset_index(drop=True)

    all_m = sorted(geno_s["marker_id"].unique())
    if len(all_m) > N_MARKERS:
        cm = set(rng.choice(all_m, size=N_MARKERS, replace=False))
        geno_s = geno_s[geno_s["marker_id"].isin(cm)].reset_index(drop=True)

    env_daily = env[env["environment_id"].isin(pheno_s["environment_id"])]
    env_tokens = tokenize_environment_stages(env_daily)
    env_feat = pivot_environment_tokens_wide(env_tokens)
    present_env = set(env_feat.index)
    pheno_s = pheno_s[pheno_s["environment_id"].isin(present_env)].reset_index(drop=True)
    print(f"  {len(pheno_s)} samples, {len(chosen)} genos, {len(all_m)} markers")

    # ── Fit LD blocks and build per-sample dict ONCE ─────────────────────
    print("\n[2] Building features...")
    blocks = fit_ld_blocks(geno_s, set(geno_s["genotype_id"]),
                           r2_threshold=0.7, max_block_size=50)
    block_order = list(dict.fromkeys(blocks["ld_block_id"]))
    gtokens = tokenize_genotype_blocks(geno_s, blocks)
    per_sample = _ordered_per_sample_dict(
        gtokens, "genotype_id", "ld_block_id", block_order, ["mean_dosage"]
    )
    n_blocks = len(block_order)
    print(f"  {n_blocks} blocks")

    # ── True result (unshuffled) ──────────────────────────────────────────
    print("\n[3] True result (unshuffled)...")
    t0 = time.time()

    def run_mlp(pheno_df, run_label):
        split = make_leave_genotype_split(
            pheno_df, n_folds=N_FOLDS, seed=SEED,
            split_version=f"perm_{run_label}",
        )
        fn = make_mlp_gxe_predict_fn(
            per_sample, env_feat, block_feature_columns=["mean_dosage"],
            d_model=D_MODEL, epochs=EPOCHS, lr=0.001, seed=SEED, device=DEVICE,
        )
        result = run_crossfit(pheno_df, split, fn)
        return {
            "rmse": float(rmse(result["y_true"], result["y_pred"])),
            "r": float(pearson_r(result["y_true"], result["y_pred"])),
        }

    true_result = run_mlp(pheno_s, "true")
    true_rmse = true_result["rmse"]
    true_r = true_result["r"]
    print(f"  True RMSE = {true_rmse:.4f}, r = {true_r:.4f} ({time.time()-t0:.0f}s)")

    # ── Permutations ──────────────────────────────────────────────────────
    print(f"\n[4] Running {N_PERMUTATIONS} permutations...")
    sys.stdout.flush()

    perm_rmses = []
    perm_rs = []
    n_completed = 0

    for perm_i in range(N_PERMUTATIONS):
        pheno_perm = pheno_s.copy()
        # Shuffle phenotype values within each fold to preserve fold structure
        # Actually: shuffle the entire phenotype column (breaks all G×E signal)
        perm_rng = np.random.default_rng(SEED + 1 + perm_i)
        perm_rng.shuffle(pheno_perm["phenotype_value"].values)

        t0 = time.time()
        result = run_mlp(pheno_perm, f"perm_{perm_i}")
        elapsed = time.time() - t0

        perm_rmses.append(result["rmse"])
        perm_rs.append(result["r"])
        n_completed += 1

        if (perm_i + 1) % 20 == 0:
            print(f"  {perm_i+1}/{N_PERMUTATIONS} complete "
                  f"(latest RMSE={result['rmse']:.4f}, {elapsed:.0f}s/perm)")
            sys.stdout.flush()

    # ── Results ────────────────────────────────────────────────────────────
    null_rmses = np.array(perm_rmses)
    null_rs = np.array(perm_rs)

    pct_below = (null_rmses < true_rmse).mean() * 100
    pct_above = (null_rmses > true_rmse).mean() * 100

    summary = {
        "n_permutations": N_PERMUTATIONS,
        "true_rmse": true_rmse,
        "true_r": true_r,
        "null_rmse_mean": float(null_rmses.mean()),
        "null_rmse_std": float(null_rmses.std()),
        "null_rmse_2.5pct": float(np.percentile(null_rmses, 2.5)),
        "null_rmse_50pct": float(np.percentile(null_rmses, 50)),
        "null_rmse_97.5pct": float(np.percentile(null_rmses, 97.5)),
        "pct_null_below_true": float(pct_below),
        "pct_null_above_true": float(pct_above),
        "null_r_mean": float(null_rs.mean()),
        "null_r_std": float(null_rs.std()),
        "true_r_percentile": float((null_rs < true_r).mean() * 100),
    }

    # Save results
    df_results = pd.DataFrame({
        "true_rmse": [true_rmse],
        "null_mean": [null_rmses.mean()],
        "null_std": [null_rmses.std()],
        "null_2.5": [float(np.percentile(null_rmses, 2.5))],
        "null_97.5": [float(np.percentile(null_rmses, 97.5))],
        "pct_below_true": [pct_below],
        "n_permutations": [N_PERMUTATIONS],
    })
    df_results.to_csv(OUT / "permutation_summary.csv", index=False)

    df_perm = pd.DataFrame({"perm_rmse": null_rmses, "perm_r": null_rs})
    df_perm.to_csv(OUT / "permutation_distribution.csv", index=False)

    print(f"\n{'=' * 70}")
    print("PERMUTATION TEST RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  True RMSE:          {true_rmse:.4f}  (r={true_r:.4f})")
    print(f"  Null distribution:  μ={null_rmses.mean():.4f}  σ={null_rmses.std():.4f}")
    print(f"  Null 2.5%:          {np.percentile(null_rmses, 2.5):.4f}")
    print(f"  Null 50%:           {np.percentile(null_rmses, 50):.4f}")
    print(f"  Null 97.5%:         {np.percentile(null_rmses, 97.5):.4f}")
    print(f"\n  True RMSE is below {pct_below:.1f}% of null distribution")
    print(f"  True r is above {summary['true_r_percentile']:.1f}% of null distribution")

    if pct_below < 2.5:
        print(f"\n  ✅ Significant at α=0.05: true RMSE < 2.5th percentile of null")
    elif pct_below < 5:
        print(f"\n  ⚠  Marginally significant: true RMSE < 5th percentile of null")
    else:
        print(f"\n  ❌ Not significant: true RMSE is in the bulk of the null distribution")

    # Standardized effect
    z = (true_rmse - null_rmses.mean()) / null_rmses.std()
    print(f"  Z-score: {z:.2f} (true RMSE is {abs(z):.1f}σ "
          f"{'below' if z < 0 else 'above'} null mean)")

    # Compare to env_mean (which doesn't use genotype signal)
    print(f"\n  Env mean RMSE (approx null lower bound, from subset data): ~2.20")
    print(f"  True RMSE vs env_mean: {true_rmse - 2.20:+.2f}")

    print(f"\n  Saved to {OUT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
