"""Gate C repeat: structured token value, 5 seeds each.

Measures mean ± std RMSE for correct LD blocks vs shuffled vs
single_marker, providing error bars for the main structured-token claim.

Usage:
    PYTHONPATH=src python3 experiments/paper3_gxe_prediction/gate_c_repeat.py
"""

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from plant_context.data.g2f_adapter import load_g2f_phenotype_plot, load_g2f_genotype_marker, load_g2f_environment_daily
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks
from plant_context.tokenizers.environment import tokenize_environment_stages
from plant_context.models.gxe_model import pivot_genotype_tokens_wide, pivot_environment_tokens_wide, make_low_rank_gxe_predict_fn
from plant_context.evaluation.splits import make_leave_genotype_split
from plant_context.evaluation.metrics import rmse, pearson_r
from plant_context.statistics.crossfit import run_crossfit

G2F_ROOT = PROJ / "data" / "external" / "g2f"
OUT = PROJ / "experiments" / "paper3_gxe_prediction" / "results_gate_c_repeat"
OUT.mkdir(parents=True, exist_ok=True)

N_GENO = 100
EPOCHS = 80
N_FOLDS = 3
SEEDS = [1234, 2345, 3456, 4567, 5678]


def run(label, blocks, geno, pheno, env_features, seed):
    tokens = tokenize_genotype_blocks(geno, blocks)
    g_feat = pivot_genotype_tokens_wide(tokens)

    p = pheno[pheno["genotype_id"].isin(g_feat.index) & pheno["environment_id"].isin(env_features.index)].reset_index(drop=True)
    split = make_leave_genotype_split(p, n_folds=N_FOLDS, seed=seed, split_version=f"gate_c_{label}_{seed}")
    fn = make_low_rank_gxe_predict_fn(g_feat, env_features, rank=4, hidden=16, epochs=EPOCHS, lr=0.01, seed=seed)
    result = run_crossfit(p, split, fn)

    return {
        "rmse": float(rmse(result["y_true"], result["y_pred"])),
        "pearson_r": float(pearson_r(result["y_true"], result["y_pred"])),
    }


def shuffle_blocks(blocks, rng):
    shuf = blocks.copy()
    for chrom in shuf["chromosome"].unique():
        mask = shuf["chromosome"] == chrom
        mids = shuf.loc[mask, "marker_id"].tolist()
        rng.shuffle(mids)
        shuf.loc[mask, "marker_id"] = mids
    return shuf


def single_marker_blocks(geno):
    m = geno[["marker_id", "chromosome", "position"]].drop_duplicates("marker_id")
    m["ld_block_id"] = m["marker_id"]
    return m.reset_index(drop=True)


def main():
    print("=" * 60)
    print("Gate C repeat: structured tokens (5 seeds)")
    print("=" * 60)
    sys.stdout.flush()

    # Load data once
    pheno = load_g2f_phenotype_plot(G2F_ROOT)
    geno = load_g2f_genotype_marker(G2F_ROOT)
    env = load_g2f_environment_daily(G2F_ROOT)

    labels = ["correct", "shuffled", "single_marker"]
    all_records = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        print(f"\n--- seed={seed} ---")
        sys.stdout.flush()

        all_g = sorted(set(pheno["genotype_id"]) & set(geno["genotype_id"]))
        chosen = set(rng.choice(all_g, size=min(N_GENO, len(all_g)), replace=False))
        pheno_s = pheno[pheno["genotype_id"].isin(chosen)].reset_index(drop=True)
        geno_s = geno[geno["genotype_id"].isin(chosen)].reset_index(drop=True)

        # Environment features
        env_daily = env[env["environment_id"].isin(pheno_s["environment_id"])]
        env_tokens = tokenize_environment_stages(env_daily)
        env_feat = pivot_environment_tokens_wide(env_tokens)

        # Single-marker blocks (same across seeds — just for comparison)
        sm_blocks = single_marker_blocks(geno_s)

        # Fit LD blocks on this seed's genotype subset
        correct_blocks = fit_ld_blocks(geno_s, set(geno_s["genotype_id"]), r2_threshold=0.7, max_block_size=50)
        shuf_rng = np.random.default_rng(seed + 100)
        shuf_blocks = shuffle_blocks(correct_blocks, shuf_rng)

        for label, blocks in [("correct", correct_blocks), ("shuffled", shuf_blocks), ("single_marker", sm_blocks)]:
            t0 = time.time()
            rec = run(label, blocks, geno_s, pheno_s, env_feat, seed)
            elapsed = time.time() - t0
            print(f"  {label:15s}: RMSE={rec['rmse']:.4f}  r={rec['pearson_r']:.4f} ({elapsed:.0f}s)")
            sys.stdout.flush()
            rec["label"] = label
            rec["seed"] = seed
            all_records.append(rec)

    df = pd.DataFrame.from_records(all_records)
    df.to_csv(OUT / "gate_c_repeat_results.csv", index=False)

    print(f"\n{'=' * 60}")
    print("Gate C Repeat Summary (mean ± std over 5 seeds):")
    for label in labels:
        vals = df[df["label"] == label]
        print(f"  {label:15s}: RMSE={vals['rmse'].mean():.4f}±{vals['rmse'].std():.4f}  "
              f"r={vals['pearson_r'].mean():.4f}±{vals['pearson_r'].std():.4f}")

    # Pairwise deltas
    c = df[df["label"] == "correct"]["rmse"]
    s = df[df["label"] == "shuffled"]["rmse"]
    m = df[df["label"] == "single_marker"]["rmse"]
    print(f"\n  Δ(shuffled - correct): {s.mean() - c.mean():+.4f} ± "
          f"{np.sqrt(s.var()/5 + c.var()/5):.4f}")
    print(f"  Δ(single - correct):   {m.mean() - c.mean():+.4f} ± "
          f"{np.sqrt(m.var()/5 + c.var()/5):.4f}")
    print(f"  Δ(single - shuffled):  {m.mean() - s.mean():+.4f} ± "
          f"{np.sqrt(m.var()/5 + s.var()/5):.4f}")
    print(f"  Saved to {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
