"""Gate C redo with block-MLP (not low-rank G×E).

The original Gate C used make_low_rank_gxe_predict_fn (a flat-feature model)
to test whether block order matters — which doesn't make sense, as the
reviewer pointed out. Here we redo the test with make_mlp_gxe_predict_fn,
which actually processes per-block tokens.

Conditions:
  1. correct — LD blocks as fitted by fit_ld_blocks
  2. shuffled — markers randomly reassigned to blocks (within chromosome)
  3. single_marker — each marker is its own block

Question: within a block-level model, does the LD-based grouping matter?
If correct ≈ shuffled → grouping quality doesn't matter (any grouping works)
If correct < shuffled → LD grouping adds value

Usage:
    PYTHONPATH=src python3 experiments/paper3_gxe_prediction/gate_c_mlp_repeat.py
"""

import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from plant_context.data.g2f_adapter import load_g2f_phenotype_plot, load_g2f_genotype_marker, load_g2f_environment_daily
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks
from plant_context.tokenizers.environment import tokenize_environment_stages
from plant_context.models.gxe_model import (
    make_mlp_gxe_predict_fn,
    make_low_rank_gxe_predict_fn,
    pivot_environment_tokens_wide,
)
from plant_context.evaluation.splits import make_leave_genotype_split
from plant_context.evaluation.metrics import rmse, pearson_r
from plant_context.statistics.crossfit import run_crossfit

G2F_ROOT = PROJ / "data" / "external" / "g2f"
OUT = PROJ / "experiments" / "paper3_gxe_prediction" / "results_gate_c_mlp"
OUT.mkdir(parents=True, exist_ok=True)

N_GENO = 100
N_MARKERS = 500
EPOCHS = 80
N_FOLDS = 3
D_MODEL = 32
DEVICE = os.environ.get("GXE_DEVICE", "cuda:1")
SEEDS = [1234, 2345, 3456, 4567, 5678]


def _ordered_per_sample_dict(lt, id_col, order_col, canonical_order, fcols):
    """Build {genotype_id: DataFrame(index=block_order, columns=fcols)}."""
    ps = {}
    for sid, grp in lt.groupby(id_col):
        idxd = grp.set_index(order_col)
        present = [c for c in canonical_order if c in idxd.index]
        ps[sid] = idxd.loc[present, fcols]
    return ps


def run_condition(label, blocks, geno, pheno, env_feat, block_order, seed):
    """Run one condition with block-MLP, return RMSE and r."""
    gtokens = tokenize_genotype_blocks(geno, blocks)
    per_sample = _ordered_per_sample_dict(
        gtokens, "genotype_id", "ld_block_id", block_order, ["mean_dosage"]
    )

    split = make_leave_genotype_split(
        pheno, n_folds=N_FOLDS, seed=seed,
        split_version=f"gate_c_mlp_{label}_{seed}",
    )
    fn = make_mlp_gxe_predict_fn(
        per_sample, env_feat, block_feature_columns=["mean_dosage"],
        d_model=D_MODEL, epochs=EPOCHS, lr=0.001, seed=seed, device=DEVICE,
    )
    result = run_crossfit(pheno, split, fn)
    return {
        "rmse": float(rmse(result["y_true"], result["y_pred"])),
        "pearson_r": float(pearson_r(result["y_true"], result["y_pred"])),
    }


def shuffle_blocks(blocks, rng):
    """Shuffle marker assignments among blocks within each chromosome."""
    shuf = blocks.copy()
    for chrom in shuf["chromosome"].unique():
        mask = shuf["chromosome"] == chrom
        mids = shuf.loc[mask, "marker_id"].tolist()
        rng.shuffle(mids)
        shuf.loc[mask, "marker_id"] = mids
    return shuf


def single_marker_blocks(geno):
    """Each marker = its own block."""
    m = geno[["marker_id", "chromosome", "position"]].drop_duplicates("marker_id")
    m["ld_block_id"] = m["marker_id"]
    return m.reset_index(drop=True)


def main():
    print("=" * 60)
    print("Gate C redo: block-MLP (5 seeds, 100 genos, 500 markers)")
    print("=" * 60)
    sys.stdout.flush()

    pheno = load_g2f_phenotype_plot(G2F_ROOT)
    geno = load_g2f_genotype_marker(G2F_ROOT)
    env = load_g2f_environment_daily(G2F_ROOT)

    all_records = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        print(f"\n--- seed={seed} ---")
        sys.stdout.flush()

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

        # Single-marker blocks
        sm_blocks = single_marker_blocks(geno_s)

        # Fit LD blocks on this seed's subset
        correct_blocks = fit_ld_blocks(
            geno_s, set(geno_s["genotype_id"]), r2_threshold=0.7, max_block_size=50
        )
        shuf_rng = np.random.default_rng(seed + 100)
        shuf_blocks = shuffle_blocks(correct_blocks, shuf_rng)

        # Block order (needed for both correct and shuffled)
        correct_order = list(dict.fromkeys(correct_blocks["ld_block_id"]))
        shuf_order = list(dict.fromkeys(shuf_blocks["ld_block_id"]))
        sm_order = list(dict.fromkeys(sm_blocks["ld_block_id"]))

        conditions = [
            ("correct", correct_blocks, correct_order),
            ("shuffled", shuf_blocks, shuf_order),
            ("single_marker", sm_blocks, sm_order),
        ]

        for label, blocks, order in conditions:
            t0 = time.time()
            rec = run_condition(label, blocks, geno_s, pheno_s, env_feat, order, seed)
            elapsed = time.time() - t0
            print(f"  {label:20s}: RMSE={rec['rmse']:.4f}  r={rec['pearson_r']:.4f} ({elapsed:.0f}s)")
            sys.stdout.flush()
            rec["label"] = label
            rec["seed"] = seed
            all_records.append(rec)

    df = pd.DataFrame.from_records(all_records)
    df.to_csv(OUT / "gate_c_mlp_results.csv", index=False)

    print(f"\n{'=' * 60}")
    print("Gate C MLP Summary (mean ± std over 5 seeds):")
    for label in ["correct", "shuffled", "single_marker"]:
        vals = df[df["label"] == label]
        print(f"  {label:20s}: RMSE={vals['rmse'].mean():.4f}±{vals['rmse'].std():.4f}  "
              f"r={vals['pearson_r'].mean():.4f}±{vals['pearson_r'].std():.4f}")

    c = df[df["label"] == "correct"]["rmse"]
    s = df[df["label"] == "shuffled"]["rmse"]
    m = df[df["label"] == "single_marker"]["rmse"]

    import math
    def d(a, b):
        sp = math.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
        return (a.mean() - b.mean()) / sp

    print(f"\n  Δ(shuffled - correct):  {s.mean() - c.mean():+.4f}  d={d(s,c):+.2f}")
    print(f"  Δ(single - correct):    {m.mean() - c.mean():+.4f}  d={d(m,c):+.2f}")
    print(f"  Δ(single - shuffled):   {m.mean() - s.mean():+.4f}  d={d(m,s):+.2f}")
    print(f"  Saved to {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
