"""Gate D: GBLUP + env_mean baselines on exact same 100-geno × 500-marker subsets.

Matches the sampling of gate_d_ablation.py exactly so we can compare
block-level encoding vs standard SOTA baselines head-to-head across 5 seeds.

Models:
  1. overall_mean — predict grand mean (trivial floor)
  2. environment_mean — per-environment mean (no genotype info)
  3. gblup — VanRaden GBLUP per-genotype mean (no environment info)
  4. gblup + env_mean — additive combination

Comparison to Gate D results (from gate_d_ablation_results.csv):
  - encoder_scratch (block Transformer + pool + env)
  - mlp_ablation (block MLP + pool + env)
  - flat_baseline (low-rank G×E, rank=8, hidden=16)

Usage:
    PYTHONPATH=src python3 experiments/paper3_gxe_prediction/gate_d_gblup_comparison.py
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
from plant_context.statistics.baselines import (
    environment_mean_predict_fn,
    make_gblup_predict_fn,
)
from plant_context.statistics.crossfit import run_crossfit
from plant_context.statistics.gblup import (
    compute_allele_frequencies,
    compute_vanraden_grm_with_frequencies,
    fit_gblup,
    pivot_genotype_marker_to_wide,
    select_gblup_lambda,
)
from plant_context.tokenizers.environment import tokenize_environment_stages

G2F_ROOT = PROJ / "data" / "external" / "g2f"
OUT = PROJ / "experiments" / "paper3_gxe_prediction" / "results_gblup_comparison"
OUT.mkdir(parents=True, exist_ok=True)

N_GENO = 100
N_MARKERS = 500
N_FOLDS = 3
SEEDS = [1234, 2345, 3456, 4567, 5678]


def run_conditional(seed):
    """Run all baseline models for one seed, return list of result dicts."""
    rng = np.random.default_rng(seed)

    # ── Load and subsample (matching Gate D exactly) ──────────────────────
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

    # Filter to environments that have weather data (same as Gate D)
    env_daily = env[env["environment_id"].isin(pheno_s["environment_id"])]
    env_tokens = tokenize_environment_stages(env_daily)
    # pivot gives lots of columns; we just need env IDs that survive
    env_feat = env_tokens.pivot_table(
        index="environment_id", columns="growth_stage",
        values="tmean_mean", aggfunc="first",
    )
    present_env = set(env_feat.index)
    pheno_s = pheno_s[pheno_s["environment_id"].isin(present_env)].reset_index(drop=True)

    print(f"\n  Data: {len(pheno_s)} samples, {len(chosen)} genos, {len(all_m)} markers")
    sys.stdout.flush()

    # ── Split (leave-genotype-out, 3 folds, same as Gate D) ───────────────
    split = make_leave_genotype_split(
        pheno_s, n_folds=N_FOLDS, seed=seed,
        split_version=f"gblup_comp_{seed}",
    )

    records = []

    # ── 1. overall_mean ───────────────────────────────────────────────────
    def overall_mean_fn(train_rows, eval_rows):
        m = train_rows["phenotype_value"].mean()
        return np.full(len(eval_rows), m)

    result = run_crossfit(pheno_s, split, overall_mean_fn)
    records.append({
        "seed": seed,
        "model": "overall_mean",
        "rmse": float(rmse(result["y_true"], result["y_pred"])),
        "pearson_r": float(pearson_r(result["y_true"], result["y_pred"])),
    })
    print(f"    overall_mean         RMSE={records[-1]['rmse']:.4f}  r={records[-1]['pearson_r']:.4f}")
    sys.stdout.flush()

    # ── 2. environment_mean ───────────────────────────────────────────────
    result = run_crossfit(pheno_s, split, environment_mean_predict_fn)
    records.append({
        "seed": seed,
        "model": "environment_mean",
        "rmse": float(rmse(result["y_true"], result["y_pred"])),
        "pearson_r": float(pearson_r(result["y_true"], result["y_pred"])),
    })
    print(f"    environment_mean     RMSE={records[-1]['rmse']:.4f}  r={records[-1]['pearson_r']:.4f}")
    sys.stdout.flush()

    # ── 3. GBLUP (full inner CV for lambda) ───────────────────────────────
    fn_gblup = make_gblup_predict_fn(
        geno_s, max_dosage=1.0,
        lambda_grid=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        n_folds=3, seed=seed,
    )
    result = run_crossfit(pheno_s, split, fn_gblup)
    records.append({
        "seed": seed,
        "model": "gblup",
        "rmse": float(rmse(result["y_true"], result["y_pred"])),
        "pearson_r": float(pearson_r(result["y_true"], result["y_pred"])),
    })
    print(f"    gblup                RMSE={records[-1]['rmse']:.4f}  r={records[-1]['pearson_r']:.4f}")
    sys.stdout.flush()

    # ── 4. GBLUP + env_mean (additive: per-geno + per-env) ────────────────
    wide_full = pivot_genotype_marker_to_wide(geno_s)
    grm_full = compute_vanraden_grm_with_frequencies(
        wide_full,
        compute_allele_frequencies(wide_full, max_dosage=1.0),
        max_dosage=1.0,
    )
    fixed_lambda = 1.0

    def gblup_plus_env_fn(train_rows, eval_rows):
        # GBLUP component (train genotypes only for allele freq)
        train_gids = set(train_rows["genotype_id"])
        wide_train = wide_full.loc[wide_full.index.isin(train_gids)]
        af = compute_allele_frequencies(wide_train, max_dosage=1.0)
        grm = compute_vanraden_grm_with_frequencies(wide_full, af, max_dosage=1.0)
        y_train_g = train_rows.groupby("genotype_id")["phenotype_value"].mean()
        y_train_g = y_train_g.reindex(grm.index)
        lam = select_gblup_lambda(grm, y_train_g, [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                                  n_folds=3, seed=seed)
        g_preds = fit_gblup(grm, y_train_g, lam)

        # Environment mean component
        env_mean = train_rows.groupby("environment_id")["phenotype_value"].mean()
        overall_mean = train_rows["phenotype_value"].mean()

        # Additive prediction
        eval_g = eval_rows["genotype_id"].map(g_preds).fillna(overall_mean).to_numpy()
        eval_e = eval_rows["environment_id"].map(env_mean).fillna(overall_mean).to_numpy()
        # Center env component (remove overall mean so no double-count)
        return eval_g + eval_e - overall_mean

    result = run_crossfit(pheno_s, split, gblup_plus_env_fn)
    records.append({
        "seed": seed,
        "model": "gblup_plus_env",
        "rmse": float(rmse(result["y_true"], result["y_pred"])),
        "pearson_r": float(pearson_r(result["y_true"], result["y_pred"])),
    })
    print(f"    gblup_plus_env       RMSE={records[-1]['rmse']:.4f}  r={records[-1]['pearson_r']:.4f}")
    sys.stdout.flush()

    return records


def main():
    print("=" * 70)
    print("Gate D — GBLUP + EnvMean baselines on exact same subsets (5 seeds)")
    print("=" * 70)
    sys.stdout.flush()

    all_records = []
    for seed in SEEDS:
        print(f"\n{'─' * 50}")
        print(f"  Seed {seed}")
        print(f"{'─' * 50}")
        sys.stdout.flush()
        records = run_conditional(seed)
        all_records.extend(records)

    df = pd.DataFrame.from_records(all_records)
    csv_path = OUT / "gblup_comparison_results.csv"
    df.to_csv(csv_path, index=False)

    # Summary per model
    print(f"\n{'=' * 70}")
    print("Baseline Summary (mean ± std over 5 seeds):")
    for model in df["model"].unique():
        v = df[df["model"] == model]
        print(f"  {model:25s}: RMSE={v['rmse'].mean():.4f}±{v['rmse'].std():.4f}  "
              f"r={v['pearson_r'].mean():.4f}±{v['pearson_r'].std():.4f}")

    # ── Comparison table ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("COMPARISON: Block-level models (Gate D) vs Baselines (this run)")
    print(f"{'=' * 70}")

    # Our Gate D results (copied from gate_d_ablation_results.csv)
    gate_d_results = {
        "encoder_scratch": {"rmse_mean": 2.7985, "rmse_std": 0.1030, "r_mean": 0.4082, "r_std": 0.0131},
        "mlp_ablation":    {"rmse_mean": 2.8408, "rmse_std": 0.0997, "r_mean": 0.3746, "r_std": 0.0277},
        "flat_baseline":   {"rmse_mean": 3.5177, "rmse_std": 0.1883, "r_mean": 0.3909, "r_std": 0.0231},
    }

    heading = f"{'Model':25s} {'RMSE':>10s} {'r':>8s}  {'Δ vs GBLUP':>12s} {'Δ vs EnvM':>12s}"
    print(f"\n{heading}")
    print('-' * len(heading))

    # Compute baseline means for delta columns
    env_m_mean = df[df["model"] == "environment_mean"]["rmse"].mean()
    gblup_mean = df[df["model"] == "gblup"]["rmse"].mean()

    for name, v in gate_d_results.items():
        d_gblup = v["rmse_mean"] - gblup_mean
        d_env = v["rmse_mean"] - env_m_mean
        print(f"  {name:25s} {v['rmse_mean']:>8.4f}  {v['r_mean']:>6.4f}  "
              f"{d_gblup:>+10.4f}   {d_env:>+10.4f}  ← block-level models")

    for model in ["overall_mean", "environment_mean", "gblup", "gblup_plus_env"]:
        v = df[df["model"] == model]
        rm = v["rmse"].mean()
        rs = v["pearson_r"].mean()
        d_gblup = rm - gblup_mean
        d_env = rm - env_m_mean
        print(f"  {model:25s} {rm:>8.4f}  {rs:>6.4f}  "
              f"{d_gblup:>+10.4f}   {d_env:>+10.4f}  ← baselines")

    print(f"\n  {'='*50}")
    print(f"  Delta = RMSE(model) - RMSE(baseline). Negative = better.")
    print(f"  GBLUP mean RMSE: {gblup_mean:.4f}, EnvMean RMSE: {env_m_mean:.4f}")
    print(f"  Yield stats (on full G2F): mean=9.62, SD=3.10 Mg/ha")
    print(f"  Save: {csv_path}")


if __name__ == "__main__":
    main()
