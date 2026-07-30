"""Full-scale G×E baselines — ALL genotypes, fixed-lambda GBLUP.

Skips low_rank_gxe (too slow on CPU for 4938 genotypes) and inner-CV
lambda selection (fixed lambda = 1.0) to keep runtime reasonable.

Usage:
    PYTHONPATH=src python3 experiments/paper3_gxe_prediction/run_full_scale.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plant_context.data.g2f_adapter import (
    load_g2f_environment_daily,
    load_g2f_genotype_marker,
    load_g2f_phenotype_plot,
)
from plant_context.evaluation.metrics import mae, pearson_r, rmse, spearman_r
from plant_context.evaluation.splits import (
    make_forward_year_split,
    make_leave_environment_split,
    make_leave_genotype_split,
    make_leave_ge_split,
)
from plant_context.statistics.baselines import (
    environment_mean_predict_fn,
    reaction_norm_predict_fn,
)
from plant_context.statistics.crossfit import run_crossfit
from plant_context.statistics.gblup import (
    compute_allele_frequencies,
    compute_vanraden_grm_with_frequencies,
    fit_gblup,
    pivot_genotype_marker_to_wide,
)
from plant_context.statistics.crossfit import FitPredictFn

CONFIG_PATH = Path(__file__).parent / "config_full.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

G2F_ROOT = Path(CONFIG["data"]["g2f_root"])
if not G2F_ROOT.is_absolute():
    G2F_ROOT = PROJECT_ROOT / G2F_ROOT

SEED = CONFIG["data"]["seed"]
FIXED_LAMBDA = 1.0

OUTPUT_DIR = Path(CONFIG["output"]["dir"])
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_gblup_full_predict_fn(
    wide_full: pd.DataFrame,
    allele_freq: pd.Series,
    max_dosage: float = 1.0,
    fixed_lambda: float = 1.0,
) -> FitPredictFn:
    """GBLUP with fixed lambda — precomputed GRM, no inner CV."""
    grm_full = compute_vanraden_grm_with_frequencies(wide_full, allele_freq, max_dosage)

    def _fit_predict(train_rows, eval_rows):
        train_gids = set(train_rows["genotype_id"])
        y_train = train_rows.groupby("genotype_id")["phenotype_value"].mean()
        y_train = y_train.reindex(grm_full.index).dropna()

        preds = fit_gblup(grm_full, y_train, fixed_lambda)
        overall_mean = train_rows["phenotype_value"].mean()
        return preds.reindex(eval_rows["genotype_id"]).fillna(overall_mean).to_numpy()

    return _fit_predict


def _make_split(split_type, phenotype_df):
    factories = {
        "leave_genotype": lambda df: make_leave_genotype_split(df, n_folds=3, seed=SEED),
        "leave_environment": lambda df: make_leave_environment_split(df, n_folds=3, seed=SEED),
        "forward_year": lambda df: make_forward_year_split(df, seed=SEED),
        "leave_ge": lambda df: make_leave_ge_split(df, n_folds=3, seed=SEED),
    }
    fn = factories.get(split_type)
    if fn is None:
        raise ValueError(f"Unknown split type: {split_type}")
    return fn(phenotype_df)


def main():
    print("=" * 60)
    print("Full-scale G×E benchmark — ALL genotypes")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    t0 = time.time()
    phenotype_df = load_g2f_phenotype_plot(G2F_ROOT)
    genotype_df = load_g2f_genotype_marker(G2F_ROOT)
    print(f"  Phenotype: {len(phenotype_df)} rows, {phenotype_df['genotype_id'].nunique()} genotypes")
    print(f"  Genotype:  {len(genotype_df)} rows, {genotype_df['genotype_id'].nunique()} genotypes")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Precompute GRM ────────────────────────────────────────────────────
    print("\n[2] Precomputing full GRM (fixed λ=1.0, no inner CV)...")
    t0 = time.time()
    wide_full = pivot_genotype_marker_to_wide(genotype_df)
    print(f"  Dosage matrix: {wide_full.shape}")
    allele_freq = compute_allele_frequencies(wide_full, max_dosage=1.0)
    gblup_fn = make_gblup_full_predict_fn(wide_full, allele_freq, max_dosage=1.0, fixed_lambda=1.0)
    print(f"  Full GRM computed in {time.time() - t0:.1f}s")

    # ── All genotypes in common ───────────────────────────────────────────
    common = sorted(
        set(phenotype_df["genotype_id"]) & set(wide_full.index)
    )
    print(f"  Genotypes in common: {len(common)}")
    phenotype_subset = phenotype_df[phenotype_df["genotype_id"].isin(common)].reset_index(drop=True)

    # ── Run ───────────────────────────────────────────────────────────────
    splits = ["leave_genotype", "leave_environment", "forward_year", "leave_ge"]
    models = [
        ("environment_mean", environment_mean_predict_fn),
        ("gblup_full", gblup_fn),
        ("reaction_norm", reaction_norm_predict_fn),
    ]

    all_records = []
    for split_type in splits:
        print(f"\n--- {split_type} ---")
        split_df = _make_split(split_type, phenotype_subset)

        for model_name, fn in models:
            t1 = time.time()
            try:
                result = run_crossfit(phenotype_subset, split_df, fn)
                metrics = {
                    "rmse": float(rmse(result["y_true"], result["y_pred"])),
                    "mae": float(mae(result["y_true"], result["y_pred"])),
                    "pearson_r": float(pearson_r(result["y_true"], result["y_pred"])),
                    "spearman_r": float(spearman_r(result["y_true"], result["y_pred"])),
                }
                rec = {
                    "model": model_name, "split": split_type,
                    "n_genotypes": len(common),
                    "n_train": int((split_df["role"] == "train").sum()),
                    "n_eval": int((split_df["role"] == "validation").sum()) + int((split_df["role"] == "test").sum()),
                    "elapsed_s": round(time.time() - t1, 1),
                    **metrics,
                }
                print(f"  {model_name:20s} RMSE={metrics['rmse']:.4f}  r={metrics['pearson_r']:.4f}  "
                      f"({rec['elapsed_s']:.1f}s)")
            except Exception as e:
                print(f"  {model_name:20s} FAILED: {e}")
                rec = {"model": model_name, "split": split_type, "error": str(e)}
            all_records.append(rec)

    # ── Save ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame.from_records(all_records)
    results_file = OUTPUT_DIR / CONFIG["output"]["results_file"]
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
