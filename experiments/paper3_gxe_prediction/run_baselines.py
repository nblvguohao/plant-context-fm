"""Paper 3: G×E baseline benchmark experiment.

Compares GBLUP, reaction norm, low-rank G×E, and environment-mean baseline
across all 4 split types on real G2F data.

Usage:
    PYTHONPATH=src python3 experiments/paper3_gxe_prediction/run_baselines.py
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
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
from plant_context.models.gxe_model import (
    make_low_rank_gxe_predict_fn,
    pivot_environment_tokens_wide,
    pivot_genotype_tokens_wide,
)
from plant_context.statistics.baselines import (
    environment_mean_predict_fn,
    make_gblup_predict_fn,
    reaction_norm_predict_fn,
)
from plant_context.statistics.crossfit import run_crossfit
from plant_context.tokenizers.environment import tokenize_environment_stages
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks

# ── Config ──────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

G2F_ROOT = Path(CONFIG["data"]["g2f_root"])
if not G2F_ROOT.is_absolute():
    G2F_ROOT = PROJECT_ROOT / G2F_ROOT

N_GENOTYPES = CONFIG["data"]["n_genotypes"]
N_FOLDS = CONFIG["data"]["n_folds"]
SEED = CONFIG["data"]["seed"]

OUTPUT_DIR = Path(CONFIG["output"]["dir"])
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / CONFIG["output"]["results_file"]

# ── Split factory ───────────────────────────────────────────────────────────


def _make_split(split_type: str, phenotype_df, n_folds, seed):
    factories = {
        "leave_genotype": make_leave_genotype_split,
        "leave_environment": make_leave_environment_split,
        "forward_year": make_forward_year_split,
        "leave_ge": make_leave_ge_split,
    }
    fn = factories.get(split_type)
    if fn is None:
        raise ValueError(f"Unknown split type: {split_type}")
    if split_type == "forward_year":
        return fn(phenotype_df, seed=seed, split_version=split_type)
    return fn(phenotype_df, n_folds=n_folds, seed=seed, split_version=split_type)


# ── Model factory ───────────────────────────────────────────────────────────


def _make_model(model_config, genotype_features, environment_features, blocks, genotype_df):
    name = model_config["name"]
    params = model_config.get("params", {})
    if name == "environment_mean":
        return name, environment_mean_predict_fn
    elif name == "gblup":
        return name, make_gblup_predict_fn(
            genotype_df,
            max_dosage=params.get("max_dosage", 1.0),
            lambda_grid=params.get("lambda_grid", [0.1, 0.5, 1.0, 5.0]),
            n_folds=params.get("n_inner_folds", 3),
            seed=SEED,
        )
    elif name == "reaction_norm":
        return name, reaction_norm_predict_fn
    elif name == "low_rank_gxe":
        return name, make_low_rank_gxe_predict_fn(
            genotype_features, environment_features,
            rank=params.get("rank", 4),
            hidden=params.get("hidden", 16),
            epochs=params.get("epochs", 150),
            lr=params.get("lr", 0.01),
            seed=SEED,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Paper 3: G×E baseline benchmark")
    print(f"Config: {CONFIG_PATH}")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    t0 = time.time()
    try:
        phenotype_df = load_g2f_phenotype_plot(G2F_ROOT)
        genotype_df = load_g2f_genotype_marker(G2F_ROOT)
        environment_daily_df = load_g2f_environment_daily(G2F_ROOT)
    except Exception as e:
        print(f"      ERROR loading G2F data: {e}")
        print("      Ensure data/external/g2f is set up correctly.")
        sys.exit(1)
    print(f"      Phenotype: {len(phenotype_df)} rows")
    print(f"      Genotype:  {len(genotype_df)} rows")
    print(f"      Weather:   {len(environment_daily_df)} rows")
    print(f"      Loaded in {time.time() - t0:.1f}s")

    # ── Subsample genotypes ──────────────────────────────────────────────
    print("\n[2/4] Subsampling genotypes...")
    rng = np.random.default_rng(SEED)
    all_genotypes = sorted(set(phenotype_df["genotype_id"]) & set(genotype_df["genotype_id"]))
    n_choose = N_GENOTYPES if N_GENOTYPES is not None else len(all_genotypes)
    n_choose = min(n_choose, len(all_genotypes))
    subset_genotypes = set(rng.choice(all_genotypes, size=n_choose, replace=False))
    phenotype_subset = phenotype_df[phenotype_df["genotype_id"].isin(subset_genotypes)].reset_index(drop=True)
    genotype_subset = genotype_df[genotype_df["genotype_id"].isin(subset_genotypes)].reset_index(drop=True)
    print(f"      Using {n_choose}/{len(all_genotypes)} genotypes")

    # ── Build features ───────────────────────────────────────────────────
    print("\n[3/4] Building features (LD blocks + environment stages)...")
    t0 = time.time()

    # LD blocks — fit on all subset genotypes (toy split for block structure)
    # In a real experiment, blocks should be fit per outer fold's train IDs
    all_train_ids = set(genotype_subset["genotype_id"])
    blocks = fit_ld_blocks(genotype_subset, all_train_ids, r2_threshold=0.7, max_block_size=50)
    genotype_tokens = tokenize_genotype_blocks(genotype_subset, blocks)
    genotype_features = pivot_genotype_tokens_wide(genotype_tokens)

    relevant_environments = set(phenotype_subset["environment_id"])
    env_daily_subset = environment_daily_df[
        environment_daily_df["environment_id"].isin(relevant_environments)
    ]
    environment_tokens = tokenize_environment_stages(env_daily_subset)
    environment_features = pivot_environment_tokens_wide(environment_tokens)

    # Filter phenotype rows to genotypes/environments that have features
    phenotype_subset = phenotype_subset[
        phenotype_subset["genotype_id"].isin(genotype_features.index)
        & phenotype_subset["environment_id"].isin(environment_features.index)
    ].reset_index(drop=True)
    print(f"      Phenotype after filtering: {len(phenotype_subset)} rows")
    print(f"      Genotype features: {genotype_features.shape}")
    print(f"      Environment features: {environment_features.shape}")
    print(f"      Built in {time.time() - t0:.1f}s")

    # ── Models ───────────────────────────────────────────────────────────
    models = CONFIG["models"]
    splits = CONFIG["splits"]
    print(f"\n[4/4] Running {len(models)} models × {len(splits)} splits...")

    all_records = []
    for split_type, model_cfg in itertools.product(splits, models):
        model_name = model_cfg["name"]
        print(f"\n  --- {model_name} @ {split_type} ---")
        try:
            t1 = time.time()

            # Make split
            split_df = _make_split(split_type, phenotype_subset, N_FOLDS, SEED)

            # Make model
            _, fit_predict = _make_model(
                model_cfg, genotype_features, environment_features, blocks, genotype_subset
            )

            # Run crossfit
            result = run_crossfit(phenotype_subset, split_df, fit_predict)

            # Compute metrics
            y_true = result["y_true"]
            y_pred = result["y_pred"]
            metrics = {
                "rmse": float(rmse(y_true, y_pred)),
                "mae": float(mae(y_true, y_pred)),
                "pearson_r": float(pearson_r(y_true, y_pred)),
                "spearman_r": float(spearman_r(y_true, y_pred)),
            }
            elapsed = time.time() - t1

            record = {
                "model": model_name,
                "split": split_type,
                "n_folds": N_FOLDS,
                "n_genotypes": n_choose,
                "n_train": int((split_df["role"] == "train").sum()),
                "n_eval": int((split_df["role"] == "validation").sum()) + int((split_df["role"] == "test").sum()),
                "elapsed_seconds": round(elapsed, 1),
                **metrics,
            }
            all_records.append(record)
            print(f"      RMSE={metrics['rmse']:.4f}, Pearson={metrics['pearson_r']:.4f} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"      FAILED: {e}")
            all_records.append({
                "model": model_name, "split": split_type, "error": str(e),
            })

    # ── Save results ─────────────────────────────────────────────────────
    results_df = pd.DataFrame.from_records(all_records)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(results_df.to_string(index=False))
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
