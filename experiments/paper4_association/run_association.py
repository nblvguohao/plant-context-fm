"""Paper 4: Association experiments — simulation calibration + G2F scan.

Two phases:
  1. Simulation: validate type-I error (null) and power (known signal).
  2. Real-data scan: run the reaction-norm GWAS pipeline on G2F.

Usage:
    PYTHONPATH=src python3 experiments/paper4_association/run_association.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plant_context.association.environment_effects import (
    adjust_multiple_testing,
    build_block_dosage_matrix,
    scan_slope_association,
)
from plant_context.data.g2f_adapter import (
    load_g2f_environment_daily,
    load_g2f_genotype_marker,
    load_g2f_phenotype_plot,
)
from plant_context.evaluation.splits import make_leave_genotype_split
from plant_context.statistics.reaction_norm import (
    compute_environment_index_from_phenotype,
    fit_reaction_norm,
)
from plant_context.tokenizers.genotype import fit_ld_blocks

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

G2F_ROOT = Path(CONFIG["data"]["g2f_root"])
if not G2F_ROOT.is_absolute():
    G2F_ROOT = PROJECT_ROOT / G2F_ROOT

N_GENOTYPES = CONFIG["data"]["n_genotypes"]
SEED = CONFIG["data"]["seed"]
FDR_THRESHOLD = CONFIG["association"]["fdr_threshold"]

OUTPUT_DIR = Path(CONFIG["output"]["dir"])
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Phase 1: Simulation ────────────────────────────────────────────────────


def _simulate_genotype_block_dosage(n_genotypes, n_blocks, rng):
    """Generate random block dosage matrix and LD block assignment."""
    dosage = pd.DataFrame(
        rng.uniform(0, 2, size=(n_genotypes, n_blocks)),
        index=[f"g{i}" for i in range(n_genotypes)],
        columns=[f"block_{j}" for j in range(n_blocks)],
    )
    block_assignment = pd.concat([
        pd.DataFrame({
            "marker_id": [f"m{j * 10 + k}" for k in range(10)],
            "ld_block_id": f"block_{j}",
            "chromosome": j % 5 + 1,
            "position": list(range(j * 100, j * 100 + 10)),
        })
        for j in range(n_blocks)
    ]).reset_index(drop=True)
    return dosage, block_assignment


def _run_null_simulation(n_genotypes, n_blocks, rng):
    """Under H0 (slope independent of dosage), return p-values."""
    dosage, _ = _simulate_genotype_block_dosage(n_genotypes, n_blocks, rng)
    slopes = pd.Series(rng.normal(5, 1, size=n_genotypes), index=dosage.index)
    return scan_slope_association(dosage, slopes)["p_value"].dropna()


def _run_power_simulation(n_genotypes, n_blocks, n_associated, effect_size, rng):
    """Under H1 (n_associated blocks have real signal), return results."""
    dosage, _ = _simulate_genotype_block_dosage(n_genotypes, n_blocks, rng)
    slopes = rng.normal(5, 1, size=n_genotypes)
    for k in range(n_associated):
        slopes += effect_size * dosage.iloc[:, k].to_numpy()
    slopes_series = pd.Series(slopes, index=dosage.index)
    return scan_slope_association(dosage, slopes_series)


def simulate(config, rng):
    """Run type-I error and power simulations."""
    sim_cfg = config["simulation"]
    print("\n[Phase 1] Simulation calibration")
    print(f"  Null:     {sim_cfg['n_simulations']} reps, "
          f"{sim_cfg['n_genotypes_null']} genotypes, {sim_cfg['n_blocks_null']} blocks")
    print(f"  Power:    {sim_cfg['n_simulations']} reps, "
          f"{sim_cfg['n_genotypes_power']} genotypes, {sim_cfg['n_blocks_power']} blocks, "
          f"{sim_cfg['n_associated_blocks']} causal, effect={sim_cfg['effect_size']}")

    # Null simulation
    null_p = []
    t0 = time.time()
    for i in range(sim_cfg["n_simulations"]):
        p_vals = _run_null_simulation(
            sim_cfg["n_genotypes_null"], sim_cfg["n_blocks_null"],
            rng,
        )
        null_p.extend(p_vals.tolist())
    null_p = np.array(null_p)
    # Type-I error at alpha=0.05
    type1_05 = (null_p < 0.05).mean()
    type1_01 = (null_p < 0.01).mean()
    print(f"  Null completed in {time.time() - t0:.1f}s")
    print(f"  Type-I error @ 0.05: {type1_05:.4f} (expected 0.05)")
    print(f"  Type-I error @ 0.01: {type1_01:.4f} (expected 0.01)")

    # Power simulation
    power_records = []
    t0 = time.time()
    for i in range(sim_cfg["n_simulations"]):
        result = _run_power_simulation(
            sim_cfg["n_genotypes_power"], sim_cfg["n_blocks_power"],
            sim_cfg["n_associated_blocks"], sim_cfg["effect_size"],
            rng,
        )
        adj = adjust_multiple_testing(result, alpha=0.05)
        # Which blocks were called significant?
        power_records.append({
            "n_significant": adj["significant_0.05"].sum(),
            "causal_detected": (
                adj.loc[[f"block_{k}" for k in range(sim_cfg["n_associated_blocks"])],
                         "significant_0.05"].sum()
            ),
        })

    power_df = pd.DataFrame.from_records(power_records)
    power_detected = power_df["causal_detected"].mean()
    power_any = (power_df["causal_detected"] >= 1).mean()
    false_positives = (power_df["n_significant"] - power_df["causal_detected"]).mean()
    print(f"  Power completed in {time.time() - t0:.1f}s")
    print(f"  Power (mean causal detected): {power_detected:.3f} / {sim_cfg['n_associated_blocks']}")
    print(f"  Power (any causal detected):  {power_any:.3f}")
    print(f"  Avg false positives:          {false_positives:.2f}")

    sim_results = pd.DataFrame({
        "type1_error_05": [type1_05],
        "type1_error_01": [type1_01],
        "power_mean": [power_detected],
        "power_any": [power_any],
        "avg_false_positives": [false_positives],
    })
    sim_file = OUTPUT_DIR / CONFIG["output"]["simulation_file"]
    sim_results.to_csv(sim_file, index=False)
    print(f"  Simulation results saved to {sim_file}")
    return sim_results


# ── Phase 2: G2F real-data scan ────────────────────────────────────────────


def scan_g2f(config, rng):
    """Run reaction-norm GWAS scan on G2F data."""
    g2f_root = Path(CONFIG["data"]["g2f_root"])
    if not g2f_root.is_absolute():
        g2f_root = PROJECT_ROOT / g2f_root

    print("\n[Phase 2] G2F association scan")
    t0 = time.time()

    # Load data
    phenotype_df = load_g2f_phenotype_plot(g2f_root)
    genotype_df = load_g2f_genotype_marker(g2f_root)
    print(f"  Data loaded: {len(phenotype_df)} pheno, {len(genotype_df)} geno rows")

    # Subsample (null = all genotypes)
    n_choose = config["data"]["n_genotypes"]
    all_genotypes = sorted(set(phenotype_df["genotype_id"]) & set(genotype_df["genotype_id"]))
    if n_choose is None:
        n_choose = len(all_genotypes)
    subset_genotypes = set(rng.choice(all_genotypes, size=n_choose, replace=False))
    phenotype_subset = phenotype_df[phenotype_df["genotype_id"].isin(subset_genotypes)].reset_index(drop=True)
    genotype_subset = genotype_df[genotype_df["genotype_id"].isin(subset_genotypes)].reset_index(drop=True)
    print(f"  Using {n_choose} genotypes")

    # Split — use leave_genotype_out so reaction norm is safe
    split_df = make_leave_genotype_split(phenotype_subset, n_folds=3, seed=SEED, split_version="association")

    # Fit reaction norm on train + compute per-genotype slopes
    fold0 = split_df[(split_df["outer_split_type"] == "leave_genotype") & (split_df["outer_fold"] == 0)]
    train_rows = phenotype_subset[phenotype_subset["sample_id"].isin(
        fold0.loc[fold0["role"] == "train", "sample_id"]
    )]
    env_index = compute_environment_index_from_phenotype(train_rows)
    rn_params = fit_reaction_norm(train_rows, env_index)
    slopes = rn_params["b"].dropna()
    print(f"  Reaction norm slopes: {len(slopes)} genotypes with identifiable slopes")

    # Build LD blocks + dosage matrix
    train_genotype_ids = set(train_rows["genotype_id"])
    blocks = fit_ld_blocks(genotype_subset, train_genotype_ids, r2_threshold=0.7, max_block_size=50)
    dosage = build_block_dosage_matrix(genotype_subset, blocks)
    print(f"  LD blocks: {len(blocks['ld_block_id'].unique())}")
    print(f"  Dosage matrix: {dosage.shape}")

    # Scan association
    assoc = scan_slope_association(dosage, slopes)
    adj = adjust_multiple_testing(assoc, alpha=FDR_THRESHOLD)
    n_sig = adj["significant_0.05"].sum()
    print(f"  Association scan: {n_sig} / {len(adj)} blocks significant at FDR 0.05")

    # Add chromosome/position for Manhattan
    block_pos = (
        blocks.groupby("ld_block_id", sort=False)
        .agg(chromosome=("chromosome", "first"), min_position=("position", "min"),
             max_position=("position", "max"), n_markers=("marker_id", "nunique"))
        .reset_index()
    )
    manhattan = adj.reset_index().merge(block_pos, on="ld_block_id", how="left")
    manhattan["-log10_p"] = -np.log10(manhattan["p_value"].clip(lower=1e-300))

    scan_file = OUTPUT_DIR / CONFIG["output"]["g2f_scan_file"]
    manhattan.to_csv(scan_file, index=False)
    print(f"  Scan results saved to {scan_file}")
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Top hits
    top_hits = manhattan.sort_values("p_value").head(10)
    print("\n  Top 10 associated blocks:")
    for _, row in top_hits.iterrows():
        print(f"    {row['ld_block_id']}: chr{row['chromosome']} "
              f"pos={int(row['min_position'])}-{int(row['max_position'])}, "
              f"beta={row['beta']:.4f}, p={row['p_value']:.2e}, "
              f"q={row['q_value']:.4f}{' ***' if row.get('significant_0.05', False) else ''}")

    return manhattan


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Paper 4: Association experiments")
    print(f"Config: {CONFIG_PATH}")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    # Phase 1
    sim_results = simulate(CONFIG, rng)

    # Phase 2
    scan_results = scan_g2f(CONFIG, rng)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
