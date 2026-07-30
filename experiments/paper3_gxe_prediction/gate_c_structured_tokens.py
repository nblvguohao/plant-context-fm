"""Gate C: structured token value — with better error handling and logging."""
import sys, time, traceback
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ / "src"))

from plant_context.data.g2f_adapter import load_g2f_phenotype_plot, load_g2f_genotype_marker, load_g2f_environment_daily
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks
from plant_context.tokenizers.environment import tokenize_environment_stages
from plant_context.models.gxe_model import pivot_genotype_tokens_wide, pivot_environment_tokens_wide, make_low_rank_gxe_predict_fn
from plant_context.evaluation.splits import make_leave_genotype_split
from plant_context.evaluation.metrics import rmse, pearson_r
from plant_context.statistics.crossfit import run_crossfit

G2F_ROOT = PROJ / "data" / "external" / "g2f"
OUT = PROJ / "experiments" / "paper3_gxe_prediction" / "results_gate_c"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(1234)
N_GENO = 100
EPOCHS = 80
N_FOLDS = 3
SEED = 1234

def run(label, blocks, geno, pheno, env_features):
    sys.stdout.flush()
    print(f"\n--- {label} ---")
    t0 = time.time()
    sys.stdout.flush()

    tokens = tokenize_genotype_blocks(geno, blocks)
    print(f"  tokenized: {tokens.shape}")
    sys.stdout.flush()

    g_feat = pivot_genotype_tokens_wide(tokens)
    print(f"  features: {g_feat.shape}")
    sys.stdout.flush()

    p = pheno[pheno["genotype_id"].isin(g_feat.index) & pheno["environment_id"].isin(env_features.index)].reset_index(drop=True)
    print(f"  pheno: {len(p)} rows")
    sys.stdout.flush()

    split = make_leave_genotype_split(p, n_folds=N_FOLDS, seed=SEED, split_version="gate_c")
    fn = make_low_rank_gxe_predict_fn(g_feat, env_features, rank=4, hidden=16, epochs=EPOCHS, lr=0.01, seed=SEED)
    result = run_crossfit(p, split, fn)

    metrics = {"rmse": float(rmse(result["y_true"], result["y_pred"])), "pearson_r": float(pearson_r(result["y_true"], result["y_pred"]))}
    elapsed = time.time() - t0
    print(f"  RMSE={metrics['rmse']:.4f}  r={metrics['pearson_r']:.4f}  ({elapsed:.0f}s)")
    sys.stdout.flush()
    return {"label": label, "n_blocks": blocks["ld_block_id"].nunique(), "elapsed_s": round(elapsed, 1), **metrics}

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
    print("Gate C: Structured token value")
    print("=" * 60)
    sys.stdout.flush()

    print("\n[1] Loading data...")
    sys.stdout.flush()
    pheno = load_g2f_phenotype_plot(G2F_ROOT)
    geno = load_g2f_genotype_marker(G2F_ROOT)
    env = load_g2f_environment_daily(G2F_ROOT)

    all_g = sorted(set(pheno["genotype_id"]) & set(geno["genotype_id"]))
    chosen = set(rng.choice(all_g, size=min(N_GENO, len(all_g)), replace=False))
    pheno_s = pheno[pheno["genotype_id"].isin(chosen)].reset_index(drop=True)
    geno_s = geno[geno["genotype_id"].isin(chosen)].reset_index(drop=True)
    train_ids = set(geno_s["genotype_id"])
    print(f"  {len(chosen)} genotypes")
    sys.stdout.flush()

    print("\n[2] Environment features...")
    sys.stdout.flush()
    env_daily = env[env["environment_id"].isin(pheno_s["environment_id"])]
    env_tokens = tokenize_environment_stages(env_daily)
    env_feat = pivot_environment_tokens_wide(env_tokens)
    print(f"  {env_feat.shape}")
    sys.stdout.flush()

    print("\n[3] Running conditions...")
    sys.stdout.flush()

    records = []

    print("\n  Fitting LD blocks...")
    sys.stdout.flush()
    correct_blocks = fit_ld_blocks(geno_s, train_ids, r2_threshold=0.7, max_block_size=50)
    records.append(run("ld_blocks_correct", correct_blocks, geno_s, pheno_s, env_feat))

    shuf_rng = np.random.default_rng(SEED + 1)
    shuf_blocks = shuffle_blocks(correct_blocks, shuf_rng)
    records.append(run("ld_blocks_shuffled", shuf_blocks, geno_s, pheno_s, env_feat))

    single_blocks = single_marker_blocks(geno_s)
    records.append(run("single_marker", single_blocks, geno_s, pheno_s, env_feat))

    import pandas as pd
    df = pd.DataFrame.from_records(records)
    csv_path = OUT / "gate_c_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    print(f"\n{'=' * 60}")
    for r in records:
        d = ""
        if r["label"] != "ld_blocks_correct":
            cr = records[0]["rmse"]
            delta = r["rmse"] - cr
            d = f"  (ΔRMSE={delta:+.4f})"
        print(f"  {r['label']:25s}: RMSE={r['rmse']:.4f}  r={r['pearson_r']:.4f}  {d}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
