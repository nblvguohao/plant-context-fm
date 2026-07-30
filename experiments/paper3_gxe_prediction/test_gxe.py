"""Minimal test: does low_rank_gxe training complete successfully on server?"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from plant_context.data.g2f_adapter import load_g2f_phenotype_plot, load_g2f_genotype_marker, load_g2f_environment_daily
from plant_context.tokenizers.genotype import fit_ld_blocks, tokenize_genotype_blocks
from plant_context.tokenizers.environment import tokenize_environment_stages
from plant_context.models.gxe_model import pivot_genotype_tokens_wide, pivot_environment_tokens_wide, make_low_rank_gxe_predict_fn
from plant_context.evaluation.splits import make_leave_genotype_split
from plant_context.statistics.crossfit import run_crossfit

G2F_ROOT = Path("data/external/g2f")
rng = np.random.default_rng(1234)

print("Loading...")
pheno = load_g2f_phenotype_plot(G2F_ROOT)
geno = load_g2f_genotype_marker(G2F_ROOT)
env = load_g2f_environment_daily(G2F_ROOT)

print("Subsampling 80 genotypes...")
all_g = sorted(set(pheno["genotype_id"]) & set(geno["genotype_id"]))
chosen = set(rng.choice(all_g, size=80, replace=False))
pheno_s = pheno[pheno["genotype_id"].isin(chosen)].reset_index(drop=True)
geno_s = geno[geno["genotype_id"].isin(chosen)].reset_index(drop=True)

print("Fitting LD blocks...")
t0 = time.time()
blocks = fit_ld_blocks(geno_s, set(geno_s["genotype_id"]), r2_threshold=0.7, max_block_size=50)
n_blocks = len(blocks["ld_block_id"].unique())
print(f"  {n_blocks} blocks in {time.time()-t0:.1f}s")

print("Tokenizing genotypes...")
t0 = time.time()
tokens = tokenize_genotype_blocks(geno_s, blocks)
g_features = pivot_genotype_tokens_wide(tokens)
print(f"  {g_features.shape} in {time.time()-t0:.1f}s")

print("Tokenizing environments...")
e_tokens = tokenize_environment_stages(env)
e_features = pivot_environment_tokens_wide(e_tokens)
print(f"  {e_features.shape}")

pheno_s = pheno_s[
    pheno_s["genotype_id"].isin(g_features.index) &
    pheno_s["environment_id"].isin(e_features.index)
].reset_index(drop=True)

split = make_leave_genotype_split(pheno_s, n_folds=3, seed=1234, split_version="test")
print(f"  pheno rows: {len(pheno_s)}")

print("Training low_rank_gxe (50 epochs)...")
fn = make_low_rank_gxe_predict_fn(g_features, e_features, rank=4, hidden=16, epochs=50, lr=0.01, seed=1234)
t0 = time.time()
result = run_crossfit(pheno_s, split, fn)
rmse = float(np.sqrt(np.mean((result["y_true"] - result["y_pred"]) ** 2)))
print(f"  RMSE={rmse:.4f} in {time.time()-t0:.1f}s")
print("SUCCESS")
