# PlantContext-FM

Structured foundation models for plant communities and genotype-by-environment
phenotypes. Doctoral research project spanning (1) plant community context
modeling and (2) crop G×E prediction and environment-dependent association,
joined by a shared structured-tokenization and evaluation methodology.

Authoritative planning documents (do not duplicate their content elsewhere):

- `docs/plant_context_fm_doctoral_execution_plan_zh.md` — doctoral execution plan
- `docs/plant_context_fm_TDD_zh.md` — technical design document (data contracts,
  tokenizer design, split protocol, TDD test plan, Go/No-Go gates)
- `CLAUDE.md` — operating rules for coding agents working in this repository

## Current execution baseline

- Python: 3.10 (`F:\Anaconda\envs\tree-py310`)
- PyTorch: 2.5.1+cu121 (CUDA 12.1, GPU0 RTX 3060 12GB available)
- Test command: `python -m pytest` (from repo root)
- Current status: 278 passed, 6 skipped (all skips are genotype-VCF-not-parsed)

## Relationship to sibling projects

This repository does not redistribute or duplicate data or model code from
other projects on this machine. Instead:

- `external/srg_gxe` is a symlink to the existing `ResidualGxE-Former` (SRG-GxE)
  repository, which is the working codebase for the G×E residual-learning
  chapter (Paper 3). This project audits and extends it against the stricter
  data contracts and leakage tests in the TDD rather than reimplementing it.
- `data/external/g2f` is populated from the raw G2F 2020-2023 release under
  `data/raw/` via `scripts/process_g2f_raw_to_processed.py`. It currently
  covers 2020/2022/2023 only (2021 weather is missing in the raw files);
  `genotype.parquet` is an empty placeholder until the 3.8 GB VCF is parsed.
- Plant community/vegetation-plot data is present: sPlotOpen (95,104 plots,
  42,677 species, CC BY 4.0, downloaded 2026-07-29) under `data/community/raw`
  and `data/community/extracted`, both git-ignored. Provenance, DOI, checksum,
  and citation requirements are frozen in `data/manifests/community_sPlotOpen.yaml`
  (tracked).

Everything under `external/` and `data/external/` is git-ignored: this
repository never commits paths, filenames, or directory layouts belonging to
sibling projects.

## Repository layout

See `docs/plant_context_fm_TDD_zh.md` §3 for the full intended layout
(tokenizers, statistics, association, evaluation, tracking modules, and the
unit/contracts/leakage/integration/statistical/regression test tree already
scaffolded here).

## Reproducing a result

Every result should be reproducible from the recorded commit, config, data
hash, split version, and seed. The minimum metadata to record is:

1. `git rev-parse HEAD`
2. the experiment config file (or script arguments)
3. hashes of the input parquet/CSV tables used
4. `split_version` passed to the split constructors
5. the RNG seed

Example for the bridge smoke-test:

```bash
source /f/Anaconda/etc/profile.d/conda.sh
conda activate tree-py310
PYTHONPATH=src python experiments/bridge_experiments/smoke_bridge_gpu.py
# writes experiments/bridge_experiments/results_smoke_bridge_gpu/smoke_bridge_gpu_results.csv
```

## Status

Scaffolding plus first-pass data processing, tokenizers, models, and smoke
tests. The G2F genotype VCF has not been parsed yet, and 2021 weather is
missing from the raw release, so full-scale G×E experiments are gated behind
data completion.
