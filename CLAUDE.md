# CLAUDE.md — PlantContext-FM operating rules

This file is the authoritative instruction source for any coding agent
working in this repository. Read `docs/plant_context_fm_doctoral_execution_plan_zh.md`
and `docs/plant_context_fm_TDD_zh.md` before making design decisions; they are
the source of truth for scope, data contracts, split protocol, and Go/No-Go
gates. Do not restate their content here — link to sections instead.

## 0. Never expose host/server information

- Never write the machine's IP address, hostname, SSH user, or any absolute
  path outside this repo (e.g. sibling project paths) into a file that is
  tracked by git. Use `docs/server_local.md` (git-ignored) for anything like
  that, and reference it generically ("see the local server notes") in
  tracked files.
- Do not print environment variables containing credentials or tokens.
- Before any `git push`, check `git status` and the diff for anything that
  looks like a real path outside `/data/lgh/plant-context-fm`, an IP, or a
  username, and stop to ask if found.

## 1. Relationship to sibling projects on this machine

Three related projects already exist alongside this one. Do not duplicate or
fork their code; reference them explicitly instead.

- **SRG-GxE / ResidualGxE-Former** (`external/srg_gxe`, symlink, git-ignored):
  the actively-developed codebase for the G×E residual-learning chapter
  (Paper 3 / TDD §6.4, §15 item 8). Treat it as the starting point to audit
  against this project's stricter data contracts (TDD §4, §8, §10.2–10.3),
  not something to reimplement from scratch. It has its own git history and
  its own `PLAN.md`/`TDD.md` (English, narrower G×E-only scope) — do not
  edit files inside `external/srg_gxe` from this repo's tooling; open a
  branch in that repo directly if a change is warranted there.
- **PlantOmics-FM v2.0** (not symlinked here): a separate, currently stalled
  (~3 months idle) project targeting a different paper, using LoRA-adapted
  genomic foundation models (PlantCaduceus, partially-downloaded Evo2-7B) on
  Arabidopsis multi-omics data. Its pretrained `PlantCaduceus_l32` weights
  are a candidate encoder for this project's GenotypeTokenizer
  (adapter-on-foundation-model path, consistent with TDD §1.2 non-goal
  "no training a large DNA foundation model from scratch") — but this is a
  candidate to evaluate, not an assumed dependency. Do not copy its
  multi-omics fusion/training code.
- **AID-X** (different machine, not on this server): a completed, submitted
  project (Plant Phenomics) using the same underlying G2F/FIP1 raw data.
  Only its raw data tables are in scope for reuse; its VIB-GxE /
  ResidualGxEFormer model code is intentionally not reused — this project's
  model code is written fresh against the TDD's contracts.

## 2. Data policy

- `data/external/g2f` and `data/external/fip1` are symlinks into SRG-GxE's
  processed outputs — read from them, never write into them.
- Nothing under `data/` (raw, processed, interim, external) is committed to
  git. Only `data/manifests/*.{json,yaml}` (hashes, versions, counts) and
  `data/contracts/` schema definitions are tracked.
- Plant community/vegetation-plot data (sPlotOpen etc.) is not yet present.
  Do not start Paper 1/2 (community model) work until it is confirmed present
  under `data/external/community/` or similar, with a manifest.

## 3. Hardware

2× NVIDIA A100 80GB, ~251GB RAM, `/data` volume with multi-TB free — this
exceeds the MVP budget in the execution plan (§11), so default to the "full"
resource tier described there, but still gate large training runs behind the
12-week MVP decision points (execution plan §7, §12) rather than jumping
straight to large-scale pretraining.

- Long-running training jobs (>30 min) should be smoke-tested on a tiny
  synthetic slice first (TDD §10.6).
- Check `nvidia-smi` before claiming a GPU; this machine hosts multiple
  unrelated projects for other users.

## 4. Development process

Follow TDD §10 literally: write a failing test, write the minimal
implementation, pass unit tests, pass the synthetic end-to-end integration
test, only then run on real data. Leakage tests (TDD §10.3) and statistical
tests on controlled simulations (TDD §10.4) are required before any result
is reported, not optional polish.
