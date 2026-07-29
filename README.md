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

## Relationship to sibling projects

This repository does not redistribute or duplicate data or model code from
other projects on this machine. Instead:

- `external/srg_gxe` is a symlink to the existing `ResidualGxE-Former` (SRG-GxE)
  repository, which is the working codebase for the G×E residual-learning
  chapter (Paper 3). This project audits and extends it against the stricter
  data contracts and leakage tests in the TDD rather than reimplementing it.
- `data/external/g2f` and `data/external/fip1` are symlinks into SRG-GxE's own
  `data/processed/` outputs (phenotype/genotype/environment/weather tables,
  splits, residual targets). They are the canonical processed data for this
  machine; nothing here re-derives them independently.
- Plant community/vegetation-plot data (e.g. sPlotOpen) is not yet present on
  this machine and must be obtained separately before Paper 1/2 work can
  start; see `docs/data_inventory.md` (to be created) once it lands.

Everything under `external/` and `data/external/` is git-ignored: this
repository never commits paths, filenames, or directory layouts belonging to
sibling projects.

## Repository layout

See `docs/plant_context_fm_TDD_zh.md` §3 for the full intended layout
(tokenizers, statistics, association, evaluation, tracking modules, and the
unit/contracts/leakage/integration/statistical/regression test tree already
scaffolded here).

## Status

Scaffolding only — directory structure, symlinks to existing data/code, and
planning docs. No pipeline code has been written yet.
