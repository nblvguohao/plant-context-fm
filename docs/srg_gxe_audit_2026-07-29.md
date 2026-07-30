# SRG-GxE audit (TDD 15 item 8)

Date: 2026-07-29
Scope: `external/srg_gxe` (symlink to the SRG-GxE working copy on the local server; see `docs/server_local.md` for the absolute path), GitHub `nblvguohao/SRG-GxE`,
branch `syntax-aware-pretraining` at commit `92a91d5` plus a large amount of
uncommitted local work -- see [Repo state](#0-repo-state)).

Per the earlier decision recorded in this project's history: SRG-GxE is
Paper 3's existing codebase. This is not a request to reimplement it; it is
an audit against this project's data contracts, splits, and leakage tests,
to decide what to build on directly and what needs fixing first.

## 0. Repo state

`git status` in `external/srg_gxe` shows 7 modified tracked files and
**~100 untracked files**, including an entire undocumented `src/gxe_audit/`
package, ~50 numbered scripts, a `RESULTS.md`, `PLAN.md`, `TDD.md`, and a
`tests/` directory with 20 files. None of this is captured in a git commit.
Practically: almost everything interesting about this codebase right now
-- including the audit findings below -- exists only on disk, not in any
reviewable commit. Before treating any of this as a stable base to build on,
it needs to be committed (on its own branch, in that repo, not here).

## 1. SRG-GxE already ran a rigorous self-audit -- read RESULTS.md first

`RESULTS.md` (dated 2026-07-28, one day before this audit) is titled
"Experiment Log and G2F Benchmark Audit" and is, on its own, most of what
this TDD item asks for. Headline findings, verified independently below
where practical:

- **GraphGxE (their production model) is significantly *worse* than
  XGBoost/LightGBM/RF/GBLUP on leave_genotype** (corrected p=0.0035,
  Nadeau-Bengio resampled t-test). On leave_environment/forward_year the
  comparison is either underpowered or invalid (see below) -- not "inconclusive
  in our favor", just genuinely inconclusive.
- **The "syntax" (token-order) effect does not exist** in their setup: 15
  independent runs shuffling SNP marker columns and weather-day order each
  produced RMSE *slightly lower* than the normal order (opposite of the
  hypothesized direction), not distinguishable from seed noise at any
  conventional threshold. An earlier claim of a real order effect, based on
  one run per condition, is explicitly withdrawn.
- **A leaky model beats every model in the study.** A "leaky" additive
  main-effects baseline (genotype/environment means fit on train+val+test
  instead of the outer train fold only) achieves forward_year test RMSE
  1.917 -- better than XGBoost, LightGBM, and GraphGxE, with no G×E
  modeling at all. This is the direct analogue of this project's leakage
  tests (`tests/leakage/`), run empirically rather than structurally.
- **forward_year and leave_year are deterministic splits: n=1, not n=3.**
  Test sets are bit-identical across all three "seeds" (Jaccard 1.0 on
  18,274 test rows). Every reported "mean±sd over 3 seeds" for these two
  split types is actually reporting model-initialization noise on one fixed
  test set.
- **n=3 seeds cannot resolve the model differences being compared.** The
  Nadeau-Bengio-corrected minimum detectable effect on leave_environment is
  1.94 RMSE against observed gaps of 0.11-0.34 -- 5-17x below the detection
  floor.
- **The weather window misses the maize reproductive period.** Because
  `weather_daily.parquet` lacked a `day_after_planting` column, their
  preprocessing defaulted to calendar days 1-180 (Jan 1 - Jun 28), which
  never sees July-August flowering/grain-fill. Fixing the window to
  DOY 105-285 raises weather-to-yield environment-level R² from 0.259 to
  0.315, but weather alone still explains at most ~31% of environment-level
  yield variance.
- Section 9 of RESULTS.md is a 10-item "known issues" table with severities
  the authors already assigned; several are High.

None of this reads as self-serving -- it reads as someone finding out their
own headline result doesn't hold up and reporting that plainly. That is
exactly the standard this project's TDD asks for, and it is already done for
the core G×E prediction claim.

## 2. Independent verification performed here

### 2.1 The audit's own verification tooling is currently broken

`scripts/verify_leakage.py` and `src/gxe_audit/leakage_check.py` (an
otherwise well-designed empirical leakage-sensitivity checker -- see
[3](#3-a-good-idea-worth-borrowing) below) both crash:

```
KeyError: 'rmse'
```

The actual `outputs/statistics/leakage_quantification.csv` has a
`main_rmse` column, not `rmse`, and lacks the `naive_identical_to_foldsafe`
column both scripts check for first. Running `python scripts/verify_leakage.py`
today does not verify anything -- it crashes before printing a verdict,
despite its own docstring calling this "the paper's core finding."

### 2.2 Manually redoing the check (correct column name) confirms the RESULTS.md claims

Re-ran the L1/L2/L3 logic directly against `main_rmse` across all 7 split
types x 3 seeds in the actual CSV:

- **L1** (fold-safe == naive on test rows): passes exactly (diff=0.000000)
  for every split type and seed.
- **L2** (leaky < fold-safe on extrapolation splits): passes for all of
  forward_year, leave_environment, leave_ge, leave_year, spatial_block,
  with gaps of 29.6%-43.5%.
- **L3** (leaky ~= fold-safe on non-extrapolation splits): passes for
  leave_genotype (gap 0.086) and random (gap 0.090), both under the 0.15
  threshold.
- Directly confirmed the n=1 finding: forward_year's and leave_year's
  foldsafe/leaky `main_rmse` values are bit-identical (3.394422 / 1.917250)
  across seeds 1234, 2345, 3456.

So: the underlying data supports every claim in RESULTS.md section 2. The
verification *script* just cannot currently confirm this on its own; someone
has to redo it by hand, as done here, until the column-name drift is fixed
in that repo.

### 2.3 Test suite: 127/130 pass

`pytest tests/` in that repo: 3 failures.

- `test_fold_preprocessing.py::test_stage_summary_weather_uses_dap_windows_and_train_standardization`:
  an output-shape assertion off by one window (`(2, 6, 13)` vs expected
  `(2, 5, 13)`) -- looks like stage-window boundary logic changed after the
  test was written, not a leakage issue.
- `test_graph_gxe.py::TestMultiHeadGraphAttention::test_forward_shape` and
  `test_backward`: `forward()` now requires a positional `n_g` argument the
  tests don't pass -- a signature changed after the test was written
  (`graph_gxe.py` is one of the 7 modified-but-uncommitted files noted in
  [0](#0-repo-state)).

None of the three look like correctness/leakage bugs; they look like tests
not yet updated after in-progress refactoring. Still, "127/130" should not
be read as "fully green" without checking which 3 failed.

## 3. A good idea worth borrowing

`src/gxe_audit/leakage_check.py`'s method -- construct a deliberately
"leaky" variant of the pipeline and confirm it does (extrapolation splits)
or does not (non-extrapolation splits) look artificially better than the
fold-safe variant -- is a genuinely different and complementary check to
this project's own `tests/leakage/test_split_leakage.py`. This project's
leakage tests check a *necessary* condition (no ID overlap in the split
object itself); SRG-GxE's checks a *sufficient* one empirically (the whole
computational pipeline, not just the split, actually behaves as if it
respects fold safety). Worth adding an equivalent "leaky-variant" check to
this project once there is a full training pipeline to run it against --
not urgent now, since there is no such pipeline yet.

## 4. Cross-checked against this project's own pipeline

- **Weather-window truncation ([1](#1-srg-gxe-already-ran-a-rigorous-self-audit----read-resultsmd-first)):**
  `g2f_adapter.environment_daily_to_contract` computes `days_after_planting`
  itself from `environment.parquet`'s `estimated_planting_doy`, rather than
  assuming a fixed calendar-day window or relying on a `day_after_planting`
  column being present in the weather table. `EnvironmentTokenizer` then
  stages by cumulative GDD over however many post-planting days exist, with
  no hard day-count cutoff. This project should not have SRG-GxE's specific
  bug, but it has not been checked against real per-environment harvest
  dates either -- there is no upper bound on the window at all right now
  (documented already in `g2f_adapter.py`'s docstring).
- **forward_year seed-is-a-no-op: this project had the identical gap**,
  found and fixed as part of this audit. `make_forward_year_split` accepted
  a `seed` argument and recorded it into the output, but never used it --
  chronological order has nothing to seed. Fixed: the docstring now says so
  explicitly, and `test_forward_year_split_is_identical_regardless_of_seed`
  pins it down (three different seeds produce byte-identical splits). This
  would otherwise have been the same trap SRG-GxE fell into and had to
  retroactively document.

## 5. Implications for the doctoral plan

- **H1 (structured token order improves OOD generalization)** has a direct,
  well-powered null result against it from SRG-GxE's own 15-run shuffle
  test, for their architecture and data. That does not mean H1 is false for
  this project's different tokenizers/model, but it is real prior evidence
  to weigh, and a null result here would not be a new finding -- it would
  corroborate an existing one. Worth citing directly if Paper 2/3 ends up
  reporting the same thing.
- **Statistical power is a real, structural constraint, not a detail.**
  3-seed designs cannot resolve the model gaps typically seen in this
  domain (5-17x below MDE in SRG-GxE's own numbers). Paper 3 experiments
  built on `run_crossfit` should plan seed counts (or an alternative design)
  with this in mind from the start, not discover it after the fact.
- **GraphGxE underperforming simple baselines on leave_genotype** is
  consistent with this project's own low-rank G×E model result direction
  (TDD 10.4: no systematic gain without genuine interaction signal) and
  worth citing as independent, larger-scale corroboration rather than
  treating SRG-GxE and this project's Paper 3 chapter as needing to
  "beat" it -- the more defensible framing is a joint simplicity-audit
  finding across two independent implementations.
- **Gate B (baseline credibility, TDD 12) reads as: pass, with caveats.**
  The core empirical claims replicate under independent recomputation.
  The caveats (broken verification script, 3 failing tests, ~100 uncommitted
  files, n=1 mislabeled as n=3 until the authors caught it) are exactly the
  kind of thing that should be fixed in that repo, on its own branch, before
  anything here cites its numbers as final.
