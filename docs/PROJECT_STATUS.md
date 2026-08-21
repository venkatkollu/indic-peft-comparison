# Project status and continuation notes

Last updated: 2026-08-18

This document records what has been checked and changed so work can continue
without repeating the audit.

## Current project scope

The maintained primary study ends at notebook `06-results-analysis.ipynb`:

`00` environment → `01` dataset validation → `02` preprocessing → `03`
infrastructure checks → `04` pooled LR search → `05` 108-run primary sweep →
`06` corrected analysis.

The post-06 hybrid IA³+LoRA experiments are exploratory and are stored under
`archive/exploratory-hybrid/`. They are excluded from the primary conclusions.

## Work completed

- Audited all project notebooks, report materials, result CSVs, and saved PEFT
  adapter artifacts.
- Confirmed the primary result table contains 108 runs and the saved primary
  sweep contains 108 adapter configurations.
- Reorganized the repository:
  - primary notebooks are in `notebooks/`;
  - primary outputs are in `results/`;
  - report materials are in `docs/report-materials/`;
  - hybrid notebooks/results are in `archive/exploratory-hybrid/`.
- Replaced the old README with a source-of-truth project overview.
- Added `docs/validated-results.md` with claims supported by the stored primary
  results and explicit limitations.
- Added warnings to older report drafts so stale claims are not reused as final
  results.
- Replaced the old notebook 06 analysis with a corrected version.
- Corrected three-seed intervals using a two-sided t interval with `df=2` and
  `t*=4.302652729` instead of the old normal multiplier `1.96`.
- Replaced unsupported crossover claims with descriptive rank-order changes in
  `results/06-results-analysis/rank_order_changes.csv`.
- Corrected the collapse rule to include accuracy within `0.001` of the
  one-class baseline. The corrected count is 24/108 (22.2%): LoRA 8, DoRA 7,
  IA³ 9.
- Regenerated the corrected summary tables and learning-curve plot.
- Added `notebooks/07-test-set-evaluation.ipynb` for held-out evaluation of all
  108 saved models without retraining.

## Current primary findings

The stored primary results are validation-set results, not final held-out test
results. Mean macro-F1 winners are:

| Budget | Hindi | Telugu |
|---:|---|---|
| 50 | DoRA | DoRA |
| 100 | DoRA | IA³ |
| 500 | IA³ | IA³ |
| 1,000 | IA³ | IA³ |
| 2,000 | IA³ | IA³ |
| 20,000 | IA³ | LoRA |

Therefore, do not claim that IA³ wins every budget from 500 onward in both
languages, and do not claim 100% cross-language agreement.

## Important limitations still open

- Notebook 04 selects learning rates using validation data, and notebook 05
  reports validation performance on that same split.
- The held-out test split has not yet been evaluated.
- Notebook 03 still refers to `train_392702.parquet`, which notebook 02 does
  not create; fix or mark that stale check before final submission.
- Training epochs vary by budget, so budget and optimizer-update exposure are
  not completely isolated.
- The primary memory field measures a no-gradient forward pass, not true peak
  training memory.
- The raw/processed datasets are not committed to this repository.

## Next action: run notebook 07

Recommended environment: Kaggle, because the original data and model setup are
Kaggle-based. Attach the saved outputs of notebooks 02 and 05 as notebook
inputs. In notebook 07, the Kaggle paths normally have this form:

```python
DATA_ROOT = Path(
    "/kaggle/input/notebooks/venkatkolluu/02-data-preprocessingv2/data/processed"
)
PRIMARY_RESULTS_ROOT = Path(
    "/kaggle/input/notebooks/venkatkolluu/05-full-experiment-sweep"
)
```

Confirm the actual mounted paths first:

```python
!find /kaggle/input -type f \( -name "test.parquet" -o -name "experiment_results.csv" \)
```

`PRIMARY_RESULTS_ROOT` must directly contain `experiment_results.csv` and
`adapters/`. Notebook 07 evaluates the saved adapters on Hindi/Telugu
`test.parquet` and writes `test_results.csv`, `test_summary_with_ci.csv`, and
`test_ranking_table.csv` to its output directory. Do not use test results to
choose a new learning rate or training configuration.

## Files to use as the current source of truth

- Primary raw runs: `results/05-full-experiment-sweep/experiment_results.csv`
- Corrected validation summary: `results/06-results-analysis/summary_with_ci.csv`
- Corrected validation rankings: `results/06-results-analysis/ranking_table.csv`
- Corrected rank changes: `results/06-results-analysis/rank_order_changes.csv`
- Corrected plot: `results/06-results-analysis/learning_curves.png`
- Evidence-based interpretation: `docs/validated-results.md`

The original notebook 06 and its superseded plot are preserved under
`archive/legacy-analysis/` for traceability and should not be used for final
reporting.
