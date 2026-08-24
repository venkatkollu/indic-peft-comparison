# Validated primary-study results (notebooks 00–06)

This document is the source of truth for claims drawn from the maintained
primary experiment. It is based on the files in `results/` and replaces older
headline claims in report-writing materials.

## What was run

The full sweep contains 108 validation-set runs:

`3 methods × 2 languages × 6 training budgets × 3 seeds`.

The methods are LoRA, DoRA, and IA³. Each run uses XLM-RoBERTa-base with a
trainable classifier head. Learning rates selected by the pooled search are
LoRA `1e-4`, DoRA `1e-4`, and IA³ `5e-3`.

## Mean macro-F1 ranking

| Budget | Hindi winner | Telugu winner |
|---:|---|---|
| 50 | DoRA | DoRA |
| 100 | DoRA | IA³ |
| 500 | IA³ | IA³ |
| 1,000 | IA³ | IA³ |
| 2,000 | IA³ | IA³ |
| 20,000 | IA³ | LoRA |

The full values are in `results/06-results-analysis/ranking_table.csv`.
Thus, the supported descriptive statement is that IA³ is the highest-mean
method at the intermediate 500–2,000 budgets in both languages, and in all
reported Hindi budgets from 500 onward. It is not correct to claim that IA³
wins every budget from 500 onward in both languages.

## Important interpretation limits

- These are validation-set results. The available `test.parquet` split is not
  used by the main sweep, and the validation split is also used in learning-rate
  selection. Do not present the metrics as final held-out test performance.
- The corrected analysis records 24 of 108 runs (22.2%) at or within 0.001 of
  the one-class chance-accuracy baseline (approximately 0.3333). The collapse
  counts are LoRA 8, DoRA 7, and IA³ 9. The older exact-rounding rule counted
  only 19 and missed equivalent cases caused by slight validation class-count
  imbalance.
- `summary_with_ci.csv` uses a two-sided t interval for three seeds (`df=2`,
  `t*=4.302652729`). These intervals describe seed variation, not pairwise
  significance; do not infer a winner from interval overlap alone.
- `rank_order_changes.csv` records descriptive changes in rank order. The old
  crossover count is retired because its code did not apply the calculated
  confidence-interval separation condition.
- The epoch schedule varies by budget, so training budget and the number of
  optimizer updates are not fully isolated.

## Follow-up work

The hybrid IA³+LoRA experiments are kept in the active project tree under
`notebooks/08-hybrid-formal.ipynb`, `notebooks/09-hybrid-test-evaluation.ipynb`,
and the corresponding result folders under `results/08 — Hybrid IA3+LoRA/` and
`results/09 — Hybrid test-set eval/`. They are follow-on analyses and should be
read as supplemental work rather than as replacements for the validated primary
study.
