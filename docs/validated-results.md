# Final project results summary

This document consolidates the full evidence from the maintained project: the
primary validation sweep and the later hybrid held-out test comparison. It is
intended to be the one-page summary for reporting and interpretation.

## 1) Scope of the study

The project includes three phases:

1. Primary validation sweep in notebooks `00–06`
2. Held-out test evaluation in notebook `07`
3. Hybrid IA³+LoRA follow-up and final comparison in notebooks `08–10`

The main primary sweep is the validated validation-set experiment. The hybrid
results are a separate follow-up analysis on the held-out test split and are
therefore reported as a distinct comparison, not as a replacement for the
primary validation claims.

## 2) Primary validation-set results (the main study)

The full primary sweep contains 108 validation-set runs:

`3 methods × 2 languages × 6 training budgets × 3 seeds`

The methods are LoRA, DoRA, and IA³. The maintained validation analysis uses
XLM-RoBERTa-base with a trainable classifier head. The pooled search selected
LoRA `1e-4`, DoRA `1e-4`, and IA³ `5e-3`.

### Mean macro-F1 ranking by budget

| Budget | Hindi winner | Telugu winner |
|---:|---|---|
| 50 | DoRA | DoRA |
| 100 | DoRA | IA³ |
| 500 | IA³ | IA³ |
| 1,000 | IA³ | IA³ |
| 2,000 | IA³ | IA³ |
| 20,000 | IA³ | LoRA |

The supported descriptive reading is that IA³ is the highest-mean method at the
intermediate 500–2,000 budgets in both languages, and in all reported Hindi
budgets from 500 onward. It is not correct to claim that IA³ wins every budget
from 500 onward in both languages.

The full primary values are in `results/06-results-analysis/ranking_table.csv`.
The corrected validation summary is in `results/06-results-analysis/summary_with_ci.csv`.

## 3) Held-out test-set comparison and hybrid follow-up

The hybrid IA³+LoRA experiments are kept in the active project tree under:

- `notebooks/08-hybrid-formal.ipynb`
- `notebooks/09-hybrid-test-evaluation.ipynb`
- `notebooks/10-final-comparison.ipynb`
- `results/08 — Hybrid IA3+LoRA/`
- `results/09 — Hybrid test-set eval/`
- `results/10-final-comparison/`

These are follow-on analyses and should be interpreted as a supplemental,
test-set comparison rather than a replacement for the validated primary study.

### Head-to-head result: hybrid vs best primary method

The final comparison uses the tested cells in `results/10-final-comparison/hybrid_vs_best_primary.csv`.
The verified test-set outcome is:

- Hybrid wins 8/12 budget-language cells
- Win rate: 67%
- Wins occur at 100, 500, 2,000, and 20,000 in both languages
- Losses occur at 50 and 1,000 in both languages

This means the hybrid method improves over the best primary approach in the
majority of the held-out test cells.

### Final comparison summary table

| Language | Budget | Best primary | Hybrid | Result |
|---|---:|---:|---:|---|
| Hindi | 50 | 0.2211 (DoRA) | 0.1667 | Loss |
| Hindi | 100 | 0.1987 (LoRA) | 0.2758 | Win |
| Hindi | 500 | 0.3756 (IA³) | 0.4361 | Win |
| Hindi | 1000 | 0.4012 (IA³) | 0.3492 | Loss |
| Hindi | 2000 | 0.4901 (IA³) | 0.5425 | Win |
| Hindi | 20000 | 0.6700 (IA³) | 0.6913 | Win |
| Telugu | 50 | 0.2026 (LoRA) | 0.1667 | Loss |
| Telugu | 100 | 0.2033 (LoRA) | 0.2754 | Win |
| Telugu | 500 | 0.3539 (IA³) | 0.4060 | Win |
| Telugu | 1000 | 0.2635 (IA³) | 0.2577 | Loss |
| Telugu | 2000 | 0.4010 (IA³) | 0.4752 | Win |
| Telugu | 20000 | 0.6280 (LoRA) | 0.6539 | Win |

The complete source data for the final comparison is in
`results/10-final-comparison/`.

## 4) Final interpretation

The project supports a two-level interpretation:

- For the primary validation sweep, IA³ is the strongest mean-performing method at
  the intermediate budgets and is the most defensible main validation result.
- For the held-out test comparison, the hybrid IA³+LoRA method is the stronger
  final test-set approach in the majority of budget-language cells.

Taken together, the strongest overall claim is that the primary validation study
identifies IA³ as the leading baseline family, while the later hybrid test-set
analysis shows that a hybrid IA³+LoRA configuration can improve over the best
primary method in the majority of held-out settings.

## 5) Important caveats

- These are validation-set results for the primary sweep. The available
  `test.parquet` split is not used by the main sweep, and the validation split is
  also used in learning-rate selection.
- `summary_with_ci.csv` uses a two-sided t interval for three seeds (`df=2`,
  `t*=4.302652729`). These intervals describe seed variation, not pairwise
  significance.
- The corrected analysis records 24 of 108 runs (22.2%) at or within 0.001 of
  the one-class chance-accuracy baseline (approximately 0.3333). The collapse
  counts are LoRA 8, DoRA 7, and IA³ 9.
- The hybrid configuration uses a dual-LR setup that was not validated with the
  same LR-search protocol used for the primary methods; that asymmetry should be
  disclosed in any write-up.
- The epoch schedule varies by budget, so training budget and the number of
  optimizer updates are not fully isolated.

This document supersedes older draft claims and is the maintained summary for the
full project evidence set.
