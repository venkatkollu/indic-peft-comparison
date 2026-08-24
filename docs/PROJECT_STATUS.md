# Project status and continuation notes

Last updated: 2026-08-24

This document records the current project state after the final validation audit,
held-out test evaluation, and hybrid comparison work.

## Current project scope

The maintained project is organized around the full notebook sequence in
`notebooks/`:

`00` environment → `01` dataset validation → `02` preprocessing → `03`
infrastructure checks → `04` pooled LR search → `05` 108-run primary sweep →
`06` corrected analysis → `07` held-out evaluation → `08` hybrid formal study →
`09` hybrid test evaluation → `10` final comparison.

The hybrid IA³+LoRA work remains in the active project tree under
`notebooks/08-hybrid-formal.ipynb`, `notebooks/09-hybrid-test-evaluation.ipynb`,
and `notebooks/10-final-comparison.ipynb`, with the corresponding outputs in
`results/08 — Hybrid IA3+LoRA/`, `results/09 — Hybrid test-set eval/`, and
`results/10-final-comparison/`. It is treated as a follow-up study with its own
held-out test comparison and is not a replacement for the primary validation
main findings.

## Work completed

- Audited the project notebooks, saved outputs, and documentation structure.
- Confirmed the maintained primary sweep contains 108 validation-set runs.
- Reorganized the repository around the active notebook sequence and current
  result folders.
- Corrected the README and source-of-truth documentation to separate primary and
  hybrid findings.
- Added `docs/validated-results.md` as the current summary for the main study,
  with explicit interpretation limits and the hybrid follow-up clearly marked as
  separate.
- Corrected the validation analysis to use the supported summary and interval
  logic: two-sided t interval with `df=2` and `t*=4.302652729`.
- Corrected the collapse rule so 24/108 runs (22.2%) are treated as equivalent
  near-baseline cases: LoRA 8, DoRA 7, IA³ 9.
- Regenerated the corrected validation rankings and result summary files.
- Executed the final comparison workflow in `notebooks/10-final-comparison.ipynb`.
- Verified the final test-set head-to-head result: hybrid wins 8/12 cells (67%).
- Committed and pushed the final documentation and results update to GitHub.

## Current findings

### Primary validation-set results

These are the validated results from the main sweep on the validation split. They
are not final held-out test-set results.

| Budget | Hindi | Telugu |
|---:|---|---|
| 50 | DoRA | DoRA |
| 100 | DoRA | IA³ |
| 500 | IA³ | IA³ |
| 1,000 | IA³ | IA³ |
| 2,000 | IA³ | IA³ |
| 20,000 | IA³ | LoRA |

This supports the claim that IA³ is the strongest mean-performing method at the
intermediate 500–2,000 budgets in both languages and the dominant Hindi method
from 500 onward, but it does not justify claiming IA³ wins every budget from 500
onward in both languages.

### Hybrid held-out test comparison

The hybrid IA³+LoRA method was evaluated against the best primary method for
each budget-language cell on the held-out test split.

Verified result:

- Hybrid wins 8/12 cells
- Win rate: 67%
- Wins occur at budgets 100, 500, 2,000, and 20,000 in both languages
- Losses occur at budgets 50 and 1,000 in both languages

This is a separate result from the primary validation study and should not be
mixed into the main validation summary table.

## Important limitations and caveats

- The primary sweep uses validation data for learning-rate selection and is
  therefore validation-set analysis, not final held-out evaluation.
- The hybrid method uses a dual-LR setup that was not optimized under the same
  LR-search protocol as the primary methods.
- `summary_with_ci.csv` reports seed variability via a two-sided t interval, not
  a pairwise significance result.
- Training budgets vary in epoch schedule, so budget and optimizer-update exposure
  are not fully isolated.
- The raw or processed datasets are not stored in this repository as a full data
  asset; the project depends on the preserved saved experiment outputs and
  notebook-generated artifacts.

## Current source-of-truth files

- Primary validation summary: `results/06-results-analysis/summary_with_ci.csv`
- Primary rankings: `results/06-results-analysis/ranking_table.csv`
- Final hybrid comparison: `results/10-final-comparison/hybrid_vs_best_primary.csv`
- Project overview: `README.md`
- Supported interpretation notes: `docs/validated-results.md`

The current source of truth is the corrected validation outputs plus the final
hybrid test comparison, and the project should be reported using that split.
