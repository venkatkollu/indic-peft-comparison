# Project status

Last updated: 2026-08-28

## Archive scope

This is an organized, self-contained capstone archive. It contains one
experiment notebook, the final CSV exports, figures, report outputs,
documentation, and the historical scripts used to build or audit the report.
It does **not** contain the original raw datasets, adapter checkpoints, or the
full multi-notebook experiment tree referenced by some historical files.

## Preserved final artifacts

- `results/final_primary_results.csv`: 108 seed-level primary validation rows.
- `results/final_summary_with_ci.csv`: 36 aggregated primary rows with
  three-seed t intervals.
- `results/final_hybrid_comparison.csv`: 12 held-out hybrid comparison cells.
- `figures/`: four report figures.
- `reports/final_capstone_report.docx`: generated final capstone report.
- `notebooks/05-full-experiment-sweep.ipynb`: preserved sweep notebook.

## Current findings

The primary validation sweep compares LoRA, DoRA, and IA³ across two languages,
six budgets, and three seeds (3 × 2 × 6 × 3 = 108 runs). The highest mean
macro-F1 method by budget is:

| Budget | Hindi | Telugu |
|---:|---|---|
| 50 | DoRA | DoRA |
| 100 | DoRA | IA³ |
| 500 | IA³ | IA³ |
| 1,000 | IA³ | IA³ |
| 2,000 | IA³ | IA³ |
| 20,000 | IA³ | LoRA |

The separate hybrid IA³+LoRA held-out comparison wins 8 of 12 cells (67%).
This must not be merged into the primary validation findings.

## Interpretation limits

- The primary data are validation-set results and should not be presented as a
  final held-out evaluation.
- Three-seed intervals describe variability; they are not pairwise significance
  tests.
- The hybrid configuration did not use the same learning-rate selection protocol
  as the primary methods.
- Historical scripts reference a missing original project location and raw
  experiment files. They are retained for provenance, not direct execution.

See `validated-results.md` for the full maintained interpretation.
