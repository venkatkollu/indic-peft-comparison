# Does PEFT Method Ranking Hold Across Script Families?

A data-efficiency study comparing LoRA, DoRA, and IA³ on Hindi and Telugu NLI, using
XLM-RoBERTa-base and the IndicXNLI benchmark.

## Research question

Prior PEFT comparison studies (Frontiers in Big Data 2025; PROPOR 2026) evaluate
LoRA/DoRA/IA³ at a single fixed dataset size, on Latin-script languages only. This
project asks two things no prior study has jointly tested:

1. Does the relative ranking of these three PEFT methods hold across different
   annotation budgets (50 to 20,000 examples)?
2. Does that ranking hold across script families — Hindi (Devanagari, Indo-Aryan)
   vs. Telugu (Telugu script, Dravidian)?

## Repository structure

```
notebooks/
  00-environment-setup.ipynb        Environment + GPU verification, library installs
  01-dataset-validation.ipynb       IndicXNLI download, EDA, split-overlap checks
  02-data-preprocessing.ipynb       Nested stratified budget subsets (50-20,000), saved as parquet
  03-infrastructure-validation.ipynb  PEFT config checks, tokenizer/truncation analysis, dry run
  04-hyperparameter-search.ipynb    Per-method LR search, pooled across both languages
  05-full-experiment-sweep.ipynb    Full 108-run sweep (3 methods x 2 languages x 6 budgets x 3 seeds)
  06-results-analysis.ipynb         Aggregation, 95% CIs, learning curves, crossover detection

results/
  experiment_results.csv    Raw per-run results (108 rows)
  summary_with_ci.csv       Mean/std/95% CI per (method, language, budget)
  ranking_table.csv         Best-performing method per (language, budget)
  compute_efficiency.csv    Trainable params / peak GPU memory / training time per method
  collapsed_runs.csv        Runs that failed to learn (see Limitations below)
  learning_curves.png       F1 vs. budget, one panel per language, with 95% CI bands

docs/
  methodology-notes.md      Design decisions, deviations from the original proposal, and why
```

## Setup

Each notebook was developed and run on Kaggle (GPU: Tesla T4 or P100, 16GB).
Notebooks 02, 04, and 05 depend on each other's committed outputs being attached
as Kaggle "Notebook" inputs, in this order: `02 -> 04 -> 05 -> 06`.

Key libraries: `transformers`, `peft`, `datasets`, `accelerate`, `torch`.

## Headline finding

**IA³ outperforms LoRA and DoRA at every budget from 500 samples upward, in both
Hindi and Telugu** -- a consistent crossover, not noise, since it holds across both
script families. LoRA and DoRA track each other closely throughout and are nearly
indistinguishable in final accuracy, despite DoRA's added computational cost.

| Budget | Hindi best | Telugu best |
|---|---|---|
| 500    | IA3 | IA3 |
| 1000   | IA3 | IA3 |
| 2000   | IA3 | IA3 |
| 20000  | IA3 | IA3 |

IA³ also shows a lower compute footprint in trainable parameters (657K vs. LoRA's
888K and DoRA's 906K) but a higher wall-clock training time in this implementation,
and a higher failure rate at the smallest budgets (see Limitations).

Full numbers: `results/summary_with_ci.csv` and `results/ranking_table.csv`.
Plots: `results/learning_curves.png`.

## Limitations (documented, not hidden)

- **"Full dataset" budget capped at 20,000** samples (of ~392,702 available),
  as a compute-constrained trade-off. This deviates from the literal wording of
  the original proposal ("the full dataset") and should be read as "a
  high-resource budget," not the complete training set.
- **13 of 108 runs (12%) failed to learn** (accuracy stuck at random chance,
  0.333). These cluster heavily in IA3 at the smallest budgets (50-100 samples,
  9 of 13 cases) and are documented in `results/collapsed_runs.csv` rather than
  discarded. This itself may be a finding: IA3's much higher learning rate
  (5e-3, vs. 1e-4 for LoRA/DoRA, chosen by a pooled hyperparameter search)
  appears to trade some small-budget stability for a stronger performance
  ceiling at moderate-to-large budgets.
- Learning rates were searched once per method, pooled across both languages
  (not per-language), to keep language a clean, uncontaminated variable in the
  cross-script comparison.
- See `docs/methodology-notes.md` for a full account of debugging decisions,
  including a classifier-head-freezing bug found and fixed during development.

## Status

Notebooks 00-06 complete. Remaining work: full write-up (Results/Discussion
sections), comparison against the Frontiers (2025) and PROPOR (2026) anchor
studies' reported rankings, and submission formatting for an ACL/EMNLP-style
venue.
