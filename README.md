# PEFT Ranking Across Indic Script Families

> Final project overview for the IndicXNLI PEFT study, including the primary LoRA/DoRA/IA³ sweep and the hybrid IA³+LoRA follow-up.

## Current status

This repository contains the full reproducible study and the final comparison workflow.
The primary analysis is the validation-set study in notebooks 00–06, while notebooks
07–10 cover held-out test evaluation and the hybrid method comparison.

The authoritative interpretation is in [docs/validated-results.md](docs/validated-results.md).
The project history and continuation notes are in [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).

## Primary findings

### 1) Primary validation-set results (source of truth for the main study)

These are the validated results from the main LoRA/DoRA/IA³ sweep on the validation split.
They are not the held-out test-set results.

| Budget | Hindi winner | Telugu winner |
|-------:|-------------|--------------|
|     50 | DoRA        | DoRA         |
|    100 | DoRA        | IA³          |
|    500 | IA³         | IA³          |
|  1,000 | IA³         | IA³          |
|  2,000 | IA³         | IA³          |
| 20,000 | IA³         | LoRA         |

Important: this table reflects the main validation analysis only. See
[docs/validated-results.md](docs/validated-results.md) for the supported claims,
caveats, and limitations.

### 2) Hybrid comparison on the held-out test set

This is a separate analysis that compares the hybrid IA³+LoRA method against the best
primary method for each budget-language cell.

The final notebook is [notebooks/10-final-comparison.ipynb](notebooks/10-final-comparison.ipynb).
The generated comparison outputs are in [results/10-final-comparison](results/10-final-comparison).

Verified head-to-head result on the test set:

- Hybrid wins 8/12 budget-language cells
- Win rate: 67%
- Wins occur at budgets 100, 500, 2000, and 20000 in both languages
- Losses occur at budget 50 and budget 1000 in both languages
- The 50-budget loss is collapse-driven; the 1000-budget loss is a real IA³-specific effect

This is a separate, test-set result. It should not be mixed with the validation-set table above.

## Repository layout

```text
notebooks/
  00-environment-setup.ipynb
  01-dataset-validation.ipynb
  02-data-preprocessingv2.ipynb
  03-experimental-infrastructure-validation.ipynb
  04-hyperparameter-search.ipynb
  05-full-experiment-sweep.ipynb
  06-results-analysis.ipynb
  07-test-set-evaluation.ipynb
  08-hybrid-formal.ipynb
  09-hybrid-test-evaluation.ipynb
  10-final-comparison.ipynb

results/
  04-hyperparameter-search/
  05-full-experiment-sweep/
  06-results-analysis/
  07-test-set-evaluation/
  08 — Hybrid IA3+LoRA/
  09 — Hybrid test-set eval/
  10-final-comparison/
  final_learning_curves.png

docs/
  PROJECT_STATUS.md
  validated-results.md
  report-materials/

LICENSE
requirements.txt
```

## Key artifacts

- Primary raw results: `results/05-full-experiment-sweep/experiment_results.csv`
- Corrected validation summary: `results/06-results-analysis/summary_with_ci.csv`
- Corrected rankings: `results/06-results-analysis/ranking_table.csv`
- Final held-out comparison outputs: `results/10-final-comparison/`
- Final learning-curve plot: `results/final_learning_curves.png`

## Getting started

```bash
# Clone the repository
git clone https://github.com/venkatkollu/indic-peft-comparison.git
cd indic-peft-comparison

# Install dependencies
pip install -r requirements.txt
```

The notebooks were executed on Kaggle and contain Kaggle-specific input paths.
Download IndicXNLI and recreate the processed subsets with notebooks 01–02 before
rerunning notebooks 04–06 or the later test/hybrid notebooks.

## Reproducibility notes

- Validation results are based on the stored primary sweep and should be interpreted
  using the caveats in [docs/validated-results.md](docs/validated-results.md).
- Test-set notebooks operate on saved outputs in `results/07-test-set-evaluation/`
  and `results/09 — Hybrid test-set eval/`.
- The hybrid notebook uses a dual-LR setup that was not validated with the same LR
  search protocol used for the primary methods; that asymmetry should be disclosed.
- The trained PEFT adapter weights are not committed to the repository; only the
  adapter config files are stored with the experiment artifacts.

## License

This project is licensed under the [MIT License](LICENSE).
