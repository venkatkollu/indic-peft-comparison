# PEFT Ranking Across Indic Script Families

> **Parameter-efficient fine-tuning comparison of LoRA, DoRA, and IA³ for
> 3-class IndicXNLI natural-language inference in Hindi and Telugu using
> XLM-RoBERTa-base.**

## Highlights

| Budget | Hindi winner | Telugu winner |
|-------:|-------------|--------------|
|     50 | DoRA        | DoRA         |
|    100 | DoRA        | IA³          |
|    500 | IA³         | IA³          |
|  1,000 | IA³         | IA³          |
|  2,000 | IA³         | IA³          |
| 20,000 | IA³         | LoRA         |

*Validation-set results (mean macro-F1 across 3 seeds). See
[validated-results.md](docs/validated-results.md) for supported claims and
known limitations.*

## Experimental scope

- **Methods:** LoRA, DoRA, IA³
- **Languages:** Hindi (`hi`) and Telugu (`te`)
- **Training budgets:** 50 · 100 · 500 · 1,000 · 2,000 · 20,000 examples
- **Seeds:** 42, 123, 456
- **Total runs:** 108 validation-set experiments

## Getting started

```bash
# Clone the repository
git clone https://github.com/venkatkollu/indic-peft-comparison.git
cd indic-peft-comparison

# Install dependencies
pip install -r requirements.txt
```

The notebooks were executed on Kaggle and contain Kaggle-specific input paths.
Download IndicXNLI and recreate the processed subsets with notebooks 01–02
before rerunning notebooks 04–06.

## Repository layout

```
notebooks/                         Primary reproducible pipeline (00–06)
  00-environment-setup.ipynb
  01-dataset-validation.ipynb
  02-data-preprocessingv2.ipynb
  03-experimental-infrastructure-validation.ipynb
  04-hyperparameter-search.ipynb
  05-full-experiment-sweep.ipynb
  06-results-analysis.ipynb
  07-test-set-evaluation.ipynb      Held-out evaluation of the saved 108 models

results/
  04-hyperparameter-search/        LR sweep and selected method-level LRs
  05-full-experiment-sweep/        108-run CSV, JSONL log, and adapter configs
  06-results-analysis/             Aggregates, rankings, plot, and collapse list

docs/
  validated-results.md             Source-of-truth interpretation of results 00–06
  PROJECT_STATUS.md                Audit log and continuation notes
  report-materials/                Earlier report-writing materials; verify against
                                   validated-results.md before reusing

archive/exploratory-hybrid/        Post-06 hybrid work; excluded from the primary study
archive/legacy-analysis/           Original notebook 06 preserved for traceability
```

## Primary result files

| File | Description |
|------|-------------|
| `results/05-full-experiment-sweep/experiment_results.csv` | All 108 runs |
| `results/06-results-analysis/summary_with_ci.csv` | Seed means, SDs, corrected 3-seed *t* intervals |
| `results/06-results-analysis/ranking_table.csv` | Highest mean macro-F1 per cell |
| `results/06-results-analysis/collapsed_runs.csv` | 24 runs at/near chance-accuracy baseline (~0.333) |
| `results/06-results-analysis/learning_curves.png` | Learning curve plot |

## Adapter weights

The trained PEFT adapter weights (`.safetensors`, `.pt`) are **not** committed
to this repository to keep the clone size small. Adapter configuration files
(`adapter_config.json`, `adapter_info.json`) are included. The full adapter
weights are available on the HuggingFace Hub.

## Reproducibility notes

- The notebooks were executed on Kaggle with Kaggle-specific input paths.
- The raw and processed datasets are not committed here. Download IndicXNLI and
  recreate the processed subsets with notebooks 01–02 before rerunning
  notebooks 04–06.
- Notebook 04 selects learning rates using validation data, and notebook 05
  reports validation performance on the same split.

## License

This project is licensed under the [MIT License](LICENSE).
