# Corrected analysis outputs

This directory contains the validated primary-study summaries used for the final
analysis. `summary_with_ci.csv` uses two-sided 95% t intervals with three seeds
(`df=2`, `t*=4.302652729`). These intervals quantify seed variation and must not
be treated as pairwise significance tests.

`rank_order_changes.csv` records descriptive changes in mean-method order. It
does not establish statistical crossovers.

`collapsed_runs.csv` uses `np.isclose(accuracy, 1/3, atol=0.001)` to include
equivalent constant-class outcomes when the validation labels are not perfectly
balanced.

The main notebook that produced these outputs is `notebooks/06-results-analysis.ipynb`.
Use the environment in `requirements.txt` and the source files under `results/` to
reproduce or extend the analysis. The current interpretation is governed by
`docs/validated-results.md`.
