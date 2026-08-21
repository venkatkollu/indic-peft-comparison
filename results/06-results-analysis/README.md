# Corrected analysis outputs

`summary_with_ci.csv` now uses two-sided 95% t intervals with three seeds
(`df=2`, `t*=4.302652729`). These intervals quantify seed variation and must
not be treated as pairwise significance tests.

`rank_order_changes.csv` records descriptive changes in mean-method order. It
does not establish statistical crossovers.

`collapsed_runs.csv` uses `np.isclose(accuracy, 1/3, atol=0.001)` to include
equivalent constant-class outcomes when the validation labels are not perfectly
balanced.

The old normal-interval plot and original analysis notebook are preserved under
`archive/legacy-analysis/`. Run `notebooks/06-results-analysis.ipynb` in an
environment with the dependencies in `requirements.txt` to generate the
corrected learning-curve plot.
