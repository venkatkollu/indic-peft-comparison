from pathlib import Path
import pandas as pd
ROOT=Path('/home/ubuntu/indic-peft-comparison')
files=[
'results/04-hyperparameter-search/lr_sweep_raw_results.csv',
'results/05-full-experiment-sweep/experiment_results.csv',
'results/06-results-analysis/summary_with_ci.csv',
'results/06-results-analysis/ranking_table.csv',
'results/06-results-analysis/compute_efficiency.csv',
'results/06-results-analysis/collapsed_runs.csv',
'results/06-results-analysis/rank_order_changes.csv',
'results/07-test-set-evaluation/test_results.csv',
'results/07-test-set-evaluation/test_summary_with_ci.csv',
'results/07-test-set-evaluation/test_ranking_table.csv',
'results/08 — Hybrid IA3+LoRA/hybrid_experiment_results.csv',
'results/09 — Hybrid test-set eval/hybrid_test_results.csv',
'results/10-final-comparison/hybrid_vs_best_primary.csv',
'results/10-final-comparison/test_summary_with_ci_all_methods.csv',
]
out=[]
for rel in files:
 p=ROOT/rel
 try:
  df=pd.read_csv(p)
  out.append(f'\n## {rel}\nshape={df.shape}\ncolumns={list(df.columns)}\n')
  out.append(df.to_string(index=False, max_rows=200))
 except Exception as e: out.append(f'\n## {rel}\nERROR {e}')
(ROOT/'audit'/'validated_tables.txt').write_text('\n'.join(out))
print(ROOT/'audit'/'validated_tables.txt')
