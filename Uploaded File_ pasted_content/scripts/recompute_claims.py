from pathlib import Path
import pandas as pd, numpy as np
T_CRIT_DF2 = 4.302652729
ROOT=Path('/home/ubuntu/indic-peft-comparison')
out=[]
primary=pd.read_csv(ROOT/'results/05-full-experiment-sweep/experiment_results.csv')
assert len(primary)==108
assert set(primary.method)=={'lora','dora','ia3'}
assert set(primary.language)=={'hi','te'}
assert set(primary.budget)=={50,100,500,1000,2000,20000}
assert set(primary.seed)=={42,123,456}
means=primary.groupby(['language','budget','method']).macro_f1.mean().reset_index()
winners=means.loc[means.groupby(['language','budget']).macro_f1.idxmax()].sort_values(['language','budget'])
out.append('PRIMARY ASSERTIONS: PASS')
out.append('PRIMARY WINNERS BY MEAN MACRO-F1')
out.append(winners.to_string(index=False))
collapse=primary[np.isclose(primary.accuracy,1/3,atol=.001)]
out.append(f'COLLAPSE: {len(collapse)}/108 = {len(collapse)/108:.4%}')
out.append(collapse.groupby('method').size().to_string())
summary=primary.groupby(['method','language','budget']).macro_f1.agg(['mean','std','count']).reset_index()
summary['ci95']=T_CRIT_DF2*summary['std']/np.sqrt(summary['count'])
out.append('PRIMARY SUMMARY WITH RECOMPUTED CI')
out.append(summary.to_string(index=False))
hybrid=pd.read_csv(ROOT/'results/10-final-comparison/hybrid_vs_best_primary.csv')
assert len(hybrid)==12
assert hybrid.hybrid_wins.sum()==8
out.append(f'HYBRID: {int(hybrid.hybrid_wins.sum())}/12 = {hybrid.hybrid_wins.mean():.4%}')
out.append(hybrid[['language','budget','best_primary_method','best_primary','hybrid_ia3_lora','hybrid_delta','hybrid_wins']].to_string(index=False))
for f in ['results/06-results-analysis/summary_with_ci.csv','results/06-results-analysis/ranking_table.csv','results/06-results-analysis/compute_efficiency.csv','results/06-results-analysis/collapsed_runs.csv']:
 df=pd.read_csv(ROOT/f); out.append(f'\n{f} shape={df.shape}\n{df.to_string(index=False)}')
(ROOT/'audit'/'recomputed_claims.txt').write_text('\n\n'.join(out))
print('PASS; wrote',ROOT/'audit'/'recomputed_claims.txt')
