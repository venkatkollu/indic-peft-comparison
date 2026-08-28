from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt, numpy as np
ROOT=Path('/home/ubuntu/indic-peft-comparison'); OUT=ROOT/'audit'/'figures'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'legend.fontsize':9,'figure.dpi':160})
primary=pd.read_csv(ROOT/'results/05-full-experiment-sweep/experiment_results.csv')
summary=primary.groupby(['method','language','budget']).macro_f1.agg(['mean','std']).reset_index()
labels={'lora':'LoRA','dora':'DoRA','ia3':'IA³'}; colors={'lora':'#1f77b4','dora':'#d62728','ia3':'#2ca02c'}
# learning curves
fig,axs=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for ax,lang,title in zip(axs,['hi','te'],['Hindi','Telugu']):
 for m in ['lora','dora','ia3']:
  d=summary[(summary.language==lang)&(summary.method==m)].sort_values('budget')
  ax.errorbar(d.budget,d['mean'],yerr=d['std'],marker='o',lw=2,c=colors[m],label=labels[m],capsize=3)
 ax.set_xscale('log'); ax.set_xticks([50,100,500,1000,2000,20000]); ax.set_xticklabels(['50','100','500','1k','2k','20k']); ax.set_title(title); ax.set_xlabel('Training examples'); ax.grid(alpha=.25)
axs[0].set_ylabel('Validation macro-F1'); axs[1].legend(frameon=False); fig.suptitle('Primary validation learning curves (mean ± one standard deviation)',y=1.02); fig.tight_layout(); fig.savefig(OUT/'fig_learning_curves.png',bbox_inches='tight'); plt.close(fig)
# rankings heatmap-like matrix
wins=summary.loc[summary.groupby(['language','budget'])['mean'].idxmax()].copy(); order=[50,100,500,1000,2000,20000]
fig,ax=plt.subplots(figsize=(9,2.8)); mat=np.array([[{'lora':0,'dora':1,'ia3':2}[wins[(wins.language==l)&(wins.budget==b)].iloc[0].method] for b in order] for l in ['hi','te']])
from matplotlib.colors import ListedColormap
ax.imshow(mat,cmap=ListedColormap(['#6baed6','#fc9272','#74c476']),vmin=0,vmax=2,aspect='auto')
for i,l in enumerate(['Hindi','Telugu']):
 for j,b in enumerate(order): ax.text(j,i,labels[wins[(wins.language==['hi','te'][i])&(wins.budget==b)].iloc[0].method],ha='center',va='center',fontweight='bold')
ax.set_xticks(range(6),['50','100','500','1k','2k','20k']); ax.set_yticks(range(2),['Hindi','Telugu']); ax.set_xlabel('Training budget'); ax.set_title('Winning primary method by language and budget'); fig.tight_layout(); fig.savefig(OUT/'fig_ranking_matrix.png',bbox_inches='tight'); plt.close(fig)
# compute efficiency
comp=primary.groupby('method').agg(trainable_params=('trainable_params','first'),memory_gb=('peak_gpu_memory_gb','mean'),time_sec=('training_time_sec','mean')).reset_index()
fig,axs=plt.subplots(1,3,figsize=(10,3.6)); x=np.arange(3); names=[labels[m] for m in comp.method]
for ax,col,title,ylabel in zip(axs,['trainable_params','memory_gb','time_sec'],['Trainable parameters','Peak GPU memory','Mean run time'],['Parameters','GB','Seconds']):
 ax.bar(x,comp[col],color=[colors[m] for m in comp.method]); ax.set_xticks(x,names); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.25)
 for i,v in enumerate(comp[col]): ax.text(i,v,f'{v:,.0f}' if col=='trainable_params' else f'{v:.2f}',ha='center',va='bottom',fontsize=8)
fig.suptitle('Primary computational measurements'); fig.tight_layout(); fig.savefig(OUT/'fig_compute_comparison.png',bbox_inches='tight'); plt.close(fig)
# hybrid deltas
h=pd.read_csv(ROOT/'results/10-final-comparison/hybrid_vs_best_primary.csv'); h['cell']=h.language.map({'hi':'Hindi','te':'Telugu'})+' '+h.budget.astype(str)
fig,ax=plt.subplots(figsize=(10,4)); cols=['#2ca02c' if v else '#d62728' for v in h.hybrid_wins]; ax.bar(np.arange(len(h)),h.hybrid_delta,color=cols); ax.axhline(0,color='black',lw=.8); ax.set_xticks(np.arange(len(h)),h.cell,rotation=45,ha='right'); ax.set_ylabel('Hybrid macro-F1 − best primary macro-F1'); ax.set_title('Held-out hybrid comparison by budget and language'); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(OUT/'fig_hybrid_delta.png',bbox_inches='tight'); plt.close(fig)
print('created',len(list(OUT.glob('*.png'))),'figures in',OUT)
