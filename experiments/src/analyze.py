"""Summarize the frozen experiment and paired conditional block uncertainty."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import json,pickle,itertools
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from data import ROOT,TASKS,load_task
from train import SEEDS
BASELINES=['xgb_clean','xgb_group','xgb_independent','rf_group','ridge_group','impute_xgb','tsk_full','tsk_group','mlp_group']
OUT=ROOT/'experiments/results/analysis'

def bootstrap_task(task):
 with threadpool_limits(1):
  arrays={}; truth=None
  for seed in SEEDS:
   with np.load(ROOT/f'experiments/results/evaluation/{task}_{seed}_predictions.npz') as z:
    truth=z['truth'].astype(float);keep=z['severity']==2
    for model in ['sfr']+BASELINES:arrays.setdefault(model,[]).append((z[model][keep].astype(float)-truth)**2)
  arrays={k:np.concatenate(v,axis=0) for k,v in arrays.items()};n=len(truth);rows=[]
  for length in [12,24,48]:
   starts=np.arange(0,n,length);lens=np.minimum(length,n-starts);B=len(starts)
   rng=np.random.default_rng(81373+length);counts=rng.multinomial(B,np.ones(B)/B,size=2000).astype(float);den=counts@lens
   stats={}
   for model,e in arrays.items():
    sums=np.add.reduceat(e,starts,axis=1);stats[model]=np.sqrt((counts@sums.T)/den[:,None]).mean(axis=1)
   for model in BASELINES:
    gain=100*(1-stats['sfr']/stats[model]);point=100*(1-np.sqrt(arrays['sfr'].mean(axis=1)).mean()/np.sqrt(arrays[model].mean(axis=1)).mean())
    lo,hi=np.quantile(gain,[.025,.975]);rows.append(dict(task=task,baseline=model,block_length=length,gain_percent=point,lower=lo,upper=hi,replicates=2000,blocks=B))
  return rows

def summarize():
 OUT.mkdir(exist_ok=True,parents=True);df=pd.read_csv(ROOT/'experiments/results/all_metrics.csv')
 per=df.groupby(['task','model','seed','k'],as_index=False).agg(rmse=('rmse','mean'),nrmse=('nrmse','mean'),worst=('rmse','max'),mae=('mae','mean'))
 per.to_csv(OUT/'per_seed.csv',index=False)
 agg=per.groupby(['task','model','k'],as_index=False).agg(rmse=('rmse','mean'),rmse_seed_sd=('rmse','std'),nrmse=('nrmse','mean'),worst=('worst','mean'),mae=('mae','mean'))
 agg.to_csv(OUT/'summary.csv',index=False)
 models=[];validation=[]
 for task in TASKS:
  for seed in SEEDS:
   for kind in ['fuzzy','linear']+BASELINES:
    with (ROOT/f'experiments/results/models/{task}_{seed}_{kind}.pkl').open('rb') as f:r=pickle.load(f)
    if kind in ['fuzzy','linear']:
     for name,p in r['portfolios'].items():
      nm=('sfr' if kind=='fuzzy' else 'linear_portfolio') if name=='survival' else name
      models.append(dict(task=task,seed=seed,model=nm,experts=len(p['ids']),rules=p['rules'],parameters=p['parameters'],weights=len(p['weights']),certificate=p['certificate'],seconds=r['seconds'],validation_score=p['validation_score'],supports=json.dumps([r['bank'].supports[i] for i in p['ids']]),coefficients=json.dumps(p['weights'].tolist())))
    else:
     validation.append(dict(task=task,seed=seed,model=kind,validation_score=r['validation_score'],seconds=r['seconds'],configuration=json.dumps(r['configuration'],default=lambda x:float(x))))
 pd.DataFrame(models).to_csv(OUT/'portfolios.csv',index=False);v=pd.DataFrame(validation);v.to_csv(OUT/'baseline_configuration.csv',index=False)
 selected=v.groupby(['task','seed'],group_keys=False).apply(lambda x:x.loc[x.validation_score.idxmin()]).reset_index(drop=True)
 selected.to_csv(OUT/'validation_selected_baseline.csv',index=False)
 bm=per.merge(selected[['task','seed','model']],on=['task','seed','model']);bm['model']='validation_selected_baseline';bm.to_csv(OUT/'selected_baseline_test.csv',index=False)
 # Outage-level seed-averaged performance, preserving all masks.
 df.groupby(['task','model','k','failed_groups'],as_index=False).agg(rmse=('rmse','mean'),nrmse=('nrmse','mean')).to_csv(OUT/'per_mask.csv',index=False)

if __name__=='__main__':
 summarize();rows=[]
 with ProcessPoolExecutor(4) as ex:
  for f in as_completed([ex.submit(bootstrap_task,t) for t in TASKS]):rows+=f.result();print('completed comparisons',len(rows),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'bootstrap.csv',index=False)
