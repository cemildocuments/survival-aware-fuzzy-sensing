import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,itertools,json,pickle,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from data import ROOT,TASKS,load_task
from train import SEEDS
from fuzzy import predict_portfolio

KINDS=['fuzzy','linear','xgb_clean','xgb_group','xgb_independent','rf_group','ridge_group','impute_xgb','tsk_full','tsk_group','mlp_group']
def evaluate(job):
 task,seed=job;folder=ROOT/'experiments/results/evaluation';folder.mkdir(parents=True,exist_ok=True);dest=folder/f'{task}_{seed}.csv'
 if dest.exists():return dest.name+' cached'
 with threadpool_limits(1):
  X,y,g,s,meta=load_task(task);test=s[3];Xraw=X[test];yraw=y[test];G=len(g)
  scenarios=list(itertools.chain.from_iterable(itertools.combinations(range(G),k) for k in range(4)))
  rows=[];out={'truth':yraw.astype(np.float32),'mask_codes':np.array([sum(1<<g for g in f) for f in scenarios]),'severity':np.array([len(f) for f in scenarios])}
  for kind in KINDS:
   path=ROOT/'experiments/results/models'/f'{task}_{seed}_{kind}.pkl'
   with path.open('rb') as fp:r=pickle.load(fp)
   sc=r['scale'];Z=(Xraw-sc['x_mean'])/sc['x_scale'];ys=sc['y_scale'];ym=sc['y_mean']
   if kind in ['fuzzy','linear']:
    bank=r['bank'];P=bank.predict_bank(Z)
    predictions={('sfr' if kind=='fuzzy' else 'linear_portfolio') if name=='survival' else name:np.stack([predict_portfolio(bank,P,p['ids'],p['weights'],f) for f in scenarios]) for name,p in r['portfolios'].items()}
    if kind=='fuzzy':
     pp=r['portfolios']['survival'];raw=[]
     for f in scenarios:
      Zf=Z.copy()
      for gg in f:Zf[:,g[gg]]=0
      raw.append(predict_portfolio(bank,bank.predict_bank(Zf),pp['ids'],pp['weights'],(),False))
     predictions['no_gate']=np.stack(raw)
   else:predictions={kind:np.stack([r['model'].predict(Z,f) for f in scenarios])}
   for name,p in predictions.items():
    p=p*ys+ym;out[name]=p.astype(np.float32)
    for j,f in enumerate(scenarios):
     err=p[j]-yraw
     rows.append({'task':task,'seed':seed,'model':name,'failed_groups':'+'.join(map(str,f)) or 'clean','k':len(f),'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(np.abs(err))),'nrmse':float(np.sqrt(np.mean(err**2))/ys),'n':len(yraw),'fit_seconds':r['seconds']})
  # Fitted training-mean predictor is a real reference in physical units.
  p=np.full_like(yraw,ym)
  for f in scenarios:
   err=p-yraw;rows.append({'task':task,'seed':seed,'model':'mean','failed_groups':'+'.join(map(str,f)) or 'clean','k':len(f),'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(np.abs(err))),'nrmse':float(np.sqrt(np.mean(err**2))/ys),'n':len(yraw),'fit_seconds':0.})
  np.savez_compressed(folder/f'{task}_{seed}_predictions.npz',**out)
  pd.DataFrame(rows).to_csv(dest,index=False)
  return f'{task} seed={seed} masks={len(scenarios)} comparisons={len(rows)}'

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=6);args=a.parse_args()
 missing=[ROOT/'experiments/results/models'/f'{t}_{s}_{k}.pkl' for t in TASKS for s in SEEDS for k in KINDS if not (ROOT/'experiments/results/models'/f'{t}_{s}_{k}.pkl').exists()]
 if missing:raise RuntimeError(f'{len(missing)} training jobs unfinished; evaluation requires all models')
 with ProcessPoolExecutor(args.workers) as ex:
  for f in as_completed([ex.submit(evaluate,(t,s)) for t in TASKS for s in SEEDS]):print(f.result(),flush=True)
 dfs=[pd.read_csv(ROOT/'experiments/results/evaluation'/f'{t}_{s}.csv') for t in TASKS for s in SEEDS]
 pd.concat(dfs,ignore_index=True).to_csv(ROOT/'experiments/results/all_metrics.csv',index=False)
