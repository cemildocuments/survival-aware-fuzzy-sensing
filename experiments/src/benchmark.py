"""Serial inference timings and complete structural coverage audits."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import pickle,time,itertools,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from data import ROOT,TASKS,load_task
from train import SEEDS

def predict_active(r,X,failed=()):
 b=r['bank'];p=r['portfolios']['survival'];active=[(j,w) for j,w in zip(p['ids'],p['weights']) if not set(b.supports[j]).intersection(failed)]
 if not active:return np.zeros(len(X))
 return sum(w*b.models[j].predict(X[:,b.features[j]]) for j,w in active)/sum(w for j,w in active)

def main():
 rows=[];cover=[]
 with threadpool_limits(1):
  for task in TASKS:
   X,y,g,s,meta=load_task(task)
   for seed in SEEDS:
    with (ROOT/f'experiments/results/models/{task}_{seed}_fuzzy.pkl').open('rb') as f:r=pickle.load(f)
    for name,p in r['portfolios'].items():
     for k in range(4):
      n=0;total=0;surv=[]
      for fail in itertools.combinations(range(len(g)),k):
       z=int(r['bank'].survive(fail)[p['ids']].sum());n+=int(z==0);total+=1;surv.append(z)
      cover.append(dict(task=task,seed=seed,model='sfr' if name=='survival' else name,k=k,fallback_masks=n,total_masks=total,minimum_survivors=min(surv),mean_survivors=np.mean(surv)))
   for kind in ['fuzzy','linear','xgb_group','rf_group','ridge_group','tsk_group','mlp_group','impute_xgb']:
    with (ROOT/f'experiments/results/models/{task}_11_{kind}.pkl').open('rb') as f:r=pickle.load(f)
    sc=r['scale'];Z=(X[s[3]][:1024]-sc['x_mean'])/sc['x_scale']
    for failed in [(),(0,1)]:
     fun=(lambda:predict_active(r,Z,failed)) if kind in ['fuzzy','linear'] else (lambda:r['model'].predict(Z,failed))
     for _ in range(2):fun()
     durations=[]
     for _ in range(9):
      start=time.perf_counter();fun();durations.append(time.perf_counter()-start)
     rows.append(dict(task=task,model={'fuzzy':'sfr','linear':'linear_portfolio'}.get(kind,kind),k=len(failed),batch=len(Z),milliseconds=1000*np.median(durations),q25=1000*np.quantile(durations,.25),q75=1000*np.quantile(durations,.75),seed=11))
    print(task,kind,flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'experiments/results/timing.csv',index=False)
 pd.DataFrame(cover).to_csv(ROOT/'experiments/results/coverage.csv',index=False)
if __name__=='__main__':main()
