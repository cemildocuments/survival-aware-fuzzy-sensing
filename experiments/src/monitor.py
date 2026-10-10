import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import itertools,pickle
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from data import ROOT,TASKS,load_task
from train import SEEDS
from fuzzy import predict_variable

def run(job):
 task,seed=job;X,y,g,s,meta=load_task(task);test=s[3];X=X[test];y=y[test];G=len(g);rows=[]
 with threadpool_limits(1):
  models={}
  for k in ['fuzzy','xgb_group','mlp_group','tsk_group']:
   with (ROOT/'experiments/results/models'/f'{task}_{seed}_{k}.pkl').open('rb') as f:models[k]=pickle.load(f)
  sc=models['fuzzy']['scale'];Z=(X-sc['x_mean'])/sc['x_scale'];ys=sc['y_scale'];ym=sc['y_mean'];n=len(Z)
  rng=np.random.default_rng(1000+seed);pairs=list(itertools.combinations(range(G),2));order=rng.permutation(len(pairs));truth=np.zeros((n,G),bool)
  blocks=(n+63)//64
  for b in range(blocks):truth[b*64:min((b+1)*64,n),list(pairs[order[b%len(pairs)]])]=True
  corrupt=Z.copy()
  for gg,c in enumerate(g):corrupt[:,c]+=3*truth[:,gg,None]
  # Paired randomness makes adjacent monitor settings directly comparable.
  u=np.repeat(rng.random((blocks,G)),64,axis=0)[:n]
  for sensitivity in [0,.25,.5,.75,1]:
   for fpr in [0,.02,.1]:
    flagged=np.where(truth,u<sensitivity,u<fpr);availability=(~flagged).astype(float)
    masked=corrupt.copy()
    for gg,c in enumerate(g):masked[np.ix_(flagged[:,gg],c)]=0
    r=models['fuzzy'];p=r['portfolios']['survival'];P=r['bank'].predict_bank(masked)
    predictions={'sfr':predict_variable(r['bank'],P,p['ids'],p['weights'],availability)}
    for k in ['xgb_group','mlp_group','tsk_group']:predictions[k]=models[k]['model'].predict(corrupt,availability=availability)
    for name,pred in predictions.items():
     err=pred*ys+ym-y;rows.append({'task':task,'seed':seed,'model':name,'sensitivity':sensitivity,'false_alarm':fpr,'rmse':np.mean(err**2)**.5,'nrmse':np.mean(err**2)**.5/ys,'mae':np.abs(err).mean()})
  # An episode trace for exactly the same fixed protocol, not an optimized example.
  if seed==11:
   flag=truth;av=(~flag).astype(float);masked=corrupt.copy()
   for gg,c in enumerate(g):masked[np.ix_(flag[:,gg],c)]=0
   r=models['fuzzy'];p=r['portfolios']['survival'];pred=predict_variable(r['bank'],r['bank'].predict_bank(masked),p['ids'],p['weights'],av)*ys+ym
   np.savez_compressed(ROOT/f'experiments/results/{task}_episode.npz',truth=y,prediction=pred,failed=truth)
 return rows
if __name__=='__main__':
 rows=[]
 with ProcessPoolExecutor(5) as ex:
  for f in as_completed([ex.submit(run,(t,s)) for t in TASKS for s in SEEDS]):rows+=f.result();print(len(rows),flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'experiments/results/monitor.csv',index=False)
