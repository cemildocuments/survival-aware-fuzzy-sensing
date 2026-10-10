import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,pickle,hashlib,resource
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from concurrent.futures import ProcessPoolExecutor,as_completed
from data import ROOT,TASKS,load_task
from fuzzy import Bank,select_portfolio,predict_portfolio,certificate,optimize_weights
from baselines import train_baseline

SEEDS=[11,23,37,51,71]

def prepare(task):
 X,y,g,s,meta=load_task(task);fit,cal,val,test=s
 mu=X[fit].mean(0);sd=np.maximum(X[fit].std(0),1e-8);ym=y[fit].mean();ys=y[fit].std()
 return (X-mu)/sd,(y-ym)/ys,g,s,meta,{'x_mean':mu,'x_scale':sd,'y_mean':ym,'y_scale':ys}

def scorep(bank,P,y,ids,w):
 e=[np.mean((predict_portfolio(bank,P,ids,w,f)-y)**2) for f in [()]+[(g,) for g in range(bank.G)]]
 return .5*e[0]+.5*np.mean(e[1:])+.15*max(e[1:])

def run(job):
 task,seed,kind=job;folder=ROOT/'experiments/results/models';folder.mkdir(parents=True,exist_ok=True);path=folder/f'{task}_{seed}_{kind}.pkl'
 if path.exists():return str(path.name)+' cached'
 start=time.perf_counter()
 with threadpool_limits(1):
  X,y,g,s,meta,scale=prepare(task);fit,cal,val,test=s
  result={'task':task,'seed':seed,'kind':kind,'meta':meta,'scale':scale}
  if kind in ['fuzzy','linear']:
   bank=Bank(g,seed,kind=='linear').fit(X[fit],y[fit],X[cal],y[cal]);Pc=bank.predict_bank(X[cal]);Pv=bank.predict_bank(X[val]);portfolios={};trace={}
   for name,robust,coverage,random in [('survival',True,True,False),('clean_select',False,False,False),('no_cover',True,False,False),('random',True,True,True)]:
    if kind=='linear' and name!='survival':continue
    best=(np.inf,None)
    for budget in [4,6,8]:
     ids,w=select_portfolio(bank,Pc,y[cal],budget,robust,coverage,seed,random);score=scorep(bank,Pv,y[val],ids,w)
     item={'ids':ids,'weights':w,'validation_score':score,'certificate':certificate(bank,ids),'parameters':bank.parameters(ids),'rules':sum(bank.chosen[i][0] for i in ids)}
     trace[f'{name}_{budget}']=item
     if score<best[0]:best=score,item
    portfolios[name]=best[1]
   if kind=='fuzzy':
    ids=list(range(len(bank.supports)));R=Pc-y[cal,None];C=R.T@R/len(cal);gate=np.stack([bank.survive(())]+[bank.survive((i,)) for i in range(len(g))]);w,_=optimize_weights(C,gate,ids,np.mean(y[cal]**2),True)
    portfolios['full_bank']={'ids':ids,'weights':w,'validation_score':scorep(bank,Pv,y[val],ids,w),'certificate':certificate(bank,ids),'parameters':bank.parameters(ids),'rules':sum(z[0] for z in bank.chosen)}
   result.update(bank=bank,portfolios=portfolios,trace=trace)
  else:
   if kind=='mlp_group':
    from neural import train_neural
    model,cfg,score=train_neural(X[np.r_[fit,cal]],y[np.r_[fit,cal]],X[val],y[val],g,seed)
   else:model,cfg,score=train_baseline(kind,X[np.r_[fit,cal]],y[np.r_[fit,cal]],X[val],y[val],g,seed)
   result.update(model=model,configuration=cfg,validation_score=score)
  result['seconds']=time.perf_counter()-start;result['max_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  with path.open('wb') as f:pickle.dump(result,f,protocol=5)
  return f'{task} seed={seed} {kind} seconds={result["seconds"]:.2f}'

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=6);a.add_argument('--gpu',action='store_true');a.add_argument('--tasks',nargs='*',default=TASKS);args=a.parse_args()
 kinds=['mlp_group'] if args.gpu else ['fuzzy','linear','xgb_clean','xgb_group','xgb_independent','rf_group','ridge_group','impute_xgb','tsk_full']
 jobs=[(t,s,k) for t in args.tasks for s in SEEDS for k in kinds]
 with ProcessPoolExecutor(args.workers) as pool:
  for future in as_completed([pool.submit(run,j) for j in jobs]):print(future.result(),flush=True)
