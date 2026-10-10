import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import itertools,json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from threadpoolctl import threadpool_limits
from data import ROOT
from fuzzy import Bank,select_portfolio,predict_portfolio
from baselines import train_baseline

def run(job):
 rho,seed=job;rng=np.random.default_rng(seed);n=6000;z=rng.normal(size=n);groups=[[2*i,2*i+1] for i in range(6)];parts=[]
 for g in range(6):
  a=1 if g<2 else rho
  parts.append(np.column_stack([a*z+.25*rng.normal(size=n),a*np.sin(z)+.25*rng.normal(size=n)]))
 X=np.column_stack(parts);y=np.sin(2*z)+.3*z**2+.1*rng.normal(size=n)
 fit=np.arange(3000);cal=np.arange(3000,3900);val=np.arange(3900,4800);test=np.arange(4800,6000)
 xm=X[fit].mean(0);xs=X[fit].std(0);ym=y[fit].mean();ys=y[fit].std();X=(X-xm)/xs;y=(y-ym)/ys
 with threadpool_limits(1):
  bank=Bank(groups,seed).fit(X[fit],y[fit],X[cal],y[cal]);Pc=bank.predict_bank(X[cal]);P=bank.predict_bank(X[test]);port={}
  for name,robust,cov in [('sfr',True,True),('clean_select',False,False),('no_cover',True,False)]:port[name]=select_portfolio(bank,Pc,y[cal],6,robust,cov,seed)
  model,cfg,loss=train_baseline('xgb_group',X[np.r_[fit,cal]],y[np.r_[fit,cal]],X[val],y[val],groups,seed)
  rows=[]
  for k in range(5):
   for f in itertools.combinations(range(6),k):
    predictions={name:predict_portfolio(bank,P,ids,w,f) for name,(ids,w) in port.items()};predictions['xgb_group']=model.predict(X[test],f)
    for name,p in predictions.items():rows.append({'rho':rho,'seed':seed,'model':name,'k':k,'failed_groups':str(f),'nrmse':np.mean((p-y[test])**2)**.5})
 return rows
if __name__=='__main__':
 rows=[]
 with ProcessPoolExecutor(4) as ex:
  for f in as_completed([ex.submit(run,(rho,s)) for rho in [0,.3,.6,.9] for s in [11,23,37,51,71]]):rows+=f.result();print(len(rows),flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'experiments/results/synthetic.csv',index=False)
