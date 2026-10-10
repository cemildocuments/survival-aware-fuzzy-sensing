import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from data import TASKS,load_task
from fuzzy import Bank,TSK,predict_portfolio,certificate,loss_from_cov

def main():
 rng=np.random.default_rng(17);X=rng.normal(size=(200,6));y=np.sin(X[:,0])+X[:,2]
 b=Bank([[0,1],[2,3],[4,5]]).fit(X[:140],y[:140],X[140:],y[140:]);ids=[0,1,2];w=np.array([.2,.3,.5]);P=b.predict_bank(X)
 assert certificate(b,ids)==2
 for k in range(3):
  for f in itertools.combinations(range(3),k):
   Z=X.copy()
   for g in f:Z[:,b.groups[g]]=1e6
   p=predict_portfolio(b,b.predict_bank(Z),ids,w,f)
   q=predict_portfolio(b,P,ids,w,f)
   assert np.allclose(p,q,rtol=1e-9,atol=1e-9)
   R=P[:,ids]-y[:,None];C=R.T@R/len(y);gate=b.survive(f)[ids][None,:]
   risk=loss_from_cov(C,gate,w,np.mean(y*y))[0]
   assert np.allclose(risk,np.mean((q-y)**2))
 assert np.all(predict_portfolio(b,P,ids,w,(0,1,2))==0)
 for task in TASKS:
  X,y,g,s,m=load_task(task);assert len(np.unique(np.concatenate(s)))==len(y)
  assert all(s[i][-1]<s[i+1][0] for i in range(3))
  assert m['target'] not in m['features']
  assert len(set(sum(g,[])))==X.shape[1]
 print('PASS: outage invariance, survival certificate, exact covariance risk, all-failed fallback, disjoint ordered splits, target exclusion, complete group partition')
if __name__=='__main__':main()
