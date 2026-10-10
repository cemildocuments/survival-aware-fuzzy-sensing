"""Sparse reliability-gated Takagi-Sugeno portfolios."""
import itertools,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

class TSK:
 def __init__(self,rules=4,alpha=.001,seed=0):self.rules=rules;self.alpha=alpha;self.seed=seed
 def design(self,X):
  log=-.5*np.sum(((X[:,None,:]-self.centers)/self.widths)**2,axis=2)
  u=softmax(log,axis=1)
  return (u[:,:,None]*np.c_[np.ones(len(X)),X][:,None,:]).reshape(len(X),-1)
 def fit(self,X,y):
  km=KMeans(self.rules,n_init=3,max_iter=100,random_state=self.seed).fit(X)
  self.centers=km.cluster_centers_
  self.widths=np.stack([np.maximum(np.std(X[km.labels_==k],axis=0),.35) for k in range(self.rules)])
  D=self.design(X);self.coef=np.linalg.solve(D.T@D+len(X)*self.alpha*np.eye(D.shape[1]),D.T@y)
  return self
 def predict(self,X):return self.design(X)@self.coef
 def nparams(self):return self.centers.size+self.widths.size+self.coef.size

class Bank:
 def __init__(self,groups,seed=0,linear=False):
  self.groups=groups; self.G=len(groups);self.seed=seed;self.linear=linear
  self.supports=list(itertools.chain.from_iterable(itertools.combinations(range(self.G),k) for k in [1,2]))+[tuple(range(self.G))]
  self.features=[sum([groups[g] for g in s],[]) for s in self.supports]
  self.incidence=np.array([[g in s for g in range(self.G)] for s in self.supports],bool)
 def fit(self,X,y,Xc,yc):
  self.models=[];self.chosen=[]
  for b,cols in enumerate(self.features):
   best=(np.inf,None,None)
   settings=[(1,a) for a in [.0001,.001,.01]] if self.linear else [(r,a) for r in [2,4,8] for a in [.0001,.001,.01]]
   for r,a in settings:
    model=Ridge(alpha=len(X)*a) if self.linear else TSK(r,a,self.seed+b*31)
    model.fit(X[:,cols],y)
    loss=np.mean((model.predict(Xc[:,cols])-yc)**2)
    if loss<best[0]:best=(loss,model,(r,a))
   self.models.append(best[1]);self.chosen.append(best[2])
  return self
 def predict_bank(self,X):return np.column_stack([m.predict(X[:,c]) for m,c in zip(self.models,self.features)])
 def survive(self,failed):
  return ~self.incidence[:,list(failed)].any(axis=1) if len(failed) else np.ones(len(self.supports),bool)
 def parameters(self,selected):
  return sum((len(self.features[i])+1 if self.linear else self.models[i].nparams()) for i in selected)

def masks(G,k):return list(itertools.combinations(range(G),k))

def loss_from_cov(C,gate,w,priorloss):
 v=gate*w[None,:];z=v.sum(axis=1);out=np.full(len(gate),priorloss)
 good=z>1e-12
 out[good]=np.einsum('si,ij,sj->s',v[good],C,v[good])/z[good]**2
 return out

def objective(C,gate,w,priorloss,robust=True):
 e=loss_from_cov(C,gate,w,priorloss)
 return .5*e[0]+.5*e[1:].mean()+.15*e[1:].max() if robust else e[0]

def optimize_weights(C,gate,ids,priorloss,robust=True):
 ids=np.array(ids);c=C[np.ix_(ids,ids)];g=gate[:,ids];k=len(ids)
 def fun(w):return objective(c,g,w,priorloss,robust)
 res=minimize(fun,np.full(k,1/k),method='SLSQP',bounds=[(min(.02,.5/k),1)]*k,constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':100,'ftol':1e-9})
 w=res.x if res.success else np.full(k,1/k)
 return w,fun(w)

def select_portfolio(bank,P,y,budget=8,robust=True,coverage=True,seed=0,random=False):
 R=P-y[:,None];C=R.T@R/len(y);priorloss=float(np.mean(y*y));gate=np.stack([bank.survive(())]+[bank.survive((g,)) for g in range(bank.G)])
 B=len(bank.supports);budget=min(budget,B)
 if random:
  rng=np.random.default_rng(seed); ids=list(rng.choice(bank.G,3,replace=False));rest=[j for j in range(B) if j not in ids];ids+=list(rng.choice(rest,budget-3,replace=False))
  w,_=optimize_weights(C,gate,ids,priorloss,robust)
  return ids,w
 if coverage:
  choices=list(itertools.combinations(range(bank.G),3))
  vals=[objective(C[np.ix_(c,c)],gate[:,c],np.ones(3)/3,priorloss,robust) for c in choices]
  ids=list(choices[int(np.argmin(vals))])
 else:
  vals=[objective(C[j:j+1,j:j+1],gate[:,j:j+1],np.ones(1),priorloss,robust) for j in range(B)];ids=[int(np.argmin(vals))]
 while len(ids)<budget:
  opts=[]
  for j in range(B):
   if j in ids:continue
   c=ids+[j];w,v=optimize_weights(C,gate,c,priorloss,robust);opts.append((v,j,w))
  _,j,w=min(opts,key=lambda x:x[0]);ids.append(j)
 w,_=optimize_weights(C,gate,ids,priorloss,robust)
 return ids,w

def predict_portfolio(bank,P,ids,w,failed=(),gate=True):
 a=bank.survive(failed)[ids] if gate else np.ones(len(ids));v=w*a
 return P[:,ids]@v/v.sum() if v.sum()>1e-12 else np.zeros(len(P))

def predict_variable(bank,P,ids,w,availability):
 g=np.ones((len(P),len(ids)))
 for j,b in enumerate(ids):g[:,j]=availability[:,bank.supports[b]].prod(axis=1)
 v=g*w;den=v.sum(axis=1);return np.divide((P[:,ids]*v).sum(axis=1),den,out=np.zeros(len(P)),where=den>1e-12)

def certificate(bank,ids):
 for k in range(1,bank.G+1):
  if any(not bank.survive(f)[ids].any() for f in masks(bank.G,k)):return k-1
 return bank.G
