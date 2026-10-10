"""Predictors trained with equal information about sensor availability."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from fuzzy import TSK

class InputModel:
 def __init__(self,model,groups,kind='zero',imputer=None):self.model=model;self.groups=groups;self.kind=kind;self.imputer=imputer
 def transform(self,X,availability=None):
  if availability is None:availability=np.ones((len(X),len(self.groups)))
  Z=X.copy()
  for g,cols in enumerate(self.groups):Z[np.ix_(availability[:,g]<.5,cols)]=np.nan if self.kind in ['native','iterative'] else 0.
  if self.imputer is not None:Z=self.imputer.transform(Z)
  return np.c_[Z,availability] if self.kind!='fuzzy' else Z
 def predict(self,X,failed=(),availability=None):
  if availability is None:
   availability=np.ones((len(X),len(self.groups)));availability[:,list(failed)]=0
  return self.model.predict(self.transform(X,availability))

def augment(X,y,groups,seed,mode='group'):
 rng=np.random.default_rng(seed);parts=[X.copy()];av=[np.ones((len(X),len(groups)))];ys=[y]
 for _ in range(3):
  Z=X.copy();a=np.ones((len(X),len(groups)))
  if mode=='group':
   bad=rng.integers(len(groups),size=len(X))
   for g,cols in enumerate(groups):
    rows=bad==g;Z[np.ix_(rows,cols)]=np.nan;a[rows,g]=0
  else:
   missing=rng.random(X.shape)<1/len(groups);Z[missing]=np.nan
   for g,cols in enumerate(groups):a[:,g]=(~missing[:,cols]).mean(axis=1)
  parts.append(Z);av.append(a);ys.append(y)
 return np.c_[np.vstack(parts),np.vstack(av)],np.concatenate(ys)

def validation_score(model,X,y,G):
 losses=[np.mean((model.predict(X)-y)**2)]+[np.mean((model.predict(X,(g,))-y)**2) for g in range(G)]
 return .5*losses[0]+.5*np.mean(losses[1:])+.15*max(losses[1:])

def train_baseline(name,X,y,Xv,yv,groups,seed):
 best=(np.inf,None,None);G=len(groups)
 if name in ['xgb_group','xgb_independent']:
  Z,Y=augment(X,y,groups,seed,'group' if name=='xgb_group' else 'independent')
 else:Z,Y=np.c_[X,np.ones((len(X),G))],y
 if name.startswith('xgb'):
  for depth in [3,5,7]:
   for reg in [1.,10.]:
    m=XGBRegressor(n_estimators=350,max_depth=depth,learning_rate=.05,reg_lambda=reg,min_child_weight=5,subsample=.9,colsample_bytree=1,tree_method='hist',max_bin=128,n_jobs=1,random_state=seed).fit(Z,Y)
    model=InputModel(m,groups,'native');score=validation_score(model,Xv,yv,G)
    if score<best[0]:best=(score,model,{'depth':depth,'reg_lambda':reg,'trees':350})
 elif name=='rf_group':
  Z,Y=augment(X,y,groups,seed);Z=np.nan_to_num(Z)
  for leaf in [2,5,15]:
   for frac in [.5,1.]:
    m=RandomForestRegressor(n_estimators=200,min_samples_leaf=leaf,max_features=frac,n_jobs=1,random_state=seed).fit(Z,Y)
    model=InputModel(m,groups);score=validation_score(model,Xv,yv,G)
    if score<best[0]:best=(score,model,{'leaf':leaf,'features':frac,'trees':200})
 elif name=='ridge_group':
  Z,Y=augment(X,y,groups,seed);Z=np.nan_to_num(Z)
  for alpha in [.001,.01,.1,1,10,100]:
   m=Ridge(alpha=alpha*len(Z)).fit(Z,Y);model=InputModel(m,groups);score=validation_score(model,Xv,yv,G)
   if score<best[0]:best=(score,model,{'alpha':alpha})
 elif name=='impute_xgb':
  imputer=IterativeImputer(max_iter=10,random_state=seed,sample_posterior=False,skip_complete=False).fit(X)
  for depth in [3,5,7]:
   for reg in [1.,10.]:
    m=XGBRegressor(n_estimators=350,max_depth=depth,learning_rate=.05,reg_lambda=reg,min_child_weight=5,subsample=.9,tree_method='hist',max_bin=128,n_jobs=1,random_state=seed).fit(Z,Y)
    model=InputModel(m,groups,'iterative',imputer);score=validation_score(model,Xv,yv,G)
    if score<best[0]:best=(score,model,{'depth':depth,'reg_lambda':reg})
 elif name in ['tsk_full','tsk_group']:
  if name=='tsk_group':
   ZX,YX=augment(X,y,groups,seed);ZX=np.nan_to_num(ZX)
  else:ZX,YX=X,y
  for r in [4,8,16]:
   for alpha in [.0001,.001]:
    m=TSK(r,alpha,seed).fit(ZX,YX);model=InputModel(m,groups,'fuzzy' if name=='tsk_full' else 'zero');score=validation_score(model,Xv,yv,G)
    if score<best[0]:best=(score,model,{'rules':r,'alpha':alpha})
 else:raise ValueError(name)
 return best[1],best[2],best[0]
