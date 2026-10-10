"""GPU-capable group-dropout neural comparator with validation stopping."""
import copy,time
import numpy as np
import torch
from torch import nn

class NeuralModel:
 def __init__(self,net,groups):self.net=net.cpu();self.groups=groups
 def predict(self,X,failed=(),availability=None):
  if availability is None:
   availability=np.ones((len(X),len(self.groups)));availability[:,list(failed)]=0
  Z=X.copy()
  for g,cols in enumerate(self.groups):Z[np.ix_(availability[:,g]<.5,cols)]=0
  self.net.eval()
  with torch.no_grad():return self.net(torch.tensor(np.c_[Z,availability],dtype=torch.float32)).numpy().ravel()

def train_neural(X,y,Xv,yv,groups,seed,device='mps'):
 torch.set_num_threads(1);device=device if device=='cpu' or torch.backends.mps.is_available() else 'cpu'
 Xt=torch.tensor(X,dtype=torch.float32,device=device);yt=torch.tensor(y[:,None],dtype=torch.float32,device=device);G=len(groups);d=X.shape[1]
 index=torch.tensor([next(g for g,c in enumerate(groups) if j in c) for j in range(d)],device=device)
 Zvs=[]
 for f in [()]+[(g,) for g in range(G)]:
  a=np.ones((len(Xv),G));a[:,list(f)]=0;z=Xv.copy()
  for g,c in enumerate(groups):z[:,c]*=a[:,g,None]
  Zvs.append(np.c_[z,a])
 V=torch.tensor(np.vstack(Zvs),dtype=torch.float32,device=device);Yv=torch.tensor(np.tile(yv,G+1),dtype=torch.float32,device=device)
 best=(np.inf,None,None)
 for width in [32,64]:
  for decay in [1e-4,1e-2,1e-1]:
   torch.manual_seed(seed);net=nn.Sequential(nn.Linear(d+G,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1)).to(device)
   opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=decay)
   valbest=np.inf;state=None;wait=0;bestep=0
   for epoch in range(1,151):
    net.train();order=torch.randperm(len(X),device=device)
    for ids in order.split(1024):
     a=torch.ones((len(ids),G),device=device);bad=torch.randint(G,(len(ids),),device=device);rows=torch.arange(len(ids),device=device);drop=torch.rand(len(ids),device=device)<.75;a[rows[drop],bad[drop]]=0
     z=torch.cat([Xt[ids]*a[:,index],a],1);pred=net(z);loss=((pred-yt[ids])**2).mean();opt.zero_grad();loss.backward();opt.step()
    if epoch%5==0:
     net.eval()
     with torch.no_grad():
      # Chunk large inference arrays to avoid unnecessary accelerator allocation.
      p=torch.cat([net(v).ravel() for v in V.split(8192)]);e=((p-Yv)**2).reshape(G+1,len(Xv)).mean(1);score=(.5*e[0]+.5*e[1:].mean()+.15*e[1:].max()).item()
     if score<valbest-1e-6:valbest=score;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};wait=0;bestep=epoch
     else:wait+=1
     if wait>=8:break
   net.load_state_dict(state)
   if valbest<best[0]:best=(valbest,NeuralModel(copy.deepcopy(net),groups),{'width':width,'weight_decay':decay,'epochs':bestep,'device':device})
   del net,opt
 return best[1],best[2],best[0]
