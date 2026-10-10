"""Restore the published neural weights for accelerator-independent replay."""
from pathlib import Path
import json,pickle
import numpy as np
import torch
from torch import nn
from neural import NeuralModel
from data import ROOT

def restore():
 folder=ROOT/'reference/neural_weights';dest=ROOT/'experiments/results/models';dest.mkdir(exist_ok=True,parents=True)
 for p in sorted(folder.glob('*.json')):
  record=json.loads(p.read_text());state=np.load(p.with_suffix('.npz'));g=record['groups'];width=record['configuration']['width'];d=len(record['scale']['x_mean'])
  net=nn.Sequential(nn.Linear(d+len(g),width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
  net.load_state_dict({k:torch.tensor(state[k]) for k in state.files});record.pop('groups');record['model']=NeuralModel(net,g)
  for k in ['x_mean','x_scale']:record['scale'][k]=np.array(record['scale'][k])
  with (dest/(p.stem+'.pkl')).open('wb') as f:pickle.dump(record,f,protocol=5)
  print('Restored',p.stem)
if __name__=='__main__':restore()
