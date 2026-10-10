"""Compare regenerated outage scores with the numerical reference release."""
import argparse
import numpy as np
import pandas as pd
from data import ROOT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--rtol',type=float,default=1e-5);a=p.parse_args()
 keys=['task','seed','model','k','failed_groups']
 expected=pd.read_csv(ROOT/'reference/results/all_metrics.csv').set_index(keys).sort_index()
 actual=pd.read_csv(ROOT/'experiments/results/all_metrics.csv').set_index(keys).sort_index()
 assert actual.index.equals(expected.index),'Different experimental rows'
 for name in ['rmse','mae','nrmse']:
  np.testing.assert_allclose(actual[name],expected[name],rtol=a.rtol,atol=1e-7,err_msg=name)
 print(f'PASS: {len(actual)} outage scores match the reference at relative tolerance {a.rtol:g}.')
