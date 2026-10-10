"""Public data, deterministic partitions, and sensor-group definitions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
TASKS=['air_co','air_benzene','air_no2','turbine_co','turbine_nox','appliances']

def load_task(task):
 if task.startswith('air_'):
  df=pd.read_csv(ROOT/'data/raw/air_quality/AirQualityUCI.csv',sep=';',decimal=',').dropna(subset=['Date'])
  df['timestamp']=pd.to_datetime(df.Date+' '+df.Time,format='%d/%m/%Y %H.%M.%S')
  df=df.sort_values('timestamp').reset_index(drop=True).replace(-200,np.nan)
  groups=[[s] for s in ['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)']]+[['T','RH','AH']]
  names=['S1','S2','S3','S4','S5','Weather']; target={'air_co':'CO(GT)','air_benzene':'C6H6(GT)','air_no2':'NO2(GT)'}[task]
  unit={'air_co':'mg/m3','air_benzene':'ug/m3','air_no2':'ug/m3'}[task]
  original=len(df);df=df.dropna(subset=sum(groups,[])+[target]).reset_index(drop=True)
  n=len(df);splits=np.split(np.arange(n),[int(.5*n),int(.65*n),int(.8*n)])
  time=df.timestamp.astype(str).to_numpy()
 elif task.startswith('turbine_'):
  frames=[]
  for year in range(2011,2016):
   f=pd.read_csv(ROOT/f'data/raw/gas_turbine/gt_{year}.csv'); f['year']=year;f['row_in_year']=np.arange(len(f));frames.append(f)
  df=pd.concat(frames,ignore_index=True);original=len(df)
  groups=[['AT','AP','AH'],['AFDP','GTEP','CDP'],['TIT','TAT'],['TEY']]
  names=['Ambient','Pressure','Temperature','Energy'];target='CO' if task.endswith('_co') else 'NOX';unit='mg/m3'
  mid=np.where(df.year.to_numpy()==2013)[0];splits=[np.where(df.year.to_numpy()<=2012)[0],mid[:len(mid)//2],mid[len(mid)//2:],np.where(df.year.to_numpy()>=2014)[0]]
  time=(df.year.astype(str)+':'+df.row_in_year.astype(str)).to_numpy()
 else:
  df=pd.read_csv(ROOT/'data/raw/appliances/energydata_complete.csv').sort_values('date').reset_index(drop=True);original=len(df)
  groups=[[f'T{i}',f'RH_{i}'] for i in range(1,10)]+[['T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint']]
  names=[f'Node{i}' for i in range(1,10)]+['Weather'];target='Appliances';unit='Wh'
  n=len(df);splits=np.split(np.arange(n),[int(.5*n),int(.65*n),int(.8*n)]);time=df.date.to_numpy()
 features=sum(groups,[]);ixgroups=[[features.index(s) for s in group] for group in groups]
 X=df[features].to_numpy(float);y=df[target].to_numpy(float)
 assert np.isfinite(X).all() and np.isfinite(y).all()
 meta={'task':task,'target':target,'unit':unit,'original_rows':original,'retained_rows':len(df),'features':features,'groups':groups,'group_names':names,'split_sizes':[len(x) for x in splits],'split_ranges':[[str(time[ix[0]]),str(time[ix[-1]])] for ix in splits]}
 return X,y,ixgroups,splits,meta

def save_metadata():
 d=ROOT/'data/processed';d.mkdir(exist_ok=True)
 for task in TASKS:
  X,y,g,s,m=load_task(task);(d/f'{task}.json').write_text(json.dumps(m,indent=2))
  np.savez_compressed(d/f'{task}_splits.npz',fit=s[0],calibration=s[1],validation=s[2],test=s[3])
  print(task,m['retained_rows'],m['split_sizes'])
if __name__=='__main__':save_metadata()
