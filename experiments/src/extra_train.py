from train import run,SEEDS,TASKS
from concurrent.futures import ProcessPoolExecutor,as_completed
if __name__=='__main__':
 with ProcessPoolExecutor(2) as pool:
  for f in as_completed([pool.submit(run,(t,s,'tsk_group')) for t in TASKS for s in SEEDS]):print(f.result(),flush=True)
