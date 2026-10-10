"""Execute the complete fixed protocol; existing trained models are reusable."""
from pathlib import Path
import argparse,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--gpu-workers',type=int,default=2);p.add_argument('--reference-neural',action='store_true',help='Replay published neural weights instead of retraining the GPU comparator');a=p.parse_args()
 (ROOT/'experiments/results').mkdir(exist_ok=True,parents=True)
 run('experiments/src/download_data.py');run('experiments/tests/test_invariants.py')
 if a.reference_neural:run('experiments/src/neural_reference.py')
 cpu=subprocess.Popen([sys.executable,'experiments/src/train.py','--workers',str(a.workers)],cwd=ROOT)
 gpu=subprocess.Popen([sys.executable,'experiments/src/train.py','--gpu','--workers',str(a.gpu_workers)],cwd=ROOT)
 rc=cpu.wait();rg=gpu.wait()
 if rc or rg:raise RuntimeError(f'Training exited with codes {rc}, {rg}')
 run('experiments/src/extra_train.py');run('experiments/src/evaluate.py','--workers',str(a.workers))
 run('experiments/src/monitor.py');run('experiments/src/synthetic.py');run('experiments/src/analyze.py');run('experiments/src/benchmark.py')
 print('Completed all experiments. Summary tables: experiments/results/analysis/')
if __name__=='__main__':main()
